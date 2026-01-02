from absl import app, flags, logging
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

main_dir = Path(os.environ.get('MAIN_DIR', '.'))

flags.DEFINE_string("llm_model_name", "gpt2", "Model name.")
flags.DEFINE_integer("ckpt_step", -1, "Checkpoint step.")
flags.DEFINE_integer("llm_layer", 5, "Layer to analyze.")
flags.DEFINE_string("dataset_name", "truthfulqa", "Dataset name.")
flags.DEFINE_float("density", 0.05, "Threshold density for adjacency.")
flags.DEFINE_string("disease_pattern", "epilepsy_like", "Disease preset.")
flags.DEFINE_integer("num_clusters", 8, "Clusters used for module-based metrics.")
FLAGS = flags.FLAGS


def _threshold_adj(C: np.ndarray, density: float) -> np.ndarray:
    A = C.copy()
    thr = np.percentile(np.abs(A), 100 - density * 100)
    A[np.abs(A) < thr] = 0.0
    np.fill_diagonal(A, 0.0)
    return (A != 0).astype(np.int8)


def _cluster_labels(n: int, k: int) -> np.ndarray:
    x = np.linspace(0, 1, n).reshape(-1, 1)
    km = KMeans(n_clusters=k, n_init=5, random_state=42)
    return km.fit_predict(x)


def _segregation(C: np.ndarray, labels: np.ndarray) -> float:
    same = labels.reshape(-1, 1) == labels.reshape(1, -1)
    W = np.abs(C)
    within = W[same]
    return float(within.mean()) if within.size else 0.0


def _aggregation(C: np.ndarray, labels: np.ndarray) -> float:
    same = labels.reshape(-1, 1) == labels.reshape(1, -1)
    W = np.abs(C)
    between = W[~same]
    return float(between.mean()) if between.size else 0.0


def _clustering_coeff(A: np.ndarray) -> float:
    n = A.shape[0]
    # Average local clustering: triangles over possible triples
    coeffs = []
    for i in range(n):
        nbrs = np.where(A[i] == 1)[0]
        k = len(nbrs)
        if k < 2:
            coeffs.append(0.0)
            continue
        sub = A[np.ix_(nbrs, nbrs)]
        triangles = sub.sum() // 2  # undirected
        coeffs.append(float(triangles) / (k * (k - 1)))
    return float(np.mean(coeffs)) if coeffs else 0.0


def _avg_shortest_path(A: np.ndarray) -> float:
    # Unweighted distances
    G = csr_matrix(A)
    D = shortest_path(G, directed=False, unweighted=True)
    # Exclude inf distances
    finite = D[np.isfinite(D)]
    if finite.size == 0:
        return float('inf')
    # Exclude zeros on diagonal
    finite = finite[finite > 0]
    return float(finite.mean()) if finite.size else float('inf')


def _small_worldness(A: np.ndarray) -> float:
    C = _clustering_coeff(A)
    L = _avg_shortest_path(A)
    n = A.shape[0]
    p = A.sum() / (n * (n - 1))  # density approx
    # ER baseline
    Ar = (np.random.rand(n, n) < p).astype(np.int8)
    Ar = np.triu(Ar, 1)
    Ar = Ar + Ar.T
    Cr = _clustering_coeff(Ar)
    Lr = _avg_shortest_path(Ar)
    if Cr == 0 or Lr == 0 or not np.isfinite(L) or not np.isfinite(Lr):
        return 0.0
    return float((C / Cr) / (L / Lr))


def _modularity(A: np.ndarray, labels: np.ndarray) -> float:
    # Newman-Girvan modularity on unweighted adjacency
    m = A.sum() / 2.0
    if m == 0:
        return 0.0
    k = A.sum(axis=1)
    Q = 0.0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if labels[i] == labels[j]:
                Q += (A[i, j] - (k[i] * k[j]) / (2.0 * m))
    return float(Q / (2.0 * m))


def _load_corr_paths(dataset_dir: Path, llm_layer: int):
    healthy = {}
    patho = {}
    for q_dir in dataset_dir.iterdir():
        if q_dir.is_dir():
            h = q_dir / f"layer_{llm_layer}_corr.npy"
            p = q_dir / f"layer_{llm_layer}_corr_patho_{FLAGS.disease_pattern}.npy"
            if h.exists() and p.exists():
                q = int(q_dir.name)
                healthy[q] = h
                patho[q] = p
    keys = sorted(set(healthy.keys()) & set(patho.keys()))
    return [(k, healthy[k], patho[k]) for k in keys]


def main(_):
    logging.info("="*10)
    logging.info("Neuropathology Graph Metrics (healthy vs pathological)")
    logging.info("="*10)

    sanitized_model_name = FLAGS.llm_model_name.replace('/', '_').replace('-', '_').replace('.', '_')
    if FLAGS.ckpt_step == -1:
        model_dir = sanitized_model_name
    else:
        model_dir = f"{sanitized_model_name}_step{FLAGS.ckpt_step}"
    dataset_dir = main_dir / "data/hallucination" / FLAGS.dataset_name / model_dir
    reports_dir = main_dir / "reports" / "neuropathology_analysis" / FLAGS.disease_pattern
    os.makedirs(reports_dir, exist_ok=True)

    triples = _load_corr_paths(dataset_dir, FLAGS.llm_layer)
    if not triples:
        logging.error("No matching healthy/pathological matrices found.")
        return

    logging.info(f"Found {len(triples)} sample pairs to analyze.")

    metrics_rows = []
    gsum_h = None
    gsum_p = None

    for q_idx, h_path, p_path in tqdm(triples, desc="Computing graph metrics", unit="sample"):
        Ch = np.load(h_path)
        Cp = np.load(p_path)
        labels = _cluster_labels(Ch.shape[0], FLAGS.num_clusters)
        Ah = _threshold_adj(Ch, FLAGS.density)
        Ap = _threshold_adj(Cp, FLAGS.density)

        row = {
            "question_id": q_idx,
            "segregation_healthy": _segregation(Ch, labels),
            "segregation_patho": _segregation(Cp, labels),
            "aggregation_healthy": _aggregation(Ch, labels),
            "aggregation_patho": _aggregation(Cp, labels),
            "clustering_healthy": _clustering_coeff(Ah),
            "clustering_patho": _clustering_coeff(Ap),
            "pathlen_healthy": _avg_shortest_path(Ah),
            "pathlen_patho": _avg_shortest_path(Ap),
            "smallworld_healthy": _small_worldness(Ah),
            "smallworld_patho": _small_worldness(Ap),
            "modularity_healthy": _modularity(Ah, labels),
            "modularity_patho": _modularity(Ap, labels),
        }
        metrics_rows.append(row)
        gsum_h = Ch if gsum_h is None else gsum_h + Ch
        gsum_p = Cp if gsum_p is None else gsum_p + Cp

    # Save per-sample metrics
    logging.info(f"Saving metrics for {len(metrics_rows)} samples...")
    df = pd.DataFrame(metrics_rows)
    df.to_csv(reports_dir / f"metrics_layer_{FLAGS.llm_layer}.csv", index=False)
    np.save(reports_dir / f"metrics_layer_{FLAGS.llm_layer}.npy", df.to_dict(orient='list'))

    # Summary stats
    summary = {}
    for col in df.columns:
        if col == "question_id":
            continue
        x = df[col].to_numpy()
        summary[col] = {
            "mean": float(np.nanmean(x)),
            "std": float(np.nanstd(x)),
            "min": float(np.nanmin(x)),
            "max": float(np.nanmax(x)),
            "median": float(np.nanmedian(x)),
        }
    (reports_dir / f"metrics_summary_layer_{FLAGS.llm_layer}.json").write_text(__import__("json").dumps(summary, indent=2), encoding="utf-8")

    # Group FC matrices
    Cg_h = gsum_h / len(metrics_rows)
    Cg_p = gsum_p / len(metrics_rows)
    np.save(reports_dir / f"group_fc_metrics_healthy_layer_{FLAGS.llm_layer}.npy", Cg_h)
    np.save(reports_dir / f"group_fc_metrics_patho_{FLAGS.disease_pattern}_layer_{FLAGS.llm_layer}.npy", Cg_p)

    def _plot_heat(mat, title, path):
        vmax = np.percentile(np.abs(mat), 99) if mat.size else 1.0
        plt.figure(figsize=(6, 5))
        plt.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        plt.colorbar(); plt.title(title)
        plt.tight_layout(); plt.savefig(path); plt.close()

    _plot_heat(Cg_h, f"Group FC Healthy (metrics) layer {FLAGS.llm_layer}", reports_dir / f"group_fc_metrics_healthy_layer_{FLAGS.llm_layer}.png")
    _plot_heat(Cg_p, f"Group FC Patho {FLAGS.disease_pattern} (metrics) layer {FLAGS.llm_layer}", reports_dir / f"group_fc_metrics_patho_{FLAGS.disease_pattern}_layer_{FLAGS.llm_layer}.png")

    # Histograms and boxplots
    def _save_hist_box(df, col_base):
        hcol = f"{col_base}_healthy"; pcol = f"{col_base}_patho"
        if hcol not in df.columns or pcol not in df.columns:
            return
        plt.figure(figsize=(6, 4))
        plt.hist(df[hcol].dropna(), bins=40, alpha=0.6, label="healthy")
        plt.hist(df[pcol].dropna(), bins=40, alpha=0.6, label="patho")
        plt.legend(); plt.title(col_base)
        plt.tight_layout(); plt.savefig(reports_dir / f"hist_{col_base}_layer_{FLAGS.llm_layer}.png"); plt.close()

        plt.figure(figsize=(5, 4))
        data = [df[hcol].dropna().to_numpy(), df[pcol].dropna().to_numpy()]
        plt.boxplot(data, labels=["healthy", "patho"]) ; plt.title(col_base)
        plt.tight_layout(); plt.savefig(reports_dir / f"box_{col_base}_layer_{FLAGS.llm_layer}.png"); plt.close()

    for base in ["segregation", "aggregation", "clustering", "pathlen", "smallworld", "modularity"]:
        _save_hist_box(df, base)

    logging.info("="*10)
    logging.info("✓ Neuropathology graph metrics complete")
    logging.info("="*10)


if __name__ == "__main__":
    app.run(main)
