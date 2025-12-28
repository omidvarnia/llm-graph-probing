from absl import app, flags, logging
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from tqdm import tqdm

from sklearn.cluster import KMeans
from scipy import sparse

import torch
from torch_geometric.utils import dense_to_sparse

from .neuropathology_config import PRESETS, DiseaseConfig


main_dir = Path(os.environ.get('MAIN_DIR', '.'))

flags.DEFINE_string("llm_model_name", "gpt2", "The name of the LLM model.")
flags.DEFINE_integer("ckpt_step", -1, "The checkpoint step.")
flags.DEFINE_integer("llm_layer", 5, "The layer to transform.")
flags.DEFINE_float("density", 0.05, "Network density used for sparsification and visualization.")
flags.DEFINE_string("dataset_name", "truthfulqa", "Dataset name.")

flags.DEFINE_string("disease_pattern", "epilepsy_like", "Disease preset: epilepsy_like, dementia_like, autism_like.")
flags.DEFINE_integer("num_clusters", 8, "Number of clusters for segregation.")
flags.DEFINE_float("within_scale", 1.3, "Scale factor for within-module correlations.")
flags.DEFINE_float("between_scale", 0.5, "Scale factor for between-module correlations.")
flags.DEFINE_float("rewiring_prob", 0.15, "Edge rewiring probability to reduce small-worldness.")
flags.DEFINE_integer("distance_threshold", 50, "Optional index-distance threshold for aggregation decrease.")

flags.DEFINE_boolean("save_sparse", True, "Whether to save sparse pathological graphs.")
flags.DEFINE_boolean("overwrite", False, "Overwrite existing pathological files.")
FLAGS = flags.FLAGS


def _save_heatmap(mat: np.ndarray, title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vmax = np.percentile(np.abs(mat), 99) if mat.size else 1.0
    plt.figure(figsize=(6, 5))
    plt.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _cluster_indices(n: int, num_clusters: int) -> np.ndarray:
    """Cluster neuron indices into modules using 1D position features.

    Returns an array of shape (n,) with integer cluster labels in [0, num_clusters).
    """
    x = np.linspace(0, 1, n).reshape(-1, 1)
    km = KMeans(n_clusters=num_clusters, n_init=5, random_state=42)
    labels = km.fit_predict(x)
    return labels


def increase_segregation(C: np.ndarray, labels: np.ndarray, within_scale: float, between_scale: float) -> np.ndarray:
    """Strengthen within-module correlations and weaken between-module correlations.

    Args:
        C: Healthy correlation matrix (n x n).
        labels: Cluster labels per node.
        within_scale: Multiplier for within-cluster entries (>1 increases segregation).
        between_scale: Multiplier for between-cluster entries (<1 decreases integration).
    Returns:
        Transformed correlation matrix.
    """
    C_new = C.copy()
    same = labels.reshape(-1, 1) == labels.reshape(1, -1)
    C_new[same] *= within_scale
    C_new[~same] *= between_scale
    np.fill_diagonal(C_new, 1.0)
    C_new = np.clip(C_new, -1.0, 1.0)
    return C_new


def decrease_aggregation(C: np.ndarray, labels: np.ndarray, scaling: float, distance_threshold: int | None) -> np.ndarray:
    """Reduce long-range connectivity strength between modules.

    If `distance_threshold` is provided, edges with |i-j| > threshold are scaled.
    Otherwise, between-cluster edges are scaled.
    """
    C_new = C.copy()
    n = C.shape[0]
    if distance_threshold is not None:
        I, J = np.ogrid[:n, :n]
        mask = np.abs(I - J) > distance_threshold
        C_new[mask] *= scaling
    else:
        same = labels.reshape(-1, 1) == labels.reshape(1, -1)
        C_new[~same] *= scaling
    np.fill_diagonal(C_new, 1.0)
    C_new = np.clip(C_new, -1.0, 1.0)
    return C_new


def _threshold_to_adj(C: np.ndarray, density: float) -> np.ndarray:
    """Threshold correlation to an unweighted adjacency at given density."""
    A = C.copy()
    percentile = density * 100.0
    thr = np.percentile(np.abs(A), 100 - percentile)
    A[np.abs(A) < thr] = 0.0
    np.fill_diagonal(A, 0.0)
    A_bin = (A != 0).astype(np.int8)
    return A_bin


def reduce_small_worldness(C: np.ndarray, rewiring_prob: float, density: float) -> np.ndarray:
    """Randomly rewire edges of the thresholded adjacency to reduce clustering.

    A simple edge swap scheme is applied to the binary adjacency derived from `C`.
    The resulting adjacency is mapped back to weights by copying corresponding entries
    from the original matrix when available; otherwise, small random weights are assigned.
    """
    rng = np.random.default_rng(42)
    A = _threshold_to_adj(C, density)
    n = A.shape[0]
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] == 1]
    num_swaps = int(len(edges) * rewiring_prob)
    if num_swaps == 0:
        return C.copy()

    # Perform random edge swaps to break local triangles
    for _ in range(num_swaps):
        if len(edges) < 2:
            break
        (u1, v1), (u2, v2) = rng.choice(edges, size=2, replace=False)
        # Propose new pairs (u1, v2) and (u2, v1), avoiding self-loops and duplicates
        if u1 == v2 or u2 == v1:
            continue
        if A[u1, v2] == 0 and A[u2, v1] == 0:
            # Remove old edges
            A[u1, v1] = A[v1, u1] = 0
            A[u2, v2] = A[v2, u2] = 0
            # Add new edges
            A[u1, v2] = A[v2, u1] = 1
            A[u2, v1] = A[v1, u2] = 1
            # Update edge list (approximate)
            edges.remove((u1, v1))
            edges.remove((u2, v2))
            edges.append((min(u1, v2), max(u1, v2)))
            edges.append((min(u2, v1), max(u2, v1)))

    # Map adjacency back to a weighted matrix
    C_new = np.zeros_like(C)
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 1:
                w = C[i, j]
                if w == 0.0:
                    w = rng.normal(loc=0.0, scale=0.05)
                C_new[i, j] = C_new[j, i] = w
    np.fill_diagonal(C_new, 1.0)
    C_new = np.clip(C_new, -1.0, 1.0)
    return C_new


def make_pathological_connectivity(
    C: np.ndarray,
    cfg: DiseaseConfig,
    density: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compose transformations to produce a pathological matrix.

    Returns:
        (C_patho, labels) where labels are module assignments used in metrics.
    """
    n = C.shape[0]
    labels = _cluster_indices(n, cfg.num_clusters)
    C1 = increase_segregation(C, labels, cfg.within_scale, cfg.between_scale)
    C2 = decrease_aggregation(C1, labels, scaling=cfg.between_scale, distance_threshold=cfg.distance_threshold)
    C3 = reduce_small_worldness(C2, rewiring_prob=cfg.rewiring_prob, density=density)
    return C3, labels


def _save_sparse_graph(C: np.ndarray, out_dir: Path, llm_layer: int, density: float, disease_name: str) -> None:
    adj = C.copy()
    percentile_threshold = density * 100
    threshold = np.percentile(np.abs(adj), 100 - percentile_threshold)
    adj[np.abs(adj) < threshold] = 0
    np.fill_diagonal(adj, 1.0)
    edge_index, edge_attr = dense_to_sparse(torch.from_numpy(adj))
    edge_index = edge_index.numpy()
    edge_attr = edge_attr.numpy()
    density_tag = f"{int(round(density * 100)):02d}"
    np.save(out_dir / f"layer_{llm_layer}_sparse_{density_tag}_patho_{disease_name}_edge_index.npy", edge_index)
    np.save(out_dir / f"layer_{llm_layer}_sparse_{density_tag}_patho_{disease_name}_edge_attr.npy", edge_attr)


def _load_all_corr_paths(dataset_dir: Path, llm_layer: int) -> Dict[int, Path]:
    paths = {}
    for q_dir in dataset_dir.iterdir():
        if q_dir.is_dir():
            p = q_dir / f"layer_{llm_layer}_corr.npy"
            if p.exists():
                paths[int(q_dir.name)] = p
    return dict(sorted(paths.items()))


def main(_):
    logging.info("="*60)
    logging.info("Neuropathology Connectivity Simulation")
    logging.info("="*60)

    # Resolve paths
    sanitized_model_name = FLAGS.llm_model_name.replace('/', '_').replace('-', '_').replace('.', '_')
    if FLAGS.ckpt_step == -1:
        model_dir = sanitized_model_name
    else:
        model_dir = f"{sanitized_model_name}_step{FLAGS.ckpt_step}"
    dataset_dir = main_dir / "data/hallucination" / FLAGS.dataset_name / model_dir
    reports_dir = main_dir / "reports" / "neuropathology_analysis" / FLAGS.disease_pattern
    os.makedirs(reports_dir, exist_ok=True)

    # Build config
    base_cfg = PRESETS.get(FLAGS.disease_pattern, DiseaseConfig(name=FLAGS.disease_pattern))
    cfg = DiseaseConfig(
        name=base_cfg.name,
        num_clusters=FLAGS.num_clusters or base_cfg.num_clusters,
        within_scale=FLAGS.within_scale or base_cfg.within_scale,
        between_scale=FLAGS.between_scale or base_cfg.between_scale,
        rewiring_prob=FLAGS.rewiring_prob or base_cfg.rewiring_prob,
        distance_threshold=FLAGS.distance_threshold or base_cfg.distance_threshold,
    )

    logging.info(f"Dataset dir: {dataset_dir}")
    logging.info(f"Reports dir: {reports_dir}")
    logging.info(f"Config: {cfg}")

    corr_paths = _load_all_corr_paths(dataset_dir, FLAGS.llm_layer)
    if not corr_paths:
        logging.error("No healthy correlation matrices found. Run hallucination.compute_llm_network first.")
        return

    logging.info(f"Found {len(corr_paths)} healthy correlation matrices to transform.")

    # Accumulate for group averages
    healthy_sum = None
    patho_sum = None
    count = 0

    # Process each sample
    for q_idx, corr_path in tqdm(corr_paths.items(), desc="Generating pathological connectivity", unit="sample"):
        out_path = corr_path.parent / f"layer_{FLAGS.llm_layer}_corr_patho_{cfg.name}.npy"
        if out_path.exists() and not FLAGS.overwrite:
            C_patho = np.load(out_path)
            C_healthy = np.load(corr_path)
        else:
            C_healthy = np.load(corr_path)
            C_patho, labels = make_pathological_connectivity(C_healthy, cfg, FLAGS.density)
            np.save(out_path, C_patho)
            if FLAGS.save_sparse:
                _save_sparse_graph(C_patho, corr_path.parent, FLAGS.llm_layer, FLAGS.density, cfg.name)

        healthy_sum = C_healthy if healthy_sum is None else healthy_sum + C_healthy
        patho_sum = C_patho if patho_sum is None else patho_sum + C_patho
        count += 1

    # Group averages
    if count > 0:
        logging.info(f"Processed {count} samples. Computing group averages...")
        Cg_healthy = healthy_sum / count
        Cg_patho = patho_sum / count
        np.save(reports_dir / f"group_fc_healthy_layer_{FLAGS.llm_layer}.npy", Cg_healthy)
        np.save(reports_dir / f"group_fc_patho_{cfg.name}_layer_{FLAGS.llm_layer}.npy", Cg_patho)
        _save_heatmap(Cg_healthy, f"Group FC Healthy (layer {FLAGS.llm_layer})", reports_dir / f"group_fc_healthy_layer_{FLAGS.llm_layer}.png")
        _save_heatmap(Cg_patho, f"Group FC Patho {cfg.name} (layer {FLAGS.llm_layer})", reports_dir / f"group_fc_patho_{cfg.name}_layer_{FLAGS.llm_layer}.png")
        logging.info("Saved group FC matrices and heatmaps.")

    logging.info("\n" + "="*60)
    logging.info("✓ Neuropathology connectivity generation complete")
    logging.info("="*60)


if __name__ == "__main__":
    app.run(main)
