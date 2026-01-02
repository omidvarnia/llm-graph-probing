from absl import app, flags, logging
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Tuple
from tqdm import tqdm

import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc, precision_recall_curve
from scipy.stats import ttest_rel, wilcoxon

from torch_geometric.utils import dense_to_sparse

from hallucination.dataset import prepare_data
from utils.model_utils import get_num_nodes
from utils.probing_model import GCNProbe


main_dir = Path(os.environ.get('MAIN_DIR', '.'))

flags.DEFINE_string("dataset_name", "truthfulqa", "Dataset name.")
flags.DEFINE_string("llm_model_name", "gpt2", "LLM model name.")
flags.DEFINE_integer("ckpt_step", -1, "Checkpoint step.")
flags.DEFINE_integer("llm_layer", 5, "LLM layer.")
flags.DEFINE_string("probe_input", "corr", "Probe input type (corr or activation). Use corr for GCN.")
flags.DEFINE_float("density", 0.05, "Network density for sparsification.")
flags.DEFINE_string("disease_pattern", "epilepsy_like", "Disease preset.")
flags.DEFINE_integer("num_layers", 3, "Number of GCN layers.")
flags.DEFINE_integer("hidden_channels", 32, "Hidden channels.")
flags.DEFINE_integer("gpu_id", 0, "GPU ID.")
flags.DEFINE_integer("num_workers", 2, "DataLoader workers.")
flags.DEFINE_integer("prefetch_factor", 2, "DataLoader prefetch.")
flags.DEFINE_float("test_set_ratio", 0.2, "Test split ratio.")
flags.DEFINE_integer("seed", None, "Random seed (None for random).")
FLAGS = flags.FLAGS


def _load_dense(path: Path) -> np.ndarray:
    return np.load(path)


def _threshold_to_sparse(C: np.ndarray, density: float):
    A = C.copy()
    thr = np.percentile(np.abs(A), 100 - density * 100)
    A[np.abs(A) < thr] = 0.0
    np.fill_diagonal(A, 1.0)
    edge_index, edge_attr = dense_to_sparse(torch.from_numpy(A))
    return edge_index.long(), edge_attr.float()


class PathoTruthfulQADataset(torch.utils.data.Dataset):
    """Load pathological graphs built from transformed connectivity.

    Reads files saved by `hallucination.neuropathology_connectivity`:
    - Dense:  layer_{L}_corr_patho_{disease}.npy
    - Sparse: layer_{L}_sparse_{density}_patho_{disease}_edge_index.npy / edge_attr.npy
    """

    def __init__(self, llm_model_name, ckpt_step, llm_layer, density, indices, dataset_name, disease_name, use_saved_sparse=True):
        self.llm_model_name = llm_model_name
        self.ckpt_step = ckpt_step
        self.llm_layer = llm_layer
        self.density = density
        self.indices = indices
        self.dataset_name = dataset_name
        self.disease_name = disease_name
        self.use_saved_sparse = use_saved_sparse

        sanitized_model_name = self.llm_model_name.replace('/', '_').replace('-', '_').replace('.', '_')
        if self.ckpt_step == -1:
            model_dir = sanitized_model_name
        else:
            model_dir = f"{sanitized_model_name}_step{self.ckpt_step}"
        self.data_dir = main_dir / "data/hallucination" / self.dataset_name / model_dir

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        q_idx = self.indices[idx]
        q_dir = self.data_dir / str(q_idx)
        y = torch.from_numpy(np.load(q_dir / "label.npy")).long()

        if self.use_saved_sparse:
            density_tag = f"{int(round(self.density * 100)):02d}"
            ei = torch.from_numpy(np.load(q_dir / f"layer_{self.llm_layer}_sparse_{density_tag}_patho_{self.disease_name}_edge_index.npy")).long()
            ea = torch.from_numpy(np.load(q_dir / f"layer_{self.llm_layer}_sparse_{density_tag}_patho_{self.disease_name}_edge_attr.npy")).float()
            num_nodes = int(ei.max().item()) + 1
        else:
            C = _load_dense(q_dir / f"layer_{self.llm_layer}_corr_patho_{self.disease_name}.npy")
            ei, ea = _threshold_to_sparse(C, self.density)
            num_nodes = C.shape[0]

        x = torch.arange(num_nodes)
        from torch_geometric.data import Data
        return Data(x=x, edge_index=ei, edge_attr=ea, y=y)


def _load_best_model(device, dataset_name, llm_model_name, ckpt_step, llm_layer, density, hidden_channels, num_layers, probe_input):
    sanitized_model_name = llm_model_name.replace('/', '_').replace('-', '_').replace('.', '_')
    if ckpt_step == -1:
        model_dir = sanitized_model_name
    else:
        model_dir = f"{sanitized_model_name}_step{ckpt_step}"
    save_model_name = f"hallucination/{dataset_name}/{model_dir}"
    density_tag = f"{int(round(density * 100)):02d}"
    model_path = main_dir / f"saves/{save_model_name}/layer_{llm_layer}" / f"best_model_density-{density_tag}_dim-{hidden_channels}_hop-{num_layers}_input-{probe_input}.pth"
    num_nodes = get_num_nodes(llm_model_name, llm_layer)
    model = GCNProbe(num_nodes=num_nodes, hidden_channels=hidden_channels, num_layers=num_layers, dropout=0.0, num_output=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def _eval_condition(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    probs = []
    labels = []
    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluating", unit="batch"):
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            p = F.softmax(out, dim=-1)[:, 1]  # hallucination prob
            probs.append(p.cpu().numpy())
            labels.append(data.y.cpu().numpy())
    probs = np.concatenate(probs, axis=0)
    labels = np.concatenate(labels, axis=0)
    return probs, labels


def main(_):
    logging.info("="*10)
    logging.info("Neuropathology Evaluation (probe on healthy vs pathological graphs)")
    logging.info("="*10)

    device = torch.device(f"cuda:{FLAGS.gpu_id}")
    sanitized_model_name = FLAGS.llm_model_name.replace('/', '_').replace('-', '_').replace('.', '_')
    if FLAGS.ckpt_step == -1:
        model_dir = sanitized_model_name
    else:
        model_dir = f"{sanitized_model_name}_step{FLAGS.ckpt_step}"

    # Build test indices
    dataset_filename = main_dir / "data/hallucination" / f"{FLAGS.dataset_name}.csv"
    train_split, test_split = prepare_data(dataset_filename, FLAGS.test_set_ratio, FLAGS.seed)

    # Healthy dataset (match training input type)
    from hallucination.dataset import TruthfulQADataset
    healthy_ds = TruthfulQADataset(
        FLAGS.llm_model_name,
        FLAGS.ckpt_step,
        FLAGS.llm_layer,
        FLAGS.density,
        test_split,
        from_sparse_data=True,  # prefer saved sparse graphs for speed
        in_memory=True,
        dataset_name=FLAGS.dataset_name,
    )
    from torch_geometric.loader import DataLoader
    healthy_loader = DataLoader(healthy_ds, batch_size=32, shuffle=False, num_workers=FLAGS.num_workers, prefetch_factor=FLAGS.prefetch_factor)

    # Pathological dataset
    patho_ds = PathoTruthfulQADataset(
        FLAGS.llm_model_name,
        FLAGS.ckpt_step,
        FLAGS.llm_layer,
        FLAGS.density,
        test_split,
        FLAGS.dataset_name,
        FLAGS.disease_pattern,
        use_saved_sparse=True,
    )
    patho_loader = DataLoader(patho_ds, batch_size=32, shuffle=False, num_workers=FLAGS.num_workers, prefetch_factor=FLAGS.prefetch_factor)

    # Load trained probe (trained on healthy graphs)
    logging.info("Loading trained probe...")
    model = _load_best_model(device, FLAGS.dataset_name, FLAGS.llm_model_name, FLAGS.ckpt_step, FLAGS.llm_layer, FLAGS.density, FLAGS.hidden_channels, FLAGS.num_layers, FLAGS.probe_input)

    # Evaluate
    logging.info("Evaluating on healthy graphs...")
    p_h, y = _eval_condition(model, healthy_loader, device)
    logging.info("Evaluating on pathological graphs...")
    p_p, y2 = _eval_condition(model, patho_loader, device)
    assert np.array_equal(y, y2)

    # Metrics
    y_pred_h = (p_h >= 0.5).astype(int)
    y_pred_p = (p_p >= 0.5).astype(int)
    acc_h = accuracy_score(y, y_pred_h)
    acc_p = accuracy_score(y, y_pred_p)
    prec_h, rec_h, f1_h, _ = precision_recall_fscore_support(y, y_pred_h, average='binary', zero_division=0)
    prec_p, rec_p, f1_p, _ = precision_recall_fscore_support(y, y_pred_p, average='binary', zero_division=0)

    # Paired tests
    diff = p_p - p_h
    t_stat, t_p = ttest_rel(p_p, p_h)
    w_stat, w_p = wilcoxon(diff)

    # ROC & PR curves
    fpr_h, tpr_h, _ = roc_curve(y, p_h)
    fpr_p, tpr_p, _ = roc_curve(y, p_p)
    auc_h = auc(fpr_h, tpr_h)
    auc_p = auc(fpr_p, tpr_p)
    pr_h, rc_h, _ = precision_recall_curve(y, p_h)
    pr_p, rc_p, _ = precision_recall_curve(y, p_p)

    # Save reports
    reports_dir = main_dir / "reports" / "neuropathology_analysis" / FLAGS.disease_pattern
    os.makedirs(reports_dir, exist_ok=True)

    np.save(reports_dir / f"probs_healthy_layer_{FLAGS.llm_layer}.npy", p_h)
    np.save(reports_dir / f"probs_patho_{FLAGS.disease_pattern}_layer_{FLAGS.llm_layer}.npy", p_p)

    plt.figure(figsize=(6, 4))
    plt.hist(p_h, bins=40, alpha=0.6, label="healthy")
    plt.hist(p_p, bins=40, alpha=0.6, label="patho")
    plt.legend(); plt.title("Hallucination probability distribution")
    plt.tight_layout(); plt.savefig(reports_dir / f"prob_hist_layer_{FLAGS.llm_layer}.png"); plt.close()

    plt.figure(figsize=(5, 5))
    plt.plot(fpr_h, tpr_h, label=f"healthy AUC={auc_h:.3f}")
    plt.plot(fpr_p, tpr_p, label=f"patho AUC={auc_p:.3f}")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC curves")
    plt.legend(); plt.tight_layout(); plt.savefig(reports_dir / f"roc_layer_{FLAGS.llm_layer}.png"); plt.close()

    plt.figure(figsize=(5, 5))
    plt.plot(rc_h, pr_h, label="healthy")
    plt.plot(rc_p, pr_p, label="patho")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR curves")
    plt.legend(); plt.tight_layout(); plt.savefig(reports_dir / f"pr_layer_{FLAGS.llm_layer}.png"); plt.close()

    summary = {
        "accuracy": {"healthy": float(acc_h), "patho": float(acc_p)},
        "precision": {"healthy": float(prec_h), "patho": float(prec_p)},
        "recall": {"healthy": float(rec_h), "patho": float(rec_p)},
        "f1": {"healthy": float(f1_h), "patho": float(f1_p)},
        "auc": {"healthy": float(auc_h), "patho": float(auc_p)},
        "paired_tests": {
            "ttest_rel": {"stat": float(t_stat), "p": float(t_p)},
            "wilcoxon": {"stat": float(w_stat), "p": float(w_p)},
        },
        "delta_probs_mean": float(np.mean(diff)),
        "delta_probs_std": float(np.std(diff)),
        "artifacts": {
            "probs_healthy": str(reports_dir / f"probs_healthy_layer_{FLAGS.llm_layer}.npy"),
            "probs_patho": str(reports_dir / f"probs_patho_{FLAGS.disease_pattern}_layer_{FLAGS.llm_layer}.npy"),
            "prob_hist": str(reports_dir / f"prob_hist_layer_{FLAGS.llm_layer}.png"),
            "roc": str(reports_dir / f"roc_layer_{FLAGS.llm_layer}.png"),
            "pr": str(reports_dir / f"pr_layer_{FLAGS.llm_layer}.png"),
        },
    }
    (reports_dir / "summary_eval.json").write_text(__import__("json").dumps(summary, indent=2), encoding="utf-8")

    logging.info("="*10)
    logging.info("✓ Neuropathology evaluation complete")
    logging.info("="*10)


if __name__ == "__main__":
    app.run(main)
