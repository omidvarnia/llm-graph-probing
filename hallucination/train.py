from absl import app, flags, logging
import os
from tqdm import tqdm
from pathlib import Path
from collections import Counter
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from hallucination.dataset import get_truthfulqa_dataloader, get_truthfulqa_linear_dataloader
from hallucination.utils import test_fn, select_device
from utils.probing_model import GCNProbe as GCNClassifier, MLPProbe as MLPClassifier
from utils.model_utils import get_num_nodes
import pandas as pd

flags.DEFINE_enum("dataset_name", "truthfulqa", ["truthfulqa", "halueval", "medhallu", "helm"], "Name of the dataset.")
flags.DEFINE_float("density", 1.0, "The density of the network/features.")
flags.DEFINE_boolean("from_sparse_data", False, "Whether to use sparse data.")
flags.DEFINE_string("llm_model_name", "qwen2.5-0.5b", "The name of the LLM model.")
flags.DEFINE_integer("ckpt_step", -1, "The checkpoint step.")
flags.DEFINE_integer("llm_layer", 12, "The layer of the LLM model.")
flags.DEFINE_string("probe_input", "activation", "The input type for linear probing: activation, activation_avg, corr, word2vec_average, word2vec_token_count, or perplexity.")
flags.DEFINE_integer("batch_size", 16, "The batch size.")
flags.DEFINE_integer("eval_batch_size", 16, "The evaluation batch size.")
flags.DEFINE_integer("num_workers", 4, "Number of workers.")
flags.DEFINE_integer("prefetch_factor", 4, "Prefetch factor.")
flags.DEFINE_integer("hidden_channels", 32, "The hidden channels.")
flags.DEFINE_integer("num_layers", 1, "The number of GNN layers.")
flags.DEFINE_float("dropout", 0.0, "The dropout rate.")
flags.DEFINE_float("lr", 0.001, "The learning rate.")
flags.DEFINE_float("weight_decay", 1e-5, "The weight decay.")
flags.DEFINE_integer("num_epochs", 100, "The number of epochs.")
flags.DEFINE_float("test_set_ratio", 0.2, "The ratio of the test set.")
flags.DEFINE_boolean("in_memory", True, "In-memory dataset.")
flags.DEFINE_integer("early_stop_patience", 20, "The patience for early stopping.")
flags.DEFINE_integer("gpu_id", 0, "The GPU ID.")
flags.DEFINE_boolean("resume", False, "Whether to resume training from the best model.")
flags.DEFINE_integer("seed", 42, "The random seed.")
flags.DEFINE_float("label_smoothing", 0.1, "Label smoothing factor.")
flags.DEFINE_float("gradient_clip", 1.0, "Gradient clipping value.")
FLAGS = flags.FLAGS

main_dir = Path(os.environ.get('MAIN_DIR', '.')) if 'MAIN_DIR' in os.environ else Path('.')

def train_model(model, train_data_loader, test_data_loader, optimizer, scheduler, writer, save_model_name, device):
    accuracy, precision, recall, f1, cm = test_fn(model, test_data_loader, device, num_layers=FLAGS.num_layers)
    # Only empty GPU cache if using CUDA
    if device.type == "cuda":
        torch.cuda.empty_cache()
    logging.info(f"Initial Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    for metric, value in zip(["accuracy", "precision", "recall", "f1"], [accuracy, precision, recall, f1]):
        writer.add_scalar(f"test/{metric}", value, 0)

    density_tag = f"{int(round(FLAGS.density * 100)):02d}"
    model_save_path = main_dir / f"saves/{save_model_name}/layer_{FLAGS.llm_layer}" / f"best_model_density-{density_tag}_dim-{FLAGS.hidden_channels}_hop-{FLAGS.num_layers}_input-{FLAGS.probe_input}.pth"
    os.makedirs(model_save_path.parent, exist_ok=True)

    best_metrics = {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "cm": cm.copy(),
        "epoch": 0
    }
    epochs_no_improve = 0
    
    # Use class-weighted loss with label smoothing
    def loss_fn(output, target):
        # Check for NaN/Inf in output before computing loss
        if torch.isnan(output).any() or torch.isinf(output).any():
            logging.warning(f"NaN/Inf in model output! Output shape: {output.shape}, unique values: {torch.unique(output)[:10]}")
            logging.warning(f"Target shape: {target.shape}, unique values: {torch.unique(target)}")
            # Return a fallback loss instead of NaN
            return torch.tensor(0.0, requires_grad=True, device=output.device)
        
        loss = F.cross_entropy(output, target, label_smoothing=FLAGS.label_smoothing)
        
        # Check for NaN loss and log details
        if torch.isnan(loss):
            logging.warning(f"NaN loss detected in cross_entropy! Output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
        return loss

    def get_loss_batch_size_graph(data):
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        
        # Validate output
        if torch.isnan(out).any() or torch.isinf(out).any():
            logging.warning(f"NaN/Inf in model output! out shape: {out.shape}, dtype: {out.dtype}")
            logging.warning(f"Model device: {next(model.parameters()).device}, Data device: {data.x.device}")
            return torch.tensor(0.0, requires_grad=True, device=device), data.num_graphs
        
        loss = loss_fn(out, data.y)
        batch_size = data.num_graphs
        return loss, batch_size

    def get_loss_batch_size_linear(data):
        x, y = data
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        batch_size = x.shape[0]
        return loss, batch_size

    if FLAGS.num_layers > 0:
        get_loss_batch_size = get_loss_batch_size_graph
    else:
        get_loss_batch_size = get_loss_batch_size_linear

    for epoch in tqdm(range(FLAGS.num_epochs), position=0, desc="Training"):
        logging.info("\n" + "-" * 60)
        logging.info(f"Epoch {epoch + 1}/{FLAGS.num_epochs}")
        logging.info("-" * 60)
        model.train()
        total_loss = 0.0
        num_graphs = 0
        for data in tqdm(train_data_loader, position=1, desc=f"Epoch {epoch + 1}", leave=False):
            optimizer.zero_grad()
            batch_size = 0
            loss, batch_size = get_loss_batch_size(data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), FLAGS.gradient_clip)
            optimizer.step()
            total_loss += loss.item() * batch_size
            num_graphs += batch_size
        
        avg_loss = total_loss / num_graphs
        logging.info(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
        writer.add_scalar("train/loss", avg_loss, epoch + 1)
        # Only empty GPU cache if using CUDA
        if device.type == "cuda":
            torch.cuda.empty_cache()

        accuracy, precision, recall, f1, cm = test_fn(model, test_data_loader, device, num_layers=FLAGS.num_layers)
        # Only empty GPU cache if using CUDA
        if device.type == "cuda":
            torch.cuda.empty_cache()
        logging.info(f"Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        logging.info(f"Confusion Matrix:\n{cm}")
        for metric, value in zip(["accuracy", "precision", "recall", "f1"], [accuracy, precision, recall, f1]):
            writer.add_scalar(f"test/{metric}", value, epoch + 1)
        scheduler.step(f1)

        if f1 > best_metrics["f1"]:
            for metric, value in zip(["accuracy", "precision", "recall", "f1"], [accuracy, precision, recall, f1]):
                best_metrics[metric] = value
            best_metrics["epoch"] = epoch + 1
            best_metrics["cm"] = cm.copy()
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"New best model saved to: {model_save_path} (F1: {f1:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= FLAGS.early_stop_patience:
                logging.info(f"Early stopping at epoch {epoch + 1}")
                break
    
    logging.info(f"Best Epoch: {best_metrics['epoch']}")
    for metric in ["accuracy", "precision", "recall", "f1"]:
        logging.info(f"Best Test {metric.capitalize()}: {best_metrics[metric]:.4f}")
    logging.info(f"Best Confusion Matrix:\n{best_metrics['cm']}")
    
    logging.info("\n" + "="*80)
    logging.info("STEP 3 COMPLETE: Probe Training & Evaluation")
    logging.info("="*80)
    logging.info("✓ Probe training completed successfully")
    logging.info("="*80 + "\n\n")
 

def main(_):
    # Log PyTorch Geometric backend device FIRST (before any other output)
    from utils.probing_model import PYG_DEVICE_INFO
    logging.info(f"PyTorch Geometric Backend: {PYG_DEVICE_INFO}")
    
    logging.info("\n" + "="*80)
    logging.info("STEP 3: HALLUCINATION DETECTION PROBE TRAINING & EVALUATION")
    logging.info("="*80)
    
    # ===== DEVICE & CONFIGURATION =====
    device = select_device(FLAGS.gpu_id)
    logging.info(f"\n{'─'*80}")
    logging.info(f"Device Configuration:")
    logging.info(f"  Device Type: {device.type.upper()}")
    logging.info(f"  Device Index: {device.index if device.index is not None else 0}")
    logging.info(f"  PyTorch CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"  GPU Name: {torch.cuda.get_device_name(FLAGS.gpu_id)}")
        logging.info(f"  GPU Memory: {torch.cuda.get_device_properties(FLAGS.gpu_id).total_memory / 1e9:.1f} GB")
    
    logging.info(f"Dataset: {FLAGS.dataset_name}")
    logging.info(f"Model: {FLAGS.llm_model_name}")
    logging.info("="*80 + "\n")

    if FLAGS.ckpt_step == -1:
        model_dir = FLAGS.llm_model_name
    else:
        model_dir = f"{FLAGS.llm_model_name}_step{FLAGS.ckpt_step}"
    save_model_name = f"hallucination/{FLAGS.dataset_name}/{model_dir}"
    
    # Log layer analysis header
    logging.info(f"\n{'─'*80}")
    logging.info(f"Layer Analysis Configuration:")
    logging.info(f"  Layer ID: {FLAGS.llm_layer}")
    logging.info(f"  Probe Input: {FLAGS.probe_input}")
    logging.info(f"  Network Density: {FLAGS.density:.1%}")
    logging.info(f"  From Sparse Data: {FLAGS.from_sparse_data}")
    logging.info(f"  Architecture: {FLAGS.num_layers}-layer GCN, {FLAGS.hidden_channels} hidden channels")
    logging.info(f"{'─'*80}\n")
    
    logging.info(f"Dataset Configuration:")
    logging.info(f"  Dataset Name: {FLAGS.dataset_name}")
    logging.info(f"  Model: {FLAGS.llm_model_name}")
    logging.info(f"  Batch Size: {FLAGS.batch_size}")
    logging.info(f"  Test Set Ratio: {FLAGS.test_set_ratio:.1%}")
    logging.info(f"")
    
    logging.info(f"Training Hyperparameters:")
    logging.info(f"  Max Epochs: {FLAGS.num_epochs}")
    logging.info(f"  Learning Rate: {FLAGS.lr}")
    logging.info(f"  Label Smoothing: {FLAGS.label_smoothing}")
    logging.info(f"  Gradient Clip: {FLAGS.gradient_clip}")
    logging.info(f"  Early Stop Patience: {FLAGS.early_stop_patience} epochs")
    logging.info(f"  Random Seed: {FLAGS.seed}")
    
    # Warn if learning rate is very small (< 1e-4)
    if FLAGS.lr < 1e-4:
        logging.warning(f"\n⚠️  WARNING: Learning rate is very small ({FLAGS.lr:.2e})!")
        logging.warning(f"             This can cause numerical instability.")
        logging.warning(f"             Consider increasing to 0.001 or higher.\n")

    # Calculate class weights for imbalanced data
    dataset_filename = main_dir / "data/hallucination" / f"{FLAGS.dataset_name}.csv"
    data = pd.read_csv(dataset_filename)
    original_data_size = len(data)
    
    logging.info(f"Class distribution: {dict(Counter(data['label']))}")

    logging.info("\nLoading data...")
    if FLAGS.num_layers > 0:
        logging.info("Using graph-based (GCN) probe")
        train_loader, test_loader = get_truthfulqa_dataloader(
            FLAGS.dataset_name,
            FLAGS.llm_model_name,
            FLAGS.ckpt_step,
            FLAGS.llm_layer,
            FLAGS.density,
            FLAGS.from_sparse_data,
            FLAGS.batch_size,
            FLAGS.eval_batch_size,
            FLAGS.num_workers,
            FLAGS.prefetch_factor,
            FLAGS.test_set_ratio,
            FLAGS.in_memory,
            FLAGS.seed,
        )
        num_nodes = get_num_nodes(FLAGS.llm_model_name, FLAGS.llm_layer)
        logging.info(f"Number of nodes: {num_nodes}")
        model = GCNClassifier(
            num_nodes=num_nodes,
            hidden_channels=FLAGS.hidden_channels,
            num_layers=FLAGS.num_layers,
            dropout=FLAGS.dropout,
            num_output=2,
        ).to(device)
    else:
        logging.info("Using linear (MLP) probe")
        train_loader, test_loader = get_truthfulqa_linear_dataloader(
            FLAGS.probe_input,
            FLAGS.dataset_name,
            FLAGS.llm_model_name,
            FLAGS.ckpt_step,
            FLAGS.llm_layer,
            FLAGS.batch_size,
            FLAGS.eval_batch_size,
            FLAGS.num_workers,
            FLAGS.prefetch_factor,
            FLAGS.test_set_ratio,
            seed=FLAGS.seed,
            feature_density=FLAGS.density,
        )
        num_nodes = get_num_nodes(FLAGS.llm_model_name, FLAGS.llm_layer, FLAGS.probe_input)
        logging.info(f"Number of features: {num_nodes}")
        model = MLPClassifier(
            num_nodes=num_nodes,
            hidden_channels=FLAGS.hidden_channels,
            num_layers=-FLAGS.num_layers,
            num_output=2,
        ).to(device)
    
    logging.info(f"Train samples: {len(train_loader.dataset)}")
    logging.info(f"Test samples: {len(test_loader.dataset)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, min_lr=1e-6)
    writer = SummaryWriter(log_dir=main_dir / f"runs/{save_model_name}/layer_{FLAGS.llm_layer}")
    writer.add_hparams(
        {
            "hidden_channels": FLAGS.hidden_channels,
            "num_layers": FLAGS.num_layers, "dropout": FLAGS.dropout,
            "batch_size": FLAGS.batch_size, "lr": FLAGS.lr, "weight_decay": FLAGS.weight_decay
        },
        {"hparam/placeholder": 0}
    )

    if FLAGS.resume:
        density_tag = f"{int(round(FLAGS.density * 100)):02d}"
        model_save_path = main_dir / f"saves/{save_model_name}/layer_{FLAGS.llm_layer}" / f"best_model_density-{density_tag}_dim-{FLAGS.hidden_channels}_hop-{FLAGS.num_layers}_input-{FLAGS.probe_input}.pth"
        logging.info(f"Resuming from: {model_save_path}")
        model.load_state_dict(torch.load(model_save_path, map_location=device))

    # ===== TRAINING =====
    logging.info("\n" + "="*80)
    logging.info("LAYER ANALYSIS: TRAINING PHASE")
    logging.info("="*80)
    logging.info(f"Layer {FLAGS.llm_layer} - Training probe on {len(train_loader.dataset)} samples")
    logging.info("="*80 + "\n")
    
    train_model(model, train_loader, test_loader, optimizer, scheduler, writer, save_model_name, device)
    
    logging.info("\n" + "="*80)
    logging.info(f"LAYER ANALYSIS COMPLETE: Layer {FLAGS.llm_layer}")
    logging.info("="*80)
    logging.info(f"✓ Training completed successfully on device: {device.type.upper()}")
    logging.info("="*80 + "\n")


if __name__ == "__main__":
    app.run(main)
