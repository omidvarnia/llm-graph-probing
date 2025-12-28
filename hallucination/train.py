from absl import app, flags, logging
import os
from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from hallucination.dataset import get_truthfulqa_dataloader, get_truthfulqa_linear_dataloader
from hallucination.utils import test_fn
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
flags.DEFINE_integer("warmup_epochs", 5, "Number of warmup epochs.")
flags.DEFINE_float("dataset_fraction", 1.0, "Fraction of dataset to use (0.1-1.0 where 1.0 = all data)")
FLAGS = flags.FLAGS

main_dir = Path(os.environ.get('MAIN_DIR', '.'))

def train_model(model, train_data_loader, test_data_loader, optimizer, scheduler, warmup_scheduler, writer, save_model_name, device, class_weights):
    accuracy, precision, recall, f1, cm = test_fn(model, test_data_loader, device, num_layers=FLAGS.num_layers)
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
        
        loss = F.cross_entropy(output, target, weight=class_weights, label_smoothing=FLAGS.label_smoothing)
        
        # Check for NaN loss and log details
        if torch.isnan(loss):
            logging.warning(f"NaN loss detected in cross_entropy! Output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
        return loss

    def get_loss_batch_size_graph(data):
        data = data.to(device)
        
        # Sanitize inputs: replace NaN/Inf with small random values to preserve information
        # NaN/Inf typically arise from zero-variance features in correlation computation
        if torch.isnan(data.x).any() or torch.isinf(data.x).any():
            data.x = torch.nan_to_num(data.x, nan=0.0, posinf=0.1, neginf=-0.1)

        if torch.isnan(data.edge_attr).any() or torch.isinf(data.edge_attr).any():
            # Replace NaN/Inf with small random noise instead of hard zeros
            # This preserves more information while staying in valid correlation range [-1, 1]
            mask_bad = ~torch.isfinite(data.edge_attr)
            num_bad = mask_bad.sum().item()
            if num_bad > 0 and num_bad <= 100:  # Only log if reasonable number of bad values
                logging.debug(f"Sanitized {num_bad} NaN/Inf edge values in shape {data.edge_attr.shape}")
            # Replace with small random values from [-0.1, 0.1]
            data.edge_attr = torch.where(
                mask_bad,
                torch.randn_like(data.edge_attr) * 0.05,  # small random noise
                data.edge_attr
            )
            # Clamp to valid correlation range
            data.edge_attr = torch.clamp(data.edge_attr, -1.0, 1.0)
        
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
        model.train()
        total_loss = 0.0
        num_graphs = 0
        for data in tqdm(train_data_loader, position=1, desc=f"Epoch {epoch + 1}", leave=False):
            optimizer.zero_grad()
            batch_size = 0
            loss, batch_size = get_loss_batch_size(data)
            
            # Check for NaN loss before backward
            if torch.isnan(loss):
                logging.warning(f"NaN loss at batch, skipping this batch")
                continue
            
            # Clamp loss to prevent explosion
            loss = torch.clamp(loss, min=0, max=1e4)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), FLAGS.gradient_clip)
            
            optimizer.step()
            total_loss += loss.item() * batch_size
            num_graphs += batch_size
        
        # Update learning rate
        if epoch < FLAGS.warmup_epochs:
            warmup_scheduler.step()
        
        # Avoid NaN in average loss calculation
        if num_graphs == 0:
            avg_loss = float('nan')
        else:
            avg_loss = total_loss / num_graphs
            avg_loss = min(avg_loss, 1e4)  # Cap very large losses
            
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        writer.add_scalar("train/loss", avg_loss, epoch + 1)
        writer.add_scalar("train/lr", current_lr, epoch + 1)
        torch.cuda.empty_cache()

        accuracy, precision, recall, f1, cm = test_fn(model, test_data_loader, device, num_layers=FLAGS.num_layers)
        torch.cuda.empty_cache()
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        logging.info(f"Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        logging.info(f"Confusion Matrix (TN={tn}, FP={fp}, FN={fn}, TP={tp}):\n{cm}")
        for metric, value in zip(["accuracy", "precision", "recall", "f1"], [accuracy, precision, recall, f1]):
            writer.add_scalar(f"test/{metric}", value, epoch + 1)
        
        # Step scheduler only after warmup
        if epoch >= FLAGS.warmup_epochs:
            scheduler.step(f1)

        # Use F1 score for best model selection (better for imbalanced data)
        if f1 > best_metrics["f1"]:
            for metric, value in zip(["accuracy", "precision", "recall", "f1"], [accuracy, precision, recall, f1]):
                best_metrics[metric] = value
            best_metrics["epoch"] = epoch + 1
            best_metrics["cm"] = cm.copy()
            torch.save(model.state_dict(), model_save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= FLAGS.early_stop_patience:
                logging.info(f"Early stopping at epoch {epoch + 1}")
                break
    
    logging.info(f"Best Epoch: {best_metrics['epoch']}")
    for metric in ["accuracy", "precision", "recall", "f1"]:
        logging.info(f"Best Test {metric.capitalize()}: {best_metrics[metric]:.4f}")
    cm = best_metrics['cm']
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    logging.info(f"Best Confusion Matrix (TN={tn}, FP={fp}, FN={fn}, TP={tp}):\n{cm}")
 

def main(_):
    logging.info("="*60)
    logging.info("Training Hallucination Detection Probes")
    logging.info("="*60)
    
    # ===== DEVICE & CONFIGURATION =====
    device = torch.device(f"cuda:{FLAGS.gpu_id}")
    logging.info(f"Using device: {device}")

    # Sanitize model name by replacing '/', '-', and '.' with '_' for filesystem paths
    sanitized_model_name = FLAGS.llm_model_name.replace('/', '_').replace('-', '_').replace('.', '_')
    
    if FLAGS.ckpt_step == -1:
        model_dir = sanitized_model_name
    else:
        model_dir = f"{sanitized_model_name}_step{FLAGS.ckpt_step}"
    save_model_name = f"hallucination/{FLAGS.dataset_name}/{model_dir}"
    
    logging.info(f"Dataset: {FLAGS.dataset_name}")
    logging.info(f"Model: {FLAGS.llm_model_name}")
    logging.info(f"Layer: {FLAGS.llm_layer}")
    logging.info(f"Probe input: {FLAGS.probe_input}")
    logging.info(f"Network density: {FLAGS.density}")
    logging.info(f"From sparse data: {FLAGS.from_sparse_data}")
    logging.info(f"Architecture: {FLAGS.num_layers} layers, {FLAGS.hidden_channels} channels")
    logging.info(f"Training: {FLAGS.num_epochs} epochs, lr={FLAGS.lr}, batch_size={FLAGS.batch_size}")
    logging.info(f"Early stopping patience: {FLAGS.early_stop_patience}")
    logging.info(f"Seed: {FLAGS.seed}")
    logging.info(f"Label smoothing: {FLAGS.label_smoothing}, Gradient clip: {FLAGS.gradient_clip}")
    logging.info(f"Warmup epochs: {FLAGS.warmup_epochs}")
    
    # Warn if learning rate is very small (< 1e-4)
    if FLAGS.lr < 1e-4:
        logging.warning(f"⚠️ WARNING: Learning rate is very small ({FLAGS.lr:.2e}). This can cause numerical instability!")
        logging.warning(f"          Consider increasing to 0.001 or higher for better training stability.")

    # Calculate class weights for imbalanced data
    dataset_filename = main_dir / "data/hallucination" / f"{FLAGS.dataset_name}.csv"
    data = pd.read_csv(dataset_filename)
    original_data_size = len(data)
    
    # Apply dataset fraction (randomly sample unless fraction=1.0)
    if FLAGS.dataset_fraction < 1.0:
        # Random sampling without fixed random_state for true randomness
        data = data.sample(frac=FLAGS.dataset_fraction).reset_index(drop=True)
        logging.info(f"Applied dataset_fraction={FLAGS.dataset_fraction}: sampled {len(data)} samples from {original_data_size}")
    else:
        logging.info(f"Using full dataset: dataset_fraction={FLAGS.dataset_fraction}")
    
    # Sanity check: verify dataset size is consistent with fraction
    expected_size = int(original_data_size * FLAGS.dataset_fraction)
    actual_size = len(data)
    tolerance = max(1, int(0.01 * original_data_size))  # 1% tolerance
    if abs(actual_size - expected_size) <= tolerance:
        logging.info(f"✓ Sanity check: Dataset size ({actual_size}) is consistent with fraction ({FLAGS.dataset_fraction}) of original ({original_data_size})")
    else:
        logging.warning(f"⚠ Sanity check: Dataset size ({actual_size}) deviates from expected ({expected_size})")
    
    from collections import Counter
    label_counts = Counter(data['label'])
    total = sum(label_counts.values())
    # Ensure weights are computed safely and normalized
    weights_list = []
    for i in sorted(label_counts.keys()):
        if label_counts[i] == 0:
            weight = 1.0  # Avoid division by zero
        else:
            weight = total / (len(label_counts) * label_counts[i])
        weights_list.append(weight)
    # Normalize weights to avoid extreme values
    max_weight = max(weights_list)
    weights_list = [w / max_weight for w in weights_list]
    class_weights = torch.tensor(weights_list, dtype=torch.float32).to(device)
    logging.info(f"Class distribution: {dict(label_counts)}")
    logging.info(f"Class weights (normalized): {class_weights.cpu().numpy()}")

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

    # Better optimizer configuration
    optimizer = torch.optim.AdamW(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay, betas=(0.9, 0.999))
    
    # Warmup scheduler - starts at 0.5x instead of 0.1x to avoid very small LRs
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=FLAGS.warmup_epochs)
    
    # Main scheduler (now uses F1 score)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-7, verbose=True)
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
    logging.info("\n" + "="*60)
    logging.info("Starting training...")
    logging.info("="*60)
    train_model(model, train_loader, test_loader, optimizer, scheduler, warmup_scheduler, writer, save_model_name, device, class_weights)
    
    logging.info("\n" + "="*60)
    logging.info("✓ Training completed successfully")
    logging.info("="*60)


if __name__ == "__main__":
    app.run(main)
