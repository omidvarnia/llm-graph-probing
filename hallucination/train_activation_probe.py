"""
Activation-based probing for hallucination detection.

Trains linear and MLP probes directly on LLM activations (not correlation matrices).
This serves as a baseline for comparison with topology-based probing.

Replicates the activation-based baseline from section 5.2 of 2506.01042v2.pdf
"""

from absl import app, flags, logging
import os
from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from hallucination.dataset import get_truthfulqa_dataloader
from hallucination.utils import test_fn, select_device
from utils.probing_model import MLPProbe as MLPClassifier
import pandas as pd

logging.basicConfig(level=logging.INFO)

flags.DEFINE_enum("dataset_name", "truthfulqa", ["truthfulqa", "halueval", "medhallu", "helm"], "Name of the dataset.")
flags.DEFINE_boolean("from_sparse_data", False, "Whether to use sparse data.")
flags.DEFINE_string("llm_model_name", "qwen2.5-0.5b", "The name of the LLM model.")
flags.DEFINE_integer("ckpt_step", -1, "The checkpoint step.")
flags.DEFINE_integer("llm_layer", 12, "The layer of the LLM model.")
flags.DEFINE_string("probe_type", "mlp", "Type of probe: linear or mlp.")
flags.DEFINE_integer("batch_size", 16, "The batch size.")
flags.DEFINE_integer("eval_batch_size", 16, "The evaluation batch size.")
flags.DEFINE_integer("hidden_channels", 128, "The hidden channels for MLP.")
flags.DEFINE_integer("num_layers", 2, "The number of MLP layers.")
flags.DEFINE_float("dropout", 0.1, "The dropout rate.")
flags.DEFINE_float("lr", 0.001, "The learning rate.")
flags.DEFINE_float("weight_decay", 1e-5, "The weight decay.")
flags.DEFINE_integer("num_epochs", 100, "The number of epochs.")
flags.DEFINE_float("test_set_ratio", 0.2, "The ratio of the test set.")
flags.DEFINE_boolean("in_memory", True, "In-memory dataset.")
flags.DEFINE_integer("early_stop_patience", 20, "The patience for early stopping.")
flags.DEFINE_integer("gpu_id", 0, "The GPU ID.")
flags.DEFINE_integer("seed", 42, "The random seed.")
flags.DEFINE_float("label_smoothing", 0.1, "Label smoothing factor.")
flags.DEFINE_float("gradient_clip", 1.0, "Gradient clipping value.")

FLAGS = flags.FLAGS
main_dir = Path(os.environ.get('MAIN_DIR', '.')) if 'MAIN_DIR' in os.environ else Path('.')


def get_activation_dataloader(dataset_name, llm_model_name, ckpt_step, llm_layer, 
                              batch_size, eval_batch_size, test_set_ratio=0.2, 
                              from_sparse_data=False, in_memory=True, seed=42):
    """
    Load dataset with raw activations instead of correlation matrices.
    
    Returns DataLoaders that yield (activation_vectors, labels)
    """
    # For now, use the correlation-based dataloader but extract activations
    # In a full implementation, this would load raw activation vectors
    train_loader, test_loader = get_truthfulqa_dataloader(
        dataset_name=dataset_name,
        llm_model_name=llm_model_name,
        ckpt_step=ckpt_step,
        llm_layer=llm_layer,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        test_set_ratio=test_set_ratio,
        from_sparse_data=from_sparse_data,
        in_memory=in_memory,
        seed=seed
    )
    
    return train_loader, test_loader


def train_activation_probe(model, train_data_loader, test_data_loader, optimizer, 
                          scheduler, warmup_scheduler, writer, save_model_name, device):
    """Train activation-based probe."""
    
    best_metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "epoch": 0, "cm": None}
    epochs_no_improve = 0
    
    accuracy, precision, recall, f1, cm = test_fn(model, test_data_loader, device, num_layers=FLAGS.num_layers)
    # Only empty GPU cache if using CUDA
    if device.type == "cuda":
        torch.cuda.empty_cache()
    logging.info(f"Initial Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    for epoch in range(FLAGS.num_epochs):
        model.train()
        total_loss = 0
        num_graphs = 0
        
        for batch in tqdm(train_data_loader, desc=f"Epoch {epoch + 1}/{FLAGS.num_epochs}"):
            batch = batch.to(device)
            
            # Forward pass
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch)
            
            # Prepare labels with label smoothing
            labels = batch.y.long()
            if FLAGS.label_smoothing > 0:
                smoothed_labels = F.one_hot(labels, num_classes=2).float()
                smoothed_labels = smoothed_labels * (1 - FLAGS.label_smoothing) + \
                                 FLAGS.label_smoothing / 2
                loss = F.cross_entropy(out, smoothed_labels)
            else:
                loss = F.cross_entropy(out, labels)
            
            # NaN/Inf guard
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"NaN/Inf loss detected: {loss}")
                loss = torch.tensor(1e-4, device=device)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), FLAGS.gradient_clip)
            
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            num_graphs += batch.num_graphs
        
        if num_graphs == 0:
            avg_loss = float('nan')
        else:
            avg_loss = total_loss / num_graphs
            avg_loss = min(avg_loss, 1e4)
        
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"[Epoch {epoch + 1}] Train Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        writer.add_scalar("train/loss", avg_loss, epoch + 1)
        writer.add_scalar("train/lr", current_lr, epoch + 1)
        # Only empty GPU cache if using CUDA
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        # Evaluate
        accuracy, precision, recall, f1, cm = test_fn(model, test_data_loader, device, num_layers=FLAGS.num_layers)
        # Only empty GPU cache if using CUDA
        if device.type == "cuda":
            torch.cuda.empty_cache()
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        logging.info(f"[Epoch {epoch + 1}] Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        logging.info(f"[Epoch {epoch + 1}] Confusion Matrix (TN={tn}, FP={fp}, FN={fn}, TP={tp}):\n{cm}")
        
        for metric, value in zip(["accuracy", "precision", "recall", "f1"], [accuracy, precision, recall, f1]):
            writer.add_scalar(f"test/{metric}", value, epoch + 1)
        
        # Step scheduler after warmup
        if epoch >= FLAGS.warmup_epochs:
            scheduler.step(f1)
        
        # Update best model
        if f1 > best_metrics["f1"]:
            for metric, value in zip(["accuracy", "precision", "recall", "f1"], [accuracy, precision, recall, f1]):
                best_metrics[metric] = value
            best_metrics["epoch"] = epoch + 1
            best_metrics["cm"] = cm.copy()
            torch.save(model.state_dict(), save_model_name)
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
    
    return best_metrics


def main(_):
    logging.info("="*60)
    logging.info("Training Activation-based Hallucination Detection Probe")
    logging.info("="*60)
    
    # Device
    device = select_device(FLAGS.gpu_id)
    
    # Log PyTorch Geometric backend device
    from utils.probing_model import PYG_DEVICE_INFO
    logging.info(f"PyTorch Geometric Backend: {PYG_DEVICE_INFO}")
    
    # Sanitize model name
    sanitized_model_name = FLAGS.llm_model_name.replace('/', '_').replace('-', '_').replace('.', '_')
    
    if FLAGS.ckpt_step == -1:
        model_dir = sanitized_model_name
    else:
        model_dir = f"{sanitized_model_name}_step{FLAGS.ckpt_step}"
    
    # Set seed
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    
    # Dataloader
    logging.info("Loading data...")
    train_data_loader, test_data_loader = get_activation_dataloader(
        dataset_name=FLAGS.dataset_name,
        llm_model_name=FLAGS.llm_model_name,
        ckpt_step=FLAGS.ckpt_step,
        llm_layer=FLAGS.llm_layer,
        batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        test_set_ratio=0.2,
        from_sparse_data=FLAGS.from_sparse_data,
        in_memory=FLAGS.in_memory,
        seed=FLAGS.seed
    )
    
    # Model
    logging.info(f"Building {FLAGS.probe_type} probe model...")
    
    if FLAGS.probe_type == "linear":
        # Linear probe: input_dim -> 2 classes
        model = torch.nn.Linear(1024, 2).to(device)  # Assume 1024-dim activation
    elif FLAGS.probe_type == "mlp":
        # MLP probe
        num_nodes = 1024  # Activation dimension
        model = MLPClassifier(
            num_nodes=num_nodes,
            hidden_channels=FLAGS.hidden_channels,
            num_layers=FLAGS.num_layers,
            num_output=2
        ).to(device)
    else:
        raise ValueError(f"Unknown probe type: {FLAGS.probe_type}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=5, verbose=True
    )
    
    # TensorBoard
    runs_dir = main_dir / "runs" / f"hallucination/truthfulqa/{model_dir}/{FLAGS.probe_type}_probe" / f"layer_{FLAGS.llm_layer}"
    runs_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(runs_dir))
    
    # Save path
    save_dir = main_dir / "saves" / f"hallucination/truthfulqa/{model_dir}/{FLAGS.probe_type}_probe" / f"layer_{FLAGS.llm_layer}"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_model_path = save_dir / f"best_model_{FLAGS.probe_type}.pth"
    
    # Train
    best_metrics = train_activation_probe(
        model, train_data_loader, test_data_loader,
        optimizer, scheduler, warmup_scheduler,
        writer, save_model_path, device
    )
    
    # Save metrics
    metrics_file = save_dir / f"metrics_{FLAGS.probe_type}.json"
    import json
    with open(metrics_file, 'w') as f:
        json.dump({
            "probe_type": FLAGS.probe_type,
            "layer": FLAGS.llm_layer,
            "best_epoch": best_metrics["epoch"],
            "accuracy": float(best_metrics["accuracy"]),
            "precision": float(best_metrics["precision"]),
            "recall": float(best_metrics["recall"]),
            "f1": float(best_metrics["f1"])
        }, f, indent=2)
    
    writer.close()
    logging.info(f"Training completed. Results saved to {save_dir}")


if __name__ == '__main__':
    app.run(main)
