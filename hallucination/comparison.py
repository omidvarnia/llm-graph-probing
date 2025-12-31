"""
Comparison of topology-based vs activation-based hallucination detection probing.

Replicates Figure 5(b-c) from section 5.2 of 2506.01042v2.pdf

Compares:
1. GCN probe on correlation matrices (topology-based) - proposed method
2. Linear probe on activations (activation-based) - baseline
3. MLP probe on activations (activation-based) - baseline
4. Coupling index distribution analysis
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_probe_metrics(reports_dir, layer_id, probe_type="topology"):
    """
    Load metrics for a probe.
    
    Args:
        reports_dir: Path to hallucination_analysis/{model_tag}
        layer_id: Layer number
        probe_type: "topology", "linear", or "mlp"
    
    Returns:
        dict with metrics
    """
    if probe_type == "topology":
        eval_log = reports_dir / f"layer_{layer_id}" / "step4_eval.log"
        if not eval_log.exists():
            return None
        
        # Parse confusion matrix from eval log
        import re
        with open(eval_log, 'r') as f:
            content = f.read()
        
        cm_match = re.search(r'\[\[(\d+)\s+(\d+)\]\s+\[(\d+)\s+(\d+)\]\]', content)
        if cm_match:
            tn, fp, fn, tp = map(int, cm_match.groups())
            total = tn + fp + fn + tp
            accuracy = (tn + tp) / total if total > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                'layer': layer_id,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'probe_type': 'GCN (Topology)'
            }
    
    elif probe_type in ["linear", "mlp"]:
        metrics_file = reports_dir / f"layer_{layer_id}" / "step4_eval.log"  # Would come from activation probe
        if not metrics_file.exists():
            return None
        
        # Parse similar to above but from activation probe results
        # This is a placeholder - actual implementation depends on where activation results are stored
        return None
    
    return None


def load_coupling_index(reports_dir, layer_id):
    """Load coupling index results for a layer."""
    coupling_file = reports_dir / f"layer_{layer_id}" / "coupling_index.json"
    
    if not coupling_file.exists():
        return None
    
    with open(coupling_file, 'r') as f:
        return json.load(f)


def create_layer_metrics_plot(reports_root_dir, layer_ids, output_dir):
    """
    Create Figure 5(b) equivalent: accuracy comparison across layers.
    
    Args:
        reports_root_dir: Path to hallucination_analysis/{model_tag}
        layer_ids: List of layer IDs to plot
        output_dir: Path to save figures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics for all layers
    topology_accuracies = []
    topology_f1_scores = []
    
    for layer_id in layer_ids:
        topo_metrics = load_probe_metrics(Path(reports_root_dir), layer_id, "topology")
        if topo_metrics:
            topology_accuracies.append(topo_metrics['accuracy'] * 100)
            topology_f1_scores.append(topo_metrics['f1'] * 100)
        else:
            topology_accuracies.append(None)
            topology_f1_scores.append(None)
    
    # Filter out None values for plotting
    valid_layers = [lid for lid, acc in zip(layer_ids, topology_accuracies) if acc is not None]
    valid_accuracies = [acc for acc in topology_accuracies if acc is not None]
    valid_f1 = [f1 for f1 in topology_f1_scores if f1 is not None]
    
    if valid_layers:
        # Figure 5(b): Accuracy across layers (line plot like reference paper)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(valid_layers, valid_accuracies, marker='o', linewidth=2.5, 
                markersize=8, label='Topology-based (GCN)', color='#2E86AB')
        ax.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Chance level (50%)', alpha=0.7)
        
        ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        ax.set_title('Hallucination Detection Accuracy Across Layers (Figure 5b)', fontsize=16, fontweight='bold')
        ax.set_xticks(valid_layers)
        ax.set_ylim([0, 100])
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add value labels on points
        for x, y in zip(valid_layers, valid_accuracies):
            ax.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                       xytext=(0, 10), ha='center', fontsize=9)
        
        fig.tight_layout()
        fig.savefig(output_dir / "hallucination_accuracy_comparison.png", dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved Figure 5(b) to {output_dir / 'hallucination_accuracy_comparison.png'}")
        
        # Save metrics to CSV
        import pandas as pd
        metrics_df = pd.DataFrame({
            'Layer': valid_layers,
            'Accuracy (%)': valid_accuracies,
            'F1 Score (%)': valid_f1
        })
        metrics_csv = output_dir / "layer_metrics_summary.csv"
        metrics_df.to_csv(metrics_csv, index=False)
        logger.info(f"✓ Saved metrics to {metrics_csv}")
        
        plt.close()


def create_comparison_figure(reports_root_dir, output_dir):
    """
    Create Figure 5(c) equivalent: coupling index distribution and per-layer analysis.
    
    Args:
        reports_root_dir: Path to hallucination_analysis/{model_tag}
        output_dir: Path to save figures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect layers from directory structure
    reports_path = Path(reports_root_dir)
    layer_dirs = sorted([d for d in reports_path.glob('layer_*') if d.is_dir()])
    layer_ids = [int(d.name.split('_')[1]) for d in layer_dirs]
    
    # Figure 5(c): Coupling index distribution
    all_coupling_indices = []
    
    for layer_id in layer_ids:
        coupling_data = load_coupling_index(Path(reports_root_dir), layer_id)
        if coupling_data:
            if 'per_sample_indices' in coupling_data:
                indices = [s['coupling_index'] for s in coupling_data['per_sample_indices']]
                all_coupling_indices.extend(indices)
    
    if all_coupling_indices:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        n, bins, patches = ax.hist(all_coupling_indices, bins=50, alpha=0.75, edgecolor='black')
        
        # Color bars based on positive/negative
        for i, patch in enumerate(patches):
            if bins[i] < 0:
                patch.set_facecolor('#E63946')  # Red for negative
            else:
                patch.set_facecolor('#2E86AB')  # Blue for positive
        
        ax.axvline(x=0, color='black', linestyle='-', linewidth=2.5, alpha=0.8)
        
        # Calculate statistics
        positive_ratio = sum(1 for c in all_coupling_indices if c > 0) / len(all_coupling_indices)
        mean_c = np.mean(all_coupling_indices)
        std_c = np.std(all_coupling_indices)
        
        ax.set_xlabel('Coupling Index (C = C_TT + C_HH - 2×C_TH)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
        ax.set_title(f'Neural Topology Coupling Index Distribution (Figure 5c)\nPositive: {positive_ratio:.1%} | Mean: {mean_c:.4f} | Std: {std_c:.4f}',
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        fig.tight_layout()
        fig.savefig(output_dir / "coupling_index_distribution.png", dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved Figure 5(c) to {output_dir / 'coupling_index_distribution.png'}")
        plt.close()
    
    # Additional plot: Coupling index per layer (line plot)
    layer_coupling_means = []
    layer_coupling_stds = []
    
    for layer_id in layer_ids:
        coupling_data = load_coupling_index(Path(reports_root_dir), layer_id)
        if coupling_data and 'per_sample_indices' in coupling_data:
            indices = [s['coupling_index'] for s in coupling_data['per_sample_indices']]
            layer_coupling_means.append(np.mean(indices))
            layer_coupling_stds.append(np.std(indices))
        else:
            layer_coupling_means.append(None)
            layer_coupling_stds.append(None)
    
    # Filter valid data
    valid_layers = [lid for lid, mean in zip(layer_ids, layer_coupling_means) if mean is not None]
    valid_means = [m for m in layer_coupling_means if m is not None]
    valid_stds = [s for s in layer_coupling_stds if s is not None]
    
    if valid_layers:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.errorbar(valid_layers, valid_means, yerr=valid_stds, marker='o', linewidth=2.5,
                   markersize=8, capsize=5, capthick=2, label='Mean ± Std', color='#2E86AB')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
        ax.set_ylabel('Coupling Index (C)', fontsize=14, fontweight='bold')
        ax.set_title('Coupling Index Across Layers', fontsize=16, fontweight='bold')
        ax.set_xticks(valid_layers)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        fig.tight_layout()
        fig.savefig(output_dir / "coupling_index_per_layer.png", dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved coupling index per layer to {output_dir / 'coupling_index_per_layer.png'}")
        plt.close()


def create_metrics_summary_table(reports_root_dir, output_dir):
    """Create a summary table of all metrics across layers and probes."""
    output_dir = Path(output_dir)
    
    layer_ids = [5, 6, 7, 8, 9, 10, 11]
    summary_data = []
    
    for layer_id in layer_ids:
        # Topology metrics
        topo_metrics = load_probe_metrics(Path(reports_root_dir), layer_id, "topology")
        if topo_metrics:
            summary_data.append({
                'Layer': layer_id,
                'Probe Type': 'GCN (Topology)',
                'Accuracy (%)': f"{topo_metrics['accuracy']*100:.2f}",
                'Precision': f"{topo_metrics['precision']:.4f}",
                'Recall': f"{topo_metrics['recall']:.4f}",
                'F1 Score': f"{topo_metrics['f1']:.4f}",
                'Above Chance': 'Yes' if topo_metrics['accuracy'] > 0.5 else 'No'
            })
        
        # Coupling index
        coupling_data = load_coupling_index(Path(reports_root_dir), layer_id)
        if coupling_data:
            summary_data.append({
                'Layer': layer_id,
                'Probe Type': 'Coupling Index',
                'C_TT': f"{coupling_data['c_tt']:.4f}",
                'C_HH': f"{coupling_data['c_hh']:.4f}",
                'C_TH': f"{coupling_data['c_th']:.4f}",
                'C': f"{coupling_data['c']:.4f}",
                'Positive %': f"{coupling_data['positive_ratio']*100:.1f}"
            })
    
    # Save as CSV
    import pandas as pd
    df = pd.DataFrame(summary_data)
    df.to_csv(output_dir / "comparison_summary.csv", index=False)
    logger.info(f"Saved summary table to {output_dir / 'comparison_summary.csv'}")
    
    # Print to console
    print("\n" + "="*100)
    print("HALLUCINATION DETECTION - TOPOLOGY VS ACTIVATION COMPARISON")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python comparison.py <reports_root_dir> [output_dir]")
        print("Example: python comparison.py results/hallucination_analysis/Qwen_Qwen2_5_0_5B/")
        sys.exit(1)
    
    reports_root = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else reports_root
    
    create_comparison_figure(reports_root, output)
    create_metrics_summary_table(reports_root, output)
