from absl import app, flags, logging
from collections import defaultdict
import os
from tqdm import tqdm

import numpy as np
import pandas as pd

from pathlib import Path

main_dir = Path(os.environ.get('MAIN_DIR', '.'))

flags.DEFINE_enum(
    "dataset_name",
    "truthfulqa",
    ["truthfulqa", "halueval", "medhallu", "helm"],
    "The dataset to analyze."
)
flags.DEFINE_string("llm_model_name", "qwen2.5-0.5b", "The name of the LLM model.")
flags.DEFINE_integer("ckpt_step", -1, "The checkpoint step.")
flags.DEFINE_integer("layer", 12, "The layer to analyze.")
flags.DEFINE_enum("feature", "corr", ["corr", "activation"], "The feature to analyze.")
FLAGS = flags.FLAGS


def load_neural_topology_data(dataset_name, model_name, ckpt_step, layer, feature_name):
    """Load neural topology data for all question IDs and their true/false labels."""
    
    # Construct model directory path
    sanitized_model_name = model_name.replace('/', '_').replace('-', '_').replace('.', '_')
    if ckpt_step == -1:
        model_dir = main_dir / "data/hallucination" / dataset_name / sanitized_model_name
    else:
        model_dir = main_dir / "data/hallucination" / dataset_name / f"{sanitized_model_name}_step{ckpt_step}"
    
    # Load the original dataset to get question_id mappings
    dataset_file = main_dir / "data/hallucination" / f"{dataset_name}.csv"
    df = pd.read_csv(dataset_file)
    
    topology_data = defaultdict(lambda: {'true': [], 'false': []})
    
    # Iterate through each row (each row index corresponds to a directory)
    for idx, row in df.iterrows():
        question_id = row['question_id']
        label = row['label']
        
        # The directory name is the row index
        data_dir = os.path.join(model_dir, str(idx))
        feature_file = os.path.join(data_dir, f"layer_{layer}_{feature_name}.npy")

        if os.path.exists(feature_file):
            feature = np.load(feature_file)

            if feature_name == "corr":
                # Get upper triangular part and flatten
                triu_indices = np.triu_indices(feature.shape[0], k=1)
                feature = feature[triu_indices]

            if label == 1:  # True answer
                topology_data[question_id]['true'].append(feature)
            else:  # False answer
                topology_data[question_id]['false'].append(feature)
    
    return topology_data


def calculate_correlation_metrics(topology_data):
    """Calculate correlation metrics for each question_id."""
    
    question_metrics = {}
    
    for question_id, data in topology_data.items():
        true_topologies = data['true']
        false_topologies = data['false']

        if len(true_topologies) < 2 or len(false_topologies) < 2:
            continue
            
        # Concatenate all topologies for this question
        all_topologies = true_topologies + false_topologies
        
        # Create correlation matrix between all topology vectors
        topology_matrix = np.array(all_topologies)
        
        # Check for constant/zero-variance vectors that would cause NaN in corrcoef
        valid_mask = np.std(topology_matrix, axis=1) > 1e-10
        if not valid_mask.all():
            # Skip questions with zero-variance vectors (constant features)
            continue
        
        corr_matrix = np.corrcoef(topology_matrix)
        
        # Additional check: skip if corrcoef still produced NaN (edge case with identical vectors)
        if np.isnan(corr_matrix).any():
            continue
        
        # Calculate indices for true-true, false-false, and true-false pairs
        num_true = len(true_topologies)
        num_false = len(false_topologies)
        
        # True-True correlations (upper triangular of true block)
        if num_true > 1:
            true_true_corrs = []
            for i in range(num_true):
                for j in range(i + 1, num_true):
                    val = corr_matrix[i, j]
                    if not np.isnan(val):
                        true_true_corrs.append(val)
            avg_true_true = np.mean(true_true_corrs) if true_true_corrs else 0
        else:
            avg_true_true = 0
            
        # False-False correlations (upper triangular of false block)
        if num_false > 1:
            false_false_corrs = []
            for i in range(num_true, num_true + num_false):
                for j in range(i + 1, num_true + num_false):
                    val = corr_matrix[i, j]
                    if not np.isnan(val):
                        false_false_corrs.append(val)
            avg_false_false = np.mean(false_false_corrs) if false_false_corrs else 0
        else:
            avg_false_false = 0
            
        # True-False correlations (cross-block)
        true_false_corrs = []
        for i in range(num_true):
            for j in range(num_true, num_true + num_false):
                val = corr_matrix[i, j]
                if not np.isnan(val):
                    true_false_corrs.append(val)
        avg_true_false = np.mean(true_false_corrs) if true_false_corrs else 0
        
        # Calculate the metric: avg_true_true + avg_false_false - 2 * avg_true_false
        intra_vs_inter_metric = avg_true_true + avg_false_false - 2 * avg_true_false
        
        # Skip if any metric is NaN
        if np.isnan(intra_vs_inter_metric) or np.isnan(avg_true_true) or np.isnan(avg_false_false) or np.isnan(avg_true_false):
            continue
        
        question_metrics[question_id] = {
            'avg_true_true': avg_true_true,
            'avg_false_false': avg_false_false,
            'avg_true_false': avg_true_false,
            'intra_vs_inter_metric': intra_vs_inter_metric,
            'num_true': num_true,
            'num_false': num_false
        }
    
    return question_metrics


def print_statistics(question_metrics):
    """Print comprehensive statistics of the metrics."""
    
    if not question_metrics:
        print("No data available for analysis.")
        return
        
    # Extract all metric values
    intra_vs_inter_values = [metrics['intra_vs_inter_metric'] for metrics in question_metrics.values()]
    true_true_values = [metrics['avg_true_true'] for metrics in question_metrics.values()]
    false_false_values = [metrics['avg_false_false'] for metrics in question_metrics.values()]
    true_false_values = [metrics['avg_true_false'] for metrics in question_metrics.values()]
    
    # Print individual question results
    print(f"\n--- Per-Question Results ---")
    for question_id in sorted(question_metrics.keys()):
        metrics = question_metrics[question_id]
        print(f"Q{question_id:3d}: {metrics['intra_vs_inter_metric']:8.6f} "
              f"(T-T: {metrics['avg_true_true']:6.4f}, "
              f"F-F: {metrics['avg_false_false']:6.4f}, "
              f"T-F: {metrics['avg_true_false']:6.4f}, "
              f"#T: {metrics['num_true']}, #F: {metrics['num_false']})")

    print(f"\n=== Neural Topology Correlation Analysis ===")
    print(f"Model: {FLAGS.llm_model_name}")
    if FLAGS.ckpt_step != -1:
        print(f"Checkpoint Step: {FLAGS.ckpt_step}")
    print(f"Layer: {FLAGS.layer}")
    print(f"Number of questions analyzed: {len(question_metrics)}")
    
    print(f"\n--- Intra vs Inter-group Metric (true-true + false-false - 2*true-false) ---")
    print(f"Mean: {np.mean(intra_vs_inter_values):.6f}")
    print(f"Std:  {np.std(intra_vs_inter_values):.6f}")
    print(f"Median: {np.median(intra_vs_inter_values):.6f}")
    print(f"Min: {np.min(intra_vs_inter_values):.6f}")
    print(f"Max: {np.max(intra_vs_inter_values):.6f}")
    print(f">0 ratio: {np.mean(np.array(intra_vs_inter_values) > 0):.4f}")
    
    print(f"\n--- True-True Correlations ---")
    print(f"Mean: {np.mean(true_true_values):.6f}")
    print(f"Std:  {np.std(true_true_values):.6f}")
    print(f"Median: {np.median(true_true_values):.6f}")
    print(f"Min: {np.min(true_true_values):.6f}")
    print(f"Max: {np.max(true_true_values):.6f}")

    print(f"\n--- False-False Correlations ---")
    print(f"Mean: {np.mean(false_false_values):.6f}")
    print(f"Std:  {np.std(false_false_values):.6f}")
    print(f"Median: {np.median(false_false_values):.6f}")
    print(f"Min: {np.min(false_false_values):.6f}")
    print(f"Max: {np.max(false_false_values):.6f}")
    
    print(f"\n--- True-False Correlations ---")
    print(f"Mean: {np.mean(true_false_values):.6f}")
    print(f"Std:  {np.std(true_false_values):.6f}")
    print(f"Median: {np.median(true_false_values):.6f}")
    print(f"Min: {np.min(true_false_values):.6f}")
    print(f"Max: {np.max(true_false_values):.6f}")

    return intra_vs_inter_values
    

def main(_):
    # Suppress PyTorch/PyG warnings
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    # Configure absl logging format
    logging.use_absl_handler()
    import logging as stdlib_logging
    absl_handler = logging.get_absl_handler()
    absl_handler.setFormatter(stdlib_logging.Formatter('%(filename)s:%(lineno)d - %(message)s'))
    logging.info("\n\n" + "="*10)
    logging.info("STEP 5: GRAPH ANALYSIS (NEURAL TOPOLOGY CORRELATION)")
    logging.info("="*10)
    
    # ===== CONFIGURATION =====
    model_name = FLAGS.llm_model_name
    # Sanitize model name for file paths (replace /, -, and . with _)
    model_tag = model_name.replace('/', '_').replace('-', '_').replace('.', '_')
    if FLAGS.ckpt_step == -1:
        model_dir = main_dir / "data/hallucination" / FLAGS.dataset_name / model_tag
    else:
        model_dir = main_dir / "data/hallucination" / FLAGS.dataset_name / f"{model_tag}_step{FLAGS.ckpt_step}"
    
    logging.info(f"Dataset: {FLAGS.dataset_name}")
    logging.info(f"Model: {FLAGS.llm_model_name}")
    logging.info("="*10 + "\n")
    
    logging.info(f"Checkpoint step: {FLAGS.ckpt_step}")
    logging.info(f"Layer: {FLAGS.layer}")
    logging.info(f"Feature: {FLAGS.feature}")
    logging.info(f"Data directory: {model_dir}")
    # Prepare results directory under MAIN_DIR/reports/hallucination_analysis/{model_tag}/layer_{layer}
    reports_dir = main_dir / "reports" / "hallucination_analysis" / model_tag / f"layer_{FLAGS.layer}"
    try:
        reports_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    
    # ===== LOAD TOPOLOGY DATA =====
    logging.info("="*10)
    logging.info("Loading neural topology data...")
    logging.info("="*10)
    topology_data = load_neural_topology_data(FLAGS.dataset_name, FLAGS.llm_model_name, FLAGS.ckpt_step, FLAGS.layer, FLAGS.feature)
    logging.info(f"Loaded data for {len(topology_data)} questions")

    # ===== COMPUTE METRICS =====
    logging.info("="*10)
    logging.info("Calculating correlation metrics...")
    logging.info("="*10)
    question_metrics = calculate_correlation_metrics(topology_data)
    logging.info(f"Computed metrics for {len(question_metrics)} questions")
    
    # ===== STATISTICS & RESULTS =====
    logging.info("="*10)
    logging.info("Statistical Analysis:")
    logging.info("="*10)
    diff = print_statistics(question_metrics)
    
    output_file = reports_dir / f"{model_tag}_layer_{FLAGS.layer}_{FLAGS.feature}_intra_vs_inter.npy"
    np.save(output_file, np.array(diff))
    logging.info(f"\nSaved results to: {output_file}")
    
    logging.info("="*10)
    logging.info("✓ Graph analysis completed successfully")
    logging.info("="*10)
    
    # ===== COUPLING INDEX ANALYSIS (Section 5.2) =====
    # Compute neural topology coupling index for hallucination detection
    logging.info("="*10)
    logging.info("Computing Neural Topology Coupling Index...")
    logging.info("(Equations 9-11 from 2506.01042v2.pdf)")
    logging.info("="*10)
    
    try:
        from hallucination.coupling_index import compute_coupling_index
        
        coupling_results = compute_coupling_index(
            topology_root=str(model_dir),
            layer_id=FLAGS.layer
        )
        
        if coupling_results:
            coupling_file = reports_dir / f"coupling_index.json"
            import json
            with open(coupling_file, 'w') as f:
                json.dump(coupling_results, f, indent=2)
            logging.info(f"Coupling index results saved to: {coupling_file}")
            
            # Summary
            logging.info("\n--- Coupling Index Summary ---")
            logging.info(f"C_TT (truthful-truthful): {coupling_results['c_tt']:.4f}")
            logging.info(f"C_HH (hallucinated-hallucinated): {coupling_results['c_hh']:.4f}")
            logging.info(f"C_TH (truthful-hallucinated): {coupling_results['c_th']:.4f}")
            logging.info(f"C (coupling index): {coupling_results['c']:.4f}")
            logging.info(f"Samples with positive coupling: {coupling_results['positive_ratio']*100:.1f}%")
    
    except Exception as e:
        logging.warning(f"Could not compute coupling index: {e}")
    
    logging.info("="*10)
    logging.info("STEP 5 COMPLETE: Graph Analysis")
    logging.info("="*10)
    logging.info("✓ Graph analysis completed successfully")
    logging.info("="*10 + "\n\n")


if __name__ == "__main__":
    app.run(main)