"""
Compute neural topology coupling index for hallucination detection.

Replicates section 5.2 of 2506.01042v2.pdf

The coupling index C measures the difference between intra-group similarity 
(truthful-to-truthful and hallucinated-to-hallucinated) and inter-group 
similarity (truthful-to-hallucinated).

Equations 9-11 from the paper:
  C_TT = AVG({ρ(A_i, A_j) | A_i, A_j ∈ A_T})
  C_HH = AVG({ρ(A_i, A_j) | A_i, A_j ∈ A_H})
  C_TH = AVG({ρ(A_i, A_j) | A_i ∈ A_T, A_j ∈ A_H})
  C = C_TT + C_HH - 2*C_TH

where:
  - A_T: set of neural topologies (correlation matrices) for truthful responses
  - A_H: set of neural topologies for hallucinated responses
  - ρ: Pearson correlation between flattened adjacency matrices
"""

import numpy as np
import logging
from pathlib import Path
from scipy.stats import pearsonr
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def flatten_matrix(matrix):
    """Flatten a 2D correlation matrix to 1D."""
    return matrix.flatten()


def compute_pairwise_correlations(matrices_list):
    """
    Compute pairwise Pearson correlations between flattened matrices.
    
    Args:
        matrices_list: List of 2D correlation matrices
    
    Returns:
        correlations: List of correlation values
    """
    correlations = []
    flattened = [flatten_matrix(m) for m in matrices_list]
    
    n = len(flattened)
    for i in range(n):
        for j in range(i + 1, n):
            try:
                corr, _ = pearsonr(flattened[i], flattened[j])
                correlations.append(corr)
            except Exception as e:
                logger.warning(f"Could not compute correlation between matrices {i} and {j}: {e}")
                continue
    
    return correlations


def compute_coupling_index(topology_root, layer_id, labels_path=None):
    """
    Compute neural topology coupling index for a given layer.
    
    Args:
        topology_root: Path to root directory containing question topologies
        layer_id: Layer number (e.g., 5, 6, 7, ...)
        labels_path: Path to CSV with labels (question_id, label). 
                    If None, will look for label in filename or assume alternating pattern
    
    Returns:
        dict: Contains C_TT, C_HH, C_TH, C, and per-sample coupling indices
    """
    
    topology_root = Path(topology_root)
    if not topology_root.exists():
        raise FileNotFoundError(f"Topology root not found: {topology_root}")
    
    # Load labels
    labels = {}
    if labels_path and Path(labels_path).exists():
        import pandas as pd
        df = pd.read_csv(labels_path)
        labels = dict(zip(df['question_id'], df['label']))
    
    # Group topologies by label
    truthful_topologies = []
    hallucinated_topologies = []
    per_sample_indices = []  # (question_id, label, coupling_index)
    
    question_dirs = sorted([d for d in topology_root.iterdir() if d.is_dir()], 
                          key=lambda x: int(x.name))
    
    logger.info(f"Found {len(question_dirs)} questions in {topology_root}")
    
    topology_data = {}
    
    for q_dir in question_dirs:
        q_id = int(q_dir.name)
        corr_path = q_dir / f"layer_{layer_id}_corr.npy"
        
        if not corr_path.exists():
            continue
        
        try:
            corr_matrix = np.load(corr_path)
            # Sanitize NaN/Inf
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Determine label (0=false, 1=true)
            if q_id in labels:
                label = labels[q_id]
            else:
                # Assume alternating pattern: even=true, odd=false (or vice versa)
                # This depends on how the dataset was constructed
                label = q_id % 2  # Adjust as needed
            
            topology_data[q_id] = {
                'matrix': corr_matrix,
                'label': label
            }
            
            if label == 1:  # Truthful
                truthful_topologies.append(corr_matrix)
            else:  # Hallucinated
                hallucinated_topologies.append(corr_matrix)
        
        except Exception as e:
            logger.warning(f"Could not load topology for question {q_id}: {e}")
            continue
    
    logger.info(f"Truthful topologies: {len(truthful_topologies)}")
    logger.info(f"Hallucinated topologies: {len(hallucinated_topologies)}")
    
    if len(truthful_topologies) < 2 or len(hallucinated_topologies) < 2:
        logger.error("Not enough topologies in each group to compute correlations")
        return None
    
    # Compute intra-group and inter-group correlations
    logger.info("Computing correlations...")
    corr_tt = compute_pairwise_correlations(truthful_topologies)
    corr_hh = compute_pairwise_correlations(hallucinated_topologies)
    
    # For inter-group: compare each truthful with each hallucinated
    corr_th = []
    for t_matrix in truthful_topologies:
        for h_matrix in hallucinated_topologies:
            try:
                corr, _ = pearsonr(flatten_matrix(t_matrix), flatten_matrix(h_matrix))
                corr_th.append(corr)
            except Exception as e:
                logger.warning(f"Could not compute inter-group correlation: {e}")
                continue
    
    # Compute coupling index
    c_tt = np.mean(corr_tt) if corr_tt else 0
    c_hh = np.mean(corr_hh) if corr_hh else 0
    c_th = np.mean(corr_th) if corr_th else 0
    
    c = c_tt + c_hh - 2 * c_th
    
    # Compute per-sample coupling index
    # For each sample, compute how much more similar it is to same-label samples than different-label samples
    for q_id, data in topology_data.items():
        matrix = data['matrix']
        label = data['label']
        
        if label == 1:  # Truthful
            # Compare to other truthful samples
            intra_corr = []
            for other_q_id, other_data in topology_data.items():
                if other_q_id != q_id and other_data['label'] == 1:
                    try:
                        corr, _ = pearsonr(flatten_matrix(matrix), 
                                         flatten_matrix(other_data['matrix']))
                        intra_corr.append(corr)
                    except:
                        continue
            
            # Compare to hallucinated samples
            inter_corr = []
            for other_q_id, other_data in topology_data.items():
                if other_data['label'] == 0:
                    try:
                        corr, _ = pearsonr(flatten_matrix(matrix), 
                                         flatten_matrix(other_data['matrix']))
                        inter_corr.append(corr)
                    except:
                        continue
        else:  # Hallucinated
            # Compare to other hallucinated samples
            intra_corr = []
            for other_q_id, other_data in topology_data.items():
                if other_q_id != q_id and other_data['label'] == 0:
                    try:
                        corr, _ = pearsonr(flatten_matrix(matrix), 
                                         flatten_matrix(other_data['matrix']))
                        intra_corr.append(corr)
                    except:
                        continue
            
            # Compare to truthful samples
            inter_corr = []
            for other_q_id, other_data in topology_data.items():
                if other_data['label'] == 1:
                    try:
                        corr, _ = pearsonr(flatten_matrix(matrix), 
                                         flatten_matrix(other_data['matrix']))
                        inter_corr.append(corr)
                    except:
                        continue
        
        if intra_corr and inter_corr:
            c_sample = np.mean(intra_corr) - np.mean(inter_corr)
            per_sample_indices.append({
                'question_id': q_id,
                'label': label,
                'coupling_index': float(c_sample),
                'intra_mean': float(np.mean(intra_corr)),
                'inter_mean': float(np.mean(inter_corr))
            })
    
    # Compute statistics
    c_indices = [s['coupling_index'] for s in per_sample_indices]
    positive_ratio = sum(1 for c in c_indices if c > 0) / len(c_indices) if c_indices else 0
    
    results = {
        'layer': layer_id,
        'c_tt': float(c_tt),
        'c_hh': float(c_hh),
        'c_th': float(c_th),
        'c': float(c),
        'num_truthful': len(truthful_topologies),
        'num_hallucinated': len(hallucinated_topologies),
        'num_samples': len(per_sample_indices),
        'positive_ratio': float(positive_ratio),
        'coupling_index_mean': float(np.mean(c_indices)) if c_indices else 0,
        'coupling_index_std': float(np.std(c_indices)) if c_indices else 0,
        'coupling_index_median': float(np.median(c_indices)) if c_indices else 0,
        'coupling_index_min': float(np.min(c_indices)) if c_indices else 0,
        'coupling_index_max': float(np.max(c_indices)) if c_indices else 0,
        'per_sample_indices': per_sample_indices[:100]  # Store first 100 for inspection
    }
    
    logger.info(f"Coupling Index (Layer {layer_id}):")
    logger.info(f"  C_TT (truthful-truthful): {c_tt:.4f}")
    logger.info(f"  C_HH (hallucinated-hallucinated): {c_hh:.4f}")
    logger.info(f"  C_TH (truthful-hallucinated): {c_th:.4f}")
    logger.info(f"  C (coupling index): {c:.4f}")
    logger.info(f"  Positive C ratio: {positive_ratio:.2%} ({sum(1 for c in c_indices if c > 0)}/{len(c_indices)})")
    
    return results


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python coupling_index.py <topology_root> <layer_id> [labels_path]")
        sys.exit(1)
    
    topology_root = sys.argv[1]
    layer_id = int(sys.argv[2])
    labels_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    results = compute_coupling_index(topology_root, layer_id, labels_path)
    
    if results:
        output_path = Path(topology_root).parent / f"coupling_index_layer_{layer_id}.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
