#!/usr/bin/env python3
"""
Comprehensive Pipeline for Hallucination Detection and Graph Statistics Analysis

This script implements the complete workflow for:
1. Data preparation (downloading TruthfulQA dataset)
2. Computing neural topology (extracting hidden-state correlations)
3. Training hallucination detection probes
4. Evaluating trained probes
5. Performing graph statistics analysis

Author: Based on LLM Graph Probing framework
Date: December 19, 2025
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import argparse
import logging
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hallucination_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class HallucinationPipeline:
    """Main pipeline class for hallucination detection and graph analysis"""
    
    def __init__(
        self,
        llm_model_name: str = "qwen2.5-0.5b",
        ckpt_step: int = -1,
        llm_layer: int = 12,
        batch_size: int = 16,
        eval_batch_size: int = 16,
        gpu_id: int = 0,
        network_density: float = 1.0,
        num_workers: int = 4,
        probe_input: str = "corr",
        num_layers: int = 1,
        hidden_channels: int = 32,
        lr: float = 0.001,
        dropout: float = 0.0
    ):
        """
        Initialize the hallucination detection pipeline.
        
        Args:
            llm_model_name: Name of the LLM model to probe (e.g., 'qwen2.5-0.5b', 'pythia-70m')
            ckpt_step: Checkpoint step for the model (-1 for final checkpoint)
            llm_layer: Layer ID to extract features from
            batch_size: Batch size for data processing
            eval_batch_size: Batch size for evaluation
            gpu_id: GPU device ID to use (will auto-adjust for Mac/CPU)
            network_density: Density of neural graph (1.0 = full graph, <1.0 = sparse)
            num_workers: Number of worker processes
            probe_input: Type of probe input ('activation' or 'corr' for correlation)
            num_layers: Number of GNN layers (>0 for GNN, 0 for linear, <0 for MLP)
            hidden_channels: Number of hidden channels in the probe
            lr: Learning rate for training
            dropout: Dropout rate
        """
        self.llm_model_name = llm_model_name
        self.ckpt_step = ckpt_step
        self.llm_layer = llm_layer
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        
        # Auto-detect best available device (CUDA → MPS → CPU)
        import torch
        if torch.cuda.is_available():
            # Respect user-provided gpu_id for CUDA
            self.gpu_id = gpu_id
            self.num_workers = num_workers
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Use Apple MPS when CUDA is unavailable
            if gpu_id >= 0:
                logger.warning("CUDA not available, defaulting to Apple MPS (gpu_id=-2)")
                self.gpu_id = -2  # sentinel for MPS
            else:
                self.gpu_id = -2 if gpu_id == -1 else gpu_id
            # Multiprocessing with MPS can be flaky; keep workers to 1
            if num_workers > 1:
                logger.warning(f"Reducing num_workers from {num_workers} to 1 on MPS/CPU to avoid multiprocessing issues")
                self.num_workers = 1
            else:
                self.num_workers = num_workers
        else:
            # CPU fallback
            if gpu_id >= 0:
                logger.warning(f"CUDA not available (detected {platform.system()}). Using CPU mode (gpu_id=-1)")
                self.gpu_id = -1
            else:
                self.gpu_id = gpu_id
            if num_workers > 1:
                logger.warning(f"Reducing num_workers from {num_workers} to 1 on MPS/CPU to avoid multiprocessing issues")
                self.num_workers = 1
            else:
                self.num_workers = num_workers
        
        self.network_density = network_density
        self.probe_input = probe_input
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.lr = lr
        self.dropout = dropout
        self.dropout = dropout
        
        # Define paths
        self.data_dir = Path("data/hallucination")
        self.dataset_file = self.data_dir / "truthfulqa-validation.csv"
        self.save_dir = Path("saves/hallucination") / self.llm_model_name / f"layer_{self.llm_layer}"
        
        logger.info(f"Initialized pipeline for model: {llm_model_name}, layer: {llm_layer}")
        if self.gpu_id == -2:
            logger.info("Running on Apple MPS")
        elif self.gpu_id == -1:
            logger.info("Running in CPU mode")
    
    def _run_command(self, command: List[str], description: str) -> bool:
        """
        Execute a shell command and handle errors.
        
        Args:
            command: List of command components
            description: Description of the step being executed
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"{'='*80}")
        logger.info(f"STEP: {description}")
        logger.info(f"Command: {' '.join(command)}")
        logger.info(f"{'='*80}")
        
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=False,
                text=True
            )
            logger.info(f"✓ {description} completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ {description} failed with error code {e.returncode}")
            logger.error(f"Error: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ {description} failed with exception: {e}")
            return False
    
    def step_1_prepare_data(self) -> bool:
        """
        STEP 1: Data Preparation
        
        Downloads and prepares the TruthfulQA validation dataset.
        This step generates a CSV file with questions, answers, and labels
        (1 for true answers, 0 for hallucinated/false answers).
        
        Output: data/hallucination/truthfulqa-validation.csv
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 1: DATA PREPARATION")
        logger.info("="*80)
        logger.info("Downloading TruthfulQA dataset from HuggingFace...")
        logger.info("Dataset will be saved to: {}".format(self.dataset_file))
        
        # Check if data already exists
        if self.dataset_file.exists():
            logger.warning(f"Dataset already exists at {self.dataset_file}")
            user_input = input("Do you want to re-download? (y/N): ").lower()
            if user_input != 'y':
                logger.info("Skipping data download")
                return True
        
        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Run the construct_dataset script
        command = [
            sys.executable, "-m", "hallucination.construct_dataset",
            "--dataset_name", "truthfulqa",
            "--output_dir", str(self.data_dir)
        ]
        
        return self._run_command(
            command,
            "Downloading and constructing TruthfulQA dataset"
        )
    
    def step_2_compute_neural_topology(
        self,
        sparse: bool = False,
        resume: bool = False,
        gpu_id_list: Optional[List[int]] = None
    ) -> bool:
        """
        STEP 2: Compute Neural Topology
        
        Extracts hidden-state correlations (neural topology) for every answer
        in the TruthfulQA dataset. This step processes the LLM and computes
        correlation matrices between neurons for each text sample.
        
        Args:
            sparse: Whether to generate sparse graphs (saves memory)
            resume: Resume from the last checkpoint if interrupted
            gpu_id_list: List of GPU IDs to use (for multi-GPU processing)
            
        Output: data/hallucination/<model_name>[_step<ckpt>]/<idx>/layer_<layer>_corr.npy
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 2: COMPUTE NEURAL TOPOLOGY")
        logger.info("="*80)
        logger.info(f"Model: {self.llm_model_name}")
        logger.info(f"Layer: {self.llm_layer}")
        logger.info(f"Network density: {self.network_density}")
        logger.info(f"Sparse mode: {sparse}")
        
        if gpu_id_list is None:
            gpu_id_list = [self.gpu_id]
        
        # Build command (use sys.executable to ensure same Python environment)
        command = [
            sys.executable, "-m", "hallucination.compute_llm_network",
            "--dataset_name", "truthfulqa",
            "--llm_model_name", self.llm_model_name,
            "--ckpt_step", str(self.ckpt_step),
            "--llm_layer", str(self.llm_layer),
            "--batch_size", str(self.batch_size),
            "--network_density", str(self.network_density),
            "--num_workers", str(self.num_workers),
        ]
        
        # Add GPU IDs
        for gpu_id in gpu_id_list:
            command.extend(["--gpu_id", str(gpu_id)])
        
        # Add optional flags
        if sparse:
            command.append("--sparse")
        if resume:
            command.append("--resume")
        
        return self._run_command(
            command,
            "Computing neural topology (hidden-state correlations)"
        )
    
    def step_3_train_probes(
        self,
        from_sparse_data: bool = False,
        num_epochs: int = 100,
        early_stop_patience: int = 20,
        test_set_ratio: float = 0.2
    ) -> bool:
        """
        STEP 3: Train Hallucination Detection Probes
        
        Trains a probe (GNN, MLP, or linear) on the extracted neural topology
        to classify truthful vs. hallucinated answers.
        
        Args:
            from_sparse_data: Whether the topology data is sparse
            num_epochs: Maximum number of training epochs
            early_stop_patience: Number of epochs without improvement before stopping
            test_set_ratio: Ratio of data to use for testing
            
        Output: saves/hallucination/<model_name>/layer_<layer>/best_model_*.pth
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 3: TRAIN HALLUCINATION DETECTION PROBES")
        logger.info("="*80)
        logger.info(f"Probe input type: {self.probe_input}")
        logger.info(f"Number of layers: {self.num_layers}")
        logger.info(f"Hidden channels: {self.hidden_channels}")
        logger.info(f"Learning rate: {self.lr}")
        logger.info(f"Dropout: {self.dropout}")
        logger.info(f"Training epochs: {num_epochs}")
        
        # Build command (use sys.executable to ensure same Python environment)
        command = [
            sys.executable, "-m", "hallucination.train",
            "--dataset_name", "truthfulqa",
            "--llm_model_name", self.llm_model_name,
            "--ckpt_step", str(self.ckpt_step),
            "--llm_layer", str(self.llm_layer),
            "--probe_input", self.probe_input,
            "--density", str(self.network_density),
            "--batch_size", str(self.batch_size),
            "--eval_batch_size", str(self.eval_batch_size),
            "--num_layers", str(self.num_layers),
            "--hidden_channels", str(self.hidden_channels),
            "--dropout", str(self.dropout),
            "--lr", str(self.lr),
            "--num_epochs", str(num_epochs),
            "--early_stop_patience", str(early_stop_patience),
            "--test_set_ratio", str(test_set_ratio),
            "--gpu_id", str(self.gpu_id),
        ]
        
        # Add optional flags
        if from_sparse_data:
            command.append("--from_sparse_data")
        
        return self._run_command(
            command,
            "Training hallucination detection probes"
        )
    
    def step_4_evaluate_probes(self) -> bool:
        """
        STEP 4: Evaluate Trained Probes
        
        Loads the best checkpoint and reports comprehensive metrics:
        - Accuracy
        - Precision
        - Recall
        - F1 score
        - Confusion matrix
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 4: EVALUATE TRAINED PROBES")
        logger.info("="*80)
        logger.info(f"Loading best model from: {self.save_dir}")
        
        # Build command (use sys.executable to ensure same Python environment)
        command = [
            sys.executable, "-m", "hallucination.eval",
            "--dataset_name", "truthfulqa",
            "--llm_model_name", self.llm_model_name,
            "--ckpt_step", str(self.ckpt_step),
            "--llm_layer", str(self.llm_layer),
            "--probe_input", self.probe_input,
            "--density", str(self.network_density),
            "--num_layers", str(self.num_layers),
            "--gpu_id", str(self.gpu_id),
        ]
        
        return self._run_command(
            command,
            "Evaluating hallucination detection probes"
        )
    
    def step_5_graph_statistics_analysis(self, feature: str = "corr") -> bool:
        """
        STEP 5: Graph Statistics Analysis
        
        Analyzes the topology differences between truthful and hallucinated answers.
        Computes:
        - Intra-answer correlations (within true answers, within false answers)
        - Inter-answer correlations (between true and false answers)
        - Statistical significance of topology differences
        
        Args:
            feature: Type of feature to analyze ('corr' or 'activation')
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 5: GRAPH STATISTICS ANALYSIS")
        logger.info("="*80)
        logger.info(f"Analyzing {feature} features at layer {self.llm_layer}")
        logger.info("Computing intra- and inter-answer topology correlations...")
        
        # Build command (use sys.executable to ensure same Python environment)
        command = [
            sys.executable, "-m", "hallucination.graph_analysis",
            "--llm_model_name", self.llm_model_name,
            "--ckpt_step", str(self.ckpt_step),
            "--layer", str(self.llm_layer),
            "--feature", feature
        ]
        
        return self._run_command(
            command,
            "Performing graph statistics analysis"
        )
    
    def run_full_pipeline(
        self,
        skip_data_prep: bool = False,
        skip_topology: bool = False,
        skip_training: bool = False,
        skip_evaluation: bool = False,
        skip_analysis: bool = False
    ) -> Dict[str, bool]:
        """
        Execute the complete hallucination detection pipeline.
        
        Args:
            skip_data_prep: Skip data preparation if already done
            skip_topology: Skip neural topology computation if already done
            skip_training: Skip training if model already trained
            skip_evaluation: Skip evaluation
            skip_analysis: Skip graph statistics analysis
            
        Returns:
            Dictionary with status of each step
        """
        logger.info("\n" + "="*80)
        logger.info("STARTING FULL HALLUCINATION DETECTION PIPELINE")
        logger.info("="*80)
        logger.info(f"Model: {self.llm_model_name}")
        logger.info(f"Layer: {self.llm_layer}")
        logger.info(f"Probe type: {self.probe_input}")
        logger.info("="*80 + "\n")
        
        results = {}
        
        # Step 1: Data Preparation
        if not skip_data_prep:
            results['data_preparation'] = self.step_1_prepare_data()
            if not results['data_preparation']:
                logger.error("Pipeline stopped due to data preparation failure")
                return results
        else:
            logger.info("Skipping Step 1: Data Preparation")
            results['data_preparation'] = True
        
        # Step 2: Compute Neural Topology
        if not skip_topology:
            results['neural_topology'] = self.step_2_compute_neural_topology()
            if not results['neural_topology']:
                logger.error("Pipeline stopped due to neural topology computation failure")
                return results
        else:
            logger.info("Skipping Step 2: Neural Topology Computation")
            results['neural_topology'] = True
        
        # Step 3: Train Probes
        if not skip_training:
            results['training'] = self.step_3_train_probes()
            if not results['training']:
                logger.error("Pipeline stopped due to training failure")
                return results
        else:
            logger.info("Skipping Step 3: Probe Training")
            results['training'] = True
        
        # Step 4: Evaluate Probes
        if not skip_evaluation:
            results['evaluation'] = self.step_4_evaluate_probes()
        else:
            logger.info("Skipping Step 4: Probe Evaluation")
            results['evaluation'] = True
        
        # Step 5: Graph Statistics Analysis
        if not skip_analysis:
            results['graph_analysis'] = self.step_5_graph_statistics_analysis()
        else:
            logger.info("Skipping Step 5: Graph Statistics Analysis")
            results['graph_analysis'] = True
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("="*80)
        for step, status in results.items():
            status_str = "✓ SUCCESS" if status else "✗ FAILED"
            logger.info(f"{step.upper()}: {status_str}")
        logger.info("="*80 + "\n")
        
        return results


def main():
    """Main function to run the pipeline with command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Hallucination Detection and Graph Statistics Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model parameters
    parser.add_argument("--llm_model_name", type=str, default="qwen2.5-0.5b",
                        help="Name of the LLM model to probe")
    parser.add_argument("--ckpt_step", type=int, default=-1,
                        help="Checkpoint step (-1 for final)")
    parser.add_argument("--llm_layer", type=int, default=12,
                        help="Layer ID to extract features from")
    
    # Data processing parameters
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for processing")
    parser.add_argument("--eval_batch_size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU device ID")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of worker processes")
    
    # Network parameters
    parser.add_argument("--network_density", type=float, default=1.0,
                        help="Density of neural graph (0.0-1.0)")
    parser.add_argument("--sparse", action="store_true",
                        help="Use sparse graph representation")
    
    # Probe parameters
    parser.add_argument("--probe_input", type=str, default="corr",
                        choices=["activation", "corr"],
                        help="Type of probe input")
    parser.add_argument("--num_layers", type=int, default=1,
                        help="Number of layers (>0: GNN, 0: linear, <0: MLP)")
    parser.add_argument("--hidden_channels", type=int, default=32,
                        help="Number of hidden channels")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout rate")
    
    # Pipeline control
    parser.add_argument("--skip_data_prep", action="store_true",
                        help="Skip data preparation step")
    parser.add_argument("--skip_topology", action="store_true",
                        help="Skip neural topology computation")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip probe training")
    parser.add_argument("--skip_evaluation", action="store_true",
                        help="Skip probe evaluation")
    parser.add_argument("--skip_analysis", action="store_true",
                        help="Skip graph statistics analysis")
    
    # Individual step execution
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4, 5],
                        help="Run only a specific step (1-5)")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = HallucinationPipeline(
        llm_model_name=args.llm_model_name,
        ckpt_step=args.ckpt_step,
        llm_layer=args.llm_layer,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        gpu_id=args.gpu_id,
        network_density=args.network_density,
        num_workers=args.num_workers,
        probe_input=args.probe_input,
        num_layers=args.num_layers,
        hidden_channels=args.hidden_channels,
        lr=args.lr,
        dropout=args.dropout
    )
    
    # Run specific step or full pipeline
    if args.step:
        logger.info(f"Running only Step {args.step}")
        if args.step == 1:
            success = pipeline.step_1_prepare_data()
        elif args.step == 2:
            success = pipeline.step_2_compute_neural_topology(sparse=args.sparse)
        elif args.step == 3:
            success = pipeline.step_3_train_probes()
        elif args.step == 4:
            success = pipeline.step_4_evaluate_probes()
        elif args.step == 5:
            success = pipeline.step_5_graph_statistics_analysis()
        
        if success:
            logger.info(f"Step {args.step} completed successfully!")
            sys.exit(0)
        else:
            logger.error(f"Step {args.step} failed!")
            sys.exit(1)
    else:
        # Run full pipeline
        results = pipeline.run_full_pipeline(
            skip_data_prep=args.skip_data_prep,
            skip_topology=args.skip_topology,
            skip_training=args.skip_training,
            skip_evaluation=args.skip_evaluation,
            skip_analysis=args.skip_analysis
        )
        
        # Exit with appropriate code
        if all(results.values()):
            logger.info("All pipeline steps completed successfully!")
            sys.exit(0)
        else:
            logger.error("Some pipeline steps failed. Check logs for details.")
            sys.exit(1)


if __name__ == "__main__":
    main()
