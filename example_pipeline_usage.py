#!/usr/bin/env python3
"""
Example Script: Running Hallucination Detection Pipeline

This script demonstrates various ways to use the HallucinationPipeline class
programmatically for different use cases.

Usage:
    python example_pipeline_usage.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import the pipeline
sys.path.insert(0, str(Path(__file__).parent))

from hallucination_pipeline import HallucinationPipeline


def example_1_basic_pipeline():
    """
    Example 1: Run the complete pipeline with default settings
    
    This is the simplest use case - just initialize and run.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Pipeline Execution")
    print("="*80)
    
    # Initialize pipeline with default parameters
    pipeline = HallucinationPipeline(
        llm_model_name="qwen2.5-0.5b",
        llm_layer=12,
        probe_input="corr",
        num_layers=1
    )
    
    # Run the complete pipeline
    results = pipeline.run_full_pipeline()
    
    # Check results
    if all(results.values()):
        print("\n✓ Pipeline completed successfully!")
    else:
        print("\n✗ Some steps failed. Check logs for details.")
    
    return results


def example_2_step_by_step():
    """
    Example 2: Run pipeline step-by-step with custom parameters
    
    This gives you more control over each step and allows you to
    check intermediate results.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Step-by-Step Execution with Custom Parameters")
    print("="*80)
    
    # Initialize with custom parameters for a smaller model
    pipeline = HallucinationPipeline(
        llm_model_name="pythia-70m",      # Smaller model for testing
        llm_layer=5,                       # Middle layer
        batch_size=8,                      # Smaller batch for limited GPU
        eval_batch_size=16,
        probe_input="corr",                # Use correlation graphs
        num_layers=2,                      # 2-layer GNN
        hidden_channels=32,
        lr=0.001,
        dropout=0.1,
        gpu_id=0
    )
    
    # Step 1: Prepare data
    print("\n--- Running Step 1: Data Preparation ---")
    if not pipeline.step_1_prepare_data():
        print("Data preparation failed!")
        return
    
    # Step 2: Compute neural topology
    print("\n--- Running Step 2: Compute Neural Topology ---")
    if not pipeline.step_2_compute_neural_topology(sparse=False, resume=False):
        print("Neural topology computation failed!")
        return
    
    # Step 3: Train probes
    print("\n--- Running Step 3: Train Hallucination Probes ---")
    if not pipeline.step_3_train_probes(num_epochs=50, early_stop_patience=10):
        print("Training failed!")
        return
    
    # Step 4: Evaluate
    print("\n--- Running Step 4: Evaluate Probes ---")
    if not pipeline.step_4_evaluate_probes():
        print("Evaluation failed!")
        return
    
    # Step 5: Graph analysis
    print("\n--- Running Step 5: Graph Statistics Analysis ---")
    if not pipeline.step_5_graph_statistics_analysis(feature="corr"):
        print("Graph analysis failed!")
        return
    
    print("\n✓ All steps completed successfully!")


def example_3_multi_layer_comparison():
    """
    Example 3: Compare hallucination detection across multiple layers
    
    This example shows how to systematically analyze different layers
    to find which layer provides the best hallucination signal.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Multi-Layer Comparison")
    print("="*80)
    
    model_name = "qwen2.5-0.5b"
    layers_to_test = [6, 12, 18, 24]  # Test early, middle, and late layers
    
    results_summary = {}
    
    for layer in layers_to_test:
        print(f"\n{'='*80}")
        print(f"Testing Layer {layer}")
        print(f"{'='*80}")
        
        pipeline = HallucinationPipeline(
            llm_model_name=model_name,
            llm_layer=layer,
            probe_input="corr",
            num_layers=1,
            batch_size=16
        )
        
        # Skip data prep after first iteration
        skip_data = layer != layers_to_test[0]
        
        # Run pipeline
        results = pipeline.run_full_pipeline(
            skip_data_prep=skip_data,
            skip_topology=False,
            skip_training=False,
            skip_evaluation=False,
            skip_analysis=True  # Skip analysis for speed
        )
        
        results_summary[f"layer_{layer}"] = results
    
    # Print summary
    print("\n" + "="*80)
    print("MULTI-LAYER COMPARISON SUMMARY")
    print("="*80)
    for layer_key, results in results_summary.items():
        status = "✓" if all(results.values()) else "✗"
        print(f"{status} {layer_key}: {results}")


def example_4_architecture_comparison():
    """
    Example 4: Compare different probe architectures
    
    Tests Linear, MLP, and GNN probes to see which performs best.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Architecture Comparison")
    print("="*80)
    
    architectures = [
        {"name": "Linear", "num_layers": 0, "lr": 0.00001},
        {"name": "MLP-2layer", "num_layers": -2, "lr": 0.001},
        {"name": "GNN-1layer", "num_layers": 1, "lr": 0.001},
        {"name": "GNN-3layer", "num_layers": 3, "lr": 0.001}
    ]
    
    for arch in architectures:
        print(f"\n{'='*80}")
        print(f"Testing {arch['name']} Architecture")
        print(f"{'='*80}")
        
        pipeline = HallucinationPipeline(
            llm_model_name="qwen2.5-0.5b",
            llm_layer=12,
            probe_input="corr",
            num_layers=arch['num_layers'],
            lr=arch['lr'],
            hidden_channels=32
        )
        
        # Only run training and evaluation (assume data and topology exist)
        results = pipeline.run_full_pipeline(
            skip_data_prep=True,
            skip_topology=True,
            skip_training=False,
            skip_evaluation=False,
            skip_analysis=True
        )
        
        print(f"\n{arch['name']} Results: {results}")


def example_5_sparse_vs_dense():
    """
    Example 5: Compare sparse vs dense graph representations
    
    Shows how sparsification affects performance and efficiency.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Sparse vs Dense Graph Comparison")
    print("="*80)
    
    configurations = [
        {"name": "Dense (100%)", "density": 1.0, "sparse": False},
        {"name": "Sparse (50%)", "density": 0.5, "sparse": True},
        {"name": "Sparse (20%)", "density": 0.2, "sparse": True},
        {"name": "Sparse (10%)", "density": 0.1, "sparse": True}
    ]
    
    for config in configurations:
        print(f"\n{'='*80}")
        print(f"Testing {config['name']}")
        print(f"{'='*80}")
        
        pipeline = HallucinationPipeline(
            llm_model_name="qwen2.5-0.5b",
            llm_layer=12,
            network_density=config['density'],
            probe_input="corr",
            num_layers=2
        )
        
        # Run topology computation and training
        results = pipeline.run_full_pipeline(
            skip_data_prep=True,
            skip_topology=False,
            skip_training=False,
            skip_evaluation=False,
            skip_analysis=True
        )
        
        print(f"\n{config['name']} Results: {results}")


def example_6_resume_interrupted():
    """
    Example 6: Resume an interrupted pipeline
    
    Demonstrates how to continue from where you left off.
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: Resume Interrupted Pipeline")
    print("="*80)
    
    pipeline = HallucinationPipeline(
        llm_model_name="qwen2.5-0.5b",
        llm_layer=12
    )
    
    # Let's say Steps 1 and 2 were completed, but training was interrupted
    # Skip completed steps and resume from training
    results = pipeline.run_full_pipeline(
        skip_data_prep=True,      # Already done
        skip_topology=True,        # Already done
        skip_training=False,       # Resume from here
        skip_evaluation=False,
        skip_analysis=False
    )
    
    print(f"\nResume Results: {results}")


def example_7_custom_analysis():
    """
    Example 7: Custom analysis workflow
    
    Shows how to run just the analysis steps you need.
    """
    print("\n" + "="*80)
    print("EXAMPLE 7: Custom Analysis Workflow")
    print("="*80)
    
    pipeline = HallucinationPipeline(
        llm_model_name="qwen2.5-0.5b",
        llm_layer=12
    )
    
    # Scenario: You already have trained models, just want to:
    # 1. Re-evaluate with different metrics
    # 2. Run graph statistics analysis
    
    print("\n--- Re-evaluating existing model ---")
    eval_success = pipeline.step_4_evaluate_probes()
    
    print("\n--- Running graph analysis on correlations ---")
    analysis_corr = pipeline.step_5_graph_statistics_analysis(feature="corr")
    
    print("\n--- Running graph analysis on activations ---")
    analysis_act = pipeline.step_5_graph_statistics_analysis(feature="activation")
    
    if eval_success and analysis_corr and analysis_act:
        print("\n✓ Custom analysis completed!")


def main():
    """Main function to run examples"""
    
    print("="*80)
    print("HALLUCINATION DETECTION PIPELINE - USAGE EXAMPLES")
    print("="*80)
    print("\nThis script demonstrates various ways to use the pipeline.")
    print("Choose which example to run:\n")
    
    examples = {
        "1": ("Basic Pipeline Execution", example_1_basic_pipeline),
        "2": ("Step-by-Step Execution", example_2_step_by_step),
        "3": ("Multi-Layer Comparison", example_3_multi_layer_comparison),
        "4": ("Architecture Comparison", example_4_architecture_comparison),
        "5": ("Sparse vs Dense Graphs", example_5_sparse_vs_dense),
        "6": ("Resume Interrupted Pipeline", example_6_resume_interrupted),
        "7": ("Custom Analysis Workflow", example_7_custom_analysis)
    }
    
    for key, (description, _) in examples.items():
        print(f"  {key}. {description}")
    
    print("\n  0. Run all examples (not recommended - very time consuming!)")
    print("  q. Quit")
    
    choice = input("\nEnter your choice: ").strip().lower()
    
    if choice == 'q':
        print("Exiting...")
        return
    
    if choice == '0':
        print("\nRunning all examples...")
        for key, (description, func) in examples.items():
            print(f"\n{'#'*80}")
            print(f"Running Example {key}: {description}")
            print(f"{'#'*80}")
            try:
                func()
            except Exception as e:
                print(f"\n✗ Example {key} failed with error: {e}")
                import traceback
                traceback.print_exc()
    elif choice in examples:
        description, func = examples[choice]
        print(f"\nRunning Example {choice}: {description}")
        try:
            func()
        except Exception as e:
            print(f"\n✗ Example failed with error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Invalid choice: {choice}")
        return
    
    print("\n" + "="*80)
    print("Example execution completed!")
    print("="*80)


if __name__ == "__main__":
    main()
