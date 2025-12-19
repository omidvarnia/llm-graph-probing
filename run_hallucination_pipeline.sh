#!/bin/bash
################################################################################
# Hallucination Detection Pipeline - Quick Start Script
#
# This script provides convenient shortcuts for running the pipeline
# with commonly used configurations.
#
# Usage:
#   ./run_hallucination_pipeline.sh [option]
#
# Options:
#   quick       - Quick test with small model (pythia-70m)
#   default     - Default setup (qwen2.5-0.5b, layer 12)
#   sparse      - Memory-efficient sparse graphs
#   multi-layer - Analyze multiple layers
#   help        - Show this help message
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Helper function to print section headers
print_header() {
    echo ""
    echo "================================================================================"
    echo "$1"
    echo "================================================================================"
    echo ""
}

# Check if Python is available
check_python() {
    if ! command -v python &> /dev/null; then
        print_error "Python is not installed or not in PATH"
        exit 1
    fi
    print_info "Python version: $(python --version)"
}

# Check if required packages are installed
check_dependencies() {
    print_info "Checking dependencies..."
    
    python -c "import torch" 2>/dev/null || {
        print_error "PyTorch is not installed. Run: pip install -r requirements.txt"
        exit 1
    }
    
    python -c "import transformers" 2>/dev/null || {
        print_error "Transformers is not installed. Run: pip install -r requirements.txt"
        exit 1
    }
    
    print_success "All dependencies are installed"
}

# Quick test with small model
run_quick_test() {
    print_header "QUICK TEST: Small Model (pythia-70m)"
    print_info "This will run a quick test with a small model"
    print_info "Estimated time: ~30 minutes"
    print_info "Estimated GPU memory: ~2-4 GB"
    echo ""
    
    python hallucination_pipeline.py \
        --llm_model_name pythia-70m \
        --llm_layer 5 \
        --batch_size 8 \
        --eval_batch_size 16 \
        --num_layers 1 \
        --hidden_channels 16 \
        --lr 0.001
    
    print_success "Quick test completed!"
}

# Default configuration
run_default() {
    print_header "DEFAULT CONFIGURATION: qwen2.5-0.5b (layer 12)"
    print_info "This will run the full pipeline with default settings"
    print_info "Estimated time: ~1-1.5 hours"
    print_info "Estimated GPU memory: ~5-8 GB"
    echo ""
    
    python hallucination_pipeline.py
    
    print_success "Default pipeline completed!"
}

# Sparse graph configuration
run_sparse() {
    print_header "SPARSE GRAPH CONFIGURATION: Memory-Efficient"
    print_info "This will run with sparse graphs (20% density)"
    print_info "Estimated time: ~40 minutes"
    print_info "Estimated GPU memory: ~2-3 GB"
    echo ""
    
    python hallucination_pipeline.py \
        --sparse \
        --network_density 0.2 \
        --batch_size 8 \
        --num_workers 2
    
    print_success "Sparse graph pipeline completed!"
}

# Multi-layer analysis
run_multi_layer() {
    print_header "MULTI-LAYER ANALYSIS"
    print_info "This will analyze layers 6, 12, 18, and 24"
    print_info "Estimated time: ~4-6 hours"
    print_info "Estimated GPU memory: ~5-8 GB"
    echo ""
    
    # Prepare data once
    print_info "Step 0: Preparing data (once)..."
    python hallucination_pipeline.py --step 1
    
    # Analyze each layer
    for layer in 6 12 18 24; do
        print_header "Processing Layer $layer"
        
        python hallucination_pipeline.py \
            --llm_layer $layer \
            --skip_data_prep \
            --llm_model_name qwen2.5-0.5b
        
        print_success "Layer $layer completed!"
    done
    
    print_success "Multi-layer analysis completed!"
    print_info "Check saves/hallucination/ for results from each layer"
}

# Run data preparation only
run_data_prep() {
    print_header "DATA PREPARATION ONLY"
    print_info "Downloading TruthfulQA dataset..."
    
    python hallucination_pipeline.py --step 1
    
    print_success "Data preparation completed!"
    print_info "Dataset saved to: data/hallucination/truthfulqa-validation.csv"
}

# Run interactive examples
run_examples() {
    print_header "INTERACTIVE EXAMPLES"
    print_info "Running example usage script..."
    
    python example_pipeline_usage.py
}

# Run custom command
run_custom() {
    print_header "CUSTOM CONFIGURATION"
    print_info "Running with custom parameters..."
    
    python hallucination_pipeline.py "$@"
}

# Show help
show_help() {
    cat << EOF
Hallucination Detection Pipeline - Quick Start Script

Usage:
  ./run_hallucination_pipeline.sh [option]

Options:
  quick           Quick test with small model (pythia-70m)
                  Time: ~30 min | Memory: ~2-4 GB
  
  default         Default configuration (qwen2.5-0.5b, layer 12)
                  Time: ~1-1.5 hours | Memory: ~5-8 GB
  
  sparse          Memory-efficient sparse graphs (20% density)
                  Time: ~40 min | Memory: ~2-3 GB
  
  multi-layer     Analyze multiple layers (6, 12, 18, 24)
                  Time: ~4-6 hours | Memory: ~5-8 GB
  
  data-only       Download and prepare data only
                  Time: ~2 min | Memory: minimal
  
  examples        Run interactive examples
  
  custom [args]   Run with custom arguments
                  Example: ./run_hallucination_pipeline.sh custom --llm_layer 18
  
  help            Show this help message

Examples:
  # Quick test
  ./run_hallucination_pipeline.sh quick
  
  # Default run
  ./run_hallucination_pipeline.sh default
  
  # Sparse graphs
  ./run_hallucination_pipeline.sh sparse
  
  # Custom configuration
  ./run_hallucination_pipeline.sh custom --llm_model_name pythia-160m --llm_layer 10

Documentation:
  - Quick Reference: QUICK_REFERENCE.md
  - Full Guide: HALLUCINATION_PIPELINE_GUIDE.md
  - Implementation: IMPLEMENTATION_SUMMARY.md

For more information, see: README.md
EOF
}

# Main script logic
main() {
    print_header "Hallucination Detection Pipeline - Quick Start"
    
    # Check Python installation
    check_python
    
    # Parse command-line arguments
    case "${1:-}" in
        quick)
            check_dependencies
            run_quick_test
            ;;
        default)
            check_dependencies
            run_default
            ;;
        sparse)
            check_dependencies
            run_sparse
            ;;
        multi-layer)
            check_dependencies
            run_multi_layer
            ;;
        data-only)
            check_dependencies
            run_data_prep
            ;;
        examples)
            check_dependencies
            run_examples
            ;;
        custom)
            check_dependencies
            shift  # Remove 'custom' argument
            run_custom "$@"
            ;;
        help|--help|-h|"")
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
    
    print_success "Script execution completed!"
}

# Run main function
main "$@"
