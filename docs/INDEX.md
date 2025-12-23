# Documentation Index

## Overview

This `docs/` folder contains comprehensive technical documentation for the **Hallucination Detection in Large Language Models using Graph Neural Networks** pipeline. All documentation is organized, cross-referenced, and includes extensive explanations in lay terms with clear variable definitions.

---

## Documentation Files

### 1. **Technical Report** (Primary Document)
📄 **File**: `hallucination_detection_technical_report.tex`
- **Format**: LaTeX (1,064 lines)
- **Sections**: 13 main sections + appendices
- **Contains**:
  - Complete methodology explanation
  - Detailed architecture specifications
  - Data characteristics and examples
  - Step-by-step pipeline walkthrough
  - Mathematical formulations
  - Configuration parameters
  - All generated figures with captions
  - Complete function references
  - Reproducibility instructions
  - Appendices with module documentation

**How to Use**:
- Compile to PDF: `pdflatex hallucination_detection_technical_report.tex` (run twice)
- Read as-is for complete technical details
- Reference specific sections for methodology details

**Key Sections**:
- Introduction: Motivation and approach
- Experimental Setup: Configuration (Table 1)
- GPT-2 Architecture: Model specs (124M parameters, 768-dim hidden, 12 layers)
- Step 1-5: Detailed explanation of each pipeline stage
- Key Definitions: Variable reference table
- Reproducibility: Complete setup and run instructions

---

### 2. **README** (Start Here)
📖 **File**: `README.md`
- **Format**: Markdown (252 lines)
- **Audience**: General readers and researchers
- **Contains**:
  - Project overview and structure
  - Report structure summary
  - Key highlights (data, model, performance stats)
  - Running instructions
  - Configuration parameters explained
  - Module and function reference
  - Generated artifacts description
  - Technical details (connectivity computation, GNN architecture)
  - Environment requirements

**How to Use**:
- Read first to understand what the pipeline does
- Reference for module organization
- Quick lookup for function documentation

**Quick Stats**:
- Dataset: 5,915 TruthfulQA Q-A pairs
- Model: GPT-2 base, layer 5, 768 neurons
- Network: 5% density (29,491 edges)
- Runtime: ~12.8 minutes (769.9 seconds)

---

### 3. **Quick Reference** (Fast Lookup)
⚡ **File**: `QUICK_REFERENCE.md`
- **Format**: Markdown (270 lines)
- **Purpose**: Fast configuration and troubleshooting
- **Contains**:
  - One-page summary
  - Key numbers table
  - Architecture diagram
  - Five-step pipeline overview
  - Configuration parameters (easy to modify)
  - Output files reference
  - Running instructions (copy-paste ready)
  - GPT-2 specs table
  - Understanding metrics
  - Troubleshooting guide
  - Key functions reference
  - Technical insights

**How to Use**:
- Copy-paste commands to run pipeline
- Reference tables for quick lookups
- Troubleshooting section for common issues
- Metrics explanation for interpreting results

---

## Document Navigation Map

```
Want to understand the project?
  → Start with README.md

Want to run the analysis?
  → Use QUICK_REFERENCE.md (Configuration & Running sections)

Want complete technical details?
  → Read hallucination_detection_technical_report.tex

Want to modify parameters?
  → Check QUICK_REFERENCE.md (Configuration Parameters section)

Want to understand metrics?
  → See QUICK_REFERENCE.md (Understanding the Numbers section)

Need function reference?
  → README.md (Module and Function Reference)
  → QUICK_REFERENCE.md (Key Functions Reference)
  → hallucination_detection_technical_report.tex (Appendix C)

Want to troubleshoot?
  → QUICK_REFERENCE.md (Troubleshooting section)
```

---

## Key Information at a Glance

| Aspect | Details |
|--------|---------|
| **Project** | Hallucination detection in LLMs via graph neural networks |
| **Model** | GPT-2 Base (124M params, 12 layers, 768 hidden dim) |
| **Layer** | Layer 5 (middle, semantic) |
| **Dataset** | TruthfulQA (5,915 Q-A pairs, 50% balance) |
| **Train/Test** | 80% / 20% (4,732 / 1,183 samples) |
| **Network** | Functional connectivity with 5% density threshold |
| **GNN** | 3-layer GCN with 32 hidden channels |
| **Loss** | Binary cross-entropy |
| **Optimizer** | Adam (lr=0.001, weight_decay=1e-5) |
| **Runtime** | ~12.8 minutes total |

---

## Documentation Structure

```
docs/
├── INDEX.md                                    (this file)
├── README.md                                   (project overview)
├── QUICK_REFERENCE.md                         (configuration & running)
└── hallucination_detection_technical_report.tex (full technical report)
```

---

## Finding Specific Information

### Data & Dataset Information
- **Dataset details**: README.md § "Dataset Preparation", QUICK_REFERENCE.md § "Key Numbers"
- **Dataset examples**: hallucination_detection_technical_report.tex § 4.1.6
- **Label distribution**: README.md, QUICK_REFERENCE.md
- **Data splitting**: hallucination_detection_technical_report.tex § 4.1.7

### Model Architecture
- **GPT-2 specs**: hallucination_detection_technical_report.tex § 3
- **Layer specifications**: QUICK_REFERENCE.md § "GPT-2 Architecture Details"
- **GNN architecture**: hallucination_detection_technical_report.tex § 5.2
- **Architecture diagram**: QUICK_REFERENCE.md § "Architecture Diagram"

### Pipeline Configuration
- **All parameters**: hallucination_detection_technical_report.tex § 2.1
- **Quick config**: QUICK_REFERENCE.md § "Configuration Parameters"
- **Changing parameters**: README.md § "Running the Analysis"

### Running the Pipeline
- **Quick start**: QUICK_REFERENCE.md § "Running the Pipeline"
- **Detailed setup**: hallucination_detection_technical_report.tex § 9 (Reproducibility)
- **Module reference**: README.md § "Module and Function Reference"

### Understanding Results
- **Output files**: README.md § "Generated Artifacts", QUICK_REFERENCE.md § "Output Files"
- **Metrics explained**: QUICK_REFERENCE.md § "Understanding the Numbers"
- **Confusion matrix**: hallucination_detection_technical_report.tex § 6.2.1
- **Timing**: hallucination_detection_technical_report.tex § 7

### Troubleshooting
- **Common issues**: QUICK_REFERENCE.md § "Troubleshooting"
- **TensorBoard**: hallucination_detection_technical_report.tex § Appendix C

---

## Content Summary by Topic

### 1. Input Parameters & Configuration
| Document | Section |
|----------|---------|
| QUICK_REFERENCE | "Configuration Parameters" |
| hallucination_detection_technical_report.tex | § 2.1 "Analysis Parameters" |
| hallucination_detection_technical_report.tex | § 9 "Reproducibility" |

### 2. GPT-2 & Neural Architecture
| Document | Section |
|----------|---------|
| hallucination_detection_technical_report.tex | § 3 "GPT-2 Model Architecture" |
| QUICK_REFERENCE | "GPT-2 Architecture Details" |
| README | "Model Architecture" |

### 3. Dataset Information
| Document | Section |
|----------|---------|
| hallucination_detection_technical_report.tex | § 4 "Step 1: Dataset Construction" |
| README | "Running the Analysis" (Dataset section) |
| QUICK_REFERENCE | "Key Numbers at a Glance" |

### 4. Network Topology (Functional Connectivity)
| Document | Section |
|----------|---------|
| hallucination_detection_technical_report.tex | § 5 "Step 2: Computing Functional Connectivity" |
| QUICK_REFERENCE | "Five-Step Pipeline" (Step 2) |
| README | "Functional Connectivity Computation" |

### 5. GNN Probe Architecture & Training
| Document | Section |
|----------|---------|
| hallucination_detection_technical_report.tex | § 6 "Step 3: GNN Probe Training" |
| README | "GNN Probe" |
| QUICK_REFERENCE | "Architecture Diagram" |

### 6. Evaluation & Metrics
| Document | Section |
|----------|---------|
| hallucination_detection_technical_report.tex | § 6.2 "Evaluation Metrics" |
| QUICK_REFERENCE | "Understanding the Numbers" |
| README | "Confusion Matrix Explanations" |

### 7. Complete Function Reference
| Document | Section |
|----------|---------|
| hallucination_detection_technical_report.tex | § 9 "Appendix A" (Modules), § 10 "Appendix B" (Functions) |
| README | "Module and Function Reference" |
| QUICK_REFERENCE | "Key Functions Reference" |

### 8. Running & Reproducibility
| Document | Section |
|----------|---------|
| QUICK_REFERENCE | "Running the Pipeline" |
| hallucination_detection_technical_report.tex | § 9 "Reproducibility" |
| README | "Running the Analysis" |

---

## Special Features

### In the Technical Report
✓ **Complete mathematical formulations** (e.g., GCN equations, loss functions)
✓ **Detailed function implementations** with pseudocode
✓ **Architectural diagrams** with layer specifications
✓ **Table references** for all key parameters and variables
✓ **Figure captions** explaining all visualization artifacts
✓ **Appendices** with module documentation
✓ **References** to academic papers and libraries

### In README
✓ **Module-level documentation** with purpose and functions
✓ **File organization** showing directory structure
✓ **Complete file path references** for outputs
✓ **Environment setup** instructions

### In Quick Reference
✓ **Copy-paste ready commands** for running
✓ **Troubleshooting guide** with common issues
✓ **Quick lookup tables** for parameters and metrics
✓ **Visual architecture diagram** (ASCII art)
✓ **Metrics explanation** in plain language

---

## Usage Scenarios

### Scenario 1: "I want to understand the paper methodology"
1. Read: README.md (overview)
2. Study: hallucination_detection_technical_report.tex (§1-7)
3. Reference: QUICK_REFERENCE.md (Five-Step Pipeline)

### Scenario 2: "I need to run the analysis with custom parameters"
1. Quick lookup: QUICK_REFERENCE.md (Configuration Parameters)
2. Execute: QUICK_REFERENCE.md (Running the Pipeline)
3. Debug: QUICK_REFERENCE.md (Troubleshooting)

### Scenario 3: "I want to understand a specific module/function"
1. Overview: README.md (Module and Function Reference)
2. Details: hallucination_detection_technical_report.tex (Appendices)
3. Source: `hallucination/*.py`, `utils/*.py`

### Scenario 4: "I need to modify the pipeline"
1. Reference: QUICK_REFERENCE.md (Configuration Parameters section)
2. Understand: hallucination_detection_technical_report.tex (relevant step section)
3. Code: Edit `my_analysis/hallucination_detection_analysis.py`

### Scenario 5: "I want to understand the results"
1. Data files: `/ptmp/aomidvarnia/analysis_results/llm_graph/reports/hallucination_analysis/`
2. Interpretation: QUICK_REFERENCE.md (Understanding the Numbers)
3. Metrics: hallucination_detection_technical_report.tex (§ 6.2)

---

## Key Numbers Summary

**Dataset**: 5,915 TruthfulQA question-answer pairs
- Train: 4,732 (80%)
- Test: 1,183 (20%)
- Balance: 50.3% truthful, 49.7% hallucinated

**Model**: GPT-2 Base
- Total parameters: 124 million
- Hidden dimension: 768
- Transformer layers: 12
- Selected layer: 5 (middle)

**Network**:
- Full edges: 589,824
- Thresholded (5%): ~29,491
- Keeps top 5% by absolute correlation

**GNN Probe**:
- Hidden channels: 32
- Number of layers: 3
- Global pooling: Mean + Max

**Training**:
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 16
- Max epochs: 10
- Early stopping patience: 20

**Computational Cost**:
- Total: 769.9 seconds (~12.8 minutes)
- Network computation: 307.6s (41%)
- Training: 289.7s (39%)
- Evaluation: 168.6s (23%)
- Dataset: 3.5s (0.4%)
- Analysis: 0.7s (0.1%)

---

## Document Metadata

| Document | Lines | Sections | Tables | Figures |
|----------|-------|----------|--------|---------|
| hallucination_detection_technical_report.tex | 1,064 | 13+appendices | 15 | references to 8 |
| README.md | 252 | 10+ | 8 | inline references |
| QUICK_REFERENCE.md | 270 | 15+ | 12 | 1 ASCII diagram |
| **Total** | **1,586** | **38+** | **35+** | **9+** |

---

## Next Steps

1. **Read README.md** for project overview (5 min)
2. **Check QUICK_REFERENCE.md** for configuration (10 min)
3. **Run the pipeline** following QUICK_REFERENCE (13 min of computation)
4. **Review results** in `/reports/hallucination_analysis/` (5 min)
5. **Study technical details** in the full report (30+ min)

---

**Documentation Created**: December 23, 2024
**Total Documentation**: 1,586 lines across 4 files
**Coverage**: Complete pipeline documentation with reproducibility instructions
