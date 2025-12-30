# Hallucination Detection Analysis Results

This directory contains the comprehensive technical report and associated figures from the hallucination detection analysis using Graph Neural Networks on the Qwen2.5-0.5B model with the TruthfulQA dataset.

## Contents

- **hallucination_detection_report.tex**: LaTeX source file for the technical report
- **figures/**: Directory containing all 14 figures referenced in the report

## Compiling the Report

To generate the PDF report, run:

```bash
pdflatex hallucination_detection_report.tex
pdflatex hallucination_detection_report.tex  # Run twice for proper references
```

Or using latexmk for automatic compilation:

```bash
latexmk -pdf hallucination_detection_report.tex
```

## Figures Index

1. `dataset_label_distribution.png` - Class distribution visualization
2. `fc_before_threshold.png` - Correlation matrix before thresholding
3. `fc_after_threshold.png` - Correlation matrix after 5% density threshold
4. `train_loss.png` - Training loss curve
5. `test_metrics.png` - Test set performance metrics
6. `intra_vs_inter_hist_layer_5.png` through `layer_11.png` - Layer-wise correlation analysis histograms
7. `step_durations.png` - Pipeline timing breakdown

## Key Findings

- **Test Accuracy**: 52.2% (near-random performance)
- **Class Separability**: Weak (mean ~0.12, std ~0.35)
- **Conclusion**: Current approach NOT viable for production

See the full report for detailed analysis, problems identified, and recommendations.

## Dataset

- Model: Qwen/Qwen2.5-0.5B
- Dataset: TruthfulQA (5,915 QA pairs)
- Layers analyzed: 5-11
- Total pipeline time: ~32 minutes

## Contact

a.omidvarnia@fz-juelich.de
