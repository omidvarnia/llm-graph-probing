"""
Disease configuration presets for neuropathology simulations on LLM functional connectivity.

Healthy vs pathological connectivity:
- Healthy connectivity is the original correlation matrix computed from LLM hidden states.
- Pathological connectivity is a transformed matrix meant to mimic disease-like network changes
  such as increased segregation, decreased aggregation, and reduced small-worldness.

This module defines `DiseaseConfig` and a preset registry to standardize experiments.
"""
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class DiseaseConfig:
    """Configuration for a disease-like neuropathology pattern.

    Attributes:
        name: Identifier for the disease preset.
        num_clusters: Number of functional modules to form.
        within_scale: Scale factor applied to within-module correlations (>1 increases segregation).
        between_scale: Scale factor applied to between-module correlations (<1 decreases aggregation).
        rewiring_prob: Probability used to randomly rewire edges to reduce small-worldness.
        distance_threshold: Optional index distance threshold used by `decrease_aggregation`.
    """

    name: str
    num_clusters: int = 8
    within_scale: float = 1.2
    between_scale: float = 0.7
    rewiring_prob: float = 0.1
    distance_threshold: Optional[int] = None


PRESETS: Dict[str, DiseaseConfig] = {
    # Increased local segregation, reduced long-range aggregation, moderate rewiring
    "epilepsy_like": DiseaseConfig(
        name="epilepsy_like",
        num_clusters=8,
        within_scale=1.3,
        between_scale=0.5,
        rewiring_prob=0.15,
        distance_threshold=50,
    ),
    # Global connectivity weakening, stronger reduction in small-world organization
    "dementia_like": DiseaseConfig(
        name="dementia_like",
        num_clusters=6,
        within_scale=1.1,
        between_scale=0.4,
        rewiring_prob=0.3,
        distance_threshold=40,
    ),
    # Over-segregation and under-integration across modules, mild rewiring
    "autism_like": DiseaseConfig(
        name="autism_like",
        num_clusters=10,
        within_scale=1.4,
        between_scale=0.6,
        rewiring_prob=0.1,
        distance_threshold=60,
    ),
}
