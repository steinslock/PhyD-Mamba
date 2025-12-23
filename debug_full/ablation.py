"""Helpers for Full/NoStage2/NoStage1 ablation logging."""

from __future__ import annotations

from typing import Dict

import torch

from .metrics import classification_metrics


def compute_ablation_metrics(
    variant_logits: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    num_classes: int,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics for each variant."""
    metrics: Dict[str, Dict[str, float]] = {}
    for name, logits in variant_logits.items():
        metrics[name] = classification_metrics(logits, labels, num_classes)
    return metrics

