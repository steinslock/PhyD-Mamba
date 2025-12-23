"""Classification metrics helpers for DFEW training."""

from __future__ import annotations

from typing import Dict

import torch


def _confusion_matrix(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Compute confusion matrix with shape (C, C)."""
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long, device=preds.device)
    for t, p in zip(targets.view(-1), preds.view(-1)):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def _accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    total = targets.numel()
    if total == 0:
        return 0.0
    correct = (preds == targets).sum().item()
    return float(correct) / float(total)


def _uar(cm: torch.Tensor) -> float:
    denom = cm.sum(dim=1).clamp_min(1)
    recall = cm.diag().float() / denom
    return float(recall.mean().item())


def _war(cm: torch.Tensor) -> float:
    total = cm.sum().item()
    if total == 0:
        return 0.0
    correct = cm.diag().sum().item()
    return float(correct) / float(total)


def classification_metrics(
    logits: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> Dict[str, float | torch.Tensor]:
    """Return accuracy/UAR/WAR and confusion matrix."""
    preds = torch.argmax(logits, dim=-1)
    cm = _confusion_matrix(preds, targets, num_classes)
    return {
        "acc": _accuracy(preds, targets),
        "uar": _uar(cm),
        "war": _war(cm),
        "confusion_matrix": cm.cpu(),
    }

