"""Loss functions for Stage-2 deviation-prototype module."""

import math
from typing import Dict, Tuple

import torch


def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute mean over valid entries; return 0 if mask is empty."""
    mask_float = mask.float()
    numerator = (values * mask_float).sum()
    denom = mask_float.sum()
    return torch.where(denom > 0, numerator / (denom + eps), torch.zeros_like(numerator))


def _gather_prototypes(
    prototypes: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """Gather prototypes at indices; indices shape (...)."""
    # prototypes: (M, C)
    m, c = prototypes.shape
    expand_proto = prototypes.view(1, 1, m, c).expand(*indices.shape, m, c)
    gather_idx = indices.unsqueeze(-1).unsqueeze(-1).expand(*indices.shape, 1, c)
    gathered = torch.gather(expand_proto, dim=-2, index=gather_idx).squeeze(-2)
    return gathered


def compute_l_con(
    outputs: Dict, delta: float = 0.2, eps: float = 1e-6
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Contrastive hinge loss using top-1/2 prototypes (stop-grad on queries)."""
    part_outputs = outputs["part_outputs"]
    part_names = outputs["part_names"]

    per_part_losses: Dict[str, torch.Tensor] = {}
    loss_maps = []
    masks = []

    for name in part_names:
        po = part_outputs[name]
        query = po["raw_deviation"].detach()  # (B,T,C)
        mask = po["valid_mask"]
        prototypes = po["prototypes"]  # (M,C)
        b, t, c = query.shape
        m = prototypes.shape[0]
        if m < 2:
            raise ValueError("num_prototypes must be at least 2 for L_con.")

        scores = torch.einsum("btc,mc->btm", query, prototypes) / math.sqrt(float(c))
        top2 = torch.topk(scores, k=2, dim=-1)
        pos_idx = top2.indices[..., 0]
        neg_idx = top2.indices[..., 1]

        pos_proto = _gather_prototypes(prototypes, pos_idx)  # (B,T,C)
        neg_proto = _gather_prototypes(prototypes, neg_idx)  # (B,T,C)

        dist_pos = (query - pos_proto).pow(2).sum(dim=-1)
        dist_neg = (query - neg_proto).pow(2).sum(dim=-1)
        loss_elem = torch.clamp_min(dist_pos - dist_neg + delta, 0.0)  # (B,T)

        loss_maps.append(loss_elem)
        masks.append(mask)
        per_part_losses[name] = masked_mean(loss_elem, mask, eps)

    total_loss = masked_mean(torch.stack(loss_maps, dim=2), torch.stack(masks, dim=2), eps)
    return total_loss, per_part_losses


def compute_l_dev(
    outputs: Dict, eps: float = 1e-6
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Deviation consistency loss comparing raw vs prototype distances."""
    part_outputs = outputs["part_outputs"]
    part_names = outputs["part_names"]

    per_part_losses: Dict[str, torch.Tensor] = {}
    loss_maps = []
    masks = []

    for name in part_names:
        po = part_outputs[name]
        query = po["raw_deviation"].detach()  # (B,T,C)
        anchor = po["anchor_deviation"].detach()  # (B,1,C)
        mask = po["valid_mask"]
        part_has_valid = po["part_has_any_valid"].view(mask.shape[0], 1).float()
        effective_mask = mask.float() * part_has_valid  # disable sample if no valid frames
        prototypes = po["prototypes"]

        b, t, c = query.shape
        anchor_exp = anchor.expand(-1, t, -1)  # (B,T,C)

        raw_dist = torch.norm(query - anchor_exp, p=1, dim=-1).detach()  # (B,T)

        scores = torch.einsum("btc,mc->btm", query, prototypes) / math.sqrt(float(c))
        pos_idx = torch.argmax(scores, dim=-1)  # (B,T)

        anchor_scores = torch.einsum("btc,mc->btm", anchor_exp, prototypes) / math.sqrt(float(c))
        anchor_idx = torch.argmax(anchor_scores, dim=-1)  # (B,T)

        pos_proto = _gather_prototypes(prototypes, pos_idx)  # (B,T,C)
        anchor_proto = _gather_prototypes(prototypes, anchor_idx)  # (B,T,C)

        proto_dist = torch.norm(pos_proto - anchor_proto, p=1, dim=-1)
        loss_elem = torch.abs(raw_dist - proto_dist)

        loss_maps.append(loss_elem)
        masks.append(effective_mask)
        per_part_losses[name] = masked_mean(loss_elem, effective_mask, eps)

    total_loss = masked_mean(torch.stack(loss_maps, dim=2), torch.stack(masks, dim=2), eps)
    return total_loss, per_part_losses
