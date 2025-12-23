"""Model wrappers to attach a classification head after fused_output."""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

from stage1.config import Stage1Config
from stage1.dinov2_backbone import DINOv2Backbone
from stage1.parser_wrapper import FaceXZooParserWrapper
from stage1.stage1_extractor import Stage1DINOv2PartExtractor
from stage1.utils import normalize_frames
from stage2.stage2_config import Stage2Config
from stage2.stage2_module import Stage2DeviationPrototypeModule
from stage2.stage2_losses import compute_l_con, compute_l_dev


class TemporalPoolingHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, pool: str = "mean") -> None:
        super().__init__()
        self.pool = pool
        if pool not in ("mean", "attn"):
            raise ValueError("pool must be 'mean' or 'attn'")
        self.attn = nn.Linear(input_dim, 1) if pool == "attn" else None
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(
        self, feats: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor | None]:
        # feats: (B,T,D), mask: (B,T)
        if feats.dim() != 3:
            raise ValueError(f"Expected feats shape (B,T,D); got {feats.shape}")
        b, t, _ = feats.shape
        if mask is None:
            mask = torch.ones(b, t, device=feats.device, dtype=feats.dtype)
        if mask.shape != (b, t):
            raise ValueError(f"Mask shape {mask.shape} incompatible with feats {feats.shape}")
        mask = mask.float()
        if self.pool == "mean":
            assert feats.shape[:2] == mask.shape, (feats.shape, mask.shape)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
            assert denom.shape == (feats.shape[0], 1), denom.shape
            pooled = (feats * mask.unsqueeze(-1)).sum(dim=1) / denom
            attn_weights = None
        else:
            scores = self.attn(feats).squeeze(-1)  # (B,T)
            scores = scores.masked_fill(mask == 0, float("-inf"))
            attn_weights = torch.softmax(scores, dim=1)
            pooled = torch.einsum("bt,btd->bd", attn_weights, feats)
        logits = self.classifier(pooled)
        return {"logits": logits, "pooled": pooled, "attn": attn_weights}


class Stage2EmotionModel(nn.Module):
    """Full pipeline: Stage1 -> Stage2 -> pooling head."""

    def __init__(
        self,
        num_classes: int = 7,
        stage1_config: Optional[Stage1Config] = None,
        stage2_config: Optional[Stage2Config] = None,
        pool: str = "mean",
        parser: Optional[FaceXZooParserWrapper] = None,
    ) -> None:
        super().__init__()
        self.stage1_config = stage1_config or Stage1Config(output_global_feats=True)
        backbone = DINOv2Backbone(
            model_name=self.stage1_config.dinov2_model_name,
            provider=self.stage1_config.dinov2_provider,
            device=self.stage1_config.device,
        )
        self.stage1 = Stage1DINOv2PartExtractor(
            config=self.stage1_config,
            parser=parser or FaceXZooParserWrapper(device=self.stage1_config.device),
            backbone=backbone,
        )
        self.stage2_config = stage2_config or Stage2Config()
        self.stage2 = Stage2DeviationPrototypeModule(self.stage2_config)
        self.part_names = list(self.stage2_config.part_names)

        self.head_input_dim = self.stage2_config.latent_dim * self.stage2_config.num_parts
        self.head = TemporalPoolingHead(self.head_input_dim, num_classes, pool=pool)

        # Variant helpers
        self.nostage2_proj = nn.ModuleList(
            [nn.Linear(self.stage2_config.d_in, self.stage2_config.latent_dim) for _ in self.part_names]
        )
        self.global_head = nn.Linear(backbone.hidden_dim, num_classes)
        self.backbone = backbone
        self._debug_print_done = False

    def forward_full(self, frames: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict:
        """Stage1 + Stage2 + head."""
        stage1_out = self.stage1(frames, labels=labels) if labels is not None else self.stage1(frames)
        stage2_out = self.stage2(stage1_out["part_feats"], stage1_out["present"])
        fused = stage2_out["fused_output"]  # (B,T,D)
        time_mask = stage2_out["present"].any(dim=2)  # (B,T)
        if not self._debug_print_done:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            if local_rank == 0:
                print("[DEBUG] fused:", fused.shape)
                print("[DEBUG] time_mask:", time_mask.shape, time_mask.dtype)
                try:
                    dbg_out = self.head(fused, mask=time_mask)
                    print("[DEBUG] logits:", dbg_out["logits"].shape)
                except Exception as exc:  # noqa: BLE001
                    print("[DEBUG] head call failed:", repr(exc))
                    raise
            self._debug_print_done = True
        head_out = self.head(fused, mask=time_mask)
        return {
            "logits": head_out["logits"],
            "pooled": head_out["pooled"],
            "head_attn": head_out["attn"],
            "stage1": stage1_out,
            "stage2": stage2_out,
            "time_mask": time_mask,
        }

    def forward(self, frames: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict:
        """Default forward routes to full pipeline for DDP compatibility."""
        return self.forward_full(frames, labels=labels)

    def _fuse_without_stage2(self, stage1_out: Dict[str, torch.Tensor]) -> Dict:
        """Use projected part features, skip deviation/prototype."""
        part_feats = stage1_out["part_feats"]  # (B,T,K,D)
        present = stage1_out["present"]  # (B,T,K)
        proj_parts: List[torch.Tensor] = []
        for idx, proj in enumerate(self.nostage2_proj):
            proj_part = proj(part_feats[:, :, idx, :])
            proj_part = proj_part * present[:, :, idx].unsqueeze(-1).float()
            proj_parts.append(proj_part)
        fused = torch.cat(proj_parts, dim=-1)
        time_mask = present.any(dim=2)
        head_out = self.head(fused, mask=time_mask)
        return {
            "logits": head_out["logits"],
            "pooled": head_out["pooled"],
            "head_attn": head_out["attn"],
            "fused": fused,
            "time_mask": time_mask,
        }

    def _forward_global_only(self, frames: torch.Tensor) -> Dict:
        """No parsing; use global DINO feature."""
        b, t = frames.shape[:2]
        frames_norm = normalize_frames(frames, normalize_to_imagenet=True)
        frames_bt = frames_norm.view(b * t, *frames_norm.shape[2:])
        tokens, cls = self.backbone(frames_bt)
        cls = cls.view(b, t, -1)
        pooled = cls.mean(dim=1)
        logits = self.global_head(pooled)
        return {
            "logits": logits,
            "pooled": pooled,
            "global_feats": cls,
            "time_mask": torch.ones(b, t, device=frames.device, dtype=torch.float32),
        }

    def forward_variants(
        self, frames: torch.Tensor, variants: Sequence[str], labels: Optional[torch.Tensor] = None
    ) -> Dict[str, Dict]:
        """Compute multiple variants in one forward pass."""
        out: Dict[str, Dict] = {}
        need_stage1 = any(v in ("full", "nostage2") for v in variants)
        stage1_out = self.stage1(frames, labels=labels) if need_stage1 else None
        if "full" in variants and stage1_out is not None:
            out["full"] = self.forward_full(frames, labels=labels)
        if "nostage2" in variants and stage1_out is not None:
            nostage2 = self._fuse_without_stage2(stage1_out)
            nostage2["stage1"] = stage1_out
            out["nostage2"] = nostage2
        if "nostage1" in variants:
            out["nostage1"] = self._forward_global_only(frames)
        return out


def compute_losses(
    outputs: Dict,
    labels: torch.Tensor,
    lambda_proto: float = 1.0,
    use_aux_losses: bool = True,
    weights: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """Compute task loss + optional Stage-2 auxiliary losses."""
    losses: Dict[str, torch.Tensor] = {}
    logits = outputs["logits"]
    if weights is not None:
        losses["task_loss"] = nn.functional.cross_entropy(logits, labels, weight=weights)
    else:
        losses["task_loss"] = nn.functional.cross_entropy(logits, labels)
    if use_aux_losses and "stage2" in outputs:
        l_con, _ = compute_l_con(outputs["stage2"])
        l_dev, _ = compute_l_dev(outputs["stage2"])
        losses["l_con"] = l_con
        losses["l_dev"] = l_dev
        losses["total"] = losses["task_loss"] + lambda_proto * (l_con + l_dev)
    else:
        losses["l_con"] = torch.tensor(0.0, device=logits.device)
        losses["l_dev"] = torch.tensor(0.0, device=logits.device)
        losses["total"] = losses["task_loss"]
    return losses
