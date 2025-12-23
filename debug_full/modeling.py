"""Model wrappers to attach a classification head after Stage3 output."""

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
from stage3.stage3_config import Stage3Config
from stage3.stage3_module import Stage3SATM


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
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
            pooled = (feats * mask.unsqueeze(-1)).sum(dim=1) / denom
            attn_weights = None
        else:
            scores = self.attn(feats).squeeze(-1)  # (B,T)
            scores = scores.masked_fill(mask == 0, float("-inf"))
            attn_weights = torch.softmax(scores, dim=1)
            pooled = torch.einsum("bt,btd->bd", attn_weights, feats)
        logits = self.classifier(pooled)
        return {"logits": logits, "pooled": pooled, "attn": attn_weights}


def _stack_stage2_mixed(stage2_out: Dict, part_names: Sequence[str]) -> torch.Tensor:
    """Stack per-part mixed deviation into [B,T,K,C]."""
    mixed_parts: List[torch.Tensor] = []
    for name in part_names:
        po = stage2_out["part_outputs"][name]
        mixed_parts.append(po["mixed_deviation"])
    return torch.stack(mixed_parts, dim=2)


class Stage3EmotionModel(nn.Module):
    """Full pipeline: Stage1 -> Stage2 -> Stage3 -> pooling head."""

    def __init__(
        self,
        num_classes: int = 7,
        stage1_config: Optional[Stage1Config] = None,
        stage2_config: Optional[Stage2Config] = None,
        stage3_config: Optional[Stage3Config] = None,
        pool: str = "mean",
        parser: Optional[FaceXZooParserWrapper] = None,
        compute_trs_default: bool = True,
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

        self.stage3_config = stage3_config or Stage3Config(
            d_model=self.stage2_config.latent_dim,
            num_parts=self.stage2_config.num_parts,
        )
        if self.stage3_config.d_model != self.stage2_config.latent_dim:
            raise ValueError(
                f"Stage3 d_model ({self.stage3_config.d_model}) must match Stage2 latent_dim ({self.stage2_config.latent_dim})."
            )
        if self.stage3_config.num_parts != self.stage2_config.num_parts:
            raise ValueError(
                f"Stage3 num_parts ({self.stage3_config.num_parts}) must match Stage2 num_parts ({self.stage2_config.num_parts})."
            )
        self.stage3 = Stage3SATM(self.stage3_config)

        self.head_input_dim = self.stage3_config.d_model * self.stage3_config.num_parts
        self.head = TemporalPoolingHead(self.head_input_dim, num_classes, pool=pool)

        # Variant helpers
        self.nostage2_proj = nn.ModuleList(
            [nn.Linear(self.stage2_config.d_in, self.stage2_config.latent_dim) for _ in self.part_names]
        )
        self.global_head = nn.Linear(backbone.hidden_dim, num_classes)
        self.backbone = backbone
        self._debug_print_done = False
        self.compute_trs_default = compute_trs_default

    def forward_full(
        self,
        frames: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        compute_trs: Optional[bool] = None,
        return_debug: bool = False,
        stage1_out: Optional[Dict] = None,
    ) -> Dict:
        """Stage1 + Stage2 + Stage3 + head."""
        if stage1_out is None:
            stage1_out = self.stage1(frames, labels=labels) if labels is not None else self.stage1(frames)
        stage2_out = self.stage2(stage1_out["part_feats"], stage1_out["present"])
        d_in = _stack_stage2_mixed(stage2_out, self.part_names)  # (B,T,K,C)
        present = stage2_out["present"]  # (B,T,K)
        compute_trs_flag = self.compute_trs_default if compute_trs is None else compute_trs
        stage3_out = self.stage3(d_in, present, compute_trs=compute_trs_flag, return_debug=return_debug)
        h = stage3_out["H"]  # (B,T,K,C)
        h_flat = h.view(h.shape[0], h.shape[1], -1)
        time_mask = present.any(dim=2)  # (B,T)

        if not self._debug_print_done:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            if local_rank == 0:
                print("[DEBUG] d_in:", d_in.shape)
                print("[DEBUG] H:", h.shape)
                print("[DEBUG] time_mask:", time_mask.shape, time_mask.dtype)
                try:
                    dbg_out = self.head(h_flat, mask=time_mask)
                    print("[DEBUG] logits:", dbg_out["logits"].shape)
                except Exception as exc:  # noqa: BLE001
                    print("[DEBUG] head call failed:", repr(exc))
                    raise
            self._debug_print_done = True

        head_out = self.head(h_flat, mask=time_mask)
        return {
            "logits": head_out["logits"],
            "pooled": head_out["pooled"],
            "head_attn": head_out["attn"],
            "stage1": stage1_out,
            "stage2": stage2_out,
            "stage3": stage3_out,
            "stage3_input": d_in,
            "time_mask": time_mask,
        }

    def forward(
        self,
        frames: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        compute_trs: Optional[bool] = None,
        return_debug: bool = False,
        stage1_out: Optional[Dict] = None,
    ) -> Dict:
        """Default forward routes to full pipeline for DDP compatibility."""
        return self.forward_full(
            frames,
            labels=labels,
            compute_trs=compute_trs,
            return_debug=return_debug,
            stage1_out=stage1_out,
        )

    def _fuse_without_stage2(self, stage1_out: Dict[str, torch.Tensor]) -> Dict:
        """Use projected part features, skip deviation/prototype (Stage3 bypassed)."""
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

    def _forward_stage3_from_stage1(
        self,
        stage1_out: Dict[str, torch.Tensor],
        compute_trs: Optional[bool] = None,
        return_debug: bool = False,
    ) -> Dict:
        """Skip Stage2; project Stage1 parts to d_model and feed Stage3."""
        part_feats = stage1_out["part_feats"]  # (B,T,K,D_in)
        present = stage1_out["present"]  # (B,T,K)
        proj_parts: List[torch.Tensor] = []
        for idx, proj in enumerate(self.nostage2_proj):
            proj_part = proj(part_feats[:, :, idx, :])
            proj_part = proj_part * present[:, :, idx].unsqueeze(-1).float()
            proj_parts.append(proj_part)
        d_in = torch.stack(proj_parts, dim=2)  # (B,T,K,C)
        if d_in.shape[-1] != self.stage3_config.d_model:
            raise ValueError(
                f"nostage2 projected dim {d_in.shape[-1]} != stage3 d_model {self.stage3_config.d_model}; please align configs."
            )
        compute_trs_flag = self.compute_trs_default if compute_trs is None else compute_trs
        stage3_out = self.stage3(d_in, present, compute_trs=compute_trs_flag, return_debug=return_debug)
        h = stage3_out["H"]
        h_flat = h.view(h.shape[0], h.shape[1], -1)
        time_mask = present.any(dim=2)
        head_out = self.head(h_flat, mask=time_mask)
        return {
            "logits": head_out["logits"],
            "pooled": head_out["pooled"],
            "head_attn": head_out["attn"],
            "stage1": stage1_out,
            "stage2": None,
            "stage3": stage3_out,
            "stage3_input": d_in,
            "time_mask": time_mask,
        }

    def _forward_no_stage3(self, stage1_out: Dict[str, torch.Tensor]) -> Dict:
        """Bypass Stage3: use Stage2 fused_output to head."""
        stage2_out = self.stage2(stage1_out["part_feats"], stage1_out["present"])
        fused = stage2_out["fused_output"]  # (B,T,K*C)
        present = stage2_out["present"]
        if fused.shape[-1] != self.head_input_dim:
            raise ValueError(
                f"nostage3 fused dim {fused.shape[-1]} != head_input_dim {self.head_input_dim}; please align configs."
            )
        time_mask = present.any(dim=2)
        head_out = self.head(fused, mask=time_mask)
        return {
            "logits": head_out["logits"],
            "pooled": head_out["pooled"],
            "head_attn": head_out["attn"],
            "stage1": stage1_out,
            "stage2": stage2_out,
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
        self,
        frames: torch.Tensor,
        variants: Sequence[str],
        labels: Optional[torch.Tensor] = None,
        compute_trs: Optional[bool] = None,
        return_debug: bool = False,
    ) -> Dict[str, Dict]:
        """Compute multiple variants in one forward pass."""
        out: Dict[str, Dict] = {}
        need_stage1 = any(v in ("full", "nostage2", "nostage3") for v in variants)
        stage1_out = self.stage1(frames, labels=labels) if need_stage1 else None
        if "full" in variants and stage1_out is not None:
            out["full"] = self.forward_full(
                frames,
                labels=labels,
                compute_trs=compute_trs,
                return_debug=return_debug,
                stage1_out=stage1_out,
            )
        if "nostage2" in variants and stage1_out is not None:
            out["nostage2"] = self._forward_stage3_from_stage1(
                stage1_out, compute_trs=compute_trs, return_debug=return_debug
            )
        if "nostage3" in variants and stage1_out is not None:
            out["nostage3"] = self._forward_no_stage3(stage1_out)
        if "nostage1+2+3" in variants:
            out["nostage1+2+3"] = self._forward_global_only(frames)
        return out


def compute_losses(
    outputs: Dict,
    labels: torch.Tensor,
    lambda_con: float = 0.05,
    lambda_dev: float = 1.0,
    lambda_trs: float = 0.02,
    use_aux_losses: bool = True,
    weights: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """Compute task loss + Stage-2 + Stage-3 losses."""
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
    else:
        device = logits.device
        losses["l_con"] = torch.tensor(0.0, device=device)
        losses["l_dev"] = torch.tensor(0.0, device=device)

    trs_loss = outputs.get("stage3", {}).get("trs_loss") if isinstance(outputs.get("stage3"), dict) else None
    if trs_loss is None:
        trs_loss = torch.tensor(0.0, device=logits.device)
    losses["l_trs"] = trs_loss

    losses["total"] = (
        losses["task_loss"] + lambda_con * losses["l_con"] + lambda_dev * losses["l_dev"] + lambda_trs * losses["l_trs"]
    )
    return losses


# Backward compatibility for earlier imports
Stage2EmotionModel = Stage3EmotionModel
