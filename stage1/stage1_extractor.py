import logging
from typing import Dict, Optional

import torch
from torch import nn

from .config import Stage1Config
from .dinov2_backbone import DINOv2Backbone
from .parser_wrapper import FaceXZooParserWrapper
from .utils import (
    compute_patch_grid,
    labels_to_part_masks,
    masks_to_patch_weights,
    normalize_frames,
    patch_weight_to_area,
)

LOG = logging.getLogger(__name__)


class Stage1DINOv2PartExtractor(nn.Module):
    """Stage-1 extractor: per-frame, per-part features using DINOv2 + parsing."""

    def __init__(
        self,
        config: Optional[Stage1Config] = None,
        parser: Optional[FaceXZooParserWrapper] = None,
        backbone: Optional[DINOv2Backbone] = None,
    ) -> None:
        super().__init__()
        self.config = config or Stage1Config()
        self.parser = parser
        self.backbone = backbone or DINOv2Backbone(
            model_name=self.config.dinov2_model_name,
            provider=self.config.dinov2_provider,
            device=self.config.device,
        )
        self.part_names = self.config.part_mapping.part_names
        self.k = len(self.part_names)

    def forward(
        self, frames: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        if frames.dim() != 5:
            raise ValueError(f"frames must have shape (B,T,3,H,W); got {frames.shape}")
        b, t, c, h, w = frames.shape
        if c != 3:
            raise ValueError("frames channel dimension must be 3 (RGB).")

        gh, gw = compute_patch_grid(h, w, self.config.patch_size)
        n = gh * gw
        LOG.debug("Processing batch=%d time=%d grid=%dx%d", b, t, gh, gw)
        backbone_device = getattr(self.backbone, "device", frames.device)
        if not isinstance(backbone_device, torch.device):
            backbone_device = torch.device(backbone_device)

        if labels is None:
            if self.parser is None:
                raise RuntimeError(
                    "No labels provided and no parser configured. "
                    "Pass labels or implement FaceXZooParserWrapper."
                )
            labels = self.parser.parse(frames)
        if labels.shape != (b, t, h, w):
            raise ValueError(
                f"labels must have shape (B,T,H,W) matching frames; got {labels.shape}"
            )

        masks_pix = labels_to_part_masks(
            labels, self.config.part_mapping.raw_to_part, self.k
        )
        weights, patch_grid = masks_to_patch_weights(masks_pix, self.config.patch_size)
        assert patch_grid == (gh, gw)
        weights = weights.to(backbone_device).float()

        frames_norm = normalize_frames(
            frames, normalize_to_imagenet=self.config.normalize_to_imagenet
        )
        frames_bt = frames_norm.view(b * t, c, h, w).to(backbone_device)

        autocast_device = "cuda" if backbone_device.type == "cuda" else "cpu"
        with torch.autocast(
            device_type=autocast_device, enabled=self.config.use_amp
        ):
            tokens_bt, cls_bt = self.backbone(frames_bt)  # (BT,N,D), (BT,D)

        if tokens_bt.dim() != 3:
            raise RuntimeError(f"Expected patch tokens of shape (BT,N,D); got {tokens_bt.shape}")
        bt, n_tokens, d = tokens_bt.shape
        if bt != b * t:
            raise RuntimeError(f"Backbone returned BT={bt}, expected {b*t}")
        if n_tokens != n:
            raise RuntimeError(
                f"Token count mismatch: got N={n_tokens}, expected grid {gh}x{gw} (N={n})."
            )

        area = patch_weight_to_area(weights)  # (B,T,K)

        tokens = tokens_bt.view(b, t, n, d)
        cls_tokens = cls_bt.view(b, t, d)

        sum_feat = torch.einsum("btnd,btkn->btkd", tokens.float(), weights)
        area_clamped = area.clamp(min=self.config.eps).unsqueeze(-1)
        part_feats = sum_feat / area_clamped

        thresholds = torch.tensor(
            [self.config.area_threshold_for_part(i) for i in range(self.k)],
            device=area.device,
            dtype=area.dtype,
        )
        present = area > thresholds.view(1, 1, self.k) * float(n)
        part_feats = torch.where(
            present.unsqueeze(-1), part_feats, torch.zeros_like(part_feats)
        )

        global_feats = (
            cls_tokens.float()
            if self.config.use_cls_as_global
            else tokens.float().mean(dim=2)
        ) if self.config.output_global_feats else None

        outputs = {
            "part_feats": part_feats,
            "present": present,
            "area": area,
            "global_feats": global_feats,
            "part_names": self.part_names,
            "patch_grid": patch_grid,
            "patch_size": self.config.patch_size,
        }
        if self.config.output_patch_weights:
            outputs["patch_weights"] = weights
        return outputs
