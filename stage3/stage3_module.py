"""Stage-3 SATM block with spatial attention, bi-directional Mamba, and TRS regularization."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .stage3_config import Stage3Config

# Patch deprecated torch.cuda.amp custom_fwd/custom_bwd to new torch.amp versions before importing mamba_ssm
try:
    import torch.cuda.amp as cuda_amp  # noqa: F401

    def _custom_fwd(*args, **kwargs):
        kwargs.setdefault("device_type", "cuda")
        return torch.amp.custom_fwd(*args, **kwargs)

    def _custom_bwd(*args, **kwargs):
        kwargs.setdefault("device_type", "cuda")
        return torch.amp.custom_bwd(*args, **kwargs)

    torch.cuda.amp.custom_fwd = _custom_fwd  # type: ignore[attr-defined]
    torch.cuda.amp.custom_bwd = _custom_bwd  # type: ignore[attr-defined]
except Exception:
    pass

try:
    from mamba_ssm import Mamba  # type: ignore
except Exception:  # noqa: BLE001
    try:
        from mamba_ssm.modules.mamba_simple import Mamba  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "mamba_ssm is required for Stage3 SATM. Please install mamba_ssm before using Stage3."
        ) from exc


class SATMBlock(nn.Module):
    """Spatial-Attention + Temporal bi-Mamba block."""

    def __init__(self, config: Stage3Config) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model

        self.spatial_norm = nn.LayerNorm(self.d_model)
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.spatial_dropout = nn.Dropout(config.dropout)

        self.mamba_fwd = Mamba(
            d_model=self.d_model,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            expand=config.mamba_expand,
        )
        self.mamba_bwd = Mamba(
            d_model=self.d_model,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            expand=config.mamba_expand,
        )
        # Learnable residual gates; start near zero so early training ≈ identity.
        self.alpha_spatial = nn.Parameter(torch.tensor(1e-3))
        self.alpha_temporal = nn.Parameter(torch.tensor(1e-3))

    def _spatial_synergy(
        self, x: torch.Tensor, mask: torch.Tensor, return_attn: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply per-frame spatial attention with key padding mask."""
        b, t, k, c = x.shape
        x_sp = x.reshape(b * t, k, c)
        m_sp = mask.reshape(b * t, k)
        pad = m_sp == 0  # True indicates missing tokens
        x_norm = self.spatial_norm(x_sp)

        attn_out = torch.zeros_like(x_sp)
        attn_weights: Optional[torch.Tensor] = None
        valid_frames = ~(pad.all(dim=1))
        if valid_frames.any():
            x_valid = x_norm[valid_frames]
            pad_valid = pad[valid_frames]
            attn_valid, attn_w_valid = self.spatial_attn(
                x_valid,
                x_valid,
                x_valid,
                key_padding_mask=pad_valid,
                need_weights=return_attn,
                average_attn_weights=True,
            )
            attn_valid = attn_valid.to(x_sp.dtype)
            attn_out[valid_frames] = self.spatial_dropout(attn_valid)
            if return_attn and attn_w_valid is not None:
                attn_w_valid = attn_w_valid.to(x_sp.dtype)
                attn_weights = torch.zeros(
                    (b * t, k, k),
                    device=x_valid.device,
                    dtype=attn_w_valid.dtype,
                )
                attn_weights[valid_frames] = attn_w_valid

        alpha = torch.clamp(self.alpha_spatial, max=float(self.config.alpha_spatial_max))
        x_spatial = x_sp + alpha * attn_out
        return x_spatial.view(b, t, k, c), None if attn_weights is None else attn_weights.view(b, t, k, k)

    def _temporal_bimamba(
        self, x: torch.Tensor, mask: torch.Tensor, return_y: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply bi-directional Mamba along time for each part."""
        b, t, k, c = x.shape
        x0 = x * mask.unsqueeze(-1)  # input-end masking
        x_tmp = x0.permute(0, 2, 1, 3).contiguous().view(b * k, t, c)

        y_fwd = self.mamba_fwd(x_tmp)

        x_rev = torch.flip(x_tmp, dims=[1])
        y_bwd = torch.flip(self.mamba_bwd(x_rev), dims=[1])

        y = y_fwd + y_bwd
        y = y.view(b, k, t, c).permute(0, 2, 1, 3).contiguous()

        alpha = torch.clamp(self.alpha_temporal, max=float(self.config.alpha_temporal_max))
        h = (x + alpha * y) * mask.unsqueeze(-1)  # output-end masking
        if return_y:
            y_fwd_out = y_fwd.view(b, k, t, c).permute(0, 2, 1, 3).contiguous()
            y_bwd_out = y_bwd.view(b, k, t, c).permute(0, 2, 1, 3).contiguous()
            y_fwd_out = y_fwd_out * mask.unsqueeze(-1)
            y_bwd_out = y_bwd_out * mask.unsqueeze(-1)
            return h, y_fwd_out, y_bwd_out
        return h

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        return_attn: bool = False,
        return_y: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass for a single SATM block.

        Args:
            x: [B, T, K, C] continuous deviation features.
            mask: [B, T, K] bool/0-1 present mask.
        """
        if x.dim() != 4:
            raise ValueError(f"x must be 4D [B,T,K,C]; got {x.shape}")
        if mask.shape != x.shape[:3]:
            raise ValueError(f"mask shape {mask.shape} incompatible with x {x.shape}")

        x_spatial, attn_w = self._spatial_synergy(x, mask, return_attn=return_attn)
        if return_y:
            h, y_fwd, y_bwd = self._temporal_bimamba(x_spatial, mask, return_y=True)
        else:
            h = self._temporal_bimamba(x_spatial, mask)
            y_fwd = None
            y_bwd = None
        return h, attn_w, y_fwd, y_bwd


class Stage3SATM(nn.Module):
    """Stage3 Spatio-Temporal Synergistic Dynamics with TRS regularization."""

    def __init__(self, config: Stage3Config) -> None:
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([SATMBlock(config) for _ in range(config.num_layers)])

    def _forward_once(
        self, d_in: torch.Tensor, mask: torch.Tensor, return_debug: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        h = d_in
        last_attn: Optional[torch.Tensor] = None
        last_y_fwd: Optional[torch.Tensor] = None
        last_y_bwd: Optional[torch.Tensor] = None
        for block in self.blocks:
            if return_debug:
                h, attn_w, y_fwd, y_bwd = block(h, mask, return_attn=True, return_y=True)
                last_attn = attn_w
                last_y_fwd = y_fwd
                last_y_bwd = y_bwd
            else:
                h, _, _, _ = block(h, mask, return_attn=False, return_y=False)
        return h, last_attn, last_y_fwd, last_y_bwd

    def forward(
        self, d_in: torch.Tensor, mask: torch.Tensor, compute_trs: bool = False, return_debug: bool = False
    ) -> Dict[str, Optional[torch.Tensor] | Optional[Dict[str, Optional[torch.Tensor]]]]:
        """Forward pass through Stage3.

        Args:
            d_in: [B, T, K, C] Stage2 fused stack (continuous deviation features).
            mask: [B, T, K] present mask; 1/True for valid parts.
            compute_trs: If True, run time-reversal path and compute TRS loss.
            return_debug: If True, return intermediate tensors for logging.

        Returns:
            Dict with keys:
                - "H": hidden states [B, T, K, C], missing entries forced to 0.
                - "trs_loss": scalar TRS loss (or None if compute_trs=False).
                - "debug": optional dict with intermediate outputs.
        """
        if d_in.dim() != 4:
            raise ValueError(f"d_in must be 4D [B,T,K,C]; got {d_in.shape}")
        if mask.shape != d_in.shape[:3]:
            raise ValueError(f"mask shape {mask.shape} incompatible with d_in {d_in.shape}")

        b, t, k, c = d_in.shape
        if k != self.config.num_parts:
            raise ValueError(f"Expected num_parts={self.config.num_parts}, but got {k}.")
        if c != self.config.d_model:
            raise ValueError(f"Expected channel dim d_model={self.config.d_model}, but got {c}.")

        m_bool = mask.bool()

        h_fwd, attn_w, y_fwd, y_bwd = self._forward_once(d_in, m_bool, return_debug=return_debug)
        trs_loss: Optional[torch.Tensor] = None
        h_bwd: Optional[torch.Tensor] = None

        if compute_trs:
            d_rev = torch.flip(d_in, dims=[1])
            m_rev = torch.flip(m_bool, dims=[1])
            h_bwd_raw, _, _, _ = self._forward_once(d_rev, m_rev, return_debug=return_debug)
            h_bwd = torch.flip(h_bwd_raw, dims=[1]) * m_bool.unsqueeze(-1)
            diff2 = (h_fwd - h_bwd).pow(2).sum(dim=-1)  # [B,T,K]
            m_float = m_bool.float()
            trs_loss = (diff2 * m_float).sum() / (m_float.sum() + self.config.trs_eps)

        debug: Optional[Dict[str, Optional[torch.Tensor]]] = None
        if return_debug:
            debug = {
                "attn_weights": attn_w,
                "y_fwd": y_fwd,
                "y_bwd": y_bwd,
                "h_fwd": h_fwd,
                "h_bwd": h_bwd,
            }

        return {"H": h_fwd, "trs_loss": trs_loss, "debug": debug}
