"""Configuration for Stage-3 SATM (Spatio-Temporal Synergistic Dynamics)."""

from dataclasses import dataclass, field
from typing import List, Optional

from stage2.stage2_config import default_part_names


@dataclass
class Stage3Config:
    """Hyperparameters for SATM + TRS regularization.

    Attributes:
        d_model: Channel dimension C for per-part representations.
        num_parts: Number of facial parts K.
        num_layers: Number of stacked SATM blocks.
        num_heads: Multi-head attention heads for spatial synergy.
        dropout: Dropout rate applied to spatial attention outputs.
        mamba_d_state: State size for Mamba SSM.
        mamba_d_conv: Convolution kernel size inside Mamba.
        mamba_expand: Expansion ratio for Mamba inner dimension.
        trs_eps: Numerical epsilon for TRS masked average.
        alpha_spatial_max: Upper bound for spatial residual gate.
        alpha_temporal_max: Upper bound for temporal residual gate.
        part_names: Optional part names; used only for validation/debug.
    """

    d_model: int = 128
    num_parts: int = 6
    num_layers: int = 1
    num_heads: int = 4
    dropout: float = 0.1
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    trs_eps: float = 1e-6
    alpha_spatial_max: float = 0.01
    alpha_temporal_max: float = 0.01
    part_names: Optional[List[str]] = field(default_factory=default_part_names)

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.num_parts <= 0 or self.num_layers <= 0:
            raise ValueError("d_model, num_parts, and num_layers must be positive.")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must be in [0, 1].")
        if self.mamba_d_state <= 0 or self.mamba_d_conv <= 0 or self.mamba_expand <= 0:
            raise ValueError("Mamba parameters must be positive.")
        if self.trs_eps <= 0:
            raise ValueError("trs_eps must be positive.")
        if self.alpha_spatial_max < 0:
            raise ValueError("alpha_spatial_max must be non-negative.")
        if self.alpha_temporal_max < 0:
            raise ValueError("alpha_temporal_max must be non-negative.")
        if self.part_names is None or len(self.part_names) != self.num_parts:
            raise ValueError(
                f"part_names length ({len(self.part_names) if self.part_names else 0}) "
                f"must equal num_parts ({self.num_parts})."
            )
