"""Configuration for Stage-2 deviation-prototype module."""

from dataclasses import dataclass, field
from typing import List, Optional


def default_part_names() -> List[str]:
    """Default six-part names for facial components."""
    return ["left_brow", "right_brow", "left_eye", "right_eye", "mouth", "full_face"]


@dataclass
class Stage2Config:
    """Hyperparameters for Stage-2 processing.

    Attributes:
        d_in: Input feature dimension per part (Stage-1 output).
        latent_dim: Projected latent dimension C.
        hidden_dim: Hidden size for the part-specific MLPs.
        num_parts: Number of facial parts K.
        num_prototypes: Number of prototypes per part M.
        eps: Small value to avoid division by zero.
        use_dropout: Whether to enable dropout inside MLPs.
        dropout_p: Dropout probability.
        activation: Activation name ("gelu" or "relu").
        part_names: Optional explicit part names; must match num_parts if provided.
    """

    d_in: int = 768
    latent_dim: int = 128
    hidden_dim: int = 256
    num_parts: int = 6
    num_prototypes: int = 32
    eps: float = 1e-6
    use_dropout: bool = True
    dropout_p: float = 0.1
    activation: str = "gelu"
    part_names: Optional[List[str]] = field(default_factory=default_part_names)
    logit_scale_init: float = 10.0
    logit_scale_max: float = 100.0

    def __post_init__(self) -> None:
        if self.d_in <= 0 or self.latent_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("d_in, latent_dim, and hidden_dim must be positive.")
        if self.num_parts <= 0 or self.num_prototypes <= 0:
            raise ValueError("num_parts and num_prototypes must be positive.")
        if self.activation not in ("gelu", "relu"):
            raise ValueError("activation must be 'gelu' or 'relu'.")
        if self.dropout_p < 0 or self.dropout_p > 1:
            raise ValueError("dropout_p must be in [0,1].")
        if self.part_names is None or len(self.part_names) != self.num_parts:
            raise ValueError(
                f"part_names length ({len(self.part_names) if self.part_names else 0}) "
                f"must equal num_parts ({self.num_parts})."
            )
