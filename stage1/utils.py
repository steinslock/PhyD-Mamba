from typing import Dict, List, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def compute_patch_grid(height: int, width: int, patch_size: int) -> Tuple[int, int]:
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"Input size ({height}x{width}) must be divisible by patch_size={patch_size}."
        )
    return height // patch_size, width // patch_size


def normalize_frames(
    frames: torch.Tensor, normalize_to_imagenet: bool = True
) -> torch.Tensor:
    """Convert frames to float in [0,1], then normalize."""
    frames = frames.float()
    if frames.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        frames = frames / 255.0
    else:
        max_val = frames.max()
        if torch.isfinite(max_val) and max_val > 1.0:
            frames = frames / 255.0
    frames = frames.clamp(0.0, 1.0)
    if not normalize_to_imagenet:
        return frames

    mean = torch.tensor(IMAGENET_MEAN, device=frames.device, dtype=frames.dtype).view(
        1, 1, 3, 1, 1
    )
    std = torch.tensor(IMAGENET_STD, device=frames.device, dtype=frames.dtype).view(
        1, 1, 3, 1, 1
    )
    return (frames - mean) / std


def labels_to_part_masks(
    labels: torch.Tensor, raw_to_part: Dict[int, Union[int, Sequence[int]]], num_parts: int
) -> torch.Tensor:
    """Convert raw label map (B,T,H,W) to one-hot masks (B,T,K,H,W)."""
    if labels.dim() != 4:
        raise ValueError(f"labels must have shape (B,T,H,W); got {labels.shape}")
    b, t, h, w = labels.shape
    masks = torch.zeros((b, t, num_parts, h, w), device=labels.device, dtype=torch.float32)
    for raw_id, part_ids in raw_to_part.items():
        if isinstance(part_ids, int):
            part_ids = (part_ids,)
        for part_id in part_ids:
            masks[:, :, part_id] += (labels == raw_id).float()
    return masks.clamp_(0.0, 1.0)


def masks_to_patch_weights(
    masks_pix: torch.Tensor, patch_size: int
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Area-pool pixel masks to patch-grid soft weights.

    Args:
        masks_pix: (B, T, K, H, W)
    Returns:
        weights: (B, T, K, N) where N = Gh*Gw
        grid: (Gh, Gw)
    """
    if masks_pix.dim() != 5:
        raise ValueError(f"masks_pix must have shape (B,T,K,H,W); got {masks_pix.shape}")
    b, t, k, h, w = masks_pix.shape
    gh, gw = compute_patch_grid(h, w, patch_size)
    masks_bt = masks_pix.view(b * t, k, h, w)
    pooled = F.avg_pool2d(masks_bt, kernel_size=patch_size, stride=patch_size)
    pooled = pooled.view(b, t, k, gh, gw)
    weights = pooled.flatten(-2, -1)  # (B,T,K,N)
    return weights, (gh, gw)


def swap_left_right_parts(
    tensor: torch.Tensor, left_right_pairs: Sequence[Tuple[int, int]], part_dim: int = -2
) -> torch.Tensor:
    """Swap left/right part dimension for horizontal flip augmentation.

    Args:
        tensor: (..., K, ...)
    """
    if not left_right_pairs:
        return tensor
    out = tensor.clone()
    nd = out.dim()
    if part_dim < 0:
        part_dim = nd + part_dim
    for left, right in left_right_pairs:
        left_idx = [slice(None)] * nd
        right_idx = [slice(None)] * nd
        left_idx[part_dim] = left
        right_idx[part_dim] = right
        tmp = out[tuple(left_idx)].clone()
        out[tuple(left_idx)] = out[tuple(right_idx)]
        out[tuple(right_idx)] = tmp
    return out


def patch_weight_to_area(weights: torch.Tensor) -> torch.Tensor:
    """Sum patch weights to obtain soft area."""
    return weights.sum(dim=-1)


def visualize_patch_mask(
    weights: torch.Tensor,
    grid: Tuple[int, int],
    title: str = "mask",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Visualize a single part mask on the patch grid (debug helper).

    Args:
        weights: 1D tensor of length N = Gh*Gw or 2D (Gh, Gw)
        grid: (Gh, Gw)
        title: plot title
        save_path: if provided, save the figure to this path
        show: whether to display the figure (default True)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("matplotlib is required for visualization.")

    gh, gw = grid
    mask = weights.view(gh, gw).detach().cpu().numpy()
    plt.imshow(mask, cmap="viridis")
    plt.colorbar()
    plt.title(title)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
