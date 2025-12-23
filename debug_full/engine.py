"""Training and evaluation loop for DFEW Stage3 classifier."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch.cuda.amp import GradScaler
from torch import amp
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from datasets import DFEWClips
from logger import TrainingLogger
from modeling import Stage3EmotionModel, compute_losses
from probe import Stage1Monitor, Stage2Monitor

EPS = 1e-6


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def build_dataloaders(
    dataset_root: str | Path,
    train_csv: str | Path,
    val_csv: str | Path,
    num_frames: int,
    image_size: int,
    batch_size: int,
    num_workers: int,
    random_sample: bool = True,
    color_jitter: float = 0.0,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    load_cached_labels: bool = False,
    label_cache_dir: Optional[str | Path] = None,
    overfit_n: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = DFEWClips(
        root=dataset_root,
        csv_path=train_csv,
        num_frames=num_frames,
        image_size=image_size,
        train=True,
        random_sample=random_sample,
        color_jitter=color_jitter,
        load_cached_labels=load_cached_labels,
        label_cache_dir=label_cache_dir,
    )
    if overfit_n > 0:
        indices = list(range(min(overfit_n, len(train_ds))))
        from torch.utils.data import Subset

        train_ds = Subset(train_ds, indices)
    val_ds = DFEWClips(
        root=dataset_root,
        csv_path=val_csv,
        num_frames=num_frames,
        image_size=image_size,
        train=False,
        random_sample=False,
        color_jitter=0.0,
        load_cached_labels=load_cached_labels,
        label_cache_dir=label_cache_dir,
    )
    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=(overfit_n <= 0))
        if distributed
        else None
    )
    val_sampler = (
        DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        if distributed
        else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False if train_sampler is not None else overfit_n <= 0,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def _lr_from_optimizer(optimizer: torch.optim.Optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _cm_add(cm: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> None:
    valid = (targets >= 0) & (targets < num_classes)
    if valid.sum() == 0:
        return
    idx = targets[valid] * num_classes + preds[valid]
    bincount = torch.bincount(idx, minlength=num_classes * num_classes)
    cm += bincount.view(num_classes, num_classes).to(cm.dtype)


def _metrics_from_cm(cm: torch.Tensor) -> Dict[str, float]:
    total = cm.sum().item()
    diag = cm.diag()
    acc = diag.sum().item() / total if total > 0 else 0.0
    denom = cm.sum(dim=1).clamp_min(1)
    uar = (diag.float() / denom).mean().item()
    return {"acc": acc, "uar": uar, "war": acc}


def dump_cls_stats(prefix: str, logits: torch.Tensor, labels: torch.Tensor, num_classes: int):
    """
    logits: (N,C)
    labels: (N,)  int
    """
    with torch.no_grad():
        preds = logits.argmax(dim=1)

        # 兼容 labels 是 1..C 或 0..C-1 的情况
        if labels.min().item() == 1 and labels.max().item() == num_classes:
            labels0 = labels - 1
        else:
            labels0 = labels

        if preds.min().item() == 1 and preds.max().item() == num_classes:
            preds0 = preds - 1
        else:
            preds0 = preds

        label_hist = torch.bincount(labels0, minlength=num_classes).cpu()
        pred_hist = torch.bincount(preds0, minlength=num_classes).cpu()

        recalls = []
        for c in range(num_classes):
            mask_c = labels0 == c
            denom = mask_c.sum().item()
            if denom == 0:
                rec = float("nan")
            else:
                rec = (preds0[mask_c] == c).float().mean().item()
            recalls.append(rec)

        uar = torch.tensor([r for r in recalls if r == r]).mean().item()
        uniq = preds0.unique().numel()

        print(f"\n[{prefix}] unique_preds={uniq}")
        print(f"[{prefix}] label_hist={label_hist.tolist()} (sum={int(label_hist.sum())})")
        print(f"[{prefix}] pred_hist ={pred_hist.tolist()} (sum={int(pred_hist.sum())})")
        print(f"[{prefix}] recall_per_class={['{:.3f}'.format(r) if r==r else 'nan' for r in recalls]}")
        print(f"[{prefix}] UAR(recomputed)={uar:.4f}")


def compute_lambda_trs(current_step: int, target: float, warmup_steps: int) -> float:
    """Linear warmup for lambda_trs."""
    if warmup_steps <= 0:
        return float(target)
    if current_step >= warmup_steps:
        return float(target)
    return float(target) * float(current_step) / float(warmup_steps)


def _compute_stage3_stats(
    stage3_out: Dict,
    d_in: torch.Tensor,
    present: torch.Tensor,
    eps: float = EPS,
) -> Optional[Dict]:
    """Extract Stage3 metrics (sums + counts) for logging."""
    if stage3_out is None or stage3_out.get("H") is None or d_in is None:
        return None
    with torch.no_grad():
        h = stage3_out["H"].detach()
        present_bool = present.detach().bool()
        mask_float = present_bool.float()

        trs = stage3_out.get("trs_loss")
        trs_sum = float(trs.detach().cpu().item()) if trs is not None else 0.0
        trs_count = 1.0 if trs is not None else 0.0

        inv_mask = ~present_bool
        if inv_mask.any():
            masked_abs_max = float(h[inv_mask].abs().max().detach().cpu().item())
        else:
            masked_abs_max = 0.0

        # Temporal smoothness
        if h.shape[1] > 1:
            mask_temporal = (present_bool[:, 1:] & present_bool[:, :-1]).float()
            d_diff = torch.sqrt(((d_in[:, 1:] - d_in[:, :-1]).detach() ** 2).sum(-1) + eps)
            h_diff = torch.sqrt(((h[:, 1:] - h[:, :-1]) ** 2).sum(-1) + eps)
            temp2_sum = float((d_diff * mask_temporal).sum().detach().cpu().item())
            temp3_sum = float((h_diff * mask_temporal).sum().detach().cpu().item())
            temp_count = float(mask_temporal.sum().detach().cpu().item())
        else:
            temp2_sum = 0.0
            temp3_sum = 0.0
            temp_count = 0.0

        debug = stage3_out.get("debug") or {}

        h_bwd = debug.get("h_bwd")
        if h_bwd is not None:
            diff = torch.sqrt(((h - h_bwd.detach()) ** 2).sum(-1) + eps)
            fb_diff_sum = float((diff * mask_float).sum().detach().cpu().item())
            fb_diff_count = float(mask_float.sum().detach().cpu().item())
            fb_diff_values = diff[present_bool].detach().cpu()
        else:
            fb_diff_sum = 0.0
            fb_diff_count = 0.0
            fb_diff_values = None

        attn_w = debug.get("attn_weights")
        if attn_w is not None:
            p = attn_w.detach()
            p_safe = p.clamp_min(eps)
            entropy = -(p_safe * p_safe.log()).sum(dim=-1)  # (B,T,K)
            entropy_sum = float((entropy * mask_float).sum().detach().cpu().item())
            entropy_count = float(mask_float.sum().detach().cpu().item())
            attn_sum = (p * mask_float.unsqueeze(-1)).sum(dim=(0, 1)).cpu()
            attn_count = mask_float.sum(dim=(0, 1)).cpu()
        else:
            entropy_sum = 0.0
            entropy_count = 0.0
            attn_sum = None
            attn_count = None

        y_fwd = debug.get("y_fwd")
        y_bwd = debug.get("y_bwd")
        if y_fwd is not None and y_bwd is not None:
            efwd = torch.norm(y_fwd.detach(), dim=-1)
            ebwd = torch.norm(y_bwd.detach(), dim=-1)
            energy_fwd_sum = float((efwd * mask_float).sum().detach().cpu().item())
            energy_bwd_sum = float((ebwd * mask_float).sum().detach().cpu().item())
            energy_count = float(mask_float.sum().detach().cpu().item())
        else:
            energy_fwd_sum = 0.0
            energy_bwd_sum = 0.0
            energy_count = 0.0

        # Residual energy ratio and cosine similarity
        diff = h - d_in
        diff_norm = torch.sqrt((diff ** 2).sum(dim=-1) + eps)  # (B,T,K)
        din_norm = torch.sqrt((d_in ** 2).sum(dim=-1) + eps)
        resid_ratio = diff_norm / (din_norm + eps)
        resid_ratio_sum = float((resid_ratio * mask_float).sum().detach().cpu().item())
        resid_count = float(mask_float.sum().detach().cpu().item())

        cos_sim = torch.nn.functional.cosine_similarity(h, d_in, dim=-1)
        cos_sum = float((cos_sim * mask_float).sum().detach().cpu().item())

        return {
            "trs_sum": trs_sum,
            "trs_count": trs_count,
            "fb_diff_sum": fb_diff_sum,
            "fb_diff_count": fb_diff_count,
            "fb_diff_values": fb_diff_values,
            "masked_output_abs_max": masked_abs_max,
            "temp2_sum": temp2_sum,
            "temp3_sum": temp3_sum,
            "temp_count": temp_count,
            "attn_entropy_sum": entropy_sum,
            "attn_entropy_count": entropy_count,
            "attn_sum": attn_sum,
            "attn_count": attn_count,
            "energy_fwd_sum": energy_fwd_sum,
            "energy_bwd_sum": energy_bwd_sum,
            "energy_count": energy_count,
            "resid_ratio_sum": resid_ratio_sum,
            "resid_count": resid_count,
            "cos_sum": cos_sum,
        }


def _metrics_from_stats(stats: Dict, eps: float = EPS) -> Dict[str, float | torch.Tensor | None]:
    """Convert raw sums/counts into scalar metrics for logging."""
    fb_mean = (
        stats["fb_diff_sum"] / (stats["fb_diff_count"] + eps) if stats.get("fb_diff_count", 0.0) > 0 else None
    )
    fb_p90 = None
    if stats.get("fb_diff_values") is not None and stats["fb_diff_values"].numel() > 0:
        fb_p90 = float(torch.quantile(stats["fb_diff_values"], 0.9).item())
    temp2_mean = stats["temp2_sum"] / (stats["temp_count"] + eps) if stats.get("temp_count", 0.0) > 0 else 0.0
    temp3_mean = stats["temp3_sum"] / (stats["temp_count"] + eps) if stats.get("temp_count", 0.0) > 0 else 0.0
    attn_entropy = (
        stats["attn_entropy_sum"] / (stats["attn_entropy_count"] + eps)
        if stats.get("attn_entropy_count", 0.0) > 0
        else None
    )
    attn_mean_map = None
    if stats.get("attn_sum") is not None and stats.get("attn_count") is not None:
        denom = stats["attn_count"].clamp_min(1.0).unsqueeze(-1)
        attn_mean_map = stats["attn_sum"] / denom
    energy_ratio = None
    if stats.get("energy_count", 0.0) > 0:
        ef = stats["energy_fwd_sum"] / (stats["energy_count"] + eps)
        eb = stats["energy_bwd_sum"] / (stats["energy_count"] + eps)
        energy_ratio = float(ef / (ef + eb + eps))

    resid_ratio_mean = (
        stats["resid_ratio_sum"] / (stats["resid_count"] + eps) if stats.get("resid_count", 0.0) > 0 else 0.0
    )
    cos_mean = stats["cos_sum"] / (stats.get("resid_count", 0.0) + eps) if stats.get("resid_count", 0.0) > 0 else 0.0

    trs_loss = stats["trs_sum"] / max(stats.get("trs_count", 1.0), eps)
    return {
        "trs_loss": trs_loss,
        "fb_diff_mean": fb_mean,
        "fb_diff_p90": fb_p90,
        "fb_diff_values": stats.get("fb_diff_values"),
        "masked_output_abs_max": stats["masked_output_abs_max"],
        "temporal_diff_stage2_mean": temp2_mean,
        "temporal_diff_stage3_mean": temp3_mean,
        "attn_entropy": attn_entropy,
        "attn_mean_map": attn_mean_map,
        "fwd_bwd_energy_ratio": energy_ratio,
        "resid_ratio_mean": resid_ratio_mean,
        "cosine_similarity_mean": cos_mean,
    }


def _log_stage3_metrics(
    logger: TrainingLogger,
    metrics: Dict[str, float | torch.Tensor | None],
    step: int,
    split: str,
    log_hist: bool,
    zero_tol: float,
    assert_zero: bool,
    lambda_trs_current: Optional[float] = None,
) -> None:
    if logger is None:
        return
    logger.log_scalar(f"{split}/stage3/trs_loss", float(metrics["trs_loss"]), step)
    if lambda_trs_current is not None:
        logger.log_scalar(f"{split}/stage3/lambda_trs_current", float(lambda_trs_current), step)
    if metrics.get("fb_diff_mean") is not None:
        logger.log_scalar(f"{split}/stage3/forward_backward_diff_mean", float(metrics["fb_diff_mean"]), step)
    if metrics.get("fb_diff_p90") is not None:
        logger.log_scalar(f"{split}/stage3/forward_backward_diff_p90", float(metrics["fb_diff_p90"]), step)
    if log_hist and metrics.get("fb_diff_values") is not None and metrics["fb_diff_values"].numel() > 0:
        logger.log_histogram(f"{split}/stage3/forward_backward_diff_hist", metrics["fb_diff_values"], step)
    logger.log_scalar(f"{split}/stage3/masked_output_abs_max", float(metrics["masked_output_abs_max"]), step)
    logger.log_scalar(
        f"{split}/stage3/temporal_diff_stage2_mean", float(metrics["temporal_diff_stage2_mean"]), step
    )
    logger.log_scalar(
        f"{split}/stage3/temporal_diff_stage3_mean", float(metrics["temporal_diff_stage3_mean"]), step
    )
    if metrics.get("attn_entropy") is not None:
        logger.log_scalar(f"{split}/stage3/spatial_attn_entropy", float(metrics["attn_entropy"]), step)
    if metrics.get("attn_mean_map") is not None:
        img = metrics["attn_mean_map"].unsqueeze(0)  # (1,K,K)
        logger.log_image(f"{split}/stage3/spatial_attn_mean_map", img, step)
    if metrics.get("fwd_bwd_energy_ratio") is not None:
        logger.log_scalar(f"{split}/stage3/fwd_bwd_energy_ratio", float(metrics["fwd_bwd_energy_ratio"]), step)
    logger.log_scalar(f"{split}/stage3/residual_energy_ratio_mean", float(metrics["resid_ratio_mean"]), step)
    logger.log_scalar(f"{split}/stage3/cosine_similarity_mean", float(metrics["cosine_similarity_mean"]), step)

    if assert_zero and metrics["masked_output_abs_max"] > zero_tol:
        raise AssertionError(
            f"Stage3 masked output not zero: {metrics['masked_output_abs_max']:.6f} > tol {zero_tol}"
        )


class Stage3MetricAggregator:
    """Accumulate Stage3 stats across a split."""

    def __init__(self, eps: float = EPS, max_hist_values: int = 50000) -> None:
        self.eps = eps
        self.max_hist_values = max_hist_values
        self.reset()

    def reset(self) -> None:
        self.trs_sum = 0.0
        self.trs_count = 0.0
        self.fb_diff_sum = 0.0
        self.fb_diff_count = 0.0
        self.fb_values: List[torch.Tensor] = []
        self.fb_value_count = 0
        self.mask_abs_max = 0.0
        self.temp2_sum = 0.0
        self.temp3_sum = 0.0
        self.temp_count = 0.0
        self.attn_entropy_sum = 0.0
        self.attn_entropy_count = 0.0
        self.attn_sum: Optional[torch.Tensor] = None
        self.attn_count: Optional[torch.Tensor] = None
        self.energy_fwd_sum = 0.0
        self.energy_bwd_sum = 0.0
        self.energy_count = 0.0
        self.resid_ratio_sum = 0.0
        self.resid_count = 0.0
        self.cos_sum = 0.0

    def update(self, stats: Optional[Dict]) -> None:
        if stats is None:
            return
        self.trs_sum += stats.get("trs_sum", 0.0)
        self.trs_count += stats.get("trs_count", 0.0)
        self.fb_diff_sum += stats.get("fb_diff_sum", 0.0)
        self.fb_diff_count += stats.get("fb_diff_count", 0.0)
        vals = stats.get("fb_diff_values")
        if vals is not None and vals.numel() > 0 and self.fb_value_count < self.max_hist_values:
            vals = vals.view(-1)
            needed = self.max_hist_values - self.fb_value_count
            if vals.numel() > needed:
                vals = vals[:needed]
            self.fb_values.append(vals)
            self.fb_value_count += vals.numel()
        self.mask_abs_max = max(self.mask_abs_max, float(stats.get("masked_output_abs_max", 0.0)))
        self.temp2_sum += stats.get("temp2_sum", 0.0)
        self.temp3_sum += stats.get("temp3_sum", 0.0)
        self.temp_count += stats.get("temp_count", 0.0)
        self.attn_entropy_sum += stats.get("attn_entropy_sum", 0.0)
        self.attn_entropy_count += stats.get("attn_entropy_count", 0.0)
        if stats.get("attn_sum") is not None and stats.get("attn_count") is not None:
            if self.attn_sum is None:
                self.attn_sum = stats["attn_sum"].clone()
                self.attn_count = stats["attn_count"].clone()
            else:
                self.attn_sum += stats["attn_sum"]
                self.attn_count += stats["attn_count"]
        self.energy_fwd_sum += stats.get("energy_fwd_sum", 0.0)
        self.energy_bwd_sum += stats.get("energy_bwd_sum", 0.0)
        self.energy_count += stats.get("energy_count", 0.0)
        self.resid_ratio_sum += stats.get("resid_ratio_sum", 0.0)
        self.resid_count += stats.get("resid_count", 0.0)
        self.cos_sum += stats.get("cos_sum", 0.0)

    def final_stats(self) -> Dict:
        fb_values = torch.cat(self.fb_values) if self.fb_values else None
        return {
            "trs_sum": self.trs_sum,
            "trs_count": self.trs_count,
            "fb_diff_sum": self.fb_diff_sum,
            "fb_diff_count": self.fb_diff_count,
            "fb_diff_values": fb_values,
            "masked_output_abs_max": self.mask_abs_max,
            "temp2_sum": self.temp2_sum,
            "temp3_sum": self.temp3_sum,
            "temp_count": self.temp_count,
            "attn_entropy_sum": self.attn_entropy_sum,
            "attn_entropy_count": self.attn_entropy_count,
            "attn_sum": self.attn_sum,
            "attn_count": self.attn_count,
            "energy_fwd_sum": self.energy_fwd_sum,
            "energy_bwd_sum": self.energy_bwd_sum,
            "energy_count": self.energy_count,
            "resid_ratio_sum": self.resid_ratio_sum,
            "resid_count": self.resid_count,
            "cos_sum": self.cos_sum,
        }


def _log_trajectory_pca(
    logger: TrainingLogger | None,
    samples: List[Tuple[torch.Tensor, torch.Tensor]],
    step: int,
    part_names: Sequence[str],
) -> None:
    if logger is None or not samples:
        return
    h, mask = samples[0]  # use the first collected sample
    h_seq = h[0]  # (T,K,C)
    mask_seq = mask[0].bool()
    flat_mask = mask_seq.view(-1)
    flat_feats = h_seq.view(-1, h_seq.shape[-1])
    valid_feats = flat_feats[flat_mask]
    if valid_feats.shape[0] < 2:
        return
    feats_center = valid_feats - valid_feats.mean(dim=0, keepdim=True)
    q = min(2, feats_center.shape[1])
    U, S, V = torch.pca_lowrank(feats_center, q=q)
    coords = torch.matmul(feats_center, V[:, :2]).cpu()
    flat_idx = torch.nonzero(flat_mask, as_tuple=False).squeeze(-1).cpu()
    k = mask_seq.shape[1]
    t_idx = (flat_idx // k).numpy()
    p_idx = (flat_idx % k).numpy()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(coords[:, 0].numpy(), coords[:, 1].numpy(), c=t_idx, cmap="viridis", s=12)
    ax.set_title("Stage3 trajectory PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("time index")
    for idx in range(min(5, len(p_idx))):
        ax.text(coords[idx, 0].item(), coords[idx, 1].item(), part_names[p_idx[idx]], fontsize=6, alpha=0.8)
    logger.log_figure("stage3/trajectory_pca", fig, step)
    plt.close(fig)


def train_one_epoch(
    model: Stage3EmotionModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    logger: TrainingLogger | None,
    lambda_con: float,
    lambda_dev: float,
    lambda_trs_target: float,
    scaler: GradScaler,
    use_amp: bool,
    log_interval: int,
    stage3_log_interval: int,
    stage1_monitor: Stage1Monitor | None,
    stage2_monitor: Stage2Monitor | None,
    num_classes: int,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
    show_progress: bool = True,
    use_aux_losses: bool = True,
    ce_weights: torch.Tensor | None = None,
    compute_trs: bool = True,
    stage3_zero_tol: float = 1e-6,
    assert_zero_missing: bool = False,
    warmup_steps_trs: int = 1000,
    start_global_step: int = 0,
) -> Dict[str, float]:
    model.train()
    total_steps = len(dataloader)
    loss_sums = torch.zeros(5, device=device)
    cm = torch.zeros((num_classes, num_classes), device=device)
    epoch_logits: List[torch.Tensor] = []
    epoch_labels: List[torch.Tensor] = []
    global_step = start_global_step

    start_time = time.time()
    iterator = tqdm(
        dataloader,
        desc=f"Train {epoch}",
        total=len(dataloader),
        disable=not (show_progress and rank == 0),
        leave=False,
        mininterval=0.1,
    )
    for step, batch in enumerate(iterator):
        if len(batch) == 4:
            frames, labels, video_ids, parsed_labels = batch
        else:
            frames, labels, video_ids = batch
            parsed_labels = torch.empty(0)
        global_step = start_global_step + epoch * total_steps + step
        frames = frames.to(device)
        labels = labels.to(device)
        parsed_labels = parsed_labels.to(device) if parsed_labels.numel() > 0 else None
        base_model = _unwrap(model)
        if parsed_labels is None and hasattr(base_model, "stage1") and getattr(base_model.stage1, "parser", None):
            base_model.stage1.parser._current_video_ids = list(video_ids)

        optimizer.zero_grad(set_to_none=True)
        ac_ctx = amp.autocast(device_type="cuda", enabled=use_amp and device.type == "cuda") if device.type == "cuda" else nullcontext()
        with ac_ctx:
            need_debug = logger is not None and (step % stage3_log_interval == 0)
            lambda_trs = compute_lambda_trs(global_step, lambda_trs_target, warmup_steps_trs)
            outputs = model(
                frames,
                parsed_labels,
                compute_trs=compute_trs,
                return_debug=need_debug,
            )
            losses = compute_losses(
                outputs,
                labels,
                lambda_con=lambda_con,
                lambda_dev=lambda_dev,
                lambda_trs=lambda_trs,
                use_aux_losses=use_aux_losses,
                weights=ce_weights,
            )
        loss_sums[0] += losses["total"]
        loss_sums[1] += losses["task_loss"]
        loss_sums[2] += losses["l_con"]
        loss_sums[3] += losses["l_dev"]
        loss_sums[4] += losses["l_trs"]

        scaler.scale(losses["total"]).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step_update(global_step)

        preds = torch.argmax(outputs["logits"], dim=-1)
        _cm_add(cm, preds, labels, num_classes)
        epoch_logits.append(outputs["logits"].detach().float().cpu())
        epoch_labels.append(labels.detach().long().cpu())

        if stage1_monitor is not None and "stage1" in outputs:
            stage1_monitor.update(outputs["stage1"], video_ids)
        if stage2_monitor is not None and "stage2" in outputs:
            stage2_monitor.update(outputs["stage2"], video_ids)

        if logger is not None:
            logger.log_scalar("stage3/lambda_trs_current", lambda_trs, global_step)
        if logger is not None and step % log_interval == 0:
            logger.log_scalar("train/loss/total_step", losses["total"].item(), global_step)
            logger.log_scalar("train/loss/task_step", losses["task_loss"].item(), global_step)
            logger.log_scalar("train/loss/l_con_step", losses["l_con"].item(), global_step)
            logger.log_scalar("train/loss/l_dev_step", losses["l_dev"].item(), global_step)
            logger.log_scalar("train/loss/l_trs_step", losses["l_trs"].item(), global_step)
            logger.log_scalar("train/lr", _lr_from_optimizer(optimizer), global_step)

        if logger is not None and global_step < 1000:
            logger.dump_text(
                f"[step {global_step}] L_task={losses['task_loss'].item():.6f} "
                f"L_dev={losses['l_dev'].item():.6f} "
                f"L_con={losses['l_con'].item():.6f} "
                f"L_trs={losses['l_trs'].item():.6f}"
            )

        if logger is not None and step % stage3_log_interval == 0:
            stage3_out = outputs.get("stage3")
            d_in = outputs.get("stage3_input")
            present = outputs.get("stage2", {}).get("present") if isinstance(outputs.get("stage2"), dict) else None
            if stage3_out is not None and d_in is not None and present is not None:
                stats = _compute_stage3_stats(stage3_out, d_in, present, eps=EPS)
                if stats is not None:
                    metrics = _metrics_from_stats(stats, eps=EPS)
                    _log_stage3_metrics(
                        logger,
                        metrics,
                        global_step,
                        split="train",
                        log_hist=True,
                        zero_tol=stage3_zero_tol,
                        assert_zero=assert_zero_missing,
                        lambda_trs_current=lambda_trs,
                    )

        if hasattr(iterator, "set_postfix") and step % max(1, log_interval // 2) == 0:
            iterator.set_postfix(loss=float(losses["total"].item()))

    step_tensor = torch.tensor([total_steps], device=device, dtype=torch.float32)
    if distributed:
        dist.all_reduce(loss_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(cm, op=dist.ReduceOp.SUM)
        dist.all_reduce(step_tensor, op=dist.ReduceOp.SUM)

    elapsed = time.time() - start_time
    total_steps_global = step_tensor.item() if step_tensor.item() > 0 else 1.0
    avg_losses = {
        "total": (loss_sums[0] / total_steps_global).item(),
        "task": (loss_sums[1] / total_steps_global).item(),
        "l_con": (loss_sums[2] / total_steps_global).item(),
        "l_dev": (loss_sums[3] / total_steps_global).item(),
        "l_trs": (loss_sums[4] / total_steps_global).item(),
    }
    metrics = _metrics_from_cm(cm)
    try:
        concat_logits = torch.cat(epoch_logits, dim=0).cpu()
        concat_labels = torch.cat(epoch_labels, dim=0).cpu()
        dump_cls_stats(f"train_epoch{epoch}", concat_logits, concat_labels, num_classes)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] dump_cls_stats train failed: {exc}")
    if logger is not None:
        logger.log_scalar("train/loss/total", avg_losses["total"], epoch)
        logger.log_scalar("train/loss/task", avg_losses["task"], epoch)
        logger.log_scalar("train/loss/l_con", avg_losses["l_con"], epoch)
        logger.log_scalar("train/loss/l_dev", avg_losses["l_dev"], epoch)
        logger.log_scalar("train/loss/l_trs", avg_losses["l_trs"], epoch)
        logger.log_scalar("train/acc", metrics["acc"], epoch)
        logger.log_scalar("train/uar", metrics["uar"], epoch)
        logger.log_scalar("train/war", metrics["war"], epoch)
        logger.dump_text(f"Epoch {epoch} train time {elapsed:.1f}s acc {metrics['acc']:.4f} uar {metrics['uar']:.4f}")

    return {
        "acc": metrics["acc"],
        "uar": metrics["uar"],
        "war": metrics["war"],
        "loss_total": avg_losses["total"],
        "loss_task": avg_losses["task"],
        "loss_l_con": avg_losses["l_con"],
        "loss_l_dev": avg_losses["l_dev"],
        "loss_l_trs": avg_losses["l_trs"],
        "time": elapsed,
        "global_step": global_step,
    }


def evaluate(
    model: Stage3EmotionModel,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    logger: TrainingLogger | None,
    lambda_con: float,
    lambda_dev: float,
    lambda_trs_target: float,
    stage1_monitor: Stage1Monitor | None,
    stage2_monitor: Stage2Monitor | None,
    num_classes: int,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
    show_progress: bool = True,
    use_aux_losses: bool = True,
    ce_weights: torch.Tensor | None = None,
    compute_trs: bool = True,
    stage3_zero_tol: float = 1e-6,
    assert_zero_missing: bool = False,
    stage3_pca_batches: int = 3,
    warmup_steps_trs: int = 1000,
    global_step: int = 0,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    loss_sums = torch.zeros(5, device=device)
    cms = {
        "full": torch.zeros((num_classes, num_classes), device=device),
        "nostage2": torch.zeros((num_classes, num_classes), device=device),
        "nostage1+2+3": torch.zeros((num_classes, num_classes), device=device),
        "nostage3": torch.zeros((num_classes, num_classes), device=device),
    }
    epoch_logits: List[torch.Tensor] = []
    epoch_labels: List[torch.Tensor] = []
    stage3_agg = Stage3MetricAggregator()
    pca_samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
    lambda_trs_eval = compute_lambda_trs(global_step, lambda_trs_target, warmup_steps_trs)

    with torch.no_grad():
        iterator = tqdm(
            dataloader,
            desc=f"Val {epoch}",
            total=len(dataloader),
            disable=not (show_progress and rank == 0),
            leave=False,
            mininterval=0.1,
        )
        for batch in iterator:
            if len(batch) == 4:
                frames, labels, video_ids, parsed_labels = batch
            else:
                frames, labels, video_ids = batch
                parsed_labels = torch.empty(0)
            frames = frames.to(device)
            labels = labels.to(device)
            parsed_labels = parsed_labels.to(device) if parsed_labels.numel() > 0 else None
            base_model = _unwrap(model)
            if parsed_labels is None and hasattr(base_model, "stage1") and getattr(base_model.stage1, "parser", None):
                base_model.stage1.parser._current_video_ids = list(video_ids)
            outs = base_model.forward_variants(
                frames,
                variants=("full", "nostage2", "nostage3", "nostage1+2+3"),
                labels=parsed_labels,
                compute_trs=compute_trs,
                return_debug=True,
            )

            full_out = outs["full"]
            losses = compute_losses(
                full_out,
                labels,
                lambda_con=lambda_con,
                lambda_dev=lambda_dev,
                lambda_trs=lambda_trs_eval,
                use_aux_losses=use_aux_losses,
                weights=ce_weights,
            )
            loss_sums[0] += losses["total"]
            loss_sums[1] += losses["task_loss"]
            loss_sums[2] += losses["l_con"]
            loss_sums[3] += losses["l_dev"]
            loss_sums[4] += losses["l_trs"]

            for name in cms:
                preds = torch.argmax(outs[name]["logits"], dim=-1)
                _cm_add(cms[name], preds, labels, num_classes)
            epoch_logits.append(outs["full"]["logits"].detach().float().cpu())
            epoch_labels.append(labels.detach().long().cpu())

            if stage1_monitor is not None and "stage1" in full_out:
                stage1_monitor.update(full_out["stage1"], video_ids)
            if stage2_monitor is not None and "stage2" in full_out:
                stage2_monitor.update(full_out["stage2"], video_ids)

            if logger is not None and len(pca_samples) < stage3_pca_batches:
                stage3_out = full_out.get("stage3")
                if stage3_out is not None and stage3_out.get("H") is not None:
                    present = full_out.get("stage2", {}).get("present")
                    if present is not None:
                        pca_samples.append(
                            (stage3_out["H"].detach().cpu()[:1], present.detach().cpu()[:1])
                        )

            if logger is not None:
                stage3_out = full_out.get("stage3")
                d_in = full_out.get("stage3_input")
                present = full_out.get("stage2", {}).get("present") if isinstance(full_out.get("stage2"), dict) else None
                if stage3_out is not None and d_in is not None and present is not None:
                    stats = _compute_stage3_stats(stage3_out, d_in, present, eps=EPS)
                    stage3_agg.update(stats)

    step_tensor = torch.tensor([len(dataloader)], device=device, dtype=torch.float32)
    if distributed:
        dist.all_reduce(loss_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(step_tensor, op=dist.ReduceOp.SUM)
        for name in cms:
            dist.all_reduce(cms[name], op=dist.ReduceOp.SUM)

    total_steps_global = step_tensor.item() if step_tensor.item() > 0 else 1.0
    avg_losses = {
        "total": (loss_sums[0] / total_steps_global).item(),
        "task": (loss_sums[1] / total_steps_global).item(),
        "l_con": (loss_sums[2] / total_steps_global).item(),
        "l_dev": (loss_sums[3] / total_steps_global).item(),
        "l_trs": (loss_sums[4] / total_steps_global).item(),
    }
    metrics_full = _metrics_from_cm(cms["full"])
    metrics_nostage2 = _metrics_from_cm(cms["nostage2"])
    metrics_nostage1 = _metrics_from_cm(cms["nostage1+2+3"])
    metrics_nostage3 = _metrics_from_cm(cms["nostage3"])
    try:
        concat_logits = torch.cat(epoch_logits, dim=0).cpu()
        concat_labels = torch.cat(epoch_labels, dim=0).cpu()
        dump_cls_stats(f"val_epoch{epoch}", concat_logits, concat_labels, num_classes)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] dump_cls_stats val failed: {exc}")

    if logger is not None:
        logger.log_scalar("val/loss/total", avg_losses["total"], epoch)
        logger.log_scalar("val/loss/task", avg_losses["task"], epoch)
        logger.log_scalar("val/loss/l_con", avg_losses["l_con"], epoch)
        logger.log_scalar("val/loss/l_dev", avg_losses["l_dev"], epoch)
        logger.log_scalar("val/loss/l_trs", avg_losses["l_trs"], epoch)

        logger.log_scalar("val/acc", metrics_full["acc"], epoch)
        logger.log_scalar("val/uar", metrics_full["uar"], epoch)
        logger.log_scalar("val/war", metrics_full["war"], epoch)
        logger.log_scalar("ablation/acc_full", metrics_full["acc"], epoch)
        logger.log_scalar("ablation/uar_full", metrics_full["uar"], epoch)
        logger.log_scalar("ablation/war_full", metrics_full["war"], epoch)
        logger.log_scalar("ablation/acc_nostage2", metrics_nostage2["acc"], epoch)
        logger.log_scalar("ablation/uar_nostage2", metrics_nostage2["uar"], epoch)
        logger.log_scalar("ablation/war_nostage2", metrics_nostage2["war"], epoch)
        logger.log_scalar("ablation/acc_nostage1+2+3", metrics_nostage1["acc"], epoch)
        logger.log_scalar("ablation/uar_nostage1+2+3", metrics_nostage1["uar"], epoch)
        logger.log_scalar("ablation/war_nostage1+2+3", metrics_nostage1["war"], epoch)
        logger.log_scalar("ablation/acc_nostage3", metrics_nostage3["acc"], epoch)
        logger.log_scalar("ablation/uar_nostage3", metrics_nostage3["uar"], epoch)
        logger.log_scalar("ablation/war_nostage3", metrics_nostage3["war"], epoch)
        logger.dump_text(
            f"Epoch {epoch} val acc {metrics_full['acc']:.4f} uar {metrics_full['uar']:.4f} "
            f"nostage2 acc {metrics_nostage2['acc']:.4f} nostage1+2+3 acc {metrics_nostage1['acc']:.4f} "
            f"nostage3 acc {metrics_nostage3['acc']:.4f}"
        )

        stage3_stats = stage3_agg.final_stats()
        metrics = _metrics_from_stats(stage3_stats, eps=EPS)
        _log_stage3_metrics(
            logger,
            metrics,
            epoch,
            split="val",
            log_hist=True,
            zero_tol=stage3_zero_tol,
            assert_zero=assert_zero_missing,
            lambda_trs_current=lambda_trs_eval,
        )
        _log_trajectory_pca(logger, pca_samples, epoch, part_names=_unwrap(model).part_names)

    return {
        "full": metrics_full,
        "nostage2": metrics_nostage2,
        "nostage1+2+3": metrics_nostage1,
        "nostage3": metrics_nostage3,
        "losses": avg_losses,
    }
