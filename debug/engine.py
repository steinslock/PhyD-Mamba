"""Training and evaluation loop for DFEW Stage2 classifier."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from contextlib import nullcontext
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from datasets import DFEWClips
from logger import TrainingLogger
from modeling import Stage2EmotionModel, compute_losses
from probe import Stage1Monitor, Stage2Monitor
from typing import Optional


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


def train_one_epoch(
    model: Stage2EmotionModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    logger: TrainingLogger | None,
    lambda_proto: float,
    scaler: GradScaler,
    use_amp: bool,
    log_interval: int,
    stage1_monitor: Stage1Monitor | None,
    stage2_monitor: Stage2Monitor | None,
    num_classes: int,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
    show_progress: bool = True,
    use_aux_losses: bool = True,
    ce_weights: torch.Tensor | None = None,
) -> Dict[str, float]:
    model.train()
    total_steps = len(dataloader)
    loss_sums = torch.zeros(4, device=device)
    cm = torch.zeros((num_classes, num_classes), device=device)
    epoch_logits: List[torch.Tensor] = []
    epoch_labels: List[torch.Tensor] = []

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
        global_step = epoch * total_steps + step
        frames = frames.to(device)
        labels = labels.to(device)
        parsed_labels = parsed_labels.to(device) if parsed_labels.numel() > 0 else None
        base_model = _unwrap(model)
        if parsed_labels is None and hasattr(base_model, "stage1") and getattr(base_model.stage1, "parser", None):
            base_model.stage1.parser._current_video_ids = list(video_ids)

        optimizer.zero_grad(set_to_none=True)
        ac_ctx = autocast(enabled=use_amp and device.type == "cuda") if device.type == "cuda" else nullcontext()
        with ac_ctx:
            outputs = model(frames, parsed_labels)
            losses = compute_losses(outputs, labels, lambda_proto=lambda_proto, use_aux_losses=use_aux_losses, weights=ce_weights)
        loss_sums[0] += losses["total"]
        loss_sums[1] += losses["task_loss"]
        loss_sums[2] += losses["l_con"]
        loss_sums[3] += losses["l_dev"]

        scaler.scale(losses["total"]).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step_update(global_step)

        preds = torch.argmax(outputs["logits"], dim=-1)
        _cm_add(cm, preds, labels, num_classes)
        epoch_logits.append(outputs["logits"].detach().float().cpu())
        epoch_labels.append(labels.detach().long().cpu())

        if stage1_monitor is not None:
            stage1_monitor.update(outputs["stage1"], video_ids)
        if stage2_monitor is not None:
            stage2_monitor.update(outputs["stage2"], video_ids)

        if logger is not None and step % log_interval == 0:
            logger.log_scalar("train/loss/total_step", losses["total"].item(), global_step)
            logger.log_scalar("train/loss/task_step", losses["task_loss"].item(), global_step)
            logger.log_scalar("train/loss/l_con_step", losses["l_con"].item(), global_step)
            logger.log_scalar("train/loss/l_dev_step", losses["l_dev"].item(), global_step)
            logger.log_scalar("train/lr", _lr_from_optimizer(optimizer), global_step)
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
        "time": elapsed,
    }


def evaluate(
    model: Stage2EmotionModel,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    logger: TrainingLogger | None,
    lambda_proto: float,
    stage1_monitor: Stage1Monitor | None,
    stage2_monitor: Stage2Monitor | None,
    num_classes: int,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
    show_progress: bool = True,
    use_aux_losses: bool = True,
    ce_weights: torch.Tensor | None = None,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    loss_sums = torch.zeros(4, device=device)
    cms = {
        "full": torch.zeros((num_classes, num_classes), device=device),
        "nostage2": torch.zeros((num_classes, num_classes), device=device),
        "nostage1": torch.zeros((num_classes, num_classes), device=device),
    }
    epoch_logits: List[torch.Tensor] = []
    epoch_labels: List[torch.Tensor] = []

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
            outs = base_model.forward_variants(frames, variants=("full", "nostage2", "nostage1"), labels=parsed_labels)

            full_out = outs["full"]
            losses = compute_losses(full_out, labels, lambda_proto=lambda_proto, use_aux_losses=use_aux_losses, weights=ce_weights)
            loss_sums[0] += losses["total"]
            loss_sums[1] += losses["task_loss"]
            loss_sums[2] += losses["l_con"]
            loss_sums[3] += losses["l_dev"]

            for name in cms:
                preds = torch.argmax(outs[name]["logits"], dim=-1)
                _cm_add(cms[name], preds, labels, num_classes)
            epoch_logits.append(outs["full"]["logits"].detach().float().cpu())
            epoch_labels.append(labels.detach().long().cpu())

            if stage1_monitor is not None and "stage1" in full_out:
                stage1_monitor.update(full_out["stage1"], video_ids)
            if stage2_monitor is not None and "stage2" in full_out:
                stage2_monitor.update(full_out["stage2"], video_ids)

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
    }
    metrics_full = _metrics_from_cm(cms["full"])
    metrics_nostage2 = _metrics_from_cm(cms["nostage2"])
    metrics_nostage1 = _metrics_from_cm(cms["nostage1"])
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

        logger.log_scalar("val/acc", metrics_full["acc"], epoch)
        logger.log_scalar("val/uar", metrics_full["uar"], epoch)
        logger.log_scalar("val/war", metrics_full["war"], epoch)
        logger.log_scalar("ablation/acc_full", metrics_full["acc"], epoch)
        logger.log_scalar("ablation/uar_full", metrics_full["uar"], epoch)
        logger.log_scalar("ablation/war_full", metrics_full["war"], epoch)
        logger.log_scalar("ablation/acc_nostage2", metrics_nostage2["acc"], epoch)
        logger.log_scalar("ablation/uar_nostage2", metrics_nostage2["uar"], epoch)
        logger.log_scalar("ablation/war_nostage2", metrics_nostage2["war"], epoch)
        logger.log_scalar("ablation/acc_nostage1", metrics_nostage1["acc"], epoch)
        logger.log_scalar("ablation/uar_nostage1", metrics_nostage1["uar"], epoch)
        logger.log_scalar("ablation/war_nostage1", metrics_nostage1["war"], epoch)
        logger.dump_text(
            f"Epoch {epoch} val acc {metrics_full['acc']:.4f} uar {metrics_full['uar']:.4f} "
            f"nostage2 acc {metrics_nostage2['acc']:.4f} nostage1 acc {metrics_nostage1['acc']:.4f}"
        )

    return {
        "full": metrics_full,
        "nostage2": metrics_nostage2,
        "nostage1": metrics_nostage1,
        "losses": avg_losses,
    }
