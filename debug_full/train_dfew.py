"""Train DFEW with Stage3 SATM + classification head."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import warnings
import types
import numpy as np
import torch.nn.functional as F
import subprocess

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import amp
import torch.cuda.amp as cuda_amp
from timm.scheduler.cosine_lr import CosineLRScheduler

# Make stage1/stage2 importable
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
for p in (PROJECT_ROOT, THIS_DIR):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from stage1.config import Stage1Config  # noqa: E402
from engine import build_dataloaders, evaluate, seed_everything, train_one_epoch  # noqa: E402
from logger import TrainingLogger, ensure_dir  # noqa: E402
from modeling import Stage3EmotionModel  # noqa: E402
from probe import Stage1Monitor, Stage2Monitor  # noqa: E402
from datasets import dfew_split_paths  # noqa: E402
from engine import _unwrap  # noqa: E402

# Suppress benign dinov2/xformers and PyTorch warnings.
warnings.filterwarnings(
    "ignore",
    message="xFormers .*SwiGLU",
    category=UserWarning,
    module=r"dinov2.layers.swiglu_ffn",
)
warnings.filterwarnings(
    "ignore",
    message="xFormers .*Attention",
    category=UserWarning,
    module=r"dinov2.layers.attention",
)
warnings.filterwarnings(
    "ignore",
    message="xFormers .*Block",
    category=UserWarning,
    module=r"dinov2.layers.block",
)
warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.",
    category=UserWarning,
    module=r"torch.functional",
)
warnings.filterwarnings(
    "ignore",
    message=r"`torch.cuda.amp.custom_fwd\(args\.\.\.\)` is deprecated\. Please use `torch\.amp.custom_fwd",
    category=FutureWarning,
    module=r"mamba_ssm",
)
warnings.filterwarnings(
    "ignore",
    message=r"`torch.cuda.amp.custom_bwd\(args\.\.\.\)` is deprecated\. Please use `torch\.amp.custom_bwd",
    category=FutureWarning,
    module=r"mamba_ssm",
)
# broader suppression for custom_fwd/custom_bwd deprecation across mamba_ssm submodules
warnings.filterwarnings(
    "ignore",
    message=r"`torch.cuda.amp.custom_fwd\(args\.\.\.\)` is deprecated",
    category=FutureWarning,
    module=r"mamba_ssm",
)
warnings.filterwarnings(
    "ignore",
    message=r"`torch.cuda.amp.custom_bwd\(args\.\.\.\)` is deprecated",
    category=FutureWarning,
    module=r"mamba_ssm",
)
warnings.filterwarnings(
    "ignore",
    message=r"`torch.cuda.amp.custom_fwd\(args\.\.\.\)` is deprecated",
    category=FutureWarning,
    module=r"mamba_ssm.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"`torch.cuda.amp.custom_bwd\(args\.\.\.\)` is deprecated",
    category=FutureWarning,
    module=r"mamba_ssm.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cuda\.amp\.custom_fwd",
    category=FutureWarning,
    module=r"mamba_ssm.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cuda\.amp\.custom_bwd",
    category=FutureWarning,
    module=r"mamba_ssm.*",
)
try:
    from torch.serialization import SourceChangeWarning

    warnings.filterwarnings("ignore", category=SourceChangeWarning)
except Exception:
    pass

# Patch deprecated torch.cuda.amp custom_fwd/custom_bwd to new torch.amp versions to avoid spam warnings from mamba_ssm.
try:
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DFEW training with Stage3 SATM head.")
    parser.add_argument("--dataset_root", type=str, default="/data/home/cqm/Project/Dataset/DFEW")
    parser.add_argument("--fold", type=int, default=1, help="DFEW fold id (1-5).")
    parser.add_argument("--output_root", type=str, default="/data/home/cqm/Project/Code/Ours/debug_full/outputs/dfew")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)

#-----------------------------Loss 权重--------------------------------------#
    parser.add_argument("--lambda_proto", type=float, default=None, help="Legacy weight for L_con + L_dev (unused by default).")
    parser.add_argument("--lambda_con", type=float, default=0.02, help="Target Weight for Stage2 L_con.")
    parser.add_argument("--lambda_dev", type=float, default=0.05, help="Target Weight for Stage2 L_dev.")
    parser.add_argument("--lambda_trs", type=float, default=0.01, help="Target Weight for Stage3 TRS loss.")

#---------------------------warmup 相关参数------------------------------------#
    parser.add_argument("--warmup_lr", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--lambda_dev_delay_epochs", type=float, default=0.0, help="Epochs after LR warmup to start L_dev warmup.在LR预热完毕后延迟多少个epoch启动。")
    parser.add_argument("--lambda_con_delay_epochs", type=float, default=2.0, help="Epochs after LR warmup to start L_con warmup.")
    parser.add_argument("--lambda_trs_delay_epochs", type=float, default=6.0, help="Epochs after LR warmup to start L_trs warmup.")
    parser.add_argument("--lambda_dev_ramp_epochs", type=float, default=3.0, help="Ramp length for L_dev warmup in epochs.一共花费多少个epoch达到最大。")
    parser.add_argument("--lambda_con_ramp_epochs", type=float, default=4.0, help="Ramp length for L_con warmup in epochs.")
    parser.add_argument("--lambda_trs_ramp_epochs", type=float, default=10.0, help="Ramp length for L_trs warmup in epochs.")

    parser.add_argument("--compute_trs", action="store_true", default=True, help="Enable Stage3 TRS loss/metrics.")
    parser.add_argument("--no_trs", dest="compute_trs", action="store_false", help="Disable Stage3 TRS computation.")
    parser.add_argument("--stage3_log_interval", type=int, default=50, help="Steps between Stage3 TB logging.")
    parser.add_argument("--stage3_zero_tol", type=float, default=1e-6, help="Tolerance for zero-masked outputs.")
    parser.add_argument("--assert_stage3_zero", action="store_true", default=False, help="Assert Stage3 masked outputs stay near zero.")
    parser.add_argument("--stage3_pca_batches", type=int, default=3, help="Val batches to sample for PCA logging.")
    parser.add_argument(
        "--pool",
        type=str,
        default="attn",
        choices=["mean", "attn"],
        help="Pooling head: attn(default part-attn), mean(legacy mean pooling).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="use_amp")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=10)

#----------------早停相关参数--------------------#
    parser.add_argument("--early_stop", action="store_true", default=True, help="Enable early stopping on val UAR.")
    parser.add_argument("--early_stop_min_epochs", type=int, default=20, help="Enable early stopping after N epochs.")
    parser.add_argument("--early_stop_patience", type=int, default=10, help="Epochs without improvement to stop.")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.001, help="Min UAR improvement to reset patience.")
    
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--probe_ids", type=str, default="00001,00002,00003")
    parser.add_argument("--gpu_ids", type=str, default="0,1", help="comma separated GPU ids")
    parser.add_argument("--dist_backend", type=str, default="nccl")
    parser.add_argument("--master_port", type=str, default="29501")
    parser.add_argument("--use_cached_labels", action="store_true", default=True)
    parser.add_argument(
        "--label_cache_dir",
        type=str,
        default="/data/home/cqm/Project/Dataset/preprocess_DFEW/labels",
    )
    parser.add_argument("--progress_bar", action="store_true", default=True)
    parser.add_argument("--launch_tensorboard", action="store_true", default=True)
    parser.add_argument("--tb_port", type=int, default=6007)
    parser.add_argument("--overfit_n", type=int, default=0, help="if >0, train on fixed subset of this size")
    return parser.parse_args()


def patch_parser_with_video_logging(parser) -> None:
    """Monkey-patch FaceXZooParserWrapper.parse to include video id in warnings."""
    if parser is None or getattr(parser, "_video_logging_patched", False):
        return
    original_device = parser.device

    def patched(frames: torch.Tensor) -> torch.Tensor:
        if frames.dim() != 5:
            raise ValueError(f"frames must be (B,T,3,H,W); got {frames.shape}")
        b, t, c, h, w = frames.shape
        frames_np = frames.detach().cpu().numpy()
        frames_np = np.clip(frames_np, 0, None)
        if frames_np.max() > 1.5:
            frames_np = frames_np / 255.0
        frames_np = (frames_np * 255.0).astype(np.uint8)

        labels_out = torch.zeros((b, t, h, w), dtype=torch.long)
        vid_list = getattr(parser, "_current_video_ids", None)
        for bi in range(b):
            vid_info = vid_list[bi] if vid_list is not None and bi < len(vid_list) else None
            for ti in range(t):
                img = frames_np[bi, ti].transpose(1, 2, 0)  # HWC RGB
                img_bgr = img[:, :, ::-1]  # BGR
                dets = parser.det_handler.inference_on_image(img_bgr)
                if dets.shape[0] == 0:
                    if vid_info is None:
                        parser.logger.warning("No face detected for sample (b=%d,t=%d); labels set to background.", bi, ti)
                    else:
                        parser.logger.warning(
                            "No face detected for sample (b=%d,t=%d,vid=%s); labels set to background.",
                            bi,
                            ti,
                            vid_info,
                        )
                    continue
                lms = parser.align_handler.inference_on_image(img_bgr, dets[0])
                lms = torch.from_numpy(lms[[104, 105, 54, 84, 90]]).float().to(original_device)
                with torch.no_grad():
                    faces = parser.parse_handler.inference_on_image(1, img_bgr, lms.unsqueeze(0))
                    seg_logits = faces["seg"]["logits"]
                    seg_probs = torch.softmax(seg_logits, dim=1)
                    seg_labels = torch.argmax(seg_probs, dim=1)
                seg_labels = seg_labels.float()
                if seg_labels.shape[-2:] != (h, w):
                    seg_labels = F.interpolate(seg_labels.unsqueeze(1), size=(h, w), mode="nearest").squeeze(1)
                labels_out[bi, ti] = seg_labels.long().cpu()
        return labels_out

    parser.parse = patched  # type: ignore[assignment]
    parser._video_logging_patched = True


def main_worker(local_rank: int, gpu_ids: list[int], args: argparse.Namespace) -> None:
    world_size = len(gpu_ids)
    distributed = world_size > 1
    device_id = gpu_ids[local_rank] if gpu_ids else None
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() and device_id is not None else "cpu")
    if torch.cuda.is_available() and device_id is not None:
        torch.cuda.set_device(device_id)

    if distributed:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", args.master_port)
        dist.init_process_group(backend=args.dist_backend, world_size=world_size, rank=local_rank)

    seed_everything(args.seed + local_rank)

    probe_ids = [vid.strip() for vid in args.probe_ids.split(",") if vid.strip()]

    train_csv, val_csv = dfew_split_paths(args.dataset_root, args.fold)
    run_name = f"dfew_fold{args.fold}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = ensure_dir(Path(args.output_root) / run_name)
    tb_dir = ensure_dir(run_dir / "tb")
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    probe_dir = ensure_dir(run_dir / "probes")

    logger = TrainingLogger(log_dir=tb_dir, text_log=run_dir / "train.log") if local_rank == 0 else None
    tb_proc = None
    if local_rank == 0 and args.launch_tensorboard:
        try:
            tb_proc = subprocess.Popen(
                ["tensorboard", "--logdir", str(tb_dir), "--port", str(args.tb_port), "--load_fast", "false"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if logger is not None:
                logger.dump_text(f"TensorBoard launched at port {args.tb_port}")
        except Exception as exc:  # noqa: BLE001
            if logger is not None:
                logger.dump_text(f"Failed to launch TensorBoard: {exc}")
    if local_rank == 0:
        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)
        if logger is not None:
            logger.writer.add_text("hparams/lambda_dev", str(args.lambda_dev))
            logger.writer.add_text("hparams/lambda_con", str(args.lambda_con))
            logger.writer.add_text("hparams/lambda_trs", str(args.lambda_trs))
            logger.writer.add_text("hparams/lambda_dev_delay_epochs", str(args.lambda_dev_delay_epochs))
            logger.writer.add_text("hparams/lambda_con_delay_epochs", str(args.lambda_con_delay_epochs))
            logger.writer.add_text("hparams/lambda_trs_delay_epochs", str(args.lambda_trs_delay_epochs))
            logger.writer.add_text("hparams/lambda_dev_ramp_epochs", str(args.lambda_dev_ramp_epochs))
            logger.writer.add_text("hparams/lambda_con_ramp_epochs", str(args.lambda_con_ramp_epochs))
            logger.writer.add_text("hparams/lambda_trs_ramp_epochs", str(args.lambda_trs_ramp_epochs))
            logger.writer.add_text("hparams/compute_trs", str(args.compute_trs))
            logger.writer.add_text("hparams/stage3_log_interval", str(args.stage3_log_interval))
            logger.writer.add_text("hparams/early_stop", str(args.early_stop))
            logger.writer.add_text("hparams/early_stop_min_epochs", str(args.early_stop_min_epochs))
            logger.writer.add_text("hparams/early_stop_patience", str(args.early_stop_patience))
            logger.writer.add_text("hparams/early_stop_min_delta", str(args.early_stop_min_delta))
            logger.dump_text(
                f"[hparams] lambda_con={args.lambda_con} lambda_dev={args.lambda_dev} "
                f"lambda_trs={args.lambda_trs} "
                f"dev_delay={args.lambda_dev_delay_epochs} con_delay={args.lambda_con_delay_epochs} "
                f"trs_delay={args.lambda_trs_delay_epochs} "
                f"dev_ramp={args.lambda_dev_ramp_epochs} con_ramp={args.lambda_con_ramp_epochs} "
                f"trs_ramp={args.lambda_trs_ramp_epochs} compute_trs={args.compute_trs} "
                f"early_stop={args.early_stop} min_epochs={args.early_stop_min_epochs} "
                f"patience={args.early_stop_patience} "
                f"min_delta={args.early_stop_min_delta}"
            )

    stage1_cfg = Stage1Config(device=str(device), output_global_feats=True)
    model = Stage3EmotionModel(pool=args.pool, stage1_config=stage1_cfg, compute_trs_default=args.compute_trs)
    model.to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device_id] if device_id is not None else None,
            output_device=device_id,
            find_unused_parameters=True,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    ce_counts = torch.tensor([1956, 1515, 2135, 1738, 1175, 116, 721], dtype=torch.float32, device=device)
    ce_weights = ce_counts.sum() / (7 * ce_counts)

    train_loader, val_loader = build_dataloaders(
        dataset_root=args.dataset_root,
        train_csv=train_csv,
        val_csv=val_csv,
        num_frames=args.num_frames,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_sample=True,
        color_jitter=0.0,
        distributed=distributed,
        rank=local_rank,
        world_size=world_size,
        load_cached_labels=args.use_cached_labels,
        label_cache_dir=args.label_cache_dir,
        overfit_n=args.overfit_n,
    )

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.epochs,
        lr_min=args.min_lr,
        warmup_lr_init=args.warmup_lr,
        warmup_t=args.warmup_epochs,
        cycle_limit=1,
        t_in_epochs=True,
    )
    try:
        scaler = amp.GradScaler(enabled=args.use_amp and device.type == "cuda")
    except TypeError:
        scaler = cuda_amp.GradScaler(enabled=args.use_amp and device.type == "cuda")

    start_epoch = 0
    global_step = 0
    best_uar = 0.0
    early_stop_counter = 0
    if args.resume is not None and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_uar = float(checkpoint.get("best_uar", 0.0))
        global_step = int(checkpoint.get("global_step", 0))
        early_stop_counter = int(checkpoint.get("early_stop_counter", 0))
        if logger is not None:
            logger.dump_text(f"Resumed from {args.resume} at epoch {start_epoch}")

    lambda_con = args.lambda_con if args.overfit_n <= 0 else 0.0
    lambda_dev = args.lambda_dev if args.overfit_n <= 0 else 0.0
    lambda_trs = args.lambda_trs if args.overfit_n <= 0 else 0.0
    compute_trs_flag = args.compute_trs and args.overfit_n <= 0

    for epoch in range(start_epoch, args.epochs):
        if distributed and isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        train_s1 = Stage1Monitor(
            model.module.part_names if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.part_names,
            output_dir=probe_dir,
            logger=logger,
            probe_ids=probe_ids,
        ) if logger is not None else None
        train_s2 = Stage2Monitor(
            model.module.part_names if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.part_names,
            num_prototypes=(
                model.module.stage2_config.num_prototypes
                if isinstance(model, torch.nn.parallel.DistributedDataParallel)
                else model.stage2_config.num_prototypes
            ),
            output_dir=probe_dir,
            logger=logger,
            probe_ids=probe_ids,
        ) if logger is not None else None
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            logger=logger,
            lambda_con=lambda_con,
            lambda_dev=lambda_dev,
            lambda_trs=lambda_trs,
            scaler=scaler,
            use_amp=args.use_amp,
            log_interval=args.log_interval,
            stage3_log_interval=args.stage3_log_interval,
            stage1_monitor=train_s1,
            stage2_monitor=train_s2,
            num_classes=7,
            distributed=distributed,
            world_size=world_size,
            rank=local_rank,
            show_progress=args.progress_bar,
            use_aux_losses=False if args.overfit_n > 0 else True,
            ce_weights=ce_weights,
            compute_trs=compute_trs_flag,
            stage3_zero_tol=args.stage3_zero_tol,
            assert_zero_missing=args.assert_stage3_zero,
            warmup_epochs_lr=args.warmup_epochs,
            lambda_dev_delay_epochs=args.lambda_dev_delay_epochs,
            lambda_con_delay_epochs=args.lambda_con_delay_epochs,
            lambda_trs_delay_epochs=args.lambda_trs_delay_epochs,
            lambda_dev_ramp_epochs=args.lambda_dev_ramp_epochs,
            lambda_con_ramp_epochs=args.lambda_con_ramp_epochs,
            lambda_trs_ramp_epochs=args.lambda_trs_ramp_epochs,
            start_global_step=global_step,
        )
        global_step = train_metrics.get("global_step", global_step)
        if train_s1 is not None:
            train_s1.summarize(epoch, split="train")
        if train_s2 is not None:
            train_s2.summarize(epoch, split="train")

        val_s1 = Stage1Monitor(
            model.module.part_names if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.part_names,
            output_dir=probe_dir,
            logger=logger,
            probe_ids=probe_ids,
        ) if logger is not None else None
        val_s2 = Stage2Monitor(
            model.module.part_names if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.part_names,
            num_prototypes=(
                model.module.stage2_config.num_prototypes
                if isinstance(model, torch.nn.parallel.DistributedDataParallel)
                else model.stage2_config.num_prototypes
            ),
            output_dir=probe_dir,
            logger=logger,
            probe_ids=probe_ids,
        ) if logger is not None else None
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            epoch=epoch,
            logger=logger,
            lambda_con=lambda_con,
            lambda_dev=lambda_dev,
            lambda_trs=lambda_trs,
            stage1_monitor=val_s1,
            stage2_monitor=val_s2,
            num_classes=7,
            distributed=distributed,
            world_size=world_size,
            rank=local_rank,
            show_progress=args.progress_bar,
            use_aux_losses=False if args.overfit_n > 0 else True,
            ce_weights=ce_weights,
            compute_trs=compute_trs_flag,
            stage3_zero_tol=args.stage3_zero_tol,
            assert_zero_missing=args.assert_stage3_zero,
            stage3_pca_batches=args.stage3_pca_batches,
            warmup_epochs_lr=args.warmup_epochs,
            lambda_dev_delay_epochs=args.lambda_dev_delay_epochs,
            lambda_con_delay_epochs=args.lambda_con_delay_epochs,
            lambda_trs_delay_epochs=args.lambda_trs_delay_epochs,
            lambda_dev_ramp_epochs=args.lambda_dev_ramp_epochs,
            lambda_con_ramp_epochs=args.lambda_con_ramp_epochs,
            lambda_trs_ramp_epochs=args.lambda_trs_ramp_epochs,
            global_step=global_step,
        )
        if val_s1 is not None:
            val_s1.summarize(epoch, split="val")
        if val_s2 is not None:
            val_s2.summarize(epoch, split="val")

        current_uar = float(val_metrics["full"]["uar"])
        early_stop_active = args.early_stop and (epoch + 1) >= args.early_stop_min_epochs
        delta = args.early_stop_min_delta if early_stop_active else 0.0
        is_best = current_uar > (best_uar + delta)
        if is_best:
            best_uar = current_uar
            early_stop_counter = 0
        elif early_stop_active:
            early_stop_counter += 1

        if logger is not None:
            state = {
                "epoch": epoch,
                "model": model.module.state_dict() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_uar": best_uar,
                "global_step": global_step,
                "early_stop_counter": early_stop_counter,
            }
            torch.save(state, checkpoints_dir / "last.pt")
            if is_best:
                torch.save(state, checkpoints_dir / "best.pt")

            if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                torch.save(state, checkpoints_dir / f"epoch_{epoch+1}.pt")

            if args.early_stop:
                logger.log_scalar("early_stop/bad_epochs", float(early_stop_counter), epoch)
                logger.log_scalar("early_stop/best_uar", float(best_uar), epoch)

        if early_stop_active and early_stop_counter >= args.early_stop_patience:
            if logger is not None:
                logger.dump_text(
                    f"Early stopping at epoch {epoch} (best_uar={best_uar:.4f}, "
                    f"patience={args.early_stop_patience})."
                )
            if local_rank == 0:
                print(
                    f"[EARLY STOP] epoch={epoch} best_uar={best_uar:.4f} "
                    f"patience={args.early_stop_patience}"
                )
            break

    if logger is not None:
        logger.close()
    if tb_proc is not None:
        tb_proc.terminate()
    if distributed:
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    if args.lambda_con is None:
        args.lambda_con = 0.05 if args.lambda_proto is None else args.lambda_proto
    if args.lambda_dev is None:
        args.lambda_dev = 1.0 if args.lambda_proto is None else args.lambda_proto
    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    if torch.cuda.is_available() and gpu_ids:
        if len(gpu_ids) > 1:
            mp.spawn(main_worker, nprocs=len(gpu_ids), args=(gpu_ids, args))
        else:
            main_worker(0, gpu_ids, args)
    else:
        # CPU or no specified GPU
        main_worker(0, gpu_ids, args)


if __name__ == "__main__":
    main()
