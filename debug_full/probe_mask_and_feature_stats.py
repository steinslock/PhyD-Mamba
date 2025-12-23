"""Probe mask and feature statistics over a few training batches."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

import sys
# Make stage1/stage2 importable
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
for p in (PROJECT_ROOT, THIS_DIR):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from engine import build_dataloaders
from modeling import Stage2EmotionModel
from stage1.config import Stage1Config
from stage1.utils import normalize_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe mask/pooled/pred stats on a few train batches.")
    parser.add_argument("--dataset_root", type=str, default="/data/home/cqm/Project/Dataset/DFEW")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=10)
    parser.add_argument("--use_cached_labels", action="store_true", default=True)
    parser.add_argument(
        "--label_cache_dir",
        type=str,
        default="/data/home/cqm/Project/Dataset/preprocess_DFEW/labels",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_csv, _ = None, None
    from datasets import dfew_split_paths  # local import to avoid cycle

    train_csv, _ = dfew_split_paths(args.dataset_root, args.fold)
    train_loader, _ = build_dataloaders(
        dataset_root=args.dataset_root,
        train_csv=train_csv,
        val_csv=train_csv,
        num_frames=16,
        image_size=224,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_sample=True,
        color_jitter=0.4,
        distributed=False,
        load_cached_labels=args.use_cached_labels,
        label_cache_dir=args.label_cache_dir,
        overfit_n=0,
    )

    s1_cfg = Stage1Config(device=str(device), output_global_feats=True)
    model = Stage2EmotionModel(pool="mean", stage1_config=s1_cfg)
    model.to(device)
    model.eval()

    zero_count = 0
    total_count = 0
    valid_sums = []
    pooled_norms = []
    pooled_var_means = []
    pred_hist = torch.zeros(7, dtype=torch.long)
    label_hist = torch.zeros(7, dtype=torch.long)

    batches_processed = 0
    with torch.no_grad():
        for batch in train_loader:
            if batches_processed >= args.num_batches:
                break
            if len(batch) == 4:
                frames, labels, video_ids, parsed_labels = batch
            else:
                frames, labels, video_ids = batch
                parsed_labels = torch.empty(0)
            frames = frames.to(device)
            labels = labels.to(device)
            parsed_labels = parsed_labels.to(device) if parsed_labels.numel() > 0 else None
            if labels.min().item() == 1 and labels.max().item() == 7:
                labels0 = labels - 1
            else:
                labels0 = labels

            print("[FRAMES RAW] dtype:", frames.dtype)
            print("[FRAMES RAW] min/max:", float(frames.min()), float(frames.max()))
            print("[FRAMES RAW] mean/std:", float(frames.mean()), float(frames.std()))

            frames_norm = normalize_frames(frames, normalize_to_imagenet=True)
            print("[FRAMES NORM] dtype:", frames_norm.dtype)
            print("[FRAMES NORM] min/max:", float(frames_norm.min()), float(frames_norm.max()))
            print("[FRAMES NORM] mean/std:", float(frames_norm.mean()), float(frames_norm.std()))

            frames_01 = frames.float()
            max_val = frames_01.max()
            if torch.isfinite(max_val) and max_val > 1.0:
                frames_01 = frames_01 / 255.0
            frames_01 = frames_01.clamp(0.0, 1.0)
            sat0 = (frames_01 <= 1e-6).float().mean().item()
            sat1 = (frames_01 >= 1.0 - 1e-6).float().mean().item()
            print(f"[FRAMES CLAMP] sat0={sat0:.4f} sat1={sat1:.4f}")

            outputs = model.forward_full(frames, parsed_labels)
            s1 = outputs["stage1"]
            s2 = outputs["stage2"]
            fused = s2["fused_output"].float()  # (B,T,640)
            time_mask = s2["present"].any(dim=2)  # (B,T)
            # feature stats
            def stat(name: str, x: torch.Tensor) -> None:
                x = x.float()
                print(f"[STAT] {name}: shape={tuple(x.shape)}  mean={x.mean().item():.4e}  std={x.std().item():.4e}  var={x.var().item():.4e}")
            stat("stage1/part_feats", s1["part_feats"])
            stat("stage2/fused_output", fused)

            mask = time_mask.float()
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
            pooled_custom = (fused * mask.unsqueeze(-1)).sum(dim=1) / denom
            stat("head/pooled", pooled_custom)

            logits = outputs.get("logits")
            if logits is not None:
                stat("head/logits", logits)

            for part, pdict in s2["part_outputs"].items():
                p = pdict["attention_probs"].float()  # (B,T,M)
                eps = 1e-8
                ent = -(p.clamp_min(eps) * (p.clamp_min(eps).log())).sum(dim=-1)
                maxp = p.max(dim=-1).values
                ent_v = ent[time_mask].mean().item() if time_mask.any() else float("nan")
                maxp_v = maxp[time_mask].mean().item() if time_mask.any() else float("nan")
                print(f"[ATTN] {part}: entropy_mean={ent_v:.4f}  maxprob_mean={maxp_v:.4f}  (uniform entropy≈{math.log(p.shape[-1]):.4f})")

            time_mask = outputs["time_mask"]  # (B,T)
            fused_valid = time_mask.sum(dim=1)  # (B,)
            zero_count += (fused_valid == 0).sum().item()
            total_count += fused_valid.numel()
            valid_sums.append(fused_valid.float().cpu())

            pooled = outputs["pooled"]  # (B,D)
            pooled_norms.append(pooled.norm(dim=1).cpu())
            pooled_var_means.append(pooled.var(dim=0, unbiased=False).mean().cpu())

            logits = outputs["logits"]
            preds = logits.argmax(dim=1)
            if preds.min().item() == 1 and preds.max().item() == 7:
                preds0 = preds - 1
            else:
                preds0 = preds
            pred_hist += torch.bincount(preds0.cpu(), minlength=7)
            label_hist += torch.bincount(labels0.cpu(), minlength=7)

            batches_processed += 1

    if total_count == 0 or batches_processed == 0:
        print("[WARN] No batches processed.")
        return

    valid_concat = torch.cat(valid_sums, dim=0)
    pooled_norm_concat = torch.cat(pooled_norms, dim=0)
    pooled_var_mean_tensor = torch.stack(pooled_var_means)

    zero_ratio = zero_count / total_count
    print(f"[MASK] batches={batches_processed} zero_ratio={zero_ratio:.4f}")
    print(f"[MASK] valid mean={valid_concat.mean().item():.3f} min={valid_concat.min().item():.3f} max={valid_concat.max().item():.3f}")

    print(f"[POOLED] norm mean={pooled_norm_concat.mean().item():.4f} std={pooled_norm_concat.std().item():.4f}")
    print(f"[POOLED] pooled_var_mean across dims: mean={pooled_var_mean_tensor.mean().item():.6f} std={pooled_var_mean_tensor.std().item():.6f}")

    print(f"[PRED] pred_hist={pred_hist.tolist()} (sum={int(pred_hist.sum())})")
    print(f"[PRED] label_hist={label_hist.tolist()} (sum={int(label_hist.sum())})")


if __name__ == "__main__":
    main()
