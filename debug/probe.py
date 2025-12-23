"""Probe visualizations and monitoring for Stage-1/Stage-2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from logger import TrainingLogger, ensure_dir


def _to_cpu(x: torch.Tensor) -> torch.Tensor:
    return x.detach().cpu()


def _plot_heatmap(mat: np.ndarray, title: str, xlabel: str, ylabel: str, xticks=None, yticks=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", origin="lower")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(range(len(xticks)))
        ax.set_xticklabels(xticks, rotation=45, ha="right")
    if yticks is not None:
        ax.set_yticks(range(len(yticks)))
        ax.set_yticklabels(yticks)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def _plot_lines(xs: np.ndarray, ys: Dict[str, np.ndarray], title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, y in ys.items():
        ax.plot(xs, y, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


class Stage1Monitor:
    """Aggregate Stage-1 stats and produce probe heatmaps."""

    def __init__(
        self,
        part_names: Sequence[str],
        output_dir: str | Path,
        logger: TrainingLogger,
        probe_ids: Optional[Sequence[str]] = None,
        max_probe: int = 4,
    ) -> None:
        self.part_names = list(part_names)
        self.k = len(self.part_names)
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.probe_ids = {str(x).zfill(5) for x in (probe_ids or [])}
        self.max_probe = max_probe

        self.present_sum = torch.zeros(self.k)
        self.area_sum = torch.zeros(self.k)
        self.area_sq = torch.zeros(self.k)
        self.area_min = torch.full((self.k,), float("inf"))
        self.area_max = torch.full((self.k,), float("-inf"))
        self.count = 0
        self.pattern_counts: Dict[str, int] = {}
        self.probe_data: Dict[str, Dict[str, np.ndarray]] = {}

    def update(self, stage1_out: Dict, video_ids: Iterable[str]) -> None:
        present = _to_cpu(stage1_out["present"]).bool()  # (B,T,K)
        area = _to_cpu(stage1_out["area"])  # (B,T,K)
        b, t, k = present.shape
        self.present_sum += present.sum(dim=(0, 1))
        self.area_sum += area.sum(dim=(0, 1))
        self.area_sq += (area ** 2).sum(dim=(0, 1))
        self.area_min = torch.minimum(self.area_min, area.view(-1, k).min(dim=0).values)
        self.area_max = torch.maximum(self.area_max, area.view(-1, k).max(dim=0).values)
        self.count += b * t

        for bi, vid in enumerate(video_ids):
            vid_str = str(vid).zfill(5)
            for ti in range(t):
                pattern = "".join("1" if present[bi, ti, pi] else "0" for pi in range(k))
                self.pattern_counts[pattern] = self.pattern_counts.get(pattern, 0) + 1
            if vid_str in self.probe_ids and len(self.probe_data) < self.max_probe and vid_str not in self.probe_data:
                self.probe_data[vid_str] = {
                    "present": present[bi].numpy(),
                    "area": area[bi].numpy(),
                }

    def summarize(self, epoch: int, split: str) -> None:
        denom = max(self.count, 1)
        present_rate = (self.present_sum / float(denom)).tolist()
        area_mean = self.area_sum / float(denom)
        area_var = self.area_sq / float(denom) - area_mean ** 2
        area_std = torch.sqrt(torch.clamp(area_var, min=0.0))

        for idx, name in enumerate(self.part_names):
            self.logger.log_scalar(f"{split}/stage1/present_rate/{name}", present_rate[idx], epoch)
            self.logger.log_scalar(f"{split}/stage1/area_mean/{name}", area_mean[idx].item(), epoch)
            self.logger.log_scalar(f"{split}/stage1/area_std/{name}", area_std[idx].item(), epoch)
            self.logger.log_scalar(f"{split}/stage1/area_min/{name}", self.area_min[idx].item(), epoch)
            self.logger.log_scalar(f"{split}/stage1/area_max/{name}", self.area_max[idx].item(), epoch)

        patterns_path = self.output_dir / split / f"missing_patterns_epoch{epoch}.json"
        patterns_path.parent.mkdir(parents=True, exist_ok=True)
        with open(patterns_path, "w", encoding="utf-8") as f:
            json.dump(self.pattern_counts, f, indent=2)

        # Probe visualizations
        for vid, data in self.probe_data.items():
            present = data["present"]  # (T,K)
            area = data["area"]
            fig_p = _plot_heatmap(
                present.T.astype(float),
                title=f"{split} present {vid}",
                xlabel="time",
                ylabel="part",
                yticks=self.part_names,
            )
            fig_a = _plot_heatmap(
                area.T,
                title=f"{split} area {vid}",
                xlabel="time",
                ylabel="part",
                yticks=self.part_names,
            )
            save_dir = ensure_dir(self.output_dir / split / f"epoch_{epoch}" / "stage1")
            fig_p.savefig(save_dir / f"{vid}_present_heatmap.png", bbox_inches="tight")
            fig_a.savefig(save_dir / f"{vid}_area_heatmap.png", bbox_inches="tight")
            self.logger.log_figure(f"{split}/stage1/present_heatmap/{vid}", fig_p, epoch)
            self.logger.log_figure(f"{split}/stage1/area_heatmap/{vid}", fig_a, epoch)
            plt.close(fig_p)
            plt.close(fig_a)


class Stage2Monitor:
    """Aggregate Stage-2 behavior and probe plots."""

    def __init__(
        self,
        part_names: Sequence[str],
        num_prototypes: int,
        output_dir: str | Path,
        logger: TrainingLogger,
        probe_ids: Optional[Sequence[str]] = None,
        max_probe: int = 4,
    ) -> None:
        self.part_names = list(part_names)
        self.num_prototypes = num_prototypes
        self.k = len(self.part_names)
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.probe_ids = {str(x).zfill(5) for x in (probe_ids or [])}
        self.max_probe = max_probe

        self.entropy_sum = torch.zeros(self.k)
        self.maxprob_sum = torch.zeros(self.k)
        self.count = torch.zeros(self.k)
        self.top1_values: List[List[torch.Tensor]] = [[] for _ in range(self.k)]

        self.reconalign_sum = torch.zeros(self.k)
        self.reconalign_sumsq = torch.zeros(self.k)
        self.reconalign_min = torch.full((self.k,), float("inf"))
        self.reconalign_max = torch.full((self.k,), float("-inf"))
        self.reconalign_count = torch.zeros(self.k)

        self.probe_data: Dict[str, Dict[str, np.ndarray]] = {}

    def update(self, stage2_out: Dict, video_ids: Iterable[str]) -> None:
        part_outputs = stage2_out["part_outputs"]
        for idx, name in enumerate(self.part_names):
            po = part_outputs[name]
            probs = _to_cpu(po["attention_probs"])  # (B,T,M)
            mask = _to_cpu(po["valid_mask"]).bool()  # (B,T)
            entropy = -(probs * (probs.clamp_min(1e-8).log())).sum(dim=-1)  # (B,T)
            maxprob = probs.max(dim=-1).values
            valid_entropy = entropy[mask]
            valid_maxprob = maxprob[mask]
            self.entropy_sum[idx] += valid_entropy.sum()
            self.maxprob_sum[idx] += valid_maxprob.sum()
            self.count[idx] += mask.sum()

            top1 = _to_cpu(po["top1_idx"])
            if mask.any():
                self.top1_values[idx].append(top1[mask])

            raw_dev = _to_cpu(po["raw_deviation"])
            recon_dev = _to_cpu(po["recon_deviation"])
            if mask.any():
                cos = F.cosine_similarity(raw_dev[mask], recon_dev[mask], dim=-1)
                self.reconalign_sum[idx] += cos.sum()
                self.reconalign_sumsq[idx] += (cos ** 2).sum()
                self.reconalign_min[idx] = torch.minimum(self.reconalign_min[idx], cos.min())
                self.reconalign_max[idx] = torch.maximum(self.reconalign_max[idx], cos.max())
                self.reconalign_count[idx] += cos.numel()

        # Probe capture: first sample of batch only (to limit memory)
        fused = _to_cpu(stage2_out["fused_output"])
        present = _to_cpu(stage2_out["present"]).bool()
        for bi, vid in enumerate(video_ids):
            vid_str = str(vid).zfill(5)
            if vid_str in self.probe_ids and len(self.probe_data) < self.max_probe and vid_str not in self.probe_data:
                per_part = {}
                for idx, name in enumerate(self.part_names):
                    po = part_outputs[name]
                    per_part[name] = {
                        "attention_probs": _to_cpu(po["attention_probs"][bi]).numpy(),
                        "top1_idx": _to_cpu(po["top1_idx"][bi]).numpy(),
                        "raw_deviation": _to_cpu(po["raw_deviation"][bi]).numpy(),
                        "recon_deviation": _to_cpu(po["recon_deviation"][bi]).numpy(),
                        "anchor_deviation": _to_cpu(po["anchor_deviation"][bi]).numpy(),
                        "prototypes": _to_cpu(po["prototypes"]).numpy(),
                        "anchor_top1": _to_cpu(po["anchor_top1_idx"][bi]).numpy(),
                        "valid_mask": _to_cpu(po["valid_mask"][bi]).numpy(),
                    }
                per_part["fused_output"] = fused[bi].numpy()
                per_part["present"] = present[bi].numpy()
                self.probe_data[vid_str] = per_part

    def summarize(self, epoch: int, split: str) -> None:
        for idx, name in enumerate(self.part_names):
            denom = max(int(self.count[idx].item()), 1)
            entropy_mean = (self.entropy_sum[idx] / denom).item()
            maxprob_mean = (self.maxprob_sum[idx] / denom).item()
            self.logger.log_scalar(f"{split}/stage2/attn_entropy/{name}", entropy_mean, epoch)
            self.logger.log_scalar(f"{split}/stage2/attn_maxprob/{name}", maxprob_mean, epoch)

            recon_count = max(int(self.reconalign_count[idx].item()), 1)
            recon_mean = (self.reconalign_sum[idx] / recon_count).item()
            recon_var = (self.reconalign_sumsq[idx] / recon_count) - (recon_mean ** 2)
            recon_std = float(np.sqrt(max(recon_var, 0.0)))
            min_val = self.reconalign_min[idx].item()
            max_val = self.reconalign_max[idx].item()
            if not np.isfinite(min_val):
                min_val = 0.0
            if not np.isfinite(max_val):
                max_val = 0.0
            self.logger.log_scalar(f"{split}/stage2/reconalign_mean/{name}", recon_mean, epoch)
            self.logger.log_scalar(f"{split}/stage2/reconalign_std/{name}", recon_std, epoch)
            self.logger.log_scalar(f"{split}/stage2/reconalign_min/{name}", min_val, epoch)
            self.logger.log_scalar(f"{split}/stage2/reconalign_max/{name}", max_val, epoch)

            if self.top1_values[idx]:
                vals = torch.cat(self.top1_values[idx], dim=0)
                self.logger.log_histogram(f"{split}/stage2/top1_idx_hist/{name}", vals, epoch)

        # Probe visualizations
        for vid, data in self.probe_data.items():
            fused = data["fused_output"]  # (T, K*C)
            present = data["present"]  # (T,K)
            t = fused.shape[0]
            xs = np.arange(t)

            save_dir = ensure_dir(self.output_dir / split / f"epoch_{epoch}" / "stage2")

            for idx, name in enumerate(self.part_names):
                part = data[name]
                probs = part["attention_probs"]  # (T,M)
                top1 = part["top1_idx"]
                valid = part["valid_mask"].astype(bool)
                recon = part["recon_deviation"]
                raw_dev = part["raw_deviation"]
                anchor_dev = part["anchor_deviation"]
                prototypes = part["prototypes"]
                anchor_top1 = int(part["anchor_top1"][0])

                # Attention heatmap and top1 curve
                fig_attn = _plot_heatmap(
                    probs,
                    title=f"{vid} attn {name}",
                    xlabel="time",
                    ylabel="prototype",
                )
                fig_top1 = _plot_lines(
                    xs,
                    {"top1": top1},
                    title=f"{vid} top1 idx {name}",
                    xlabel="time",
                    ylabel="prototype id",
                )

                fig_attn.savefig(save_dir / f"{vid}_{name}_attn_heatmap.png", bbox_inches="tight")
                fig_top1.savefig(save_dir / f"{vid}_{name}_top1_curve.png", bbox_inches="tight")
                self.logger.log_figure(f"{split}/stage2/attn_heatmap/{vid}_{name}", fig_attn, epoch)
                self.logger.log_figure(f"{split}/stage2/top1_curve/{vid}_{name}", fig_top1, epoch)
                plt.close(fig_attn)
                plt.close(fig_top1)

                # Recon alignment over time
                if valid.any():
                    cos = np.zeros_like(valid, dtype=float)
                    cos[valid] = F.cosine_similarity(
                        torch.tensor(raw_dev[valid]), torch.tensor(recon[valid]), dim=-1
                    ).numpy()
                    fig_cos = _plot_lines(
                        xs,
                        {"cos(raw,recon)": cos},
                        title=f"{vid} recon align {name}",
                        xlabel="time",
                        ylabel="cos",
                    )
                    fig_cos.savefig(save_dir / f"{vid}_{name}_reconalign.png", bbox_inches="tight")
                    self.logger.log_figure(f"{split}/stage2/reconalign/{vid}_{name}", fig_cos, epoch)
                    plt.close(fig_cos)

                # L_dev components for probe
                top1_idx = top1
                proto_pos = prototypes[top1_idx]
                anchor_proto = prototypes[anchor_top1]
                anchor_proto_exp = np.repeat(anchor_proto[None, :], t, axis=0)
                raw_dist = np.linalg.norm(raw_dev - anchor_dev, ord=1, axis=-1)
                proto_dist = np.linalg.norm(proto_pos - anchor_proto_exp, ord=1, axis=-1)
                fig_dev = _plot_lines(
                    xs,
                    {"raw_dist": raw_dist, "proto_dist": proto_dist},
                    title=f"{vid} dev raw vs proto {name}",
                    xlabel="time",
                    ylabel="L1 distance",
                )
                fig_dev.savefig(save_dir / f"{vid}_{name}_dev_raw_vs_proto.png", bbox_inches="tight")
                self.logger.log_figure(f"{split}/stage2/dev_raw_vs_proto/{vid}_{name}", fig_dev, epoch)
                plt.close(fig_dev)

                # fused_output energy per part
                c = fused.shape[1] // self.k
                part_vec = fused[:, idx * c : (idx + 1) * c]
                energy = np.linalg.norm(part_vec, axis=-1)
                fig_energy = _plot_lines(
                    xs,
                    {f"{name}_energy": energy},
                    title=f"{vid} fused energy {name}",
                    xlabel="time",
                    ylabel="L2",
                )
                fig_energy.savefig(save_dir / f"{vid}_{name}_fused_energy.png", bbox_inches="tight")
                self.logger.log_figure(f"{split}/stage2/fused_energy/{vid}_{name}", fig_energy, epoch)
                plt.close(fig_energy)
