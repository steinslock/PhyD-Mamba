"""DFEW dataset loader returning clip tensors and metadata."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


def _pad_video_id(vid: str | int) -> str:
    return str(vid).zfill(5)


def _load_csv(csv_path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            video_id, label = row[0], int(row[1])
            rows.append({"video_id": _pad_video_id(video_id), "label": label - 1})
    return rows


class DFEWClips(Dataset):
    """Dataset for DFEW clip_224x224_16f directory."""

    def __init__(
        self,
        root: str | Path,
        csv_path: str | Path,
        num_frames: int = 16,
        image_size: int = 224,
        train: bool = True,
        random_sample: bool = True,
        color_jitter: float = 0.4,
        probe_ids: Optional[Sequence[str]] = None,
        load_cached_labels: bool = False,
        label_cache_dir: Optional[str | Path] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.frames_root = self.root / "Clip" / "clip_224x224_16f"
        self.samples = _load_csv(Path(csv_path))
        self.num_frames = num_frames
        self.image_size = image_size
        self.train = train
        self.random_sample = random_sample
        self.color_jitter = color_jitter
        self.probe_ids = set(probe_ids or [])
        self.load_cached_labels = load_cached_labels
        self.label_cache_dir = Path(label_cache_dir) if label_cache_dir is not None else None
        self.color_jitter_tf = (
            transforms.ColorJitter(
                brightness=color_jitter, contrast=color_jitter, saturation=color_jitter, hue=0.1
            )
            if self.train and color_jitter > 0
            else None
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_indices(self, n: int) -> List[int]:
        if n <= 0:
            return [0] * self.num_frames
        indices: List[int] = []
        for i in range(self.num_frames):
            base = int(n * i / self.num_frames)
            if self.train and self.random_sample:
                base += int(random.random() * max(1, self.num_frames))
            idx = min(max(base, 0), n - 1)
            indices.append(idx)
        return indices

    def _transform(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB")
        img = img.resize((self.image_size, self.image_size))
        if self.train and random.random() < 0.5:
            img = TF.hflip(img)
        if self.train and self.color_jitter_tf is not None:
            img = self.color_jitter_tf(img)
        return TF.to_tensor(img)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str, torch.Tensor]:
        sample = self.samples[index]
        video_id = sample["video_id"]
        label = sample["label"]
        frame_dir = self.frames_root / video_id
        frame_paths = sorted(frame_dir.glob("*.jpg"))
        if not frame_paths:
            raise FileNotFoundError(f"No frames found for {video_id} in {frame_dir}")

        idxs = self._sample_indices(len(frame_paths))
        frames: List[torch.Tensor] = []
        for idx in idxs:
            img = Image.open(frame_paths[idx])
            frames.append(self._transform(img))

        frames_tensor = torch.stack(frames, dim=0)  # (T,3,H,W)

        cached_labels = torch.empty(0)
        if self.load_cached_labels and self.label_cache_dir is not None:
            cache_path = self.label_cache_dir / f"{video_id}.npy"
            if cache_path.exists():
                cached_np = np.load(cache_path)
                cached_labels = torch.from_numpy(cached_np).long()  # (T,H,W)

        return frames_tensor, label, video_id, cached_labels


def dfew_split_paths(dataset_root: str | Path, fold: int) -> Tuple[Path, Path]:
    root = Path(dataset_root)
    train_csv = root / "EmoLabel_DataSplit" / "train(single-labeled)" / f"set_{fold}.csv"
    test_csv = root / "EmoLabel_DataSplit" / "test(single-labeled)" / f"set_{fold}.csv"
    return train_csv, test_csv
