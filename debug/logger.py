"""TensorBoard-friendly logger for training and monitoring."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, Optional

from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    """Lightweight wrapper around SummaryWriter with path helpers."""

    def __init__(
        self,
        log_dir: str | Path,
        text_log: Optional[str | Path] = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.text_log_path = Path(text_log) if text_log is not None else None
        if self.text_log_path is not None:
            self.text_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_text(self, msg: str) -> None:
        if self.text_log_path is None:
            return
        with open(self.text_log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def log_scalar(self, name: str, value: float, step: int) -> None:
        self.writer.add_scalar(name, value, step)

    def log_scalars(self, prefix: str, values: Dict[str, float], step: int) -> None:
        for k, v in values.items():
            self.writer.add_scalar(f"{prefix}/{k}", v, step)

    def log_histogram(self, name: str, values, step: int, bins: str = "auto") -> None:
        self.writer.add_histogram(name, values, step, bins=bins)

    def log_image(self, name: str, img, step: int) -> None:
        self.writer.add_image(name, img, step, dataformats="CHW")

    def log_figure(self, name: str, fig, step: int) -> None:
        self.writer.add_figure(name, fig, step)

    def log_json(self, name: str, obj, filename: Optional[str] = None) -> None:
        """Persist JSON to disk for later inspection."""
        file = self.log_dir / (filename or f"{name}.json")
        file.parent.mkdir(parents=True, exist_ok=True)
        with open(file, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

    def dump_text(self, msg: str) -> None:
        self._write_text(msg)

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

