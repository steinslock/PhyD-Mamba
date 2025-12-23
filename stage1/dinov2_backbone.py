from typing import Optional, Tuple

import torch
from torch import nn


class DINOv2Backbone(nn.Module):
    """Wrapper around a frozen DINOv2-Base (ViT-B/14) backbone."""

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        provider: str = "auto",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        model: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.provider = provider
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model or self._load_model(model_name, provider)
        self.patch_size = (
            int(self.model.patch_size[0])
            if hasattr(self.model, "patch_size") and isinstance(self.model.patch_size, (tuple, list))
            else int(getattr(self.model, "patch_size", 14))
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.to(self.device, dtype=dtype)
        self.hidden_dim = self._infer_hidden_dim()

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return patch tokens (B,N,D) and CLS/global token (B,D)."""
        images = images.to(self.device)
        with torch.no_grad():
            tokens, cls_token = self._extract_tokens(images)
        return tokens, cls_token

    def _load_model(self, model_name: str, provider: str) -> nn.Module:
        last_err: Optional[Exception] = None
        providers = [provider] if provider != "auto" else ["torchhub", "transformers"]
        for prov in providers:
            if prov == "torchhub":
                try:
                    return torch.hub.load("facebookresearch/dinov2", model_name)
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
            elif prov == "transformers":
                try:
                    from transformers import AutoModel

                    return AutoModel.from_pretrained(model_name)
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
            else:
                raise ValueError(f"Unknown provider: {prov}")
        msg = (
            "Failed to load DINOv2 model. Install either torch hub repo "
            "'facebookresearch/dinov2' or HuggingFace 'facebook/dinov2-base'. "
            f"Last error: {last_err}"
        )
        raise RuntimeError(msg)

    def _extract_tokens(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = None
        if hasattr(self.model, "forward_features"):
            features = self.model.forward_features(images)
        else:
            features = self.model(images)

        tokens = None
        cls_token = None

        if isinstance(features, dict):
            patch_tokens = features.get("x_norm_patchtokens")
            if patch_tokens is None:
                patch_tokens = features.get("patch_tokens")
            if patch_tokens is None and "x_norm_tokens" in features:
                raw_tokens = features["x_norm_tokens"]
                if raw_tokens.dim() == 3 and raw_tokens.size(1) > 1:
                    cls_token = raw_tokens[:, 0, :]
                    patch_tokens = raw_tokens[:, 1:, :]
                else:
                    patch_tokens = raw_tokens
            tokens = patch_tokens
            if cls_token is None:
                cls_token = features.get("x_norm_clstoken")
            if cls_token is None:
                cls_token = features.get("cls_token")
        elif hasattr(features, "last_hidden_state"):
            hidden = features.last_hidden_state
            cls_token = hidden[:, 0, :]
            tokens = hidden[:, 1:, :]
        elif isinstance(features, (list, tuple)):
            if len(features) == 2 and isinstance(features[1], torch.Tensor):
                tokens = features[1]
                cls_token = features[0]
            elif len(features) > 0 and isinstance(features[0], torch.Tensor):
                tokens = features[0]
        elif torch.is_tensor(features):
            cls_token = features

        if tokens is None:
            raise RuntimeError("Unable to extract patch tokens from DINOv2 output.")
        if cls_token is None:
            cls_token = tokens.mean(dim=1)

        return tokens, cls_token

    def _infer_hidden_dim(self) -> int:
        if hasattr(self.model, "embed_dim"):
            return int(self.model.embed_dim)
        if hasattr(self.model, "config") and hasattr(self.model.config, "hidden_size"):
            return int(self.model.config.hidden_size)
        try:
            dummy = torch.zeros((1, 3, 224, 224), device=self.device)
            tokens, _ = self._extract_tokens(dummy)
            return tokens.shape[-1]
        except Exception:
            return 768
