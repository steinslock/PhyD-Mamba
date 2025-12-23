import logging
import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

LOG = logging.getLogger(__name__)
_LOGGING_CONFIGURED = False


class FaceXZooParserWrapper:
    """Face parsing wrapper built on FaceX-Zoo (face_sdk).

    This implementation mirrors the logic in ``face_parsing_extract.py``:
      - load face detection, alignment, and parsing models from face_sdk
      - for each frame, detect the first face, align it, and run the parser
      - upsample parser labels to the input frame resolution
    Assumes inputs are already face crops; if multiple faces are found, only
    the first face is used.
    """

    def __init__(
        self,
        facex_root: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.facex_root = facex_root or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "face_sdk"
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_paths()
        self.det_handler, self.align_handler, self.parse_handler, self.logger = self._load_models()

    def _setup_paths(self) -> None:
        if self.facex_root not in sys.path:
            sys.path.append(self.facex_root)

    def _load_models(self):
        import logging.config
        import yaml

        # FaceX-Zoo loaders expect cwd to be facex_root for logging.conf relative path.
        orig_cwd = os.getcwd()
        os.chdir(self.facex_root)
        try:
            from core.model_loader.face_detection.FaceDetModelLoader import (
                FaceDetModelLoader,
            )
            from core.model_handler.face_detection.FaceDetModelHandler import (
                FaceDetModelHandler,
            )
            from core.model_loader.face_alignment.FaceAlignModelLoader import (
                FaceAlignModelLoader,
            )
            from core.model_handler.face_alignment.FaceAlignModelHandler import (
                FaceAlignModelHandler,
            )
            from core.model_loader.face_parsing.FaceParsingModelLoader import (
                FaceParsingModelLoader,
            )
            from core.model_handler.face_parsing.FaceParsingModelHandler import (
                FaceParsingModelHandler,
            )

            logs_dir = os.path.abspath(os.path.join(self.facex_root, os.pardir, "logs"))
            os.makedirs(logs_dir, exist_ok=True)
            self._configure_logging(logs_dir)
            logger = logging.getLogger("api")

            with open(os.path.join(self.facex_root, "config/model_conf.yaml")) as f:
                model_conf = yaml.load(f, Loader=yaml.FullLoader)
            scene = "non-mask"
            model_path = os.path.join(self.facex_root, "models")

            logger.info("Loading FaceX-Zoo face detection/alignment/parsing models...")
            det_loader = FaceDetModelLoader(
                model_path, "face_detection", model_conf[scene]["face_detection"]
            )
            det_model, det_cfg = det_loader.load_model()
            det_handler = FaceDetModelHandler(det_model, self.device, det_cfg)

            align_loader = FaceAlignModelLoader(
                model_path, "face_alignment", model_conf[scene]["face_alignment"]
            )
            align_model, align_cfg = align_loader.load_model()
            align_handler = FaceAlignModelHandler(align_model, self.device, align_cfg)

            parse_loader = FaceParsingModelLoader(
                model_path, "face_parsing", model_conf[scene]["face_parsing"]
            )
            parse_model, parse_cfg = parse_loader.load_model()
            parse_handler = FaceParsingModelHandler(parse_model, self.device, parse_cfg)
        finally:
            os.chdir(orig_cwd)

        return det_handler, align_handler, parse_handler, logger

    def _configure_logging(self, logs_dir: str) -> None:
        global _LOGGING_CONFIGURED
        if _LOGGING_CONFIGURED:
            return
        log_file = os.path.join(logs_dir, "face_sdk.log")
        fmt = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        logging.basicConfig(
            level=logging.INFO,
            format=fmt,
            datefmt=datefmt,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            ],
        )
        _LOGGING_CONFIGURED = True

    def parse(self, frames: torch.Tensor) -> torch.Tensor:
        """Run FaceX-Zoo parsing on frames shaped (B,T,3,H,W)."""
        if frames.dim() != 5:
            raise ValueError(f"frames must be (B,T,3,H,W); got {frames.shape}")
        b, t, c, h, w = frames.shape
        frames_np = frames.detach().cpu().numpy()
        frames_np = np.clip(frames_np, 0, None)
        if frames_np.max() > 1.5:
            frames_np = frames_np / 255.0
        frames_np = (frames_np * 255.0).astype(np.uint8)

        labels_out = torch.zeros((b, t, h, w), dtype=torch.long)
        for bi in range(b):
            for ti in range(t):
                img = frames_np[bi, ti].transpose(1, 2, 0)  # HWC RGB
                img_bgr = img[:, :, ::-1]  # to BGR for FaceX-Zoo

                dets = self.det_handler.inference_on_image(img_bgr)
                if dets.shape[0] == 0:
                    self.logger.warning("No face detected for sample (b=%d,t=%d); labels set to background.", bi, ti)
                    continue

                # Use first detected face
                lms = self.align_handler.inference_on_image(img_bgr, dets[0])
                lms = torch.from_numpy(lms[[104, 105, 54, 84, 90]]).float().to(self.device)

                with torch.no_grad():
                    faces = self.parse_handler.inference_on_image(1, img_bgr, lms.unsqueeze(0))
                    seg_logits = faces["seg"]["logits"]  # 1 x C x h x w
                    seg_probs = torch.softmax(seg_logits, dim=1)
                    seg_labels = torch.argmax(seg_probs, dim=1)  # 1 x h x w

                # Resize to input frame resolution if needed
                seg_labels = seg_labels.float()
                if seg_labels.shape[-2:] != (h, w):
                    seg_labels = F.interpolate(
                        seg_labels.unsqueeze(1), size=(h, w), mode="nearest"
                    ).squeeze(1)
                labels_out[bi, ti] = seg_labels.long().cpu()

        return labels_out
