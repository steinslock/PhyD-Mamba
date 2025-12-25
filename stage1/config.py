from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union


@dataclass
class PartMappingConfig:
    """Mapping from raw parser labels to target facial-part IDs."""

    part_names: List[str]
    raw_to_part: Dict[int, Union[int, Sequence[int]]]
    left_right_pairs: List[Tuple[int, int]] = field(default_factory=list)

    def num_parts(self) -> int:
        return len(self.part_names)

    def validate(self) -> None:
        if not self.part_names:
            raise ValueError("part_names cannot be empty.")
        max_pid = self.num_parts() - 1
        for raw, pid_or_list in self.raw_to_part.items():
            part_ids = (pid_or_list,) if isinstance(pid_or_list, int) else pid_or_list
            for pid in part_ids:
                if pid < 0 or pid > max_pid:
                    raise ValueError(
                        f"raw label {raw} maps to invalid part id {pid}; expected 0..{max_pid}."
                    )
        for left, right in self.left_right_pairs:
            if left < 0 or right < 0 or left > max_pid or right > max_pid:
                raise ValueError(
                    f"left/right pair ({left},{right}) outside valid part id range 0..{max_pid}."
                )


@dataclass
class Stage1Config:
    """Configuration for Stage-1 part feature extraction."""

    patch_size: int = 14
    dinov2_model_name: str = "dinov2_vitb14"
    dinov2_provider: str = "auto"  # "auto", "torchhub", or "transformers"
    device: Optional[str] = None
    use_amp: bool = False
    min_area_ratio: float = 0.01
    small_part_area_ratio: float = 0.002
    small_part_names: Sequence[str] = ("left_eye", "right_eye", "left_brow", "right_brow")
    eps: float = 1e-6
    normalize_to_imagenet: bool = True
    use_cls_as_global: bool = True
    output_global_feats: bool = False
    include_optional_parts: bool = False
    output_patch_weights: bool = False
    part_mapping: PartMappingConfig = None  # type: ignore

    def __post_init__(self) -> None:
        if self.part_mapping is None:
            self.part_mapping = default_part_mapping(include_optional=self.include_optional_parts)
        self.part_mapping.validate()
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if self.min_area_ratio < 0 or self.small_part_area_ratio < 0:
            raise ValueError("area ratios must be non-negative.")

    def area_threshold_for_part(self, part_id: int) -> float:
        name = self.part_mapping.part_names[part_id]
        if name in self.small_part_names:
            return self.small_part_area_ratio
        return self.min_area_ratio


def default_part_mapping(include_optional: bool = False) -> PartMappingConfig:
    """Default mapping focused on six parts using FaceX-Zoo labels.

    FaceX-Zoo parsing labels (from face_parsing_extract.py):
        0: background
        1: face/skin
        2: right brow
        3: left brow,
        4: right eye
        5: left eye
        6: nose,
        7: upper lip
        8: inner mouth
        9: lower lip,
        10: hair
    We merge 7/8/9 into a single mouth part and add a full-face part.
    """

    part_names = [
        "left_brow",   # part 0
        "right_brow",  # part 1
        "left_eye",    # part 2
        "right_eye",   # part 3
        "mouth",       # part 4
        "full_face",   # part 5
    ]
    raw_to_part = {
        1: [5],       # face/skin
        3: [0, 5],    # left brow
        2: [1, 5],    # right brow
        5: [2, 5],    # left eye
        4: [3, 5],    # right eye
        6: [5],       # nose
        7: [4, 5],    # upper lip
        8: [4, 5],    # inner mouth
        9: [4, 5],    # lower lip
    }
    left_right_pairs = [(0, 1), (2, 3)]

    if include_optional:
        # Append optional parts (nose, skin, hair, background)
        base_len = len(part_names)
        part_names.extend(["nose", "skin", "hair", "background"])
        optional_map = {
            6: base_len + 0,   # nose
            1: base_len + 1,   # skin/face
            10: base_len + 2,  # hair
            0: base_len + 3,   # background
        }
        for raw_id, part_id in optional_map.items():
            if raw_id in raw_to_part:
                existing = raw_to_part[raw_id]
                if isinstance(existing, int):
                    raw_to_part[raw_id] = [existing, part_id]
                else:
                    raw_to_part[raw_id] = list(existing) + [part_id]
            else:
                raw_to_part[raw_id] = [part_id]

    return PartMappingConfig(
        part_names=part_names, raw_to_part=raw_to_part, left_right_pairs=left_right_pairs
    )
