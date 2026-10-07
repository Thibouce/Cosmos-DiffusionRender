"""Tensor plumbing between ComfyUI IMAGE batches and the DiffusionRenderer pipeline.

Pure torch, no dependency on the upstream repository, so it can be unit-tested on CPU.

Conventions
-----------
* ComfyUI ``IMAGE``: float32 ``[T, H, W, C]`` in ``[0, 1]``.
* Pipeline input: float ``[1, 3, T, H, W]`` in ``[-1, 1]``.
* Pipeline output: uint8 ``[T, H, W, 3]``.
"""

import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

GBUFFER_INDEX_MAPPING = {
    "basecolor": 0,
    "metallic": 1,
    "roughness": 2,
    "normal": 3,
    "depth": 4,
}
GBUFFER_ORDER = ["basecolor", "normal", "depth", "roughness", "metallic"]

# The Cosmos tokenizer compresses time by 8 (plus the first frame), so a chunk
# must have 8k + 1 frames; space is compressed by 8 and the DiT patchifies by 2.
TEMPORAL_FACTOR = 8
SPATIAL_MULTIPLE = 16


def round_up_chunk_length(num_frames: int) -> int:
    """Smallest valid chunk length (8k + 1) that holds ``num_frames`` frames."""
    num_frames = max(int(num_frames), 1)
    return ((num_frames - 1 + TEMPORAL_FACTOR - 1) // TEMPORAL_FACTOR) * TEMPORAL_FACTOR + 1


def effective_chunk_length(total_frames: int, chunk_frames: int) -> int:
    """Chunk length actually used: ``chunk_frames``, shrunk for short inputs.

    A single image gives 1, a 10-frame clip gives 17, a 200-frame clip gives ``chunk_frames``.
    """
    if (chunk_frames - 1) % TEMPORAL_FACTOR != 0:
        raise ValueError(f"chunk_frames must be 8k+1 (1, 9, 17, ..., 57), got {chunk_frames}")
    return min(chunk_frames, round_up_chunk_length(total_frames))


def chunk_ranges(total_frames: int, chunk_length: int) -> List[Tuple[int, int]]:
    """Non-overlapping ``[start, end)`` ranges covering ``total_frames``."""
    return [(s, min(s + chunk_length, total_frames)) for s in range(0, total_frames, chunk_length)]


def validate_resolution(height: int, width: int) -> None:
    if height % SPATIAL_MULTIPLE or width % SPATIAL_MULTIPLE:
        raise ValueError(f"height and width must be multiples of {SPATIAL_MULTIPLE}, got {height}x{width}")


def pad_frames(x: torch.Tensor, length: int, dim: int = 2) -> torch.Tensor:
    """Pad along ``dim`` to ``length`` by repeating the last frame (as the upstream dataset does)."""
    n = x.shape[dim]
    if n >= length:
        return x.narrow(dim, 0, length)
    last = x.narrow(dim, n - 1, 1)
    reps = [1] * x.ndim
    reps[dim] = length - n
    return torch.cat([x, last.repeat(*reps)], dim=dim)


def comfy_to_bcthw(images: torch.Tensor, height: int, width: int, resize_mode: str = "stretch") -> torch.Tensor:
    """ComfyUI IMAGE ``[T, H, W, C]`` in [0, 1] -> ``[1, 3, T, height, width]`` in [-1, 1].

    resize_mode:
      * ``stretch``: resize straight to (height, width), like the upstream default
        (``resize_resolution == resolution`` followed by a no-op center crop).
      * ``crop``: keep the aspect ratio, scale to cover, then center-crop.
    """
    if images.ndim != 4:
        raise ValueError(f"expected IMAGE batch [T, H, W, C], got shape {tuple(images.shape)}")
    x = images.detach().float()
    if x.shape[-1] == 1:
        x = x.repeat(1, 1, 1, 3)
    x = x[..., :3].permute(0, 3, 1, 2).contiguous()  # [T, 3, H, W]

    h, w = x.shape[-2:]
    if (h, w) != (height, width):
        if resize_mode == "stretch":
            x = F.interpolate(x, size=(height, width), mode="bilinear", antialias=True, align_corners=False)
        elif resize_mode == "crop":
            scale = max(height / h, width / w)
            rh, rw = max(height, math.ceil(h * scale)), max(width, math.ceil(w * scale))
            x = F.interpolate(x, size=(rh, rw), mode="bilinear", antialias=True, align_corners=False)
            top, left = (rh - height) // 2, (rw - width) // 2
            x = x[..., top : top + height, left : left + width]
        else:
            raise ValueError(f"unknown resize_mode {resize_mode!r}")

    x = x.clamp(0, 1) * 2.0 - 1.0
    return x.permute(1, 0, 2, 3).unsqueeze(0).contiguous()  # [1, 3, T, H, W]


def frames_to_comfy(frames: np.ndarray) -> torch.Tensor:
    """Pipeline output uint8 ``[T, H, W, 3]`` -> ComfyUI IMAGE float ``[T, H, W, 3]`` in [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(frames)).float() / 255.0


def build_data_batch(
    video_tensors: Dict[str, torch.Tensor],
    height: int,
    width: int,
    fps: int = 24,
    context_index: int = 0,
) -> Dict[str, object]:
    """Assemble the dict the upstream pipeline expects, mirroring ``VideoFramesDataset`` +
    ``dict_collation_fn`` with batch size 1.

    ``video_tensors`` maps names (``rgb``, ``basecolor``, ...) to ``[1, 3, T, H, W]`` tensors.
    """
    num_frames = next(iter(video_tensors.values())).shape[2]
    text_mask = torch.zeros(1, 512)
    text_mask[:, 0] = 1
    batch: Dict[str, object] = dict(video_tensors)
    batch.update(
        {
            "is_preprocessed": True,
            "num_frames": torch.tensor([float(num_frames)]),
            "image_size": torch.tensor([[height, width]]),
            "fps": torch.tensor([float(fps)]),
            "padding_mask": torch.zeros(1, 1, height, width),
            "t5_text_embeddings": torch.zeros(1, 512, 1024),
            "t5_text_mask": text_mask,
            "context_index": torch.full((1, 1), int(context_index), dtype=torch.long),
        }
    )
    return batch
