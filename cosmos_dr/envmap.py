"""HDRI discovery and conversion to the ``env_ldr / env_log / env_nrm`` tensors the forward model expects."""

import os
from typing import Dict, Optional

import torch

from . import runtime

HDRI_EXTENSIONS = (".hdr", ".exr")
BUNDLED_PREFIX = "bundled/"
MODELS_PREFIX = "models/"
INPUT_PREFIX = "input/"


def _scan(directory: str, prefix: str, out: Dict[str, str]) -> None:
    if not os.path.isdir(directory):
        return
    for name in sorted(os.listdir(directory)):
        if name.lower().endswith(HDRI_EXTENSIONS):
            out[prefix + name] = os.path.join(directory, name)


def list_hdris() -> Dict[str, str]:
    """``{dropdown label: absolute path}`` for the bundled examples and user folders."""
    found: Dict[str, str] = {}
    try:
        _scan(os.path.join(runtime.find_upstream_repo(), "asset", "examples", "hdri_examples"), BUNDLED_PREFIX, found)
    except RuntimeError:
        pass  # upstream not installed yet; the loader node will explain when it is run
    try:
        import folder_paths

        _scan(os.path.join(runtime.checkpoint_root(), "hdri"), MODELS_PREFIX, found)
        _scan(os.path.join(folder_paths.get_input_directory(), "hdri"), INPUT_PREFIX, found)
    except Exception:
        pass
    return found


def resolve_hdri(label: str, custom_path: str = "") -> str:
    custom_path = (custom_path or "").strip().strip('"')
    if custom_path:
        if not os.path.isfile(custom_path):
            raise FileNotFoundError(f"custom_path does not exist: {custom_path}")
        return custom_path
    path = list_hdris().get(label)
    if path is None:
        raise FileNotFoundError(f"HDRI '{label}' not found")
    return path


def compute_envmap_tensors(
    spec: dict,
    height: int,
    width: int,
    num_frames: int,
    device: torch.device,
    rotate_total_frames: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Run upstream ``process_environment_map`` and return CPU tensors ``[1, 3, T, H, W]``.

    ``rotate_total_frames``: when the light rotates over a clip split in several chunks, the rotation
    is computed over the whole clip (``T_total`` frames) so it stays continuous across chunk borders;
    the returned tensors then cover ``T_total`` frames and the caller slices them per chunk.
    """
    with runtime.upstream_context():
        # Needs nvdiffrast at import time; keep the import lazy so the pack loads without it.
        from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.utils_env_proj import process_environment_map

        total = rotate_total_frames if (spec["rotate_light"] and rotate_total_frames) else num_frames
        result = process_environment_map(
            spec["path"],
            resolution=(height, width),
            num_frames=total,
            fixed_pose=True,
            rotate_envlight=spec["rotate_light"],
            env_format=["proj"],
            env_strength=spec["strength"],
            env_flip=spec["flip"],
            env_rot=180.0 + spec["rotation_deg"],  # upstream: flip=True + rot=180 == original orientation
            device=device,
        )
    out = {}
    for key in ("env_ldr", "env_log"):
        # (T, H, W, 3) in [0, 1] -> (1, 3, T, H, W) in [-1, 1]
        out[key] = (result[key].unsqueeze(0).permute(0, 4, 1, 2, 3) * 2 - 1).cpu()
    return out


def env_normals(height: int, width: int, device: torch.device, like: torch.Tensor) -> torch.Tensor:
    """Ray-direction map (``env_nrm``), broadcast to ``like``'s shape."""
    with runtime.upstream_context():
        from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.rendering_utils import envmap_vec

        vec = envmap_vec([height, width], device=device)  # [H, W, 3]
    return vec.unsqueeze(0).unsqueeze(0).permute(0, 4, 1, 2, 3).expand_as(like)
