"""Locating the upstream code + checkpoints and managing the (heavy) pipeline lifecycle."""

import contextlib
import gc
import os
import sys
import threading
from typing import Dict

REPO_ENV = "COSMOS_DR_REPO"
CKPT_ENV = "COSMOS_DR_CHECKPOINTS"
UPSTREAM_DIRNAME = "cosmos-transfer1-diffusion-renderer"
TOKENIZER_DIRNAME = "Cosmos-Tokenize1-CV8x8x8-720p"
MODEL_NAMES = {
    "inverse": "Diffusion_Renderer_Inverse_Cosmos_7B",
    "forward": "Diffusion_Renderer_Forward_Cosmos_7B",
}

_PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_REL = os.path.join("cosmos_predict1", "diffusion", "config", "diffusion_renderer_config.py")
_lock = threading.RLock()
_pipelines: Dict[tuple, object] = {}


def find_upstream_repo() -> str:
    """Directory of the cloned ``cosmos-transfer1-diffusion-renderer`` repository."""
    candidates = [os.environ.get(REPO_ENV), os.path.join(_PACK_DIR, UPSTREAM_DIRNAME)]
    for cand in candidates:
        if cand and os.path.isfile(os.path.join(cand, _CONFIG_REL)):
            return os.path.abspath(cand)
    raise RuntimeError(
        "Upstream cosmos-transfer1-diffusion-renderer code not found. Either run `python install.py` "
        f"in the node folder, clone it to '{os.path.join(_PACK_DIR, UPSTREAM_DIRNAME)}', "
        f"or set {REPO_ENV} to your clone."
    )


def checkpoint_root() -> str:
    override = os.environ.get(CKPT_ENV)
    if override:
        return os.path.abspath(override)
    import folder_paths  # provided by ComfyUI

    return os.path.join(folder_paths.models_dir, "diffusion_renderer")


def check_checkpoints(kind: str) -> str:
    """Return the checkpoint root after verifying the files the pipeline will open."""
    root = checkpoint_root()
    model = os.path.join(root, MODEL_NAMES[kind], "model.pt")
    tokenizer = os.path.join(root, TOKENIZER_DIRNAME)
    missing = [p for p in (model, tokenizer) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing checkpoint(s):\n  " + "\n  ".join(missing) + "\nSee the README (\"Checkpoints\") or run `python install.py --download`."
        )
    return root


@contextlib.contextmanager
def upstream_context():
    """Upstream resolves its config relative to the repo root, so run inside it."""
    repo = find_upstream_repo()
    with _lock:
        if repo not in sys.path:
            sys.path.insert(0, repo)
        previous = os.getcwd()
        os.chdir(repo)
        try:
            yield repo
        finally:
            os.chdir(previous)


def get_pipeline(kind: str, offload_network: bool, offload_tokenizer: bool, keep_loaded: bool):
    """Build (or fetch from cache) the DiffusionRendererPipeline for ``kind`` (inverse|forward)."""
    root = check_checkpoints(kind)
    key = (kind, root, offload_network, offload_tokenizer)
    with _lock:
        if key in _pipelines:
            return _pipelines[key]
        # Two 7B models rarely fit together: drop whatever else is resident first.
        release_all()
        with upstream_context():
            from cosmos_predict1.diffusion.inference.diffusion_renderer_pipeline import DiffusionRendererPipeline

            pipeline = DiffusionRendererPipeline(
                checkpoint_dir=root,
                checkpoint_name=MODEL_NAMES[kind],
                offload_network=offload_network,
                offload_tokenizer=offload_tokenizer,
                offload_text_encoder_model=True,
                offload_guardrail_models=True,
            )
        if keep_loaded:
            _pipelines[key] = pipeline
        return pipeline


def release_all() -> None:
    """Drop cached pipelines and free VRAM (callers must drop their own references first)."""
    with _lock:
        _pipelines.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
