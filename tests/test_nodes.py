"""Wiring test: real node code, fake ComfyUI modules and a fake upstream repo (no GPU, no weights)."""

import importlib
import os
import sys
import textwrap
import types

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAKE_PIPELINE = '''
import numpy as np
CALLS = []

class DiffusionRendererPipeline:
    def __init__(self, **kw):
        self.kw = kw
        self.num_steps = self.guidance = None

    def generate_video(self, data_batch, normalize_normal=False, seed=None):
        CALLS.append({k: (tuple(v.shape) if hasattr(v, "shape") else v) for k, v in data_batch.items()} | {"seed": seed, "nn": normalize_normal})
        T, H, W = data_batch["rgb" if "rgb" in data_batch else "basecolor"].shape[2:]
        return np.full((T, H, W, 3), 128, dtype=np.uint8)
'''

FAKE_ENV_PROJ = '''
import torch
def process_environment_map(path, resolution, num_frames, fixed_pose, rotate_envlight, env_format, env_strength, env_flip, env_rot, device):
    H, W = resolution
    return {"env_ldr": torch.rand(num_frames, H, W, 3), "env_log": torch.rand(num_frames, H, W, 3), "args": (path, rotate_envlight, env_rot)}
'''

FAKE_RENDERING = '''
import torch
def envmap_vec(res, device=None):
    return torch.zeros(res[0], res[1], 3)
'''


@pytest.fixture()
def nodes(tmp_path, monkeypatch):
    # fake upstream repo
    repo = tmp_path / "upstream"
    pkg = repo / "cosmos_predict1" / "diffusion"
    inf = pkg / "inference"
    utils = inf / "diffusion_renderer_utils"
    (pkg / "config").mkdir(parents=True)
    utils.mkdir(parents=True)
    (pkg / "config" / "diffusion_renderer_config.py").write_text("")
    for d in (repo / "cosmos_predict1", pkg, inf, utils):
        (d / "__init__.py").write_text("")
    (inf / "diffusion_renderer_pipeline.py").write_text(FAKE_PIPELINE)
    (utils / "utils_env_proj.py").write_text(FAKE_ENV_PROJ)
    (utils / "rendering_utils.py").write_text(FAKE_RENDERING)
    hdri = repo / "asset" / "examples" / "hdri_examples"
    hdri.mkdir(parents=True)
    (hdri / "sun.hdr").write_bytes(b"")

    # fake checkpoints
    ckpt = tmp_path / "ckpt"
    for name in ("Diffusion_Renderer_Inverse_Cosmos_7B", "Diffusion_Renderer_Forward_Cosmos_7B"):
        (ckpt / name).mkdir(parents=True)
        (ckpt / name / "model.pt").write_bytes(b"")
    (ckpt / "Cosmos-Tokenize1-CV8x8x8-720p").mkdir()
    monkeypatch.setenv("COSMOS_DR_REPO", str(repo))
    monkeypatch.setenv("COSMOS_DR_CHECKPOINTS", str(ckpt))

    # fake ComfyUI
    comfy = types.ModuleType("comfy")
    utils_mod = types.ModuleType("comfy.utils")
    utils_mod.ProgressBar = type("ProgressBar", (), {"__init__": lambda self, n: None, "update": lambda self, n: None})
    mm = types.ModuleType("comfy.model_management")
    mm.throw_exception_if_processing_interrupted = lambda: None
    mm.get_torch_device = lambda: torch.device("cpu")
    comfy.utils, comfy.model_management = utils_mod, mm
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)

    # load the pack as a package, the way ComfyUI does
    spec = importlib.util.spec_from_file_location("cosmos_pack", os.path.join(ROOT, "__init__.py"), submodule_search_locations=[ROOT])
    pack = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "cosmos_pack", pack)
    spec.loader.exec_module(pack)
    yield pack, importlib.import_module("cosmos_pack.cosmos_dr.runtime")
    for m in [m for m in sys.modules if m.startswith(("cosmos_predict1", "cosmos_pack"))]:
        sys.modules.pop(m, None)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(repo)])


def test_registration(nodes):
    pack, _ = nodes
    assert set(pack.NODE_CLASS_MAPPINGS) == set(pack.NODE_DISPLAY_NAME_MAPPINGS)
    for cls in pack.NODE_CLASS_MAPPINGS.values():
        cls.INPUT_TYPES()  # must not raise


def test_inverse_chunks_and_passes(nodes):
    pack, runtime = nodes
    N = pack.NODE_CLASS_MAPPINGS
    (model,) = N["CosmosDRLoadModel"]().load("inverse", False, False, True)
    image = torch.rand(20, 32, 48, 3)  # 20 frames, chunk_frames=9 -> 3 windows
    outs = N["CosmosDRInverseRender"]().run(model, image, True, True, False, False, False, 32, 48, "stretch", 9, 15, 0.0, 7, True)
    assert [o.shape for o in outs] == [(20, 32, 48, 3)] * 5
    assert outs[0].mean() == pytest.approx(128 / 255, abs=1e-3)  # basecolor ran
    assert outs[2].max() == 0  # depth disabled -> black
    pipeline = next(iter(runtime._pipelines.values()))
    pipe_mod = sys.modules["cosmos_predict1.diffusion.inference.diffusion_renderer_pipeline"]
    calls = pipe_mod.CALLS
    assert len(calls) == 3 * 2  # 3 windows x 2 enabled passes
    assert all(c["rgb"] == (1, 3, 9, 32, 48) for c in calls)  # last window padded to 9
    assert {c["context_index"] for c in calls} == {(1, 1)} and calls[0]["seed"] == 7
    assert [c["nn"] for c in calls[:2]] == [False, True]  # basecolor, then normal
    assert pipeline.num_steps == 15
    runtime.release_all()


def test_single_image_and_release(nodes):
    pack, runtime = nodes
    N = pack.NODE_CLASS_MAPPINGS
    (model,) = N["CosmosDRLoadModel"]().load("inverse", False, False, False)
    outs = N["CosmosDRInverseRender"]().run(model, torch.rand(1, 32, 48, 3), True, True, True, True, True, 32, 48, "crop", 57, 15, 0.0, 1, False)
    assert outs[0].shape == (1, 32, 48, 3)
    assert runtime._pipelines == {}  # keep_loaded False -> nothing cached
    pipe_mod = sys.modules["cosmos_predict1.diffusion.inference.diffusion_renderer_pipeline"]
    assert pipe_mod.CALLS[0]["rgb"] == (1, 3, 1, 32, 48)


def test_forward_with_and_without_envmap(nodes):
    pack, runtime = nodes
    N = pack.NODE_CLASS_MAPPINGS
    (model,) = N["CosmosDRLoadModel"]().load("forward", False, False, False)
    g = [torch.rand(10, 32, 48, 3) for _ in range(5)]
    (env,) = N["CosmosDRLoadEnvmap"]().load("bundled/sun.hdr", 1.0, 30.0, True, True)
    assert env["path"].endswith("sun.hdr")

    (out,) = N["CosmosDRForwardRender"]().run(model, *g, 32, 48, "stretch", 9, 15, 0.0, 3, -1, envmap=env)
    assert out.shape == (10, 32, 48, 3)
    pipe_mod = sys.modules["cosmos_predict1.diffusion.inference.diffusion_renderer_pipeline"]
    c = pipe_mod.CALLS
    assert len(c) == 2  # 10 frames in windows of 9
    for call in c:
        assert call["basecolor"] == (1, 3, 9, 32, 48) and call["env_ldr"] == (1, 3, 9, 32, 48)
        assert call["env_nrm"] == (1, 3, 9, 32, 48) and call["context_index"] == (1, 1)

    c.clear()
    (out,) = N["CosmosDRForwardRender"]().run(model, *g, 32, 48, "stretch", 9, 15, 0.0, 3, 4)  # random light + static frame
    assert out.shape == (10, 32, 48, 3) and "env_ldr" not in c[0]

    with pytest.raises(ValueError, match="inverse"):
        (inv,) = N["CosmosDRLoadModel"]().load("inverse", False, False, False)
        N["CosmosDRForwardRender"]().run(inv, *g, 32, 48, "stretch", 9, 15, 0.0, 3, -1)


def test_missing_checkpoint_message(nodes, monkeypatch, tmp_path):
    pack, _ = nodes
    monkeypatch.setenv("COSMOS_DR_CHECKPOINTS", str(tmp_path / "empty"))
    with pytest.raises(FileNotFoundError, match="model.pt"):
        pack.NODE_CLASS_MAPPINGS["CosmosDRLoadModel"]().load("inverse", False, False, False)
