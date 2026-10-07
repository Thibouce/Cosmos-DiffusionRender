"""ComfyUI nodes wrapping NVIDIA's Cosmos DiffusionRenderer (inverse + forward rendering)."""

import torch

import comfy.utils
import comfy.model_management as mm

from .cosmos_dr import envmap as envmap_utils
from .cosmos_dr import runtime
from .cosmos_dr import tensors as T

CATEGORY = "Cosmos DiffusionRenderer"
RESOLUTION_INPUTS = {
    "height": ("INT", {"default": 704, "min": 64, "max": 2048, "step": 16}),
    "width": ("INT", {"default": 1280, "min": 64, "max": 2048, "step": 16}),
    "resize_mode": (["stretch", "crop"], {"default": "stretch"}),
}
CHUNK_INPUT = (
    "INT",
    {
        "default": 57,
        "min": 1,
        "max": 57,
        "step": 8,
        "tooltip": "Frames per diffusion window (8k+1; 57 is the model's native length, 1 for a single image). "
        "Longer inputs are processed in consecutive windows.",
    },
)


def _check_handle(model, kind):
    if model["kind"] != kind:
        raise ValueError(f"This node needs the {kind} model, but a {model['kind']} model was connected.")


def _set_sampling(pipeline, steps, guidance):
    pipeline.num_steps = steps
    pipeline.guidance = guidance


class CosmosDRLoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (["inverse", "forward"], {"tooltip": "inverse: image -> G-buffers. forward: G-buffers + light -> relit image."}),
                "offload_diffusion_transformer": ("BOOLEAN", {"default": False, "tooltip": "Move the 7B transformer to CPU when not sampling. Needed on ~16-24 GB GPUs."}),
                "offload_tokenizer": ("BOOLEAN", {"default": False, "tooltip": "Keep the video tokenizer on CPU between encode/decode."}),
                "keep_loaded": ("BOOLEAN", {"default": False, "tooltip": "Keep the model in VRAM after the run. Off frees VRAM so the second 7B model can load."}),
            }
        }

    RETURN_TYPES = ("COSMOS_DR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, model_type, offload_diffusion_transformer, offload_tokenizer, keep_loaded):
        runtime.check_checkpoints(model_type)  # fail early; the weights themselves load lazily on first use
        return ({"kind": model_type, "offload_network": offload_diffusion_transformer, "offload_tokenizer": offload_tokenizer, "keep_loaded": keep_loaded},)


class CosmosDRInverseRender:
    @classmethod
    def INPUT_TYPES(cls):
        passes = {name: ("BOOLEAN", {"default": True}) for name in T.GBUFFER_ORDER}
        return {
            "required": {
                "model": ("COSMOS_DR_MODEL",),
                "image": ("IMAGE",),
                **passes,
                **RESOLUTION_INPUTS,
                "chunk_frames": CHUNK_INPUT,
                "num_steps": ("INT", {"default": 15, "min": 1, "max": 100}),
                "guidance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 1000, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "normalize_normal": ("BOOLEAN", {"default": False, "tooltip": "Renormalize the predicted normals."}),
            }
        }

    RETURN_TYPES = ("IMAGE",) * 5
    RETURN_NAMES = tuple(T.GBUFFER_ORDER)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, model, image, basecolor, normal, depth, roughness, metallic, height, width, resize_mode, chunk_frames, num_steps, guidance, seed, normalize_normal):
        _check_handle(model, "inverse")
        T.validate_resolution(height, width)
        enabled = [n for n, on in zip(T.GBUFFER_ORDER, (basecolor, normal, depth, roughness, metallic)) if on]
        if not enabled:
            raise ValueError("Enable at least one G-buffer pass.")

        total = image.shape[0]
        chunk_len = T.effective_chunk_length(total, chunk_frames)
        ranges = T.chunk_ranges(total, chunk_len)
        video = T.comfy_to_bcthw(image, height, width, resize_mode)  # [1, 3, T, H, W]

        pbar = comfy.utils.ProgressBar(len(ranges) * len(enabled))
        outputs = {n: [] for n in T.GBUFFER_ORDER}
        pipeline = runtime.get_pipeline(model["kind"], model["offload_network"], model["offload_tokenizer"], model["keep_loaded"])
        try:
            _set_sampling(pipeline, num_steps, guidance)
            with runtime.upstream_context(), torch.no_grad():
                for start, end in ranges:
                    mm.throw_exception_if_processing_interrupted()
                    rgb = T.pad_frames(video[:, :, start:end], chunk_len)
                    for name in enabled:
                        batch = T.build_data_batch({"rgb": rgb.clone()}, height, width, context_index=T.GBUFFER_INDEX_MAPPING[name])
                        frames = pipeline.generate_video(data_batch=batch, normalize_normal=(name == "normal" and normalize_normal), seed=seed)
                        outputs[name].append(T.frames_to_comfy(frames[: end - start]))
                        pbar.update(1)
        finally:
            del pipeline
            if not model["keep_loaded"]:
                runtime.release_all()

        results = []
        for name in T.GBUFFER_ORDER:
            # Disabled passes return black frames of the right shape so downstream wiring never breaks.
            results.append(torch.cat(outputs[name], dim=0) if outputs[name] else torch.zeros(total, height, width, 3))
        return tuple(results)


class CosmosDRLoadEnvmap:
    @classmethod
    def INPUT_TYPES(cls):
        labels = list(envmap_utils.list_hdris().keys()) or ["(no HDRI found)"]
        return {
            "required": {
                "hdri": (labels, {"tooltip": "Bundled examples, models/diffusion_renderer/hdri, or input/hdri (.hdr/.exr)."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "rotation_deg": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0, "tooltip": "Rotate the light around the vertical axis."}),
                "flip": ("BOOLEAN", {"default": True, "tooltip": "Upstream default (needed to load the map in its original orientation)."}),
                "rotate_light": ("BOOLEAN", {"default": False, "tooltip": "Spin the light a full turn over the clip."}),
            },
            "optional": {"custom_path": ("STRING", {"default": "", "tooltip": "Absolute path to an .hdr/.exr; overrides the dropdown."})},
        }

    RETURN_TYPES = ("COSMOS_DR_ENVMAP",)
    RETURN_NAMES = ("envmap",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    @classmethod
    def VALIDATE_INPUTS(cls, hdri, custom_path=""):
        return True  # the dropdown is rebuilt from disk; resolve (and report errors) at run time

    def load(self, hdri, strength, rotation_deg, flip, rotate_light, custom_path=""):
        path = envmap_utils.resolve_hdri(hdri, custom_path)
        return ({"path": path, "strength": strength, "rotation_deg": rotation_deg, "flip": flip, "rotate_light": rotate_light},)


class CosmosDRForwardRender:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("COSMOS_DR_MODEL",),
                **{name: ("IMAGE",) for name in T.GBUFFER_ORDER},
                **RESOLUTION_INPUTS,
                "chunk_frames": CHUNK_INPUT,
                "num_steps": ("INT", {"default": 15, "min": 1, "max": 100}),
                "guidance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 1000, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Without an envmap, the seed alone randomizes the illumination."}),
                "static_frame_index": ("INT", {"default": -1, "min": -1, "max": 100000, "tooltip": "-1 off. Otherwise repeat this G-buffer frame over the clip (static scene + moving light)."}),
            },
            "optional": {"envmap": ("COSMOS_DR_ENVMAP", {"tooltip": "Leave unconnected for random illumination."})},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("relit",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, model, basecolor, normal, depth, roughness, metallic, height, width, resize_mode, chunk_frames, num_steps, guidance, seed, static_frame_index, envmap=None):
        _check_handle(model, "forward")
        T.validate_resolution(height, width)
        gbufs = {"basecolor": basecolor, "normal": normal, "depth": depth, "roughness": roughness, "metallic": metallic}
        total = max(g.shape[0] for g in gbufs.values())
        if any(g.shape[0] not in (1, total) for g in gbufs.values()):
            raise ValueError("All G-buffer inputs must have the same number of frames.")
        if static_frame_index >= total:
            raise ValueError(f"static_frame_index {static_frame_index} is out of range for {total} frames")

        chunk_len = T.effective_chunk_length(total, chunk_frames)
        ranges = T.chunk_ranges(total, chunk_len)
        videos = {}
        for name, img in gbufs.items():
            v = T.comfy_to_bcthw(img, height, width, resize_mode)
            v = v.expand(-1, -1, total, -1, -1) if v.shape[2] == 1 else v
            if static_frame_index >= 0:
                v = v[:, :, static_frame_index : static_frame_index + 1].expand(-1, -1, total, -1, -1)
            videos[name] = v

        env = None
        if envmap is not None:
            n_chunks = len(ranges)
            env = envmap_utils.compute_envmap_tensors(envmap, height, width, chunk_len, mm.get_torch_device(), rotate_total_frames=n_chunks * chunk_len)

        pbar = comfy.utils.ProgressBar(len(ranges))
        out = []
        pipeline = runtime.get_pipeline(model["kind"], model["offload_network"], model["offload_tokenizer"], model["keep_loaded"])
        try:
            _set_sampling(pipeline, num_steps, guidance)
            with runtime.upstream_context(), torch.no_grad():
                for i, (start, end) in enumerate(ranges):
                    mm.throw_exception_if_processing_interrupted()
                    batch = T.build_data_batch({k: T.pad_frames(v[:, :, start:end], chunk_len).clone() for k, v in videos.items()}, height, width)
                    if env is not None:
                        # rotating light: tensors span the whole clip; static light: a single chunk worth
                        s = i * chunk_len if env["env_ldr"].shape[2] > chunk_len else 0
                        device = mm.get_torch_device()
                        ldr = env["env_ldr"][:, :, s : s + chunk_len].to(device)
                        batch["env_ldr"] = ldr
                        batch["env_log"] = env["env_log"][:, :, s : s + chunk_len].to(device)
                        batch["env_nrm"] = envmap_utils.env_normals(height, width, device, ldr)
                    frames = pipeline.generate_video(data_batch=batch, seed=seed)
                    out.append(T.frames_to_comfy(frames[: end - start]))
                    pbar.update(1)
        finally:
            del pipeline
            if not model["keep_loaded"]:
                runtime.release_all()
        return (torch.cat(out, dim=0),)


NODE_CLASS_MAPPINGS = {
    "CosmosDRLoadModel": CosmosDRLoadModel,
    "CosmosDRInverseRender": CosmosDRInverseRender,
    "CosmosDRLoadEnvmap": CosmosDRLoadEnvmap,
    "CosmosDRForwardRender": CosmosDRForwardRender,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CosmosDRLoadModel": "Load DiffusionRenderer Model",
    "CosmosDRInverseRender": "DiffusionRenderer Inverse (G-buffers)",
    "CosmosDRLoadEnvmap": "DiffusionRenderer Load Envmap",
    "CosmosDRForwardRender": "DiffusionRenderer Forward (Relight)",
}
