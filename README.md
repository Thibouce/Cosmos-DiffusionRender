# ComfyUI – Cosmos DiffusionRenderer

ComfyUI nodes for NVIDIA's [Cosmos-Transfer1-DiffusionRenderer](https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer):

* **Inverse rendering** – image / video → G-buffers (basecolor, normal, depth, roughness, metallic).
* **Forward rendering** – G-buffers + an HDRI (or random light) → relit image / video.

The pack wraps the official `cosmos_predict1` inference code (same pipeline, same tensors), so results match the reference scripts. It does **not** reimplement the model inside ComfyUI's native Cosmos support.

> **Status:** the tensor plumbing and node wiring are covered by CPU tests (`pytest`, fake pipeline). The pack has **not** been run against the real 7B weights yet; please report anything that differs from the reference scripts.

## Requirements

* Linux, NVIDIA GPU (≥ 16 GB VRAM with offloading, ≥ 48 GB comfortable; ~27 GB peak per model), CUDA 12+
* ~70 GB disk (two 7B models + tokenizer)
* The upstream code needs **Python 3.10**, `transformer-engine==1.12.0` and (for HDRI lighting) `nvdiffrast`. They must be installed in the Python environment ComfyUI runs with.
* Hugging Face access to the NVIDIA checkpoints (`huggingface-cli login`)

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/thibouce/cosmos-diffusionrender
cd Cosmos-DiffusionRender
python install.py --extras --download   # run with ComfyUI's python
```

`install.py` clones the upstream repo into `./cosmos-transfer1-diffusion-renderer/` (or set `COSMOS_DR_REPO` to an existing clone), installs `requirements.txt`, and optionally builds transformer-engine / nvdiffrast (`--extras`) and downloads the weights (`--download`). Restart ComfyUI afterwards.

### Checkpoints

Expected layout (override the root with `COSMOS_DR_CHECKPOINTS`):

```
ComfyUI/models/diffusion_renderer/
├── Diffusion_Renderer_Inverse_Cosmos_7B/model.pt
├── Diffusion_Renderer_Forward_Cosmos_7B/model.pt
├── Cosmos-Tokenize1-CV8x8x8-720p/
└── hdri/            # optional: your own .hdr / .exr
```

## Nodes (category *Cosmos DiffusionRenderer*)

| Node | Role |
|---|---|
| **Load DiffusionRenderer Model** | Picks `inverse` or `forward`; offload switches; `keep_loaded`. Weights load lazily on first run. |
| **DiffusionRenderer Inverse (G-buffers)** | `IMAGE` → 5 `IMAGE` outputs. Toggle the passes you need (disabled passes output black frames). |
| **DiffusionRenderer Load Envmap** | HDRI dropdown (bundled examples, `models/diffusion_renderer/hdri`, `input/hdri`) or `custom_path`; strength, rotation, light spin over the clip. |
| **DiffusionRenderer Forward (Relight)** | 5 G-buffers + optional envmap → relit `IMAGE`. No envmap = random lighting driven by `seed`. |

Start from `example_workflows/image_relight.json` (load an image, edit the `LoadImage` filename).

### Notes

* **Resolution**: `height`/`width` must be multiples of 16 (default 704×1280, the model's native size). `resize_mode=stretch` matches the upstream scripts; `crop` keeps the aspect ratio and center-crops.
* **Video**: any `IMAGE` batch works. It is processed in consecutive windows of `chunk_frames` (default 57, must be 8k+1; 1 for a single image). A rotating light stays continuous across windows. There is no overlap/blending between windows, so expect small seams on long clips.
* **VRAM**: `keep_loaded=false` (default) frees the first 7B model before the second loads. On smaller GPUs enable `offload_diffusion_transformer` and `offload_tokenizer`.
* **Static scene, moving light**: set `static_frame_index` on the forward node to reuse one G-buffer frame for the whole clip, and enable `rotate_light` on the envmap node.
* The model runs under the upstream code path, not ComfyUI's model manager, so ComfyUI cannot auto-evict it; use `keep_loaded` and the offload flags.

## Tests

```bash
pip install pytest torch
pytest
```

## Licensing

Upstream code is Apache-2.0; the model weights are under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). Check both before redistributing.
