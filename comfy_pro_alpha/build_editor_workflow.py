#!/usr/bin/env python3
"""Génère un workflow ComfyUI au format éditeur (.json à glisser dans l'interface) avec le node
ComfyProAlphaGenerator, ses widgets pré-remplis, et les nodes autour (Load Video -> Get Video Components,
aperçus alpha / comp).

  python build_editor_workflow.py --video plate.mp4 --fps 24 --picks picks.json -o workflow_editor.json

Cliquer ensuite 'Load plate' dans le node pour que le picker retrouve la plaque (si picks est vide, choisir le sujet).
"""
import argparse
import json

QUALITY = ["Best (v12, 2 LTX passes)", "Fast (v7, 1 LTX pass)"]
RENDER = ["full shot", "preview chunk"]


def build(a):
    picks = open(a.picks).read().strip() if a.picks else "{}"
    json.loads(picks)  # valide
    # ordre = ordre des inputs de define_schema() (images est un lien, le reste des widgets)
    widgets = [a.fps, a.first_frame, QUALITY[0 if a.quality == "best" else 1], RENDER[0 if a.render == "full" else 1],
               picks, a.prefix, a.format, a.ltx_model, a.alpha_lora, a.text_encoder, a.video_vae, a.audio_vae,
               a.vitmatte, a.chunk, a.overlap, a.seed, a.erode, a.dilate, a.near_px, a.stabilize_radius,
               a.sam3_threshold, not a.no_fast_attention]
    nodes = [
        {"id": 1, "type": "LoadVideo", "pos": [40, 60], "size": [320, 300], "flags": {}, "order": 0, "mode": 0,
         "inputs": [], "outputs": [{"name": "VIDEO", "type": "VIDEO", "links": [1]}],
         "properties": {"Node name for S&R": "LoadVideo"}, "widgets_values": [a.video, "image"]},
        {"id": 2, "type": "GetVideoComponents", "pos": [400, 60], "size": [240, 100], "flags": {}, "order": 1, "mode": 0,
         "inputs": [{"name": "video", "type": "VIDEO", "link": 1}],
         "outputs": [{"name": "images", "type": "IMAGE", "links": [2]},
                     {"name": "audio", "type": "AUDIO", "links": None},
                     {"name": "fps", "type": "FLOAT", "links": None}],
         "properties": {"Node name for S&R": "GetVideoComponents"}, "widgets_values": []},
        {"id": 3, "type": "ComfyProAlphaGenerator", "pos": [700, 40], "size": [440, 760], "flags": {}, "order": 2, "mode": 0,
         "inputs": [{"name": "images", "type": "IMAGE", "link": 2}],
         "outputs": [{"name": "alpha", "type": "MASK", "links": [3]},
                     {"name": "comp", "type": "IMAGE", "links": [4]}],
         "properties": {"Node name for S&R": "ComfyProAlphaGenerator"}, "widgets_values": widgets},
        {"id": 4, "type": "MaskToImage", "pos": [1200, 60], "size": [200, 30], "flags": {}, "order": 3, "mode": 0,
         "inputs": [{"name": "mask", "type": "MASK", "link": 3}],
         "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [5]}],
         "properties": {"Node name for S&R": "MaskToImage"}, "widgets_values": []},
        {"id": 5, "type": "PreviewImage", "pos": [1440, 40], "size": [300, 260], "flags": {}, "order": 4, "mode": 0,
         "title": "alpha", "inputs": [{"name": "images", "type": "IMAGE", "link": 5}], "outputs": [],
         "properties": {"Node name for S&R": "PreviewImage"}, "widgets_values": []},
        {"id": 6, "type": "PreviewImage", "pos": [1200, 340], "size": [300, 260], "flags": {}, "order": 5, "mode": 0,
         "title": "comp", "inputs": [{"name": "images", "type": "IMAGE", "link": 4}], "outputs": [],
         "properties": {"Node name for S&R": "PreviewImage"}, "widgets_values": []},
    ]
    # link: [id, origin_node, origin_slot, target_node, target_slot, type]
    links = [[1, 1, 0, 2, 0, "VIDEO"], [2, 2, 0, 3, 0, "IMAGE"], [3, 3, 0, 4, 0, "MASK"],
             [4, 3, 1, 6, 0, "IMAGE"], [5, 4, 0, 5, 0, "IMAGE"]]
    return {"last_node_id": 6, "last_link_id": 5, "nodes": nodes, "links": links, "groups": [],
            "config": {}, "extra": {}, "version": 0.4}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default="plate.mp4")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--first-frame", dest="first_frame", type=int, default=1001)
    ap.add_argument("--picks")
    ap.add_argument("--quality", choices=["best", "fast"], default="best")
    ap.add_argument("--render", choices=["full", "preview"], default="full")
    ap.add_argument("--prefix", default="comfy_pro_alpha/shot")
    ap.add_argument("--format", choices=["png16", "exr"], default="png16")
    ap.add_argument("--ltx-model", default="ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors")
    ap.add_argument("--alpha-lora", default="ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors")
    ap.add_argument("--text-encoder", default="built-in empty prompt (no text encoder needed)")
    ap.add_argument("--video-vae", default="ltx-2.5-video-vae-conv-bf16.safetensors")
    ap.add_argument("--audio-vae", default="ltx-2.5-audio-vae-bf16.safetensors")
    ap.add_argument("--vitmatte", default="auto-download")
    ap.add_argument("--chunk", type=int, default=49)
    ap.add_argument("--overlap", type=int, default=9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--erode", type=int, default=10)
    ap.add_argument("--dilate", type=int, default=15)
    ap.add_argument("--near-px", dest="near_px", type=int, default=15)
    ap.add_argument("--stabilize-radius", dest="stabilize_radius", type=int, default=2)
    ap.add_argument("--sam3-threshold", dest="sam3_threshold", type=float, default=0.3)
    ap.add_argument("--no-fast-attention", action="store_true")
    ap.add_argument("-o", "--output", default="workflow_editor.json")
    a = ap.parse_args()
    json.dump(build(a), open(a.output, "w"), indent=2)
    print(f"-> {a.output}")


if __name__ == "__main__":
    main()
