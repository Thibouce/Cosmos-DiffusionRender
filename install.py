#!/usr/bin/env python3
"""Set up the Cosmos DiffusionRenderer node pack.

    python install.py                 # clone upstream + pip install requirements
    python install.py --download      # ... and download the checkpoints (needs HF login, ~56 GB)
    python install.py --extras        # ... and build transformer-engine + nvdiffrast (CUDA toolchain required)

Run it with the Python that runs ComfyUI.
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM_URL = "https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer.git"
UPSTREAM_DIR = os.path.join(HERE, "cosmos-transfer1-diffusion-renderer")


def run(*cmd, cwd=None):
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=cwd)


def default_checkpoint_dir():
    # <ComfyUI>/custom_nodes/<this pack>  ->  <ComfyUI>/models/diffusion_renderer
    return os.path.abspath(os.path.join(HERE, "..", "..", "models", "diffusion_renderer"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true", help="download checkpoints with the upstream script")
    parser.add_argument("--extras", action="store_true", help="install transformer-engine and nvdiffrast")
    parser.add_argument("--checkpoint-dir", default=default_checkpoint_dir())
    args = parser.parse_args()

    if not os.path.isdir(UPSTREAM_DIR):
        run("git", "clone", "--depth", "1", UPSTREAM_URL, UPSTREAM_DIR)
    run(sys.executable, "-m", "pip", "install", "-r", os.path.join(HERE, "requirements.txt"))

    if args.extras:
        run(sys.executable, "-m", "pip", "install", "transformer-engine[pytorch]==1.12.0")
        run(sys.executable, "-m", "pip", "install", "git+https://github.com/NVlabs/nvdiffrast.git", "--no-build-isolation")

    if args.download:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        # Fetches both 7B renderers and the Cosmos tokenizer; the script imports `scripts.*`, hence PYTHONPATH.
        os.environ["PYTHONPATH"] = UPSTREAM_DIR + os.pathsep + os.environ.get("PYTHONPATH", "")
        run(sys.executable, "scripts/download_diffusion_renderer_checkpoints.py", "--checkpoint_dir", args.checkpoint_dir, cwd=UPSTREAM_DIR)

    print("\nDone. Restart ComfyUI. Checkpoints are expected in:", args.checkpoint_dir)


if __name__ == "__main__":
    main()
