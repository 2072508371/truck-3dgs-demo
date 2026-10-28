"""Pack fused DINOv2 PCA colors with PLY centers for browser GPU lookup.

Run from this directory's repository with the torchcuda311 environment after
generating assets/features/*-dinov2.npz. Each record is xyz float32 + RGBA8.
"""

from pathlib import Path
import struct

import numpy as np
from plyfile import PlyData


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "lego-3000": "assets/lego_3000_3dgs.ply",
    "lego-10000": "assets/demos/lego-10000.ply",
    "lego-10000-clean": "assets/demos/lego-10000-clean.ply",
    "chair-3000": "assets/demos/chair-3000.ply",
    "chair-3000-clean": "assets/demos/chair-3000-clean.ply",
    "drums-3000": "assets/demos/drums-3000.ply",
    "drums-3000-clean": "assets/demos/drums-3000-clean.ply",
}

for model_id, ply_relative in MODELS.items():
    vertex = PlyData.read(str(ROOT / ply_relative))["vertex"].data
    with np.load(ROOT / "assets/features" / f"{model_id}-dinov2.npz") as features:
        colors = features["colors"]
    if colors.shape != (len(vertex), 3):
        raise ValueError(f"Feature count mismatch for {model_id}")
    records = np.empty(len(vertex), dtype=np.dtype([("xyz", "<f4", 3), ("rgba", "u1", 4)]))
    for axis, key in enumerate(("x", "y", "z")):
        records["xyz"][:, axis] = vertex[key]
    records["rgba"][:, :3] = colors
    records["rgba"][:, 3] = 255
    output = ROOT / "assets/features" / f"{model_id}-dinov2.bin"
    output.write_bytes(struct.pack("<I", len(vertex)) + records.tobytes())
    print(f"{model_id}: {len(vertex)} points, {output.stat().st_size / 1048576:.2f} MiB")
