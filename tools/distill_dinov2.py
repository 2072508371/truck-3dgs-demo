"""Fuse posed DINOv2 image descriptors onto a trained GraphDeCo Gaussian PLY.

The result is a per-Gaussian PCA color visualization, not a CUS-GS neural
feature checkpoint. CUS-GS CUDA later rasterizes these fused values from any
camera pose.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.ndimage import minimum_filter
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
from torchvision.transforms import v2


WORK = Path(__file__).resolve().parents[2]
MODELS = {
    "lego-3000": ("datasets/nerf_synthetic_lego_filtered", "assets/lego_3000_3dgs.ply"),
    "lego-10000": ("datasets/nerf_synthetic_lego_filtered", "assets/demos/lego-10000.ply"),
    "lego-10000-clean": ("datasets/nerf_synthetic_lego_filtered", "assets/demos/lego-10000-clean.ply"),
    "chair-3000": ("datasets/nerf_synthetic_chair", "assets/demos/chair-3000.ply"),
    "chair-3000-clean": ("datasets/nerf_synthetic_chair", "assets/demos/chair-3000-clean.ply"),
    "drums-3000": ("datasets/nerf_synthetic_drums", "assets/demos/drums-3000.ply"),
    "drums-3000-clean": ("datasets/nerf_synthetic_drums", "assets/demos/drums-3000-clean.ply"),
}
HUB = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
IMAGE_SIZE = 448
VIEWS = 32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", choices=MODELS, nargs="?", default="lego-10000-clean")
    args = parser.parse_args()
    data_path, model_path = MODELS[args.model_id]
    data = WORK / data_path
    model_path = WORK / "truck_project_page" / model_path
    output = WORK / "truck_project_page/assets/features" / f"{args.model_id}-dinov2.npz"
    vertex = PlyData.read(str(model_path))["vertex"].data
    xyz = np.column_stack([vertex[key] for key in ("x", "y", "z")]).astype(np.float32)
    n = len(xyz)
    sums = np.zeros((n, 384), dtype=np.float32)
    counts = np.zeros(n, dtype=np.uint16)
    scene = json.loads((data / "transforms_train.json").read_text())
    frames = scene["frames"]
    chosen = np.unique(np.linspace(0, len(frames) - 1, min(VIEWS, len(frames)), dtype=int))
    transform = v2.Compose([
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    source = str(HUB) if HUB.exists() else "facebookresearch/dinov2"
    model = torch.hub.load(source, "dinov2_vits14", source="local" if HUB.exists() else "github", pretrained=True).eval().cuda()
    fx_cache = {}
    with torch.inference_mode():
        for number, frame_id in enumerate(chosen, 1):
            frame = frames[int(frame_id)]
            image = Image.open(data / (frame["file_path"].removeprefix("./") + ".png")).convert("RGBA")
            width, height = image.size
            rgba = np.asarray(image)
            alpha = rgba[:, :, 3]
            rgb = (rgba[:, :, :3].astype(np.float32) * (alpha[:, :, None] / 255.0)).astype(np.uint8)
            image = Image.fromarray(rgb)
            features = model.forward_features(transform(image).unsqueeze(0).cuda())["x_norm_patchtokens"]
            side = int(features.shape[1] ** 0.5)
            feature_map = features.transpose(1, 2).reshape(1, 384, side, side)

            c2w = np.asarray(frame["transform_matrix"], dtype=np.float32)
            camera_xyz = (xyz - c2w[:3, 3]) @ c2w[:3, :3]
            depth = -camera_xyz[:, 2]
            fx = fx_cache.setdefault(width, width / (2 * np.tan(scene["camera_angle_x"] / 2)))
            fy = fx
            safe_depth = np.maximum(depth, 1e-4)
            u = fx * camera_xyz[:, 0] / safe_depth + width / 2
            v = height / 2 - fy * camera_xyz[:, 1] / safe_depth
            valid = (depth > 0.1) & (u >= 1) & (u < width - 1) & (v >= 1) & (v < height - 1)
            indices = np.flatnonzero(valid)
            px = np.clip(np.rint(u[indices]).astype(np.int32), 0, width - 1)
            py = np.clip(np.rint(v[indices]).astype(np.int32), 0, height - 1)
            indices = indices[alpha[py, px] > 127]
            if not len(indices):
                continue
            # A coarse nearest-center depth buffer rejects points behind the
            # visible surface before descriptors are fused across views.
            gw, gh = width // 2, height // 2
            gx = np.clip((u[indices] * gw / width).astype(np.int32), 0, gw - 1)
            gy = np.clip((v[indices] * gh / height).astype(np.int32), 0, gh - 1)
            nearest = np.full(gw * gh, np.inf, dtype=np.float32)
            flat = gy * gw + gx
            np.minimum.at(nearest, flat, depth[indices])
            near_image = minimum_filter(nearest.reshape(gh, gw), size=3)
            visible = depth[indices] <= near_image[gy, gx] + 0.06 + 0.012 * depth[indices]
            indices = indices[visible]
            if not len(indices):
                continue
            grid = np.column_stack((u[indices] / width * 2 - 1, v[indices] / height * 2 - 1)).astype(np.float32)
            sample = F.grid_sample(feature_map, torch.from_numpy(grid).cuda().view(1, -1, 1, 2), align_corners=False)
            sampled = sample.squeeze(0).squeeze(-1).T.float().cpu().numpy()
            sums[indices] += sampled
            counts[indices] += 1
            print(f"view {number:02d}/{len(chosen)}: {len(indices):6d} visible, {np.count_nonzero(counts):6d}/{n} covered", flush=True)

    covered = counts > 0
    if covered.mean() < 0.5:
        raise RuntimeError(f"Only {covered.mean():.1%} of Gaussians received features; check camera projection.")
    descriptors = sums[covered] / counts[covered, None]
    fit = descriptors[np.random.default_rng(42).choice(len(descriptors), min(30000, len(descriptors)), replace=False)]
    pca = PCA(n_components=3, svd_solver="randomized", random_state=42).fit(fit)
    projected = np.zeros((n, 3), dtype=np.float32)
    projected[covered] = pca.transform(descriptors)
    low, high = np.percentile(projected[covered], [2, 98], axis=0)
    colors = np.clip((projected - low) / np.maximum(high - low, 1e-6), 0, 1)
    # Fill unseen Gaussians with their nearest observed neighbor. This avoids
    # introducing arbitrary black holes when orbiting to less-covered views.
    if (~covered).any():
        tree = cKDTree(xyz[covered])
        _, closest = tree.query(xyz[~covered], workers=-1)
        colors[~covered] = colors[covered][closest]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, colors=(colors * 255).astype(np.uint8), counts=counts)
    print(f"saved {output}: {n} Gaussians, {covered.mean():.1%} directly covered; PCA variance {pca.explained_variance_ratio_.sum():.1%}", flush=True)


if __name__ == "__main__":
    main()
