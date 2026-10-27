"""CUS-GS CUDA depth and normal rendering for bundled scene checkpoints."""

from __future__ import annotations

import io
import math
import sys
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData


PAGE_DIR = Path(__file__).resolve().parent
WORK_DIR = PAGE_DIR.parent
RASTERIZER = WORK_DIR / "CUS-GS-uni-gs-test" / "diff-gaussian-rasterization"
PLY_PATHS = {
    "lego-100": PAGE_DIR / "assets" / "demos" / "lego-100.ply",
    "lego-1000": PAGE_DIR / "assets" / "demos" / "lego-1000.ply",
    "lego-3000": PAGE_DIR / "assets" / "lego_3000_3dgs.ply",
    "lego-10000": PAGE_DIR / "assets" / "demos" / "lego-10000.ply",
    "lego-10000-clean": PAGE_DIR / "assets" / "demos" / "lego-10000-clean.ply",
    "chair-3000": PAGE_DIR / "assets" / "demos" / "chair-3000.ply",
    "drums-3000": PAGE_DIR / "assets" / "demos" / "drums-3000.ply",
}
sys.path.insert(0, str(RASTERIZER))

from diff_gaussian_rasterization import GaussianRasterizationSettings, _RasterizeGaussians


_lock = threading.Lock()
_gaussians: dict[str, dict] = {}


def _load_gaussians(model_id: str) -> dict:
    if model_id not in PLY_PATHS:
        raise ValueError("Unknown bundled model checkpoint.")
    if model_id in _gaussians:
        return _gaussians[model_id]
    if not torch.cuda.is_available():
        raise RuntimeError("CUS-GS depth and normals require the local CUDA-enabled PyTorch environment.")
    ply_path = PLY_PATHS[model_id]
    if not ply_path.is_file():
        raise RuntimeError(f"The selected model file is missing: {ply_path.name}")
    vertex = PlyData.read(str(ply_path))["vertex"].data

    def columns(prefix: str, count: int) -> np.ndarray:
        return np.stack([vertex[f"{prefix}_{i}"] for i in range(count)], axis=1).astype(np.float32)

    xyz = np.stack([vertex[key] for key in ("x", "y", "z")], axis=1).astype(np.float32)
    scales = np.exp(columns("scale", 3))
    rotations = columns("rot", 4)
    logits = np.asarray(vertex["opacity"], dtype=np.float32)
    opacities = 1.0 / (1.0 + np.exp(-logits))
    sh0 = columns("f_dc", 3)
    colors = np.maximum(sh0 * 0.28209479177387814 + 0.5, 0.0)
    device = "cuda"
    means_t = torch.as_tensor(xyz, device=device)
    scales_t = torch.as_tensor(scales, device=device)
    rotations_t = torch.as_tensor(rotations, device=device)
    q = torch.nn.functional.normalize(rotations_t, dim=1)
    qw, qx, qy, qz = q.unbind(dim=1)
    rotation_matrix = torch.stack((
        torch.stack((1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)), dim=1),
        torch.stack((2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)), dim=1),
        torch.stack((2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)), dim=1),
    ), dim=1)
    shortest_axis = scales_t.argmin(dim=1)
    gaussian_normals = rotation_matrix.gather(2, shortest_axis[:, None, None].expand(-1, 3, 1)).squeeze(2)
    gaussian_normals = torch.nn.functional.normalize(gaussian_normals, dim=1)
    gaussians = {
        "means": means_t,
        "scales": scales_t,
        "rotations": rotations_t,
        "gaussian_normals": gaussian_normals,
        "opacities": torch.as_tensor(opacities, device=device),
        "colors": torch.as_tensor(colors, device=device),
        "query": torch.zeros((len(xyz), 160), device=device),
        "empty": torch.empty(0, device=device),
    }
    _gaussians[model_id] = gaussians
    return gaussians


def _matrix(values: list[float]) -> torch.Tensor:
    if len(values) != 16 or not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Camera view matrix must contain 16 finite numbers.")
    return torch.tensor(values, dtype=torch.float32, device="cuda").reshape(4, 4)


def _edge_aware_smooth_normals(
    normals: np.ndarray, depth: np.ndarray, alpha: np.ndarray, strength: float
) -> np.ndarray:
    if strength <= 0:
        return normals

    height, width = depth.shape
    valid = (alpha > 0.12) & (depth > 0)
    depth_sigma = 0.01 + depth * (0.025 * strength)
    normal_sum = np.zeros_like(normals)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    radius = 2
    spatial_sigma = 1.35

    for dy in range(-radius, radius + 1):
        y0, y1 = max(0, -dy), min(height, height - dy)
        ny0, ny1 = y0 + dy, y1 + dy
        for dx in range(-radius, radius + 1):
            x0, x1 = max(0, -dx), min(width, width - dx)
            nx0, nx1 = x0 + dx, x1 + dx
            center_valid = valid[y0:y1, x0:x1]
            neighbor_valid = valid[ny0:ny1, nx0:nx1]
            depth_delta = depth[ny0:ny1, nx0:nx1] - depth[y0:y1, x0:x1]
            sigma = depth_sigma[y0:y1, x0:x1]
            spatial = np.exp(-(dx * dx + dy * dy) / (2.0 * spatial_sigma**2))
            weight = spatial * np.exp(-0.5 * (depth_delta / np.maximum(sigma, 1e-4)) ** 2)
            weight *= center_valid & neighbor_valid
            weight *= np.clip(alpha[ny0:ny1, nx0:nx1], 0.15, 1.0)
            normal_sum[y0:y1, x0:x1] += normals[ny0:ny1, nx0:nx1] * weight[..., None]
            weight_sum[y0:y1, x0:x1] += weight

    smoothed = normal_sum / np.maximum(weight_sum[..., None], 1e-6)
    smoothed /= np.maximum(np.linalg.norm(smoothed, axis=2, keepdims=True), 1e-6)
    return np.where(valid[..., None], smoothed, 0.0)


def _depth_normals(
    depth: torch.Tensor, alpha: torch.Tensor, view: torch.Tensor, fovx: float, fovy: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backproject a robust depth surface and fuse camera-facing normals at three scales."""
    height, width = depth.shape
    valid = (alpha > 0.12) & (depth > 0) & torch.isfinite(depth)
    depth = torch.where(valid, depth, 0.0)
    padded_depth = F.pad(depth[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    padded_valid = F.pad(valid[None, None], (1, 1, 1, 1))[0, 0]
    neighbors = torch.stack([
        padded_depth[dy:dy + height, dx:dx + width]
        for dy in range(3) for dx in range(3)
    ])
    neighbor_valid = torch.stack([
        padded_valid[dy:dy + height, dx:dx + width]
        for dy in range(3) for dx in range(3)
    ])
    filtered = torch.where(neighbor_valid, neighbors, depth).median(dim=0).values

    y, x = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    fx = width / (2.0 * math.tan(fovx / 2.0))
    fy = height / (2.0 * math.tan(fovy / 2.0))
    points = torch.stack(((x - width / 2.0) * filtered / fx,
                          (y - height / 2.0) * filtered / fy, filtered), dim=-1)
    ray = F.normalize(points, dim=-1, eps=1e-6)
    fused = torch.zeros_like(points)
    total_weight = torch.zeros_like(depth)

    for step, weight in ((1, 0.5), (2, 0.3), (4, 0.2)):
        if 2 * step >= min(width, height):
            continue
        center = (slice(step, -step), slice(step, -step))
        left = (slice(step, -step), slice(None, -2 * step))
        right = (slice(step, -step), slice(2 * step, None))
        up = (slice(None, -2 * step), slice(step, -step))
        down = (slice(2 * step, None), slice(step, -step))
        local_depth = filtered[center]
        threshold = 0.02 + local_depth * (0.035 * step)
        trusted = valid[center]
        for neighbor in (left, right, up, down):
            trusted = trusted & valid[neighbor] & ((filtered[neighbor] - local_depth).abs() < threshold)
        horizontal = points[right] - points[left]
        vertical = points[down] - points[up]
        normal = F.normalize(torch.cross(horizontal, vertical, dim=-1), dim=-1, eps=1e-6)
        normal = torch.where((normal * ray[center]).sum(dim=-1, keepdim=True) > 0, -normal, normal)
        trusted = trusted & (torch.linalg.vector_norm(torch.cross(horizontal, vertical, dim=-1), dim=-1) > 1e-8)
        fused[center] += normal * (trusted.float() * weight)[..., None]
        total_weight[center] += trusted.float() * weight

    camera_normal = F.normalize(fused, dim=-1, eps=1e-6)
    world_normal = camera_normal @ view[:3, :3].T
    return world_normal.permute(2, 0, 1), total_weight > 0


def render(payload: dict) -> bytes:
    width = max(128, min(768, int(payload.get("width", 640))))
    height = max(128, min(768, int(payload.get("height", 480))))
    fovy = float(payload.get("fovy", math.radians(50)))
    aspect = width / height
    if not math.isfinite(fovy) or not 0.1 < fovy < 2.8:
        raise ValueError("Invalid vertical field of view.")
    fovx = 2.0 * math.atan(math.tan(fovy / 2.0) * aspect)
    camera_position = payload.get("cameraPosition", [])
    if len(camera_position) != 3 or not all(math.isfinite(float(v)) for v in camera_position):
        raise ValueError("Camera position must contain three finite numbers.")
    focus_distance = max(0.01, float(payload.get("focusDistance", 4.0)))
    normal_contrast = max(0.25, min(2.0, float(payload.get("normalContrast", 1.0))))
    normal_smoothing = max(0.0, min(2.5, float(payload.get("normalSmoothing", 1.2))))
    mode = payload.get("mode")
    normal_method = payload.get("normalMethod", "depth")
    if mode not in {"depth", "normal"}:
        raise ValueError("Mode must be depth or normal.")
    if normal_method not in {"depth", "gaussian", "hybrid"}:
        raise ValueError("Normal method must be depth, gaussian, or hybrid.")

    with _lock, torch.inference_mode():
        gaussians = _load_gaussians(str(payload.get("modelId", "lego-3000")))
        # Three.js looks down local -Z with +Y up; Uni-GS uses +Z forward and +Y down.
        three_view_transposed = _matrix(payload.get("viewMatrix", []))
        axis_flip = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], device="cuda"))
        view = three_view_transposed @ axis_flip
        znear, zfar = 0.01, 100.0
        projection = torch.zeros((4, 4), dtype=torch.float32, device="cuda")
        projection[0, 0] = 1.0 / math.tan(fovx / 2.0)
        projection[1, 1] = 1.0 / math.tan(fovy / 2.0)
        projection[2, 2] = zfar / (zfar - znear)
        projection[3, 2] = -(zfar * znear) / (zfar - znear)
        projection[2, 3] = 1.0
        settings = GaussianRasterizationSettings(
            image_height=height,
            image_width=width,
            tanfovx=math.tan(fovx / 2.0),
            tanfovy=math.tan(fovy / 2.0),
            bg=torch.full((3,), 0.5 if mode == "normal" and normal_method in {"gaussian", "hybrid"} else 1.0, device="cuda"),
            scale_modifier=1.0,
            viewmatrix=view,
            projmatrix=view @ projection,
            sh_degree=0,
            campos=torch.tensor(camera_position, dtype=torch.float32, device="cuda"),
            prefiltered=False,
            debug=False,
        )
        empty = gaussians["empty"]
        means = gaussians["means"]
        colors = gaussians["colors"]
        if mode == "normal" and normal_method in {"gaussian", "hybrid"}:
            normal_vectors = gaussians["gaussian_normals"]
            view_to_camera = torch.tensor(camera_position, dtype=torch.float32, device="cuda") - means
            facing_away = (normal_vectors * view_to_camera).sum(dim=1, keepdim=True) < 0
            normal_vectors = torch.where(facing_away, -normal_vectors, normal_vectors)
            colors = (normal_vectors + 1.0) * 0.5
        rendered, _, _, _, depth, normal, alpha, _ = _RasterizeGaussians.apply(
            means,
            torch.zeros_like(means),
            empty,
            colors,
            gaussians["opacities"],
            gaussians["scales"],
            gaussians["rotations"],
            gaussians["query"],
            empty,
            empty,
            settings,
        )
        valid_t = (alpha > 0.05) & (depth > 0)
        if mode == "depth":
            depth_np = depth.cpu().numpy()
            valid = valid_t.cpu().numpy()
            if valid.any():
                low = max(0.0, focus_distance * float(payload.get("depthNear", 0.15)))
                high = max(low + 0.01, focus_distance * float(payload.get("depthFar", 1.5)))
                gray = np.clip((depth_np - low) / (high - low), 0.0, 1.0)
            else:
                gray = np.zeros((height, width), dtype=np.float32)
            rgb = np.repeat(gray[..., None], 3, axis=2)
        else:
            depth_np = depth.cpu().numpy()
            alpha_np = alpha.cpu().numpy()
            valid = (alpha_np > 0.12) & (depth_np > 0)
            depth_normal_t, trusted_t = _depth_normals(depth, alpha, view, fovx, fovy)
            trusted = trusted_t.cpu().numpy()
            if normal_method in {"gaussian", "hybrid"}:
                alpha_safe = alpha.clamp(min=1e-6).unsqueeze(0)
                bg_contribution = (1.0 - alpha).unsqueeze(0) * 0.5
                gaussian_normal_t = ((rendered - bg_contribution) / alpha_safe) * 2.0 - 1.0
                gaussian_normal_t = torch.nn.functional.normalize(gaussian_normal_t, dim=0, eps=1e-6)
                if normal_method == "hybrid":
                    normal_t = torch.nn.functional.normalize(0.85 * depth_normal_t + 0.15 * gaussian_normal_t, dim=0, eps=1e-6)
                    normal_t = torch.where(trusted_t.unsqueeze(0), normal_t, gaussian_normal_t)
                else:
                    normal_t = gaussian_normal_t
                normal_np = normal_t.permute(1, 2, 0).cpu().numpy()
            else:
                normal_np = depth_normal_t.permute(1, 2, 0).cpu().numpy()
                valid &= trusted
            smooth_alpha = alpha_np if normal_method != "depth" else np.where(trusted, alpha_np, 0.0)
            normal_np = _edge_aware_smooth_normals(normal_np, depth_np, smooth_alpha, normal_smoothing)
            rgb = np.clip((normal_np * normal_contrast + 1.0) * 0.5, 0.0, 1.0)
            rgb[~valid] = 0.0

    image = Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=1)
    return output.getvalue()
