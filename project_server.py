"""Local project API for photo-to-3DGS jobs and generated model serving."""

from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


PAGE_DIR = Path(__file__).resolve().parent
WORK_DIR = PAGE_DIR.parent
REPO_DIR = WORK_DIR.parent
PROJECTS_DIR = WORK_DIR / "gs_projects"
TRAIN_LAUNCHER = WORK_DIR / "run_gaussian_splatting.bat"
TRAIN_ITERATIONS = 30000
MIN_PHOTOS = 20
MAX_UPLOAD_BYTES = 250 * 1024 * 1024
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


def colmap_executable() -> str | None:
    configured = os.environ.get("COLMAP_EXE")
    if configured and Path(configured).is_file():
        return configured
    return shutil.which("colmap") or shutil.which("colmap.exe") or shutil.which("COLMAP.bat")


def read_project(project_id: str) -> dict | None:
    path = PROJECTS_DIR / project_id / "project.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_project(project: dict) -> None:
    folder = PROJECTS_DIR / project["id"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "project.json").write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")


def project_list() -> list[dict]:
    items = []
    builtins = [
        ("chair-3000", "Chair · 3,000 iterations", 3000, "assets/demos/chair-3000.ply", "assets/demos/chair-3000.png", "assets/demos/chair-reference.png", 100),
        ("drums-3000", "Drums · 3,000 iterations", 3000, "assets/demos/drums-3000.ply", "assets/demos/drums-3000.png", "assets/demos/drums-reference.png", 100),
        ("lego-3000", "Lego · 3,000 iterations", 3000, "assets/lego_3000_3dgs.ply", "assets/lego-render.png", "assets/lego-gt.png", 136),
        ("lego-10000", "Lego · 10,000 iterations", 10000, "assets/demos/lego-10000.ply", "assets/demos/lego-10000.png", "assets/lego-gt.png", 136),
        ("lego-10000-clean", "Lego · 10,000 · 背景优化", 10000, "assets/demos/lego-10000-clean.ply", "assets/demos/lego-10000-clean.png", "assets/demos/lego-10000-clean-reference.png", 136),
    ]
    for project_id, name, iterations, model_rel, render_rel, reference_rel, image_count in builtins:
        model_file = PAGE_DIR / model_rel
        gaussian_count = 0
        if model_file.is_file():
            with model_file.open("rb") as stream:
                header = stream.read(4096).split(b"end_header", 1)[0].decode("ascii", errors="ignore")
            count = re.search(r"element vertex (\d+)", header)
            gaussian_count = int(count.group(1)) if count else 0
        items.append({
            "id": project_id, "name": name, "status": "demo_ready", "iterations": iterations,
            "image_count": image_count, "gaussian_count": gaussian_count,
            "model_url": f"/{model_rel}", "render_url": render_rel,
            "reference_url": reference_rel, "model_size": f"{model_file.stat().st_size / 1_000_000:.1f} MB" if model_file.is_file() else "—",
            "view_label": ("TRAINED VIEW R_101" if project_id == "lego-10000-clean" else "TRAIN VIEW 035") if project_id.startswith("lego-") else "TRAIN VIEW 000", "updated_at": "Built-in scene demo", "is_builtin": True,
        })
    if PROJECTS_DIR.exists():
        for folder in PROJECTS_DIR.iterdir():
            if folder.is_dir():
                item = read_project(folder.name)
                if item:
                    items.append(item)
    return sorted(items, key=lambda item: (not item.get("is_builtin", False), item.get("updated_at", "")), reverse=False)


def run_checked(command: list[str], log_path: Path, cwd: Path | None = None) -> None:
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write("\n$ " + subprocess.list2cmdline(command) + "\n")
        log.flush()
        if command[0].lower().endswith(".bat"):
            result = subprocess.run(subprocess.list2cmdline(command), shell=True, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=False)
        else:
            result = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {Path(command[0]).name}")


def train_project(project_id: str) -> None:
    project = read_project(project_id)
    if not project:
        return
    root = PROJECTS_DIR / project_id
    source = root / "source"
    image_dir = source / "images"
    sparse_dir = source / "sparse"
    output = root / "output"
    database = source / "colmap.db"
    log_path = root / "training.log"
    colmap = colmap_executable()
    try:
        if not colmap:
            raise RuntimeError("COLMAP is not installed or not on PATH. Install COLMAP, then restart this app.")
        if not TRAIN_LAUNCHER.is_file():
            raise RuntimeError(f"Training launcher not found: {TRAIN_LAUNCHER}")
        image_dir.mkdir(parents=True, exist_ok=True)
        sparse_dir.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True)
        with LOCK:
            project["status"] = "estimating_cameras"
            project["message"] = "COLMAP is estimating camera poses…"
            project["updated_at"] = time.strftime("%Y-%m-%d %H:%M")
            write_project(project)
        run_checked([colmap, "feature_extractor", "--database_path", str(database), "--image_path", str(image_dir), "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "OPENCV"], log_path)
        run_checked([colmap, "exhaustive_matcher", "--database_path", str(database)], log_path)
        run_checked([colmap, "mapper", "--database_path", str(database), "--image_path", str(image_dir), "--output_path", str(sparse_dir)], log_path)
        if not any(path.glob("images.bin") for path in sparse_dir.glob("*/")):
            raise RuntimeError("COLMAP did not produce a registered camera model. Add more overlapping, sharp photos and retry.")
        with LOCK:
            project["status"] = "training"
            project["message"] = f"Training 3DGS for {TRAIN_ITERATIONS:,} iterations on the local GPU…"
            write_project(project)
        run_checked([str(TRAIN_LAUNCHER), "train.py", "-s", str(source), "-m", str(output), "--eval", "--iterations", str(TRAIN_ITERATIONS), "--data_device", "cuda", "--disable_viewer"], log_path, REPO_DIR)
        run_checked([str(TRAIN_LAUNCHER), "render.py", "-m", str(output), "-s", str(source), "--skip_train"], log_path, REPO_DIR)
        run_checked([str(TRAIN_LAUNCHER), "metrics.py", "-m", str(output)], log_path, REPO_DIR)
        project["status"] = "demo_ready"
        project["message"] = "Training complete. Open the generated demo from the project library."
        project["model_url"] = f"/project-files/{project_id}/output/point_cloud/iteration_{TRAIN_ITERATIONS}/point_cloud.ply"
        render_folder = output / "test" / f"ours_{TRAIN_ITERATIONS}" / "renders"
        reference_folder = output / "test" / f"ours_{TRAIN_ITERATIONS}" / "gt"
        if render_folder.is_dir():
            first_render = next(iter(sorted(render_folder.glob("*.png"))), None)
            if first_render:
                project["render_url"] = f"/project-files/{project_id}/output/test/ours_{TRAIN_ITERATIONS}/renders/{first_render.name}"
        if reference_folder.is_dir():
            first_reference = next(iter(sorted(reference_folder.glob("*.png"))), None)
            if first_reference:
                project["reference_url"] = f"/project-files/{project_id}/output/test/ours_{TRAIN_ITERATIONS}/gt/{first_reference.name}"
        results = output / "results.json"
        if results.is_file():
            raw_metrics = json.loads(results.read_text(encoding="utf-8"))
            scene_metrics = next(iter(raw_metrics.values()), {})
            project["metrics"] = next(iter(scene_metrics.values()), {})
        model_file = output / "point_cloud" / f"iteration_{TRAIN_ITERATIONS}" / "point_cloud.ply"
        if model_file.is_file():
            with model_file.open("rb") as stream:
                header = stream.read(4096).split(b"end_header", 1)[0].decode("ascii", errors="ignore")
            count = re.search(r"element vertex (\d+)", header)
            if count:
                project["gaussian_count"] = int(count.group(1))
        project["iterations"] = TRAIN_ITERATIONS
    except Exception as error:  # Surface process failures in the library instead of losing the job.
        project["status"] = "failed"
        project["message"] = str(error)
    finally:
        project["updated_at"] = time.strftime("%Y-%m-%d %H:%M")
        write_project(project)
        with LOCK:
            JOBS.pop(project_id, None)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PAGE_DIR), **kwargs)

    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def send_json(self, status: int, value: dict | list) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def body_json(self) -> dict:
        size = int(self.headers.get("Content-Length", "0"))
        if size > 1024 * 1024:
            raise ValueError("Request body is too large")
        return json.loads(self.rfile.read(size) or b"{}")

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            self.send_json(200, {
                "colmap_ready": bool(colmap_executable()),
                "colmap_path": colmap_executable(),
                "gpu": "RTX 4060 / CUDA (configured in run_gaussian_splatting.bat)",
                "min_photos": MIN_PHOTOS,
                "iterations": TRAIN_ITERATIONS,
            })
            return
        if parsed.path == "/api/geometry/status":
            try:
                import cus_geometry  # noqa: F401
                available = True
                message = "CUS-GS CUDA renderer is available."
            except Exception as error:
                available = False
                message = str(error)
            self.send_json(200, {"available": available, "message": message})
            return
        if parsed.path == "/api/projects":
            self.send_json(200, project_list())
            return
        match = re.fullmatch(r"/api/projects/([a-f0-9-]+)/status", parsed.path)
        if match:
            project = read_project(match.group(1))
            self.send_json(200 if project else 404, project or {"error": "Project not found"})
            return
        match = re.fullmatch(r"/project-files/([a-f0-9-]+)/(.*)", unquote(parsed.path))
        if match:
            base = (PROJECTS_DIR / match.group(1)).resolve()
            target = (base / match.group(2)).resolve()
            if base not in target.parents or not target.is_file():
                self.send_error(404)
                return
            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(target.stat().st_size))
            self.end_headers()
            with target.open("rb") as stream:
                shutil.copyfileobj(stream, self.wfile)
            return
        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/geometry/render":
            try:
                import cus_geometry
                payload = self.body_json()
                image = cus_geometry.render(payload)
            except (ValueError, json.JSONDecodeError) as error:
                self.send_json(400, {"error": str(error)})
                return
            except Exception as error:
                self.send_json(503, {"error": str(error)})
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(image)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(image)
            return
        if parsed.path == "/api/projects":
            try:
                payload = self.body_json()
                name = str(payload.get("name", "")).strip()[:60]
                if not name:
                    self.send_json(400, {"error": "Enter a scene name."})
                    return
                project_id = str(uuid.uuid4())
                project = {
                    "id": project_id, "name": name, "status": "collecting_photos",
                    "image_count": 0, "images": [], "created_at": time.strftime("%Y-%m-%d %H:%M"),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M"),
                }
                write_project(project)
                self.send_json(201, project)
            except (ValueError, json.JSONDecodeError) as error:
                self.send_json(400, {"error": str(error)})
            return
        match = re.fullmatch(r"/api/projects/([a-f0-9-]+)/train", parsed.path)
        if match:
            project_id = match.group(1)
            project = read_project(project_id)
            if not project:
                self.send_json(404, {"error": "Project not found"})
                return
            if len(project.get("images", [])) < MIN_PHOTOS:
                self.send_json(400, {"error": f"At least {MIN_PHOTOS} accepted photos are required."})
                return
            if not colmap_executable():
                self.send_json(424, {"error": "COLMAP is not installed. Your photos are saved locally; install COLMAP and restart this app to start reconstruction."})
                return
            if project_id in JOBS:
                self.send_json(409, {"error": "This project is already training."})
                return
            project["status"] = "queued"
            project["message"] = "Reconstruction queued."
            write_project(project)
            worker = threading.Thread(target=train_project, args=(project_id,), daemon=True)
            JOBS[project_id] = {"thread": worker}
            worker.start()
            self.send_json(202, project)
            return
        self.send_error(404)

    def do_PUT(self):
        match = re.fullmatch(r"/api/projects/([a-f0-9-]+)/images/([^/]+)", urlparse(self.path).path)
        if not match:
            self.send_error(404)
            return
        project_id, raw_name = match.groups()
        filename = Path(unquote(raw_name)).name
        if Path(filename).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            self.send_json(415, {"error": "Only JPG, PNG, and WebP photos are accepted."})
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            self.send_json(413, {"error": "Photo is empty or exceeds the 250 MB upload limit."})
            return
        project = read_project(project_id)
        if not project:
            self.send_json(404, {"error": "Project not found"})
            return
        image_dir = PROJECTS_DIR / project_id / "source" / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        safe_name = f"{len(project.get('images', [])):04d}_{filename}"
        target = image_dir / safe_name
        remaining = length
        with target.open("wb") as output:
            while remaining:
                chunk = self.rfile.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                output.write(chunk)
                remaining -= len(chunk)
        if remaining:
            target.unlink(missing_ok=True)
            self.send_json(400, {"error": "Upload ended before the full photo arrived."})
            return
        query = parse_qs(urlparse(self.path).query)
        project.setdefault("images", []).append({
            "name": safe_name, "width": int(query.get("width", [0])[0]),
            "height": int(query.get("height", [0])[0]), "bytes": length,
        })
        project["image_count"] = len(project["images"])
        project["status"] = "photos_ready" if project["image_count"] >= MIN_PHOTOS else "collecting_photos"
        project["updated_at"] = time.strftime("%Y-%m-%d %H:%M")
        write_project(project)
        self.send_json(201, {"saved": safe_name, "image_count": project["image_count"]})


if __name__ == "__main__":
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 8766), Handler)
    print("3DGS project library: http://127.0.0.1:8766")
    print("Training GPU: RTX 4060; COLMAP:", colmap_executable() or "not found")
    server.serve_forever()
