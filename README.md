# Truck 3D Gaussian Splatting Demo

An interactive project page for viewing a trained 3D Gaussian Splatting scene. It includes the built-in Lego demo, color/depth/estimated-normal render modes, camera controls, local PLY/SPLAT/KSPLAT loading, saved camera views, model export, project-library UI, and an optional local photo-reconstruction service.

## Quick preview (Windows)

1. Install Python 3.10 or newer and enable the Python launcher (`py -3`).
2. Double-click `start_preview.bat`.
3. Open <http://127.0.0.1:8766/> if the browser did not open automatically.

This serves the static page and built-in Lego example. It does not run training or save uploaded projects. Keep the console window open while previewing; press Ctrl+C to stop.

## Local project library and reconstruction

Double-click `start_project_app.bat` to start the local API and browser page. This mode can retain uploaded photo projects under `work/gs_projects/` and start the reconstruction pipeline when the requirements below are satisfied.

Automatic reconstruction requires all of the following on the computer running this app:

- This page folder is located at `<repo>/work/truck_project_page/`.
- The sibling `<repo>/work/gaussian-splatting/` checkout, its Python dependencies, and CUDA extension builds are installed and working.
- A compatible NVIDIA GPU and CUDA/PyTorch environment are available; the original development machine used an RTX 4060.
- COLMAP is installed and available as `colmap` on `PATH`, or its executable path is set in `COLMAP_EXE` before starting the server.
- At least 20 sharp, overlapping JPG/PNG/WebP photos of a static subject are uploaded. More coverage and overlap generally improve camera registration.

GitHub Pages and `start_preview.bat` only serve static files; neither can perform COLMAP or GPU training. The project API binds to `127.0.0.1` and is not exposed to the network. Reconstruction configuration is in `project_server.py` (`TRAIN_ITERATIONS`, photo limits, and training commands).

## Contents

- `index.html`, `style.css`, `project-library.js`: project page and project library UI.
- `viewer/index.html`: interactive Three.js Gaussian Splat viewer.
- `assets/lego_3000_3dgs.ply`: built-in trained example (about 32 MB).
- `assets/lego-gt.png`, `assets/lego-render.png`: static comparison imagery.
- `project_server.py`: local API for project metadata, image uploads, COLMAP, training, and generated outputs.
- `start_preview.bat`: static preview server.
- `start_project_app.bat`: local API server and browser launch.

The viewer's normal mode estimates a screen-space normal from depth gradients. It is a view-dependent visualization, not a mesh-normal channel stored in the Gaussian PLY.

## Sharing with classmates

Zip the entire `truck_project_page` folder, including `assets/` and `viewer/`. For a quick demo they only need Python 3 and `start_preview.bat`. For photo upload/reconstruction they must also install and configure the full training stack listed above. Do not share private photos or credentials in the archive.

The viewer loads Three.js and GaussianSplats3D from jsDelivr, so the first load requires internet access. The Lego model itself is bundled locally.

## GitHub Pages

The `.github/workflows/pages.yml` workflow deploys the static project page on every push to `main`. Enable **Settings → Pages → Build and deployment → GitHub Actions** once in the repository. GitHub Pages hosts the interactive Lego demo, including all three render modes; the local photo library, `/api/*` endpoints, COLMAP, and GPU reconstruction remain available only through `start_project_app.bat` on a configured computer. The included PLY is below GitHub's 100 MB per-file limit.
