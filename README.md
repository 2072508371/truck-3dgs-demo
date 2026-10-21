# Truck 3D Gaussian Splatting Demo

An interactive project page for viewing trained 3D Gaussian Splatting scenes. The built-in selector contains three different Synthetic NeRF scenes: Chair, Drums, and Lego. Selecting one updates its own Gaussian model, reference/render images, training information, and geometry-rendering model ID. It also includes camera controls, real-time CUS-GS CUDA depth/normal modes, local PLY/SPLAT/KSPLAT loading, saved camera views, model export, project-library UI, and an optional local photo-reconstruction service.

## Quick preview (Windows)

1. Install Python 3.10 or newer and enable the Python launcher (`py -3`).
2. Double-click `start_preview.bat`.
3. Open <http://127.0.0.1:8766/> if the browser did not open automatically.

This serves the static page and built-in scene color viewers. It does not run the CUDA geometry API, training, or save uploaded projects. Keep the console window open while previewing; press Ctrl+C to stop.

## Local project library and reconstruction

Double-click `start_project_app.bat` to start the local API and browser page. This mode can retain uploaded photo projects under `work/gs_projects/` and start the reconstruction pipeline when the requirements below are satisfied.

Automatic reconstruction requires all of the following on the computer running this app:

- This page folder is located at `<repo>/work/truck_project_page/`.
- The sibling `<repo>/work/CUS-GS-uni-gs-test/` checkout and its compiled CUDA rasterizer extension are present.
- The `torchcuda311` Conda environment has CUDA-enabled PyTorch, `plyfile`, and Pillow installed. `start_project_app.bat` selects `%USERPROFILE%\miniconda3\envs\torchcuda311\python.exe` when available.
- The sibling `<repo>/work/gaussian-splatting/` checkout, its Python dependencies, and CUDA extension builds are installed and working.
- A compatible NVIDIA GPU and CUDA/PyTorch environment are available; the original development machine used an RTX 4060.
- COLMAP is installed and available as `colmap` on `PATH`, or its executable path is set in `COLMAP_EXE` before starting the server.
- At least 20 sharp, overlapping JPG/PNG/WebP photos of a static subject are uploaded. More coverage and overlap generally improve camera registration.

GitHub Pages and `start_preview.bat` only serve static files; neither can perform COLMAP or GPU training. The project API binds to `127.0.0.1` and is not exposed to the network. Reconstruction configuration is in `project_server.py` (`TRAIN_ITERATIONS`, photo limits, and training commands).

## Contents

- `index.html`, `style.css`, `project-library.js`: project page and project library UI.
- `viewer/index.html`: interactive Three.js Gaussian Splat viewer.
- `assets/lego_3000_3dgs.ply`, `assets/demos/chair-3000.ply`, `assets/demos/drums-3000.ply`: built-in trained scene models.
- `assets/demos/`: per-scene PLY models and matching training-view renders.
- `assets/lego-gt.png`, `assets/lego-render.png`: static comparison imagery.
- `project_server.py`: local API for project metadata, image uploads, COLMAP, training, and generated outputs.
- `start_preview.bat`: static preview server.
- `start_project_app.bat`: local API server and browser launch.

Depth and normal modes send the current interactive camera matrix and selected checkpoint ID to the local API. The CUS-GS CUDA rasterizer loads the matching bundled checkpoint, computes ray-ellipsoid depth and multi-scale normals backprojected from that depth, then the viewer overlays the returned frame. Camera orbit, pan, and zoom continue to work; a new frame is rendered after camera movement settles. A user-selected external PLY remains viewable in color mode but is not passed to the geometry endpoint.

The recommended local normal mode follows the UniGS depth-to-normal idea: it backprojects rendered depth, computes camera-facing normals at multiple pixel steps, rejects low-opacity/depth-discontinuity neighborhoods, and fuses trusted estimates before edge-aware smoothing. The optional hybrid and Gaussian-axis modes remain available for comparison. These are rendering improvements for existing RGB-trained checkpoints, not a substitute for UniGS depth/normal-supervised training; 3,000-iteration models can still have noisy or incomplete geometry.

## Sharing with classmates

Zip the entire `truck_project_page` folder, including `assets/` and `viewer/`. For a quick color-only demo they only need Python 3 and `start_preview.bat`. For live CUS-GS depth/normals they need the CUDA/PyTorch environment, the compiled CUS-GS rasterizer, and `start_project_app.bat`. For photo upload/reconstruction they must also install and configure the full training stack listed above. Do not share private photos or credentials in the archive.

The viewer loads Three.js and GaussianSplats3D from jsDelivr, so the first load requires internet access. All three scene models are bundled locally. Chair and Drums use the public Synthetic NeRF dataset mirror at Hugging Face; each was trained locally with GraphDeco for 3,000 iterations from 100 posed training views. The built-in comparisons use training views, not held-out benchmark results.

Dataset attribution: Chair is credited to 1DInc (CC0); Drums is credited to bryanajones (CC BY). The image/pose mirror is `rishitdagli/nerf-gs-datasets` on Hugging Face; retain these attributions when redistributing the demo.

## GitHub Pages

The `.github/workflows/pages.yml` workflow deploys the static project page on every push to `main`. Enable **Settings → Pages → Build and deployment → GitHub Actions** once in the repository. GitHub Pages serves the color viewer only; real-time CUDA depth and normals, the local photo library, `/api/*` endpoints, COLMAP, and GPU reconstruction require `start_project_app.bat` on a configured computer. A static host cannot execute this machine's CUDA renderer. The included PLY is below GitHub's 100 MB per-file limit.
