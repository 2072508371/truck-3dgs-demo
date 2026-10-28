# Truck 3D Gaussian Splatting Demo

An interactive project page for viewing trained 3D Gaussian Splatting scenes. The built-in selector contains three different Synthetic NeRF scenes: Chair, Drums, and Lego, with multiple Lego checkpoints including a background-pruned 10K variant. Selecting one updates its own Gaussian model, reference/render images, training information, and geometry-rendering model ID. It also includes camera controls, browser GPU depth, optional local CUS-GS CUDA normal rendering, local PLY/SPLAT/KSPLAT loading, saved camera views, model export, project-library UI, and an optional local photo-reconstruction service.

## Quick preview (Windows)

1. Install Python 3.10 or newer and enable the Python launcher (`py -3`).
2. Double-click `start_preview.bat`.
3. Open <http://127.0.0.1:8766/> if the browser did not open automatically.

This serves the static page and built-in scene color, browser GPU depth, and approximate browser GPU normal viewers. It does not run the CUDA geometry API, training, or save uploaded projects. Keep the console window open while previewing; press Ctrl+C to stop.

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
- `assets/demos/lego-10000-clean.ply`: 10K Lego RGB checkpoint with multi-view silhouette pruning. It removes 38,434 of 223,629 Gaussians using 100 training-view alpha masks; the original 10K model remains selectable.
- `assets/lego-gt.png`, `assets/lego-render.png`: static comparison imagery.
- `project_server.py`: local API for project metadata, image uploads, COLMAP, training, and generated outputs.
- `start_preview.bat`: static preview server.
- `start_project_app.bat`: local API server and browser launch.

The browser draws color, depth, and approximate Gaussian covariance normals directly with WebGL, so these modes follow camera orbit, pan, and zoom on a static host. With the local API available, normal mode instead sends the current camera and selected built-in checkpoint to CUS-GS CUDA, which rasterizes depth and alpha. The local server then produces normals using the selected method: multi-scale depth backprojection, the Gaussian covariance shortest axis, or a blend; an edge-aware filter is optional. A user-selected external PLY can use the browser modes but is not passed to the CUDA endpoint. Neither browser nor local mode has been established as more accurate against calibrated ground-truth normals.

The **Gaussian ellipsoids** switch in the viewer draws an instanced wireframe preview from each selected model's stored center, scale, rotation, and color. It defaults to a deterministic sample of 2,500 splats at a radius of two standard deviations; the sample count and radius are adjustable in the viewer. This preview shows the original 3D ellipsoid geometry, while the color mode uses the usual alpha-blended splat renderer. Sampling changes only the preview, not the PLY file.

On the local project server, **CUS-GS color** sends the current camera and selected built-in PLY to the CUDA rasterizer in `../CUS-GS-uni-gs-test/diff-gaussian-rasterization`. It reads the PLY spherical harmonic coefficients (up to the chosen degree) and returns a PNG for comparison with the browser color mode. This requires the configured local CUDA environment and is not available on static GitHub Pages. The wireframe ellipsoid mode remains a separate browser visualization because the CUS-GS rasterizer outputs a 2D splat image, not 3D ellipsoid meshes.

The recommended local normal mode follows the UniGS depth-to-normal idea: it backprojects rendered depth, computes camera-facing normals at multiple pixel steps, rejects low-opacity/depth-discontinuity neighborhoods, and fuses trusted estimates before edge-aware smoothing. The optional hybrid and Gaussian-axis modes remain available for comparison. These are rendering improvements for existing RGB-trained checkpoints, not a substitute for UniGS depth/normal-supervised training; 3,000-iteration models can still have noisy or incomplete geometry.

## Sharing with classmates

Zip the entire `truck_project_page` folder, including `assets/` and `viewer/`. For a quick browser demo they only need Python 3 and `start_preview.bat`. For CUS-GS CUDA normals they need the CUDA/PyTorch environment, the compiled CUS-GS rasterizer, and `start_project_app.bat`. For photo upload/reconstruction they must also install and configure the full training stack listed above. Do not share private photos or credentials in the archive.

The viewer loads Three.js and GaussianSplats3D from jsDelivr, so the first load requires internet access. All scene models are bundled locally. Chair and Drums use the public Synthetic NeRF dataset mirror at Hugging Face; each was trained locally with GraphDeco for 3,000 iterations from 100 posed training views. The built-in comparisons use RGB-trained views, not held-out RGB benchmark results.

The cleaned Lego variant uses the original 10K RGB checkpoint and removes Gaussian centers that fall outside the object silhouettes across most of the 100 mask views. It does not add normal supervision. On seven RGB-trained views omitted from mask pruning, the average object PSNR changed from 27.59 to 27.88 dB, full-image PSNR from 6.19 to 22.32 dB, and CUS-GS normal background leakage from 68.0% to 3.9%. See the sibling `work/geometry_gt_probe/` experiment scripts for reproducible evaluation. GitHub Pages provides interactive color, GPU depth, and approximate covariance-normal modes. The local CUDA service provides a separate CUS-GS rasterized normal mode with selectable depth-derived, Gaussian-axis, and blended results.

Chair and Drums also have separate cleaned variants, leaving their original 3K checkpoints available for comparison. Each filter uses 93 of the 100 posed RGBA training silhouettes with three-pixel dilation and keeps centers observed in fewer than four views. Seven RGB-trained views are omitted from mask pruning for evaluation; there are no test frames in these local datasets, so the numbers are not a novel-view benchmark. On those seven views, Chair at 35% silhouette agreement reduces the fraction of background pixels above 0.1 RGB from 90.4% to 25.5%, while foreground PSNR changes from 23.74 to 21.23 dB. Drums at 50% agreement reduces the same background fraction from 92.6% to 16.1%, while foreground PSNR changes from 18.85 to 17.51 dB. The more aggressive 75% filter cuts finer foreground structure and was not selected. The source scripts, full per-view metrics, and comparison images are in `work/geometry_gt_probe/`.

Dataset attribution: Chair is credited to 1DInc (CC0); Drums is credited to bryanajones (CC BY). The image/pose mirror is `rishitdagli/nerf-gs-datasets` on Hugging Face; retain these attributions when redistributing the demo.

## GitHub Pages

The `.github/workflows/pages.yml` workflow deploys the static project page on every push to `main`. Enable **Settings → Pages → Build and deployment → GitHub Actions** once in the repository. GitHub Pages serves interactive color, GPU depth, and approximate Gaussian covariance normals. The latter are a visualization estimate from the RGB-trained PLY, not the CUS-GS output or supervised ground-truth normals. Real-time CUDA normals, the local photo library, `/api/*` endpoints, COLMAP, and GPU reconstruction require `start_project_app.bat` on a configured computer. A static host cannot execute this machine's CUDA renderer. The included PLY files are below GitHub's 100 MB per-file limit.
