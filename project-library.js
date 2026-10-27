const dialog = document.getElementById('library-dialog');
const projectList = document.getElementById('project-list');
const photoPicker = document.getElementById('photo-picker');
const photoReview = document.getElementById('photo-review');
const createButton = document.getElementById('create-project');
const readiness = document.getElementById('train-readiness');
const viewerFrame = document.querySelector('#viewer iframe');
const isLocalApp = ['localhost', '127.0.0.1'].includes(location.hostname);
let selectedRenderMode = 'color';
const renderStatus = document.getElementById('viewer-render-status');
const demoSelect = document.getElementById('demo-select');
const builtinDemos = [
  { id: 'chair-3000', name: 'Chair · 3,000 iterations', displayName: 'Chair Scene', scene: 'Synthetic NeRF Chair', iterations: 3000, image_count: 100, gaussian_count: 142302, model_size: '35.3 MB', model_url: 'assets/demos/chair-3000.ply', render_url: 'assets/demos/chair-3000.png', reference_url: 'assets/demos/chair-reference.png', view_label: 'TRAIN VIEW 000', description: 'A locally trained reconstruction from the distinct Synthetic NeRF Chair multi-view dataset. This demo uses 100 posed training views and its own Gaussian model.' },
  { id: 'chair-3000-clean', name: 'Chair · 3,000 · 背景清理', displayName: 'Chair Scene · Cleaned', scene: 'Synthetic NeRF Chair', iterations: 3000, image_count: 100, gaussian_count: 96871, model_size: '24.0 MB', model_url: 'assets/demos/chair-3000-clean.ply', render_url: 'assets/demos/chair-3000-clean.png', reference_url: 'assets/demos/chair-3000-clean-reference.png', view_label: 'TRAIN VIEW 003 · MASK-OMITTED', description: 'The original 3,000-iteration Chair checkpoint, with Gaussian centers filtered by 93 posed training silhouettes at 35% agreement. Seven RGB-trained views were omitted from pruning for checking. Background noise is reduced, with some loss in fine foreground detail.' },
  { id: 'drums-3000', name: 'Drums · 3,000 iterations', displayName: 'Drums Scene', scene: 'Synthetic NeRF Drums', iterations: 3000, image_count: 100, gaussian_count: 113725, model_size: '28.2 MB', model_url: 'assets/demos/drums-3000.ply', render_url: 'assets/demos/drums-3000.png', reference_url: 'assets/demos/drums-reference.png', view_label: 'TRAIN VIEW 000', description: 'A locally trained reconstruction from the distinct Synthetic NeRF Drums multi-view dataset. This demo uses 100 posed training views and its own Gaussian model.' },
  { id: 'drums-3000-clean', name: 'Drums · 3,000 · 背景清理', displayName: 'Drums Scene · Cleaned', scene: 'Synthetic NeRF Drums', iterations: 3000, image_count: 100, gaussian_count: 47347, model_size: '11.7 MB', model_url: 'assets/demos/drums-3000-clean.ply', render_url: 'assets/demos/drums-3000-clean.png', reference_url: 'assets/demos/drums-3000-clean-reference.png', view_label: 'TRAIN VIEW 003 · MASK-OMITTED', description: 'The original 3,000-iteration Drums checkpoint, with Gaussian centers filtered by 93 posed training silhouettes at 50% agreement. Seven RGB-trained views were omitted from pruning for checking. Background noise is reduced while most drum structure remains.' },
  { id: 'lego-3000', name: 'Lego · 3,000 iterations', displayName: 'Lego Scene', scene: 'Synthetic NeRF Lego', iterations: 3000, image_count: 136, gaussian_count: 134826, model_size: '33.4 MB', model_url: 'assets/lego_3000_3dgs.ply', render_url: 'assets/lego-render.png', reference_url: 'assets/lego-gt.png', view_label: 'TRAIN VIEW 035', description: 'A locally trained reconstruction from the Synthetic NeRF Lego multi-view dataset. The viewer loads the trained Gaussian PLY and supports orbit, pan, zoom, depth, and live CUS-GS normals.' },
  { id: 'lego-10000', name: 'Lego · 10,000 iterations', displayName: 'Lego Scene', scene: 'Synthetic NeRF Lego', iterations: 10000, image_count: 136, gaussian_count: 223629, model_size: '55.5 MB', model_url: 'assets/demos/lego-10000.ply', render_url: 'assets/demos/lego-10000.png', reference_url: 'assets/lego-gt.png', view_label: 'TRAIN VIEW 035', description: 'Longer RGB training improves the reconstruction and relative depth structure compared with the 3,000-iteration Lego model. Native UniGS rendering was tested separately but is not used here.' },
  { id: 'lego-10000-clean', name: 'Lego · 10,000 · 背景优化', displayName: 'Lego Scene · Cleaned', scene: 'Synthetic NeRF Lego', iterations: 10000, image_count: 136, gaussian_count: 185195, model_size: '45.9 MB', model_url: 'assets/demos/lego-10000-clean.ply', render_url: 'assets/demos/lego-10000-clean.png', reference_url: 'assets/demos/lego-10000-clean-reference.png', view_label: 'TRAINED VIEW R_101', description: 'The same 10,000-iteration RGB model, with background Gaussians pruned using 100 training-view silhouette masks. This reduces floaters in live color, depth, and normal modes. It is not a newly trained normal-supervised model.' }
];
let demoPickerBound = false;
let pickerProjects = [];
let selectedPhotos = [];
let acceptedPhotos = [];
let serverStatus;

async function api(path, options) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`);
  return body;
}

function setRenderMode(mode) {
  if (!['color', 'depth', 'normal'].includes(mode)) return;
  selectedRenderMode = mode;
  document.querySelectorAll('[data-render-mode]').forEach(button => {
    button.setAttribute('aria-pressed', String(button.dataset.renderMode === mode));
  });
  renderStatus.textContent = mode === 'normal'
    ? (isLocalApp ? '正在计算本机 CUS-GS 法向图…' : '浏览器 GPU 近似法向图；可随相机实时转动')
    : mode === 'depth' ? '浏览器 GPU 实时深度图' : '交互彩色渲染';
  viewerFrame.contentWindow?.postMessage({ type: '3dgs-render-mode', mode }, location.origin);
}

window.addEventListener('message', event => {
  if (event.origin !== location.origin || event.source !== viewerFrame.contentWindow) return;
  if (event.data?.type === '3dgs-viewer-ready') setRenderMode(selectedRenderMode);
  if (event.data?.type === '3dgs-geometry-progress') renderStatus.textContent = '正在按当前相机重新计算 CUS-GS 几何图…';
  if (event.data?.type === '3dgs-geometry-rendered') renderStatus.textContent = 'CUS-GS CUDA 几何图已同步到当前相机';
  if (event.data?.type === '3dgs-geometry-error') {
    setRenderMode('color');
    renderStatus.textContent = `实时深度/法向不可用：${event.data.message}`;
  }
  if (event.data?.type === '3dgs-normal-source' && selectedRenderMode === 'normal') {
    renderStatus.textContent = event.data.source === 'browser'
      ? '浏览器 GPU 近似法向图；可随相机实时转动'
      : '正在计算本机 CUS-GS 法向图…';
  }
});

document.querySelectorAll('[data-render-mode]').forEach(button => {
  button.addEventListener('click', () => setRenderMode(button.dataset.renderMode));
});
if (!isLocalApp) document.getElementById('open-library').hidden = true;
if (!isLocalApp) renderStatus.textContent = '公开页面支持浏览器 GPU 法向估计；本机 CUDA 法向需要本地服务';

function qualityOf(file) {
  return new Promise(resolve => {
    const image = new Image();
    const objectUrl = URL.createObjectURL(file);
    image.onload = () => {
      const canvas = document.createElement('canvas');
      const scale = Math.min(1, 240 / image.width);
      canvas.width = Math.max(3, Math.round(image.width * scale));
      canvas.height = Math.max(3, Math.round(image.height * scale));
      const context = canvas.getContext('2d', { willReadFrequently: true });
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
      const gray = new Float32Array(canvas.width * canvas.height);
      for (let i = 0; i < gray.length; i++) gray[i] = (pixels[i * 4] * 0.299 + pixels[i * 4 + 1] * 0.587 + pixels[i * 4 + 2] * 0.114);
      let sum = 0, sumSq = 0, count = 0;
      for (let y = 1; y < canvas.height - 1; y++) for (let x = 1; x < canvas.width - 1; x++) {
        const i = y * canvas.width + x;
        const lap = gray[i - canvas.width] + gray[i - 1] - 4 * gray[i] + gray[i + 1] + gray[i + canvas.width];
        sum += lap; sumSq += lap * lap; count++;
      }
      const blurScore = count ? sumSq / count - (sum / count) ** 2 : 0;
      URL.revokeObjectURL(objectUrl);
      resolve({ file, width: image.naturalWidth, height: image.naturalHeight, blurScore });
    };
    image.onerror = () => { URL.revokeObjectURL(objectUrl); resolve({ file, width: 0, height: 0, blurScore: 0 }); };
    image.src = objectUrl;
  });
}

function assess(photo) {
  const reasons = [];
  if (photo.width < 1000 || photo.height < 700) reasons.push('分辨率低于 1000×700');
  if (photo.blurScore < 18) reasons.push('可能模糊或细节不足');
  return reasons;
}

async function inspectPhotos(files) {
  selectedPhotos = [...files];
  acceptedPhotos = [];
  createButton.disabled = true;
  photoReview.textContent = '正在检查照片尺寸与清晰度…';
  const checked = [];
  for (let start = 0; start < selectedPhotos.length; start += 4) {
    checked.push(...await Promise.all(selectedPhotos.slice(start, start + 4).map(qualityOf)));
  }
  const failures = checked.flatMap(photo => assess(photo).map(reason => `${photo.file.name}: ${reason}`));
  acceptedPhotos = checked.filter(photo => assess(photo).length === 0);
  const countNote = `${acceptedPhotos.length}/${checked.length} 张通过基础质检；至少需要 20 张合格照片。`;
  photoReview.textContent = failures.length ? `${countNote} 未通过：${failures.slice(0, 3).join('；')}${failures.length > 3 ? '…' : ''}` : countNote;
  createButton.disabled = acceptedPhotos.length < 20 || !document.getElementById('project-name').value.trim() || !document.getElementById('coverage-confirm').checked;
}

function renderProjectCard(project) {
  const card = document.createElement('article');
  card.className = 'project-card';
  const details = document.createElement('div');
  const title = document.createElement('h4');
  title.textContent = project.name;
  const status = document.createElement('p');
  const statusNames = { demo_ready: 'Demo 可用', photos_ready: '照片已保存', collecting_photos: '待补充照片', estimating_cameras: 'COLMAP 位姿估计中', training: 'GPU 训练中', queued: '任务排队中', failed: project.message || '任务失败' };
  status.textContent = `${statusNames[project.status] || project.status} · ${project.image_count || 0} 张照片`;
  details.append(title, status);
  card.append(details);
  if (project.status === 'demo_ready') {
    const open = document.createElement('button');
    open.type = 'button'; open.textContent = project.is_builtin ? '切换到此 Demo' : '打开 Demo';
    open.addEventListener('click', () => {
      if (project.is_builtin) selectDemo(project.id);
      else selectProject(project);
      dialog.close();
      document.getElementById('viewer-section').scrollIntoView({ behavior: 'smooth' });
    });
    card.append(open);
  } else if (project.image_count >= 20 && ['photos_ready', 'failed'].includes(project.status)) {
    const train = document.createElement('button');
    train.type = 'button'; train.textContent = '启动重建';
    train.addEventListener('click', async () => {
      try {
        await api(`/api/projects/${project.id}/train`, { method: 'POST' });
        readiness.textContent = '重建任务已启动；COLMAP 位姿估计与 3DGS 训练状态会显示在项目卡片中。';
      } catch (error) {
        readiness.textContent = error.message;
      }
      await refreshProjects();
    });
    card.append(train);
  }
  return card;
}

async function refreshProjects() {
  try {
    const projects = await api('/api/projects');
    configureDemoPicker(projects);
    projectList.replaceChildren(...projects.map(renderProjectCard));
  } catch {
    projectList.innerHTML = '<p class="library-muted">本地项目服务未连接。请用 start_project_app.bat 启动项目库。</p>';
  }
}

async function checkReadiness() {
  try {
    serverStatus = await api('/api/status');
    readiness.classList.toggle('ready', serverStatus.colmap_ready);
    readiness.textContent = serverStatus.colmap_ready
      ? `本机 GPU 可用 · COLMAP 已就绪 · ${serverStatus.iterations.toLocaleString()} 次训练`
      : '本机 GPU 可用，但未检测到 COLMAP。照片仍可保存；配置 COLMAP 后重启本地服务即可自动估计相机位姿并训练。';
  } catch {
    readiness.textContent = '本地服务未连接；项目管理和照片上传需要通过 start_project_app.bat 打开。';
  }
}

document.getElementById('open-library').addEventListener('click', async () => {
  dialog.showModal();
  await Promise.all([refreshProjects(), checkReadiness()]);
});
document.getElementById('close-library').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
photoPicker.addEventListener('change', () => inspectPhotos(photoPicker.files || []));
document.getElementById('project-name').addEventListener('input', () => {
  createButton.disabled = acceptedPhotos.length < 20 || !document.getElementById('project-name').value.trim() || !document.getElementById('coverage-confirm').checked;
});
document.getElementById('coverage-confirm').addEventListener('change', () => {
  createButton.disabled = acceptedPhotos.length < 20 || !document.getElementById('project-name').value.trim() || !document.getElementById('coverage-confirm').checked;
});

createButton.addEventListener('click', async () => {
  createButton.disabled = true;
  createButton.textContent = '正在保存照片…';
  try {
    const project = await api('/api/projects', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: document.getElementById('project-name').value.trim() }) });
    for (let index = 0; index < acceptedPhotos.length; index++) {
      const photo = acceptedPhotos[index];
      photoReview.textContent = `正在保存照片 ${index + 1}/${acceptedPhotos.length}…`;
      const url = `/api/projects/${project.id}/images/${encodeURIComponent(photo.file.name)}?width=${photo.width}&height=${photo.height}`;
      const response = await fetch(url, { method: 'PUT', headers: { 'Content-Type': photo.file.type }, body: photo.file });
      if (!response.ok) throw new Error((await response.json()).error || '照片保存失败');
    }
    photoReview.textContent = `${acceptedPhotos.length} 张合格照片已保存在本机，正在准备重建…`;
    await refreshProjects();
    try {
      await api(`/api/projects/${project.id}/train`, { method: 'POST' });
      readiness.textContent = '任务已启动。可在“已保存的 Demo”中查看相机估计与训练状态。';
    } catch (error) {
      readiness.textContent = error.message;
    }
    await refreshProjects();
  } catch (error) {
    photoReview.textContent = `创建失败：${error.message}`;
    createButton.disabled = false;
  } finally {
    createButton.textContent = '创建项目并检查照片';
    createButton.disabled = acceptedPhotos.length < 20 || !document.getElementById('project-name').value.trim() || !document.getElementById('coverage-confirm').checked;
  }
});

function number(value) { return Number(value || 0).toLocaleString(); }

function configureDemoPicker(projects = []) {
  pickerProjects = projects;
  for (const demo of builtinDemos) {
    const saved = projects.find(project => project.id === demo.id && project.is_builtin);
    if (saved) Object.assign(demo, {
      gaussian_count: saved.gaussian_count || demo.gaussian_count,
      model_size: saved.model_size || demo.model_size,
      image_count: saved.image_count || demo.image_count,
      model_url: saved.model_url?.replace(/^\/assets\//, 'assets/') || demo.model_url,
      render_url: saved.render_url || demo.render_url,
      reference_url: saved.reference_url || demo.reference_url,
      view_label: saved.view_label || demo.view_label
    });
  }
  const current = demoSelect.value;
  const options = [...builtinDemos, ...projects.filter(project => project.status === 'demo_ready' && !project.is_builtin)];
  demoSelect.replaceChildren(...options.map(project => new Option(project.name, project.id)));
  if (!demoPickerBound) {
    demoSelect.addEventListener('change', () => {
      if (builtinDemos.some(item => item.id === demoSelect.value)) selectDemo(demoSelect.value);
      else {
        const project = pickerProjects.find(item => item.id === demoSelect.value);
        if (project) selectProject(project);
      }
    });
    demoPickerBound = true;
  }
  if ([...demoSelect.options].some(option => option.value === current)) demoSelect.value = current;
}

function applyDemo(demo, isUploaded = false) {
  const iterations = Number(demo.iterations || 30000);
  const name = demo.name || 'Untitled scene';
  const modelUrl = new URL(demo.model_url, location.href).href;
  const renderUrl = demo.render_url || demo.reference_url || 'assets/lego-render.png';
  const referenceUrl = demo.reference_url || renderUrl;
  const points = Number(demo.gaussian_count || 0);
  const size = demo.model_size || 'Gaussian PLY';
  const escaped = name.replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);
  document.title = `${name} | 3D Gaussian Splatting`;
  document.querySelector('meta[name="description"]').content = `${name}: an interactive 3D Gaussian Splatting reconstruction, with scene-specific training information and renders.`;
  document.querySelector('h1').innerHTML = `Reconstructing the<br><em>${demo.displayName || escaped}</em> with 3D Gaussian Splatting`;
  document.querySelector('.hero-copy > .eyebrow').innerHTML = `<span class="live-dot"></span>${isUploaded ? 'LOCAL PHOTO RECONSTRUCTION' : 'SYNTHETIC NERF · NOVEL VIEW SYNTHESIS'}`;
  document.querySelector('.hero-deck').textContent = demo.description || `A locally trained 3D Gaussian Splatting reconstruction of ${name}.`;
  const heroImage = document.querySelector('.hero-art img');
  heroImage.src = renderUrl;
  heroImage.alt = `${name} 3DGS training render`;
  document.querySelector('.coord-a').textContent = isUploaded ? `${demo.image_count || 0} INPUT PHOTOS` : `${demo.image_count || 136} TRAIN VIEWS`;
  document.querySelector('.coord-b').textContent = `GAUSSIAN FIELD /// ${iterations >= 1000 ? `${iterations / 1000}K` : iterations}`;
  document.querySelector('.art-caption span:first-child').textContent = isUploaded ? 'GENERATED DEMO' : 'REAL TRAINING RENDER';
  document.querySelector('.art-caption span:last-child').textContent = points ? `${number(points)} GAUSSIANS` : 'GAUSSIAN PLY';
  document.querySelector('.metric-grid').innerHTML = `<div class="metric"><strong>${number(iterations)}</strong><span>TRAINING ITERATIONS</span></div><div class="metric"><strong>${points ? number(points) : '—'}</strong><span>GAUSSIAN POINTS</span></div><div class="metric"><strong>${size}</strong><span>PLY MODEL SIZE</span></div>`;
  document.getElementById('overview-title').textContent = `From posed ${demo.scene || name} views to a navigable scene`;
  document.querySelector('.overview-copy .lead').textContent = isUploaded
    ? `A custom 3D Gaussian Splatting reconstruction of ${name}, generated from ${demo.image_count || 0} uploaded photos.`
    : `${demo.description || `A 3D Gaussian Splatting reconstruction of ${name}.`} This page is showing the selected checkpoint.`;
  document.querySelectorAll('.overview-copy p')[2].textContent = isUploaded
    ? `The model was trained locally for ${number(iterations)} iterations. Camera poses were estimated from the uploaded overlapping views with COLMAP.`
    : `This saved model contains ${points ? number(points) : 'an unreported number of'} Gaussians after ${number(iterations)} iterations. The viewer, render images, and metrics on this page all refer to this selected checkpoint.`;
  document.getElementById('viewer-scene-name').textContent = name;
  document.querySelector('.wordmark').setAttribute('aria-label', `${name} project home`);
  document.getElementById('footer-scene-name').textContent = `${name.toLocaleUpperCase()} · 3D GAUSSIAN SPLATTING`;
  document.querySelector('#viewer iframe').title = `Interactive ${name} Gaussian Splatting viewer`;
  const previewUrl = new URL(renderUrl, location.href);
  const params = new URLSearchParams({ embedded: '1', model: modelUrl, name, demoId: demo.id || '', iterations: String(iterations), preview: previewUrl.pathname, v: `demo-${demo.id || 'custom'}-${Date.now()}` });
  viewerFrame.src = `viewer/index.html?${params}`;
  const resultImages = document.querySelectorAll('.result-image');
  resultImages[0].src = referenceUrl;
  resultImages[0].alt = `${name} reference image`;
  resultImages[1].src = renderUrl;
  resultImages[1].alt = `${name} Gaussian Splatting rendering`;
  document.querySelector('.result-frame:first-child figcaption span').textContent = isUploaded ? 'INPUT PHOTO' : `GROUND TRUTH · ${demo.view_label || 'TRAIN VIEW'}`;
  document.querySelector('.result-frame:first-child figcaption code').textContent = referenceUrl.split('/').pop();
  document.querySelector('.result-frame:last-child figcaption span').textContent = `3DGS RENDER · ${demo.view_label || 'SELECTED VIEW'}`;
  document.querySelector('.result-frame:last-child figcaption code').textContent = renderUrl.split('/').pop();
  document.getElementById('result-view-label').innerHTML = `${demo.view_label || 'TRAIN VIEW'} <b>·</b> ${number(iterations)} ITERATIONS`;
  document.querySelector('.asset-note').textContent = isUploaded
    ? 'Generated locally from this project’s photos. Metrics are produced by the 3DGS evaluation script when available.'
    : `Both images show the selected ${number(iterations)}-iteration checkpoint and its matching trained view. This is not a held-out RGB benchmark.`;
  document.querySelector('.metric-table-wrap tbody').innerHTML = `<tr><td>${escaped}</td><td>${isUploaded ? 'Uploaded photos' : demo.view_label || 'Training view'}</td><td>${number(iterations)}</td><td>${points ? number(points) : '—'}</td><td>${size}</td></tr>`;
  if (![...demoSelect.options].some(option => option.value === demo.id)) demoSelect.add(new Option(name, demo.id));
  demoSelect.value = demo.id;
}

function selectDemo(demoId) {
  const demo = builtinDemos.find(item => item.id === demoId);
  if (!demo) return;
  applyDemo(demo);
  const url = new URL(location.href);
  url.searchParams.set('demo', demo.id);
  url.searchParams.delete('project');
  history.replaceState({}, '', url);
  refreshProjects();
}

function selectProject(project) {
  applyDemo(project, true);
  const url = new URL(location.href);
  url.searchParams.set('project', project.id);
  url.searchParams.delete('demo');
  history.replaceState({}, '', url);
}

async function loadSelectedDemo() {
  const params = new URLSearchParams(location.search);
  const projectId = params.get('project');
  if (projectId) {
    try {
      const project = await api(`/api/projects/${encodeURIComponent(projectId)}/status`);
      if (project.status === 'demo_ready') selectProject(project);
      return;
    } catch { return; }
  }
  const requested = params.get('demo') || 'chair-3000';
  selectDemo(builtinDemos.some(item => item.id === requested) ? requested : 'chair-3000');
}

if (isLocalApp) {
  configureDemoPicker();
  loadSelectedDemo();
  setInterval(() => { if (dialog.open) refreshProjects(); }, 3000);
} else {
  configureDemoPicker();
  loadSelectedDemo();
}
