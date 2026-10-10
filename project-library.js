const dialog = document.getElementById('library-dialog');
const projectList = document.getElementById('project-list');
const photoPicker = document.getElementById('photo-picker');
const photoReview = document.getElementById('photo-review');
const createButton = document.getElementById('create-project');
const readiness = document.getElementById('train-readiness');
const viewerFrame = document.querySelector('#viewer iframe');
const isLocalApp = ['localhost', '127.0.0.1'].includes(location.hostname);
let selectedRenderMode = 'color';
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
  viewerFrame.contentWindow?.postMessage({ type: '3dgs-render-mode', mode }, location.origin);
}

window.addEventListener('message', event => {
  if (event.origin !== location.origin || event.source !== viewerFrame.contentWindow) return;
  if (event.data?.type === '3dgs-viewer-ready') setRenderMode(selectedRenderMode);
});

document.querySelectorAll('[data-render-mode]').forEach(button => {
  button.addEventListener('click', () => setRenderMode(button.dataset.renderMode));
});
if (!isLocalApp) document.getElementById('open-library').hidden = true;

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
    open.type = 'button'; open.textContent = '打开 Demo';
    open.addEventListener('click', () => { window.location.href = project.is_builtin ? '/' : `/?project=${encodeURIComponent(project.id)}`; });
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

async function openGeneratedProject() {
  const projectId = new URLSearchParams(location.search).get('project');
  if (!projectId || projectId === 'lego-3000') return;
  try {
    const project = await api(`/api/projects/${encodeURIComponent(projectId)}/status`);
    if (project.status !== 'demo_ready') return;
    document.title = `${project.name} | 3D Gaussian Splatting`;
    document.querySelector('h1').innerHTML = `Reconstructing<br><em>${project.name.replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char])}</em> with 3D Gaussian Splatting`;
    document.querySelector('.hero-deck').textContent = 'A locally trained 3D Gaussian Splatting scene reconstructed from your photos.';
    document.querySelector('.hero-art img').src = project.render_url || 'assets/lego-render.png';
    document.querySelector('.hero-art img').alt = `${project.name} 3DGS render`;
    document.querySelector('.coord-a').textContent = `${project.image_count} INPUT PHOTOS`;
    document.querySelector('.coord-b').textContent = 'TRAINED GAUSSIAN FIELD';
    document.querySelector('.art-caption span:first-child').textContent = 'GENERATED DEMO';
    document.querySelector('.art-caption span:last-child').textContent = 'LOCAL GPU TRAINING';
    const scores = project.metrics || {};
    document.querySelector('.metric-grid').innerHTML = `<div class="metric"><strong>${Number(project.iterations || 30000).toLocaleString()}</strong><span>TRAINING ITERATIONS</span></div><div class="metric"><strong>${project.image_count}</strong><span>INPUT PHOTOS</span></div><div class="metric"><strong>${scores.PSNR ? Number(scores.PSNR).toFixed(2) : 'Ready'}</strong><span>${scores.PSNR ? 'TEST PSNR · dB' : 'MODEL STATUS'}</span></div>`;
    document.querySelector('.overview-copy .lead').textContent = `A custom 3D Gaussian Splatting reconstruction of ${project.name}, generated from ${project.image_count} uploaded photos.`;
    document.querySelectorAll('.overview-copy p')[2].textContent = `The model was trained locally for ${Number(project.iterations || 30000).toLocaleString()} iterations. Camera poses were estimated from the uploaded overlapping views with COLMAP.`;
    viewerFrame.src = `viewer/index.html?embedded=1&model=${encodeURIComponent(project.model_url)}&name=${encodeURIComponent(project.name)}`;
    if (project.reference_url) document.querySelector('.result-image').src = project.reference_url;
    if (project.render_url) document.querySelectorAll('.result-image')[1].src = project.render_url;
    document.querySelector('.asset-note').textContent = 'Generated locally from this project’s photos. Metrics are produced by the 3DGS evaluation script when available.';
    document.querySelector('.metric-table-wrap tbody').innerHTML = `<tr><td>${project.name}</td><td>Test split</td><td>${Number(project.iterations || 30000).toLocaleString()}</td><td>${project.gaussian_count || '—'}</td><td>Gaussian PLY</td></tr>`;
  } catch { /* A missing local project leaves the bundled example visible. */ }
}

if (isLocalApp) {
  openGeneratedProject();
  setInterval(() => { if (dialog.open) refreshProjects(); }, 3000);
}
