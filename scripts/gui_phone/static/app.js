// WBCD Console — Full Pipeline + Waterfall Feed Labeler

let S = {
    token: null,
    selectedTemplate: null,
    logPolling: null,
    currentJobId: null,
    datasets: [],
    // Labeler
    dsPath: null,
    dsMeta: null,
    globalCats: [],
    feedPage: 1,
    feedTotal: 0,
    feedLoading: false,
    feedDone: false,
    gpuCache: null,
    // Pipeline
    pipeGpuData: [],
    pipeSelectedGPUs: new Set(),
    pipeDatasets: [],
    pipeRunning: false,
};
const PER_PAGE = 10;
const PIPE_STEPS = ['compute_norm_stats', 'postprocess_norm_stats', 'train'];

const $ = id => document.getElementById(id);
const showToast = (m, e) => { const t=$('toast'); t.textContent=m; t.className=`toast${e?' error':''}`; setTimeout(()=>t.className='toast hidden',2200); };
const fmtDate = iso => { if(!iso)return'-'; const d=new Date(iso); return `${d.getMonth()+1}/${d.getDate()} ${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`; };

// API
const api = async (u,m='GET',b=null) => {
    const h={}; if(S.token)h['Authorization']=`Bearer ${S.token}`; if(b)h['Content-Type']='application/json';
    const r=await fetch(u,{method:m,headers:h,body:b?JSON.stringify(b):null});
    if(r.status===401){logout();throw new Error('Unauthorized')}
    const d=await r.json(); if(!r.ok)throw new Error(d.detail||'Error'); return d;
};
const apiQ = async(...a)=>{try{return await api(...a)}catch(e){return null}};

// Nav
const switchView=id=>{document.querySelectorAll('.view').forEach(v=>v.classList.add('hidden'));$(id).classList.remove('hidden')};
const switchPage=(id,title)=>{
    document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n=>n.classList.remove('active'));
    const p=$(id);if(p)p.classList.add('active');
    const n=document.querySelector(`.nav-item[data-target="${id}"]`);if(n)n.classList.add('active');
    $('page-title').textContent=title; loadPage(id);
};
const loadPage=id=>{
    if(id==='datasets') loadDatasets();
    else if(id==='train') loadTrainPage();
    else if(id==='task'){loadGpuStrip();loadJobs();pollPipelineStatus();}
};

// Auth
const login=async e=>{if(e)e.preventDefault();try{const r=await api('/api/login','POST',{password:$('password').value});S.token=r.token;switchView('app-view');switchPage('datasets','Datasets');startGpu();}catch(e){$('login-error').textContent='Wrong password'}};
const logout=async()=>{S.token=null;stopGpu();stopPipePoll();try{await api('/api/logout','POST')}catch(e){}switchView('login-view');$('password').value=''};

// Status Panel
const toggleStatus=()=>{const p=$('status-panel'),b=$('status-panel-backdrop');if(p.classList.contains('hidden')){p.classList.remove('hidden');b.classList.remove('hidden');loadStatus();}else closeStatus()};
const closeStatus=()=>{$('status-panel').classList.add('hidden');$('status-panel-backdrop').classList.add('hidden')};
const loadStatus=async()=>{
    const el=$('status-panel-content');el.innerHTML='<div class="loading-spinner"></div>';
    try{const d=await api('/api/system');const gc=p=>p>85?'var(--dng)':p>60?'var(--wrn)':'var(--ac)';
    let h=`<div class="mini-card"><div class="mini-card-title">Host: ${d.hostname}</div><div class="mini-stat-row"><span>Up</span><span>${(d.uptime_seconds/3600).toFixed(1)}h</span></div></div>
    <div class="mini-card"><div class="mini-card-title">CPU</div><div class="mini-stat-row"><span>${d.cpu.percent}%</span><span>Load ${d.cpu.load_1m}</span></div><div class="mini-progress"><div class="mini-progress-fill" style="width:${d.cpu.percent}%;background:${gc(d.cpu.percent)}"></div></div></div>
    <div class="mini-card"><div class="mini-card-title">Mem ${d.memory.used_gb}/${d.memory.total_gb}G</div><div class="mini-stat-row"><span></span><span>${d.memory.percent}%</span></div><div class="mini-progress"><div class="mini-progress-fill" style="width:${d.memory.percent}%;background:${gc(d.memory.percent)}"></div></div></div>`;
    if(d.gpu&&d.gpu.length>0)d.gpu.forEach(g=>{const mp=Math.round(g.memory_used_mb/g.memory_total_mb*100);h+=`<div class="mini-card"><div class="mini-card-title">GPU${g.index}</div><div class="mini-stat-row"><span>${g.utilization}% ${g.temperature}°C</span><span>${g.memory_used_mb}/${g.memory_total_mb}M</span></div><div class="mini-progress"><div class="mini-progress-fill" style="width:${mp}%;background:${gc(mp)}"></div></div></div>`;});
    el.innerHTML=h;}catch(e){}
};

// GPU Badge (nav)
let _gi=null;
const startGpu=()=>{updGpu();_gi=setInterval(updGpu,15000)};
const stopGpu=()=>{if(_gi)clearInterval(_gi)};
const updGpu=async()=>{const d=await apiQ('/api/system');if(!d)return;S.gpuCache=d;const b=$('nav-gpu-badge');if(d.gpu&&d.gpu.length>0){b.textContent=Math.max(...d.gpu.map(g=>g.utilization))+'%';b.classList.remove('hidden')}else b.classList.add('hidden')};
const loadGpuStrip=async()=>{const el=$('gpu-strip');const d=S.gpuCache||await apiQ('/api/system');if(!d||!d.gpu||!d.gpu.length){el.innerHTML='';return;}el.innerHTML=d.gpu.map(g=>`<div class="gpu-card"><div class="gpu-card-name">GPU${g.index}</div><div class="gpu-card-stats"><span class="gpu-card-util">${g.utilization}%</span><span class="gpu-card-mem">${g.memory_used_mb}/${g.memory_total_mb}M</span><span class="gpu-card-temp">${g.temperature}°C</span></div></div>`).join('')};

// ════════════════════════════════════
// Datasets list
// ════════════════════════════════════
const loadDatasets=async()=>{
    const el=$('datasets-list');el.innerHTML='<div class="loading-spinner"></div>';
    try{const r=await api('/api/datasets/scan');S.datasets=r.datasets;
    el.innerHTML=r.datasets.map(ds=>{const cats=(ds.categories||[]).map(c=>`<span class="cat-tag">${c}</span>`).join('');
    return `<div class="list-item" onclick="openLabeler('${ds.dataset_root}')"><div class="item-icon">🗂️</div><div class="item-details"><div class="item-name">${ds.name}</div><div class="item-meta">${ds.total_episodes} eps</div>${cats?`<div class="item-categories">${cats}</div>`:''}</div></div>`;}).join('')||'<div style="padding:14px;text-align:center;color:var(--t2)">No datasets</div>';
    }catch(e){showToast(e.message,true)}
};

// ════════════════════════════════════
// ★ LABELER — Waterfall Feed
// ════════════════════════════════════

const openLabeler = async (path) => {
    S.dsPath = path;
    S.feedPage = 1;
    S.feedDone = false;
    S.feedLoading = false;

    $('labeler').classList.remove('hidden');
    const feed = $('labeler-feed');
    feed.innerHTML = '<div class="loading-spinner"></div>';
    $('labeler-ds-name').textContent = path.split('/').pop();
    $('labeler-count').textContent = '';

    try {
        const [meta, globalCats, epRes] = await Promise.all([
            api(`/api/dataset/meta?path=${encodeURIComponent(path)}`),
            api('/api/categories/global'),
            api(`/api/dataset/episodes?path=${encodeURIComponent(path)}&lf=all&page=1&per_page=${PER_PAGE}`)
        ]);

        S.dsMeta = meta;
        S.globalCats = globalCats.categories || [];
        S.feedTotal = epRes.total;

        const f = $('labeler-filter');
        f.innerHTML = '<option value="all">All</option><option value="unlabeled">Unlabeled</option>';
        S.globalCats.forEach(c => { f.innerHTML += `<option value="${c}">${c}</option>`; });

        $('labeler-count').textContent = `${epRes.total} eps`;

        feed.innerHTML = '';
        renderFeedCards(epRes.items);

        if (epRes.items.length >= epRes.total) {
            S.feedDone = true;
            feed.innerHTML += '<div class="feed-end">— End —</div>';
        }
    } catch(e) {
        feed.innerHTML = `<div style="padding:20px;text-align:center;color:var(--dng)">${e.message}</div>`;
    }
};

const renderFeedCards = (items) => {
    const feed = $('labeler-feed');
    items.forEach(ep => {
        const card = document.createElement('div');
        card.className = 'feed-card';
        card.id = `fc-${ep.episode_id}`;

        const labelBadge = ep.label
            ? `<span class="feed-card-label-badge">${ep.label}</span>`
            : `<span class="feed-card-label-badge unlabeled">Unlabeled</span>`;

        const chips = S.globalCats.map(c =>
            `<div class="feed-chip ${ep.label===c?'active':''}" onclick="chipClick('${ep.episode_id}','${c.replace(/'/g,"\\'")}',this)">${c}</div>`
        ).join('') + `<div class="feed-chip add-new" onclick="addGlobalCat()">+</div>`;

        card.innerHTML = `
            <div class="feed-card-video">
                <video src="${ep.head_video_url}" loop muted playsinline preload="metadata" onloadeddata="this.currentTime=0.5" onclick="this.paused?this.play():this.pause()"></video>
            </div>
            <div class="feed-card-body">
                <div class="feed-card-header">
                    <span class="feed-card-epname">${ep.episode_id}</span>
                    ${labelBadge}
                </div>
                <div class="feed-card-chips" id="chips-${ep.episode_id}">
                    ${chips}
                </div>
            </div>`;

        feed.appendChild(card);
    });
};

const chipClick = async (epId, cat, chipEl) => {
    const chipsContainer = $(`chips-${epId}`);
    const allChips = chipsContainer.querySelectorAll('.feed-chip:not(.add-new)');

    const wasActive = chipEl.classList.contains('active');
    allChips.forEach(c => c.classList.remove('active'));
    const newLabel = wasActive ? '' : cat;
    if (!wasActive) chipEl.classList.add('active');

    try {
        await api('/api/dataset/label', 'POST', {
            dataset_root: S.dsPath,
            episode_id: epId,
            label: newLabel,
            note: ''
        });

        const card = $(`fc-${epId}`);
        const badge = card.querySelector('.feed-card-label-badge');
        if (newLabel) {
            badge.textContent = newLabel;
            badge.classList.remove('unlabeled');
        } else {
            badge.textContent = 'Unlabeled';
            badge.classList.add('unlabeled');
        }

        showToast(`✓ ${newLabel || 'Cleared'}`);
    } catch(e) {
        showToast(e.message, true);
    }
};

const addGlobalCat = async () => {
    const n = prompt("New category name:");
    if (!n || !n.trim()) return;
    const name = n.trim();

    try {
        await api('/api/dataset/category/add', 'POST', { dataset_root: S.dsPath, name });
        await api('/api/categories/sync', 'POST', { name });
        if (!S.globalCats.includes(name)) S.globalCats.push(name);

        document.querySelectorAll('.feed-card-chips').forEach(container => {
            const epId = container.id.replace('chips-', '');
            const card = $(`fc-${epId}`);
            const badge = card.querySelector('.feed-card-label-badge');
            const currentLabel = badge.classList.contains('unlabeled') ? '' : badge.textContent;

            container.innerHTML = S.globalCats.map(c =>
                `<div class="feed-chip ${currentLabel===c?'active':''}" onclick="chipClick('${epId}','${c.replace(/'/g,"\\'")}',this)">${c}</div>`
            ).join('') + `<div class="feed-chip add-new" onclick="addGlobalCat()">+</div>`;
        });

        const f = $('labeler-filter');
        if (!Array.from(f.options).some(o => o.value === name)) {
            f.innerHTML += `<option value="${name}">${name}</option>`;
        }

        showToast(`Category "${name}" added to all datasets`);
    } catch(e) { showToast(e.message, true); }
};

const onFeedScroll = async () => {
    if (S.feedLoading || S.feedDone) return;
    const feed = $('labeler-feed');
    const nearBottom = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 200;
    if (!nearBottom) return;

    S.feedLoading = true;
    S.feedPage++;

    const loader = document.createElement('div');
    loader.className = 'feed-loader';
    loader.id = 'feed-loader';
    loader.innerHTML = '<div class="loading-spinner"></div>';
    feed.appendChild(loader);

    try {
        const filter = $('labeler-filter').value;
        const r = await api(`/api/dataset/episodes?path=${encodeURIComponent(S.dsPath)}&lf=${filter}&page=${S.feedPage}&per_page=${PER_PAGE}`);
        loader.remove();
        renderFeedCards(r.items);

        const loaded = (S.feedPage - 1) * PER_PAGE + r.items.length;
        if (loaded >= r.total || r.items.length === 0) {
            S.feedDone = true;
            feed.innerHTML += '<div class="feed-end">— End —</div>';
        }
    } catch(e) {
        loader.remove();
        showToast(e.message, true);
    }
    S.feedLoading = false;
};

const labelerFilterChanged = async () => {
    S.feedPage = 1;
    S.feedDone = false;
    S.feedLoading = false;
    const feed = $('labeler-feed');
    feed.innerHTML = '<div class="loading-spinner"></div>';

    try {
        const filter = $('labeler-filter').value;
        const r = await api(`/api/dataset/episodes?path=${encodeURIComponent(S.dsPath)}&lf=${filter}&page=1&per_page=${PER_PAGE}`);
        S.feedTotal = r.total;
        $('labeler-count').textContent = `${r.total} eps`;
        feed.innerHTML = '';
        renderFeedCards(r.items);
        if (r.items.length >= r.total || r.items.length === 0) {
            S.feedDone = true;
            feed.innerHTML += r.items.length ? '<div class="feed-end">— End —</div>' : '<div class="feed-end">No episodes match</div>';
        }
    } catch(e) { showToast(e.message, true); }
};

const closeLabeler = () => {
    $('labeler').classList.add('hidden');
    $('labeler-feed').querySelectorAll('video').forEach(v => v.pause());
    $('labeler-feed').innerHTML = '';
};


// ════════════════════════════════════
// ★ TRAIN PAGE — Pipeline Configuration
// ════════════════════════════════════

const loadTrainPage = async () => {
    // Load GPU data, pipeline defaults, and datasets in parallel
    const [gpuRes, defaults, dsRes] = await Promise.all([
        apiQ('/api/pipeline/gpu'),
        apiQ('/api/pipeline/defaults'),
        apiQ('/api/pipeline/datasets'),
    ]);

    // GPU grid — render and auto-select free GPUs on first load
    if (gpuRes && gpuRes.gpus) {
        S.pipeGpuData = gpuRes.gpus;
        if (S.pipeSelectedGPUs.size === 0) {
            // Auto-select GPUs with >30GB free on first visit
            S.pipeGpuData.forEach(g => {
                if (g.memory_free_mb > 30000) S.pipeSelectedGPUs.add(g.index);
            });
        }
        renderPipeGpuGrid();
    }

    // Populate config_name dropdown
    if (defaults && defaults.config_names) {
        const sel = $('pipe-config-name');
        sel.innerHTML = '';
        defaults.config_names.forEach(n => {
            const opt = document.createElement('option');
            opt.value = n;
            opt.textContent = n;
            if (n === defaults.config_name) opt.selected = true;
            sel.appendChild(opt);
        });
    }

    // Datasets — populate native <select>
    if (dsRes && dsRes.datasets) {
        S.pipeDatasets = dsRes.datasets;
        renderPipeDatasetSelect();
    }

    // Auto-fill experiment name if empty
    if (!$('pipe-exp-name').value.trim()) {
        pipeAutoExpName();
    }

    // Check if pipeline is running and update UI
    const st = await apiQ('/api/pipeline/status');
    if (st && st.running) {
        S.pipeRunning = true;
        $('pipe-btn-start').disabled = true;
        $('pipe-btn-start').textContent = '⏳ Pipeline Running...';
    } else {
        S.pipeRunning = false;
        $('pipe-btn-start').disabled = false;
        $('pipe-btn-start').textContent = '🚀 Start Training Pipeline';
    }
};

const renderPipeGpuGrid = () => {
    const grid = $('pipe-gpu-grid');
    grid.innerHTML = '';
    S.pipeGpuData.forEach(g => {
        const pct = g.memory_total_mb > 0 ? (g.memory_used_mb / g.memory_total_mb * 100) : 0;
        const freeMB = g.memory_free_mb;
        const freeGB = (freeMB / 1024).toFixed(1);
        let color = 'green';
        if (freeMB < 10240) color = 'red';
        else if (freeMB < 30720) color = 'yellow';

        const card = document.createElement('div');
        card.className = 'gpu-sel-card' + (S.pipeSelectedGPUs.has(g.index) ? ' selected' : '');
        card.onclick = () => togglePipeGPU(g.index);
        card.innerHTML = `
            <div class="gpu-idx">GPU ${g.index}</div>
            <div class="gpu-name">${g.name}</div>
            <div class="gpu-mem">${(g.memory_used_mb/1024).toFixed(1)} / ${(g.memory_total_mb/1024).toFixed(1)} GB</div>
            <div class="mem-bar"><div class="mem-bar-fill ${color}" style="width:${pct}%"></div></div>
            <div class="gpu-free ${color}">Free: ${freeGB} GB</div>`;
        grid.appendChild(card);
    });
    updatePipeGpuText();
};

const togglePipeGPU = (idx) => {
    if (S.pipeRunning) return;
    if (S.pipeSelectedGPUs.has(idx)) S.pipeSelectedGPUs.delete(idx);
    else S.pipeSelectedGPUs.add(idx);
    renderPipeGpuGrid();
};

const updatePipeGpuText = () => {
    const arr = [...S.pipeSelectedGPUs].sort((a,b) => a-b);
    $('pipe-selected-text').textContent = arr.length ? arr.map(i => `GPU ${i}`).join(', ') : 'none (click to select)';
    if (!S.pipeRunning) {
        const fsdpEl = $('pipe-fsdp-devices');
        if (fsdpEl) fsdpEl.value = arr.length || 1;
    }
};

const autoSelectFreeGPUs = () => {
    S.pipeSelectedGPUs.clear();
    S.pipeGpuData.forEach(g => {
        if (g.memory_free_mb > 30000) S.pipeSelectedGPUs.add(g.index);
    });
    renderPipeGpuGrid();
};

const renderPipeDatasetSelect = () => {
    const sel = $('pipe-dataset-select');
    sel.innerHTML = '';
    if (S.pipeDatasets.length === 0) {
        sel.innerHTML = '<option value="" disabled selected>No datasets found</option>';
        return;
    }
    S.pipeDatasets.forEach((ds, i) => {
        const opt = document.createElement('option');
        opt.value = ds.path;
        opt.textContent = `${ds.name}  (${ds.total_episodes} eps, ${ds.total_frames} frames)`;
        if (i === 0) opt.selected = true;
        sel.appendChild(opt);
    });
    // Show meta for selected dataset
    updatePipeDsMeta();
};

const updatePipeDsMeta = () => {
    const sel = $('pipe-dataset-select');
    const path = sel.value;
    const metaEl = $('pipe-ds-meta');
    const ds = S.pipeDatasets.find(d => d.path === path);
    if (ds) {
        const taskStr = ds.tasks && ds.tasks.length > 0 ? ds.tasks.join(', ') : 'none';
        metaEl.innerHTML = `<span class="pipe-hint">${ds.path}<br>Tasks: ${taskStr}</span>`;
    } else {
        metaEl.innerHTML = '';
    }
};

// When dataset changes, update meta and auto-fill experiment name
const onDatasetSelectChange = () => {
    updatePipeDsMeta();
    pipeAutoExpName();
};

const togglePipeAdvanced = () => {
    const panel = $('pipe-adv-panel');
    const arrow = $('pipe-adv-arrow');
    panel.classList.toggle('open');
    arrow.textContent = panel.classList.contains('open') ? '▼' : '▶';
};

const pipeAutoExpName = () => {
    const sel = $('pipe-dataset-select');
    const dsPath = sel ? sel.value : '';
    const d = new Date();
    const dateStr = String(d.getFullYear()).slice(2) + String(d.getMonth()+1).padStart(2, '0') + String(d.getDate()).padStart(2, '0');
    if (!dsPath) {
        $('pipe-exp-name').value = `evorl_pi05_lora_exp_${dateStr}`;
        return;
    }
    const parts = dsPath.replace(/\/+$/, '').split('/');
    const shortName = parts[parts.length - 1].replace(/[^a-zA-Z0-9_]/g, '_').substring(0, 30);
    $('pipe-exp-name').value = `evorl_pi05_lora_${shortName}_${dateStr}`;
};


// ════════════════════════════════════
// ★ PIPELINE CONTROL — Start / Status / Cancel
// ════════════════════════════════════

const startPipeline = async () => {
    const dsPath = $('pipe-dataset-select').value;
    const expName = $('pipe-exp-name').value.trim();
    if (!dsPath) { showToast('Please select a dataset', true); return; }
    if (!expName) { showToast('Please specify experiment name', true); return; }
    if (S.pipeSelectedGPUs.size === 0) { showToast('Please select at least one GPU', true); return; }

    const body = {
        dataset_path: dsPath,
        exp_name: expName,
        config_name: $('pipe-config-name').value,
        gpu_indices: [...S.pipeSelectedGPUs].sort((a,b) => a-b),
        batch_size: parseInt($('pipe-batch-size').value) || 64,
        fsdp_devices: parseInt($('pipe-fsdp-devices').value) || S.pipeSelectedGPUs.size,
        num_train_steps: parseInt($('pipe-num-train-steps').value) || 20000,
        save_interval: parseInt($('pipe-save-interval').value) || 1000,
        min_range: parseFloat($('pipe-min-range').value) || 0.1,
        resume: $('pipe-resume').checked,
        overwrite: $('pipe-overwrite').checked,
        wandb_enabled: $('pipe-wandb').checked,
        skip_norm_stats: $('pipe-skip-norm').checked,
        skip_postprocess: $('pipe-skip-post').checked,
    };

    try {
        const r = await api('/api/pipeline/start', 'POST', body);
        if (!r.ok) { showToast(r.error || 'Failed to start', true); return; }
        S.pipeRunning = true;
        $('pipe-btn-start').disabled = true;
        $('pipe-btn-start').textContent = '⏳ Pipeline Running...';
        showToast('Pipeline started!');
        // Auto-switch to Task page
        switchPage('task', 'Task');
    } catch(e) {
        showToast(e.message || 'Network error', true);
    }
};

const cancelPipeline = async () => {
    if (!confirm('Cancel the running pipeline?')) return;
    try {
        await api('/api/pipeline/cancel', 'POST');
        showToast('Cancel requested');
    } catch(e) { showToast(e.message, true); }
};

const copyKillCmd = () => {
    const text = $('pipe-kill-text').textContent;
    navigator.clipboard.writeText(text).then(() => {
        showToast('Copied!');
    });
};

// Pipeline status polling
let _pipeInterval = null;

const startPipePoll = () => {
    if (_pipeInterval) return;
    _pipeInterval = setInterval(pollPipelineStatus, 2000);
};

const stopPipePoll = () => {
    if (_pipeInterval) { clearInterval(_pipeInterval); _pipeInterval = null; }
};

const pollPipelineStatus = async () => {
    const d = await apiQ('/api/pipeline/status');
    if (!d) return;

    S.pipeRunning = d.running;
    $('pipe-btn-cancel').disabled = !d.running;

    // Update start button on train page
    const startBtn = $('pipe-btn-start');
    if (startBtn) {
        startBtn.disabled = d.running;
        startBtn.textContent = d.running ? '⏳ Pipeline Running...' : '🚀 Start Training Pipeline';
    }

    // Step badges & logs
    PIPE_STEPS.forEach((name, i) => {
        const st = d.steps[name];
        if (!st) return;

        // Badge
        const badge = $('psb-' + name);
        if (badge) {
            badge.className = 'pipe-step-badge ' + st.status;
            badge.textContent = st.status;
        }

        // Step number circle
        const numEl = $('psn-' + i);
        if (numEl) {
            numEl.className = 'pipe-step-num';
            if (st.status === 'running') numEl.classList.add('active');
            else if (st.status === 'completed') numEl.classList.add('done');
            else if (st.status === 'failed') numEl.classList.add('fail');
            else if (st.status === 'skipped' || st.status === 'cancelled') numEl.classList.add('skip');
        }

        // Logs
        const logEl = $('psl-' + name);
        if (logEl && st.logs !== logEl._lastLogs) {
            logEl.textContent = st.logs;
            logEl._lastLogs = st.logs;
            if (st.status === 'running') logEl.scrollTop = logEl.scrollHeight;
        }
    });

    // WandB
    const wDiv = $('pipe-wandb-link');
    if (d.wandb_url) {
        wDiv.style.display = 'block';
        const a = $('pipe-wandb-url');
        a.href = d.wandb_url;
        a.textContent = d.wandb_url;
    } else {
        wDiv.style.display = 'none';
    }

    // Kill command
    const killBar = $('pipe-kill-bar');
    if (d.running && d.kill_cmd) {
        killBar.style.display = 'flex';
        $('pipe-kill-text').textContent = d.kill_cmd;
    } else {
        killBar.style.display = 'none';
    }
};


// ════════════════════════════════════
// Legacy Jobs (template-based)
// ════════════════════════════════════

const loadJobs=async()=>{const el=$('job-list');try{const j=await api('/api/jobs/list');el.innerHTML=j.map(j=>`<div class="list-item" onclick="openLogs('${j.job_id}','${j.status}')"><div class="item-icon">⚡</div><div class="item-details"><div class="item-name">${j.template} <span class="badge bg-${j.status}">${j.status}</span></div><div class="item-meta">${j.job_id} • ${fmtDate(j.start_time)}</div></div></div>`).join('')||'<div style="padding:14px;text-align:center;color:var(--t2)">No jobs</div>';}catch(e){}};
const openLogs=async(id,st)=>{S.currentJobId=id;$('log-title').textContent=`Job: ${id}`;const b=$('log-status-badge');b.textContent=st;b.className=`badge bg-${st}`;$('stop-job-btn').classList.toggle('hidden',st!=='running');$('log-container').textContent='Loading...\n';$('log-viewer').classList.remove('hidden');pollLogs(id)};
const pollLogs=async id=>{if(S.logPolling)clearInterval(S.logPolling);let off=0;const term=$('log-container');const fn=async()=>{if(S.currentJobId!==id)return;try{const d=await api(`/api/jobs/log?job_id=${id}&offset=${off}`);if(d.lines&&d.lines.length>0){if(off===0)term.textContent='';term.textContent+=d.lines.join('');term.scrollTop=term.scrollHeight;off=d.total_size;}if(Math.random()<.2){const s=await api(`/api/jobs/status?job_id=${id}`);const b=$('log-status-badge');b.textContent=s.status;b.className=`badge bg-${s.status}`;$('stop-job-btn').classList.toggle('hidden',s.status!=='running');if(s.status!=='running'&&S.logPolling)clearInterval(S.logPolling);}}catch(e){}};await fn();S.logPolling=setInterval(fn,2000)};


// ════════════════════════════════════
// Events
// ════════════════════════════════════
$('login-form').addEventListener('submit', login);
$('logout-btn').addEventListener('click', logout);
$('refresh-page-btn').addEventListener('click', ()=>{const a=document.querySelector('.page.active');if(a)loadPage(a.id)});
$('status-btn').addEventListener('click', toggleStatus);
$('status-panel-close').addEventListener('click', closeStatus);
$('status-panel-backdrop').addEventListener('click', closeStatus);
document.querySelectorAll('.nav-item').forEach(b=>b.addEventListener('click',()=>switchPage(b.dataset.target,b.querySelector('.label').textContent)));
document.querySelectorAll('.close-overlay').forEach(b=>b.addEventListener('click',e=>{e.target.closest('.overlay').classList.add('hidden');if(e.target.closest('#log-viewer')){if(S.logPolling)clearInterval(S.logPolling);S.currentJobId=null;}}));

// Labeler
$('labeler-close').addEventListener('click', closeLabeler);
$('labeler-filter').addEventListener('change', labelerFilterChanged);
$('labeler-feed').addEventListener('scroll', onFeedScroll);

// Dataset select change
const pipeSel = $('pipe-dataset-select');
if (pipeSel) {
    pipeSel.addEventListener('change', onDatasetSelectChange);
}

// Stop job button
$('stop-job-btn').addEventListener('click',async()=>{if(!S.currentJobId||!confirm('Stop?'))return;try{await api(`/api/jobs/stop?job_id=${S.currentJobId}`,'POST');showToast('Stopped')}catch(e){}});
$('btn-run-ops').addEventListener('click',async()=>{const o=document.getElementById('op-output')?.value;const items=[];if(S.opMode==='merge'){document.querySelectorAll('#source-checklist input:checked').forEach(i=>items.push(i.value));if(!items.length)return alert('Select sources')}try{await api('/api/dataset/ops/run','POST',{mode:S.opMode,output_root:o,dataset_root:S.dsPath,label_csv:S.dsPath+'/task_labels.csv',selected_sources:items});$('data-ops-overlay').classList.add('hidden');showToast('Started')}catch(e){showToast(e.message,true)}});

// Start pipeline polling when on task page
const _origSwitchPage = switchPage;

// Init
window.addEventListener('DOMContentLoaded',()=>{
    if(document.cookie.includes('token=')&&!S.token){
        S.token='dummy';switchView('app-view');switchPage('datasets','Datasets');startGpu();
    }
    // Start pipeline polling globally (lightweight)
    startPipePoll();
});

// Also refresh GPU on pipeline page every 10s
setInterval(async () => {
    const trainPage = $('train');
    if (trainPage && trainPage.classList.contains('active')) {
        const gpuRes = await apiQ('/api/pipeline/gpu');
        if (gpuRes && gpuRes.gpus) {
            S.pipeGpuData = gpuRes.gpus;
            renderPipeGpuGrid();
        }
    }
}, 10000);
