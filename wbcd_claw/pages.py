"""Mobile-first SPA: inline HTML/CSS/JS for WBCDClaw MVP."""

PAGE = r"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'/>
<meta name='viewport' content='width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no'/>
<title>WBCDClaw</title>
<style>
:root{
  --bg:#0f0f17;--bg2:#181825;--bg3:#1e1e2e;--fg:#cdd6f4;--fg2:#a6adc8;
  --accent:#89b4fa;--green:#a6e3a1;--red:#f38ba8;--yellow:#f9e2af;--surface:#313244;
  --radius:10px;
}
*{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
body{background:var(--bg);color:var(--fg);font-family:system-ui,-apple-system,sans-serif;
  font-size:15px;overflow:hidden;height:100dvh;display:flex;flex-direction:column}
button{font-family:inherit;cursor:pointer;border:none;border-radius:var(--radius);
  padding:10px 16px;font-size:14px;background:var(--surface);color:var(--fg)}
button:active{opacity:.75}
input,select,textarea{font-family:inherit;font-size:15px;background:var(--bg2);
  color:var(--fg);border:1px solid var(--surface);border-radius:var(--radius);padding:10px}

/* --- layout --- */
.tabs{display:flex;background:var(--bg2);border-top:1px solid var(--surface);
  padding:4px 0 env(safe-area-inset-bottom);flex-shrink:0}
.tabs button{flex:1;background:transparent;color:var(--fg2);padding:10px 0;
  font-size:13px;border-radius:0;border-top:2px solid transparent}
.tabs button.active{color:var(--accent);border-top-color:var(--accent)}
.page{flex:1;overflow-y:auto;display:none;padding:12px;-webkit-overflow-scrolling:touch}
.page.active{display:block}

/* --- dataset selector --- */
.ds-bar{display:flex;gap:6px;overflow-x:auto;margin-bottom:12px;padding-bottom:4px;flex-shrink:0}
.ds-bar button{white-space:nowrap;padding:8px 14px;font-size:13px;font-weight:600;
  border:2px solid var(--surface);flex-shrink:0}
.ds-bar button.active{border-color:var(--accent);background:var(--accent);color:#000}

/* --- dashboard --- */
.stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:16px}
.stat{background:var(--bg3);border-radius:var(--radius);padding:14px;text-align:center}
.stat .num{font-size:28px;font-weight:700;margin-bottom:2px}
.stat .lbl{font-size:12px;color:var(--fg2)}
.dash-btn{display:block;width:100%;padding:14px;margin-bottom:10px;
  background:var(--accent);color:#000;font-size:16px;font-weight:600;text-align:center}

/* --- sample list --- */
.filter-bar{display:flex;gap:8px;margin-bottom:10px}
.filter-bar input{flex:1;min-width:0}
.filter-bar select{width:120px}
.ep-item{background:var(--bg3);border-radius:var(--radius);padding:12px;
  margin-bottom:8px;display:flex;justify-content:space-between;align-items:center}
.ep-item .ep-id{font-weight:600;font-size:14px}
.ep-item .ep-tag{font-size:12px;padding:3px 8px;border-radius:6px;background:var(--surface);color:var(--fg2)}
.ep-item .ep-tag.labeled{background:#2a4a2e;color:var(--green)}

/* --- sample detail --- */
#detail_view{display:none;position:fixed;top:0;left:0;right:0;bottom:0;
  background:var(--bg);z-index:100;flex-direction:column}
#detail_view.open{display:flex}
.detail-top{flex:1;overflow-y:auto;padding:12px;padding-bottom:140px}
.detail-header{display:flex;align-items:center;gap:8px;margin-bottom:10px}
.detail-header .back-btn{font-size:20px;padding:8px 12px}
.detail-header .ep-title{font-size:16px;font-weight:600;flex:1}
.detail-header .ep-counter{font-size:13px;color:var(--fg2)}
.video-area{margin-bottom:10px}
.video-area video{width:100%;border-radius:var(--radius);max-height:280px;background:#000}
.video-area .cam-label{font-size:12px;color:var(--fg2);margin-bottom:4px}
.meta-row{font-size:13px;color:var(--fg2);margin-bottom:6px}
.note-area{margin-bottom:10px}
.note-area textarea{width:100%;height:60px}

/* --- bottom action bar --- */
.action-bar{position:fixed;bottom:0;left:0;right:0;background:var(--bg2);
  border-top:1px solid var(--surface);padding:8px 12px env(safe-area-inset-bottom);z-index:101}
.label-btns{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px}
.label-btns button{padding:10px 14px;font-size:14px;font-weight:600;min-width:60px}
.label-btns button.sel{background:var(--accent);color:#000}
.nav-row{display:flex;gap:8px}
.nav-row button{flex:1;padding:12px;font-size:14px}
.nav-row .save-btn{background:var(--green);color:#000;font-weight:700;flex:2}

/* --- training --- */
.cfg-card{background:var(--bg3);border-radius:var(--radius);padding:12px;margin-bottom:10px}
.cfg-card .cfg-name{font-weight:600;font-size:14px;margin-bottom:4px}
.cfg-card .cfg-meta{font-size:12px;color:var(--fg2)}
.launch-row{display:flex;gap:8px;margin-bottom:16px}
.launch-row input{flex:1}
.launch-row button{background:var(--green);color:#000;font-weight:700;padding:12px 20px}
.task-card{background:var(--bg3);border-radius:var(--radius);padding:12px;margin-bottom:8px}
.task-card .task-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px}
.task-card .task-name{font-weight:600;font-size:14px}
.task-status{font-size:12px;padding:3px 8px;border-radius:6px}
.task-status.running{background:#2d3a5e;color:var(--accent)}
.task-status.completed{background:#2a4a2e;color:var(--green)}
.task-status.failed{background:#4a2a2e;color:var(--red)}
.task-status.killed{background:#4a3a1e;color:var(--yellow)}
.task-meta{font-size:12px;color:var(--fg2);margin-bottom:4px}
.log-box{background:var(--bg);border:1px solid var(--surface);border-radius:var(--radius);
  padding:8px;max-height:300px;overflow-y:auto;font-family:'Courier New',monospace;
  font-size:11px;line-height:1.5;white-space:pre-wrap;word-break:break-all;display:none;margin-top:8px}
.log-box.open{display:block}
.toast{position:fixed;top:20px;left:50%;transform:translateX(-50%);background:var(--green);
  color:#000;padding:8px 20px;border-radius:var(--radius);font-size:14px;font-weight:600;
  z-index:200;opacity:0;transition:opacity .3s}
.toast.show{opacity:1}
</style></head><body>

<!-- pages -->
<div id='page_dash' class='page active'>
  <h2 style='margin-bottom:14px;font-size:18px'>WBCDClaw</h2>
  <div id='ds_bar_dash' class='ds-bar'></div>
  <div class='stat-grid'>
    <div class='stat'><div class='num' id='d_total'>-</div><div class='lbl'>Total</div></div>
    <div class='stat'><div class='num' id='d_labeled'>-</div><div class='lbl'>Labeled</div></div>
    <div class='stat'><div class='num' id='d_unlabeled'>-</div><div class='lbl'>Unlabeled</div></div>
    <div class='stat'><div class='num' id='d_running'>-</div><div class='lbl'>Training</div></div>
  </div>
  <button class='dash-btn' onclick='resumeReview()' id='d_resume_btn'>Continue Review</button>
  <button class='dash-btn' onclick='switchTab("train")' style='background:var(--surface);color:var(--fg)'>Launch Training</button>
</div>

<div id='page_samples' class='page'>
  <div id='ds_bar_samples' class='ds-bar'></div>
  <div class='filter-bar'>
    <input id='s_search' placeholder='Search episode...' oninput='loadSampleList()'/>
    <select id='s_filter' onchange='loadSampleList()'></select>
  </div>
  <div id='s_list'></div>
</div>

<div id='page_train' class='page'>
  <h3 style='margin-bottom:10px;font-size:16px'>Config Templates</h3>
  <div id='t_configs'></div>
  <div class='launch-row'>
    <input id='t_exp_name' placeholder='Experiment name'/>
    <button onclick='launchTrain()'>Launch</button>
  </div>
  <h3 style='margin:16px 0 10px;font-size:16px'>Tasks</h3>
  <div id='t_tasks'></div>
</div>

<!-- sample detail overlay -->
<div id='detail_view'>
  <div class='detail-top'>
    <div class='detail-header'>
      <button class='back-btn' onclick='closeDetail()'>&larr;</button>
      <span class='ep-title' id='det_title'></span>
      <span class='ep-counter' id='det_counter'></span>
    </div>
    <div id='det_videos' class='video-area'></div>
    <div id='det_meta' class='meta-row'></div>
    <div class='note-area'><textarea id='det_note' placeholder='Notes...'></textarea></div>
  </div>
  <div class='action-bar'>
    <div class='label-btns' id='det_labels'></div>
    <div class='nav-row'>
      <button onclick='navEp(-1)'>Prev</button>
      <button class='save-btn' onclick='saveLabel()'>Save & Next</button>
      <button onclick='navEp(1)'>Next</button>
    </div>
  </div>
</div>

<!-- tab bar -->
<div class='tabs'>
  <button class='active' onclick='switchTab("dash")'>Home</button>
  <button onclick='switchTab("samples")'>Samples</button>
  <button onclick='switchTab("train")'>Training</button>
</div>

<div class='toast' id='toast'></div>

<script>
const API='/api';
let curTab='dash';
let allDatasets=[];
let curDs='';
let sampleList=[];
let categories=[];
let curEpIdx=-1;
let curLabel='';
let selectedConfig=null;
let trainPollTimer=null;
let expandedTask=null;

async function j(url,opts){const r=await fetch(url,opts);return r.json();}
function dsQ(){return curDs?'ds='+encodeURIComponent(curDs):'';}

function showToast(msg,ms){
  const t=document.getElementById('toast');t.innerText=msg;
  t.classList.add('show');setTimeout(()=>t.classList.remove('show'),ms||1500);
}

// ---- dataset bar ----
function renderDsBar(containerId){
  const box=document.getElementById(containerId);
  if(allDatasets.length<=1){box.style.display='none';return;}
  box.style.display='flex';
  box.innerHTML=allDatasets.map(d=>{
    const info=d.labeled_count+'/'+d.episode_count;
    return `<button class='${d.name===curDs?"active":""}' onclick='selectDs("${d.name}")'>${d.name} <span style="opacity:.7;font-size:11px">(${info})</span></button>`;
  }).join('');
}

function selectDs(name){
  curDs=name;
  renderDsBar('ds_bar_dash');
  renderDsBar('ds_bar_samples');
  if(curTab==='dash') loadDashboard(true);
  if(curTab==='samples') loadSampleList();
}

// ---- tabs ----
function switchTab(name){
  curTab=name;
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.getElementById('page_'+name).classList.add('active');
  document.querySelectorAll('.tabs button').forEach((b,i)=>{
    b.classList.toggle('active',['dash','samples','train'][i]===name);
  });
  if(name==='dash') loadDashboard();
  if(name==='samples') loadSampleList();
  if(name==='train') loadTrainPage();
}

// ---- dashboard ----
async function loadDashboard(skipDs){
  if(!skipDs){
    const dsData=await j(API+'/samples/datasets');
    allDatasets=dsData.datasets||[];
    if(!curDs&&allDatasets.length) curDs=allDatasets[0].name;
    renderDsBar('ds_bar_dash');
    renderDsBar('ds_bar_samples');
  }
  const meta=await j(API+'/samples/meta?'+dsQ());
  document.getElementById('d_total').innerText=meta.episode_count;
  document.getElementById('d_labeled').innerText=meta.labeled_count;
  document.getElementById('d_unlabeled').innerText=meta.unlabeled_count;
  categories=meta.categories||[];
  const tr=await j(API+'/train/tasks?limit=10');
  const running=(tr.tasks||[]).filter(t=>t.status==='running').length;
  document.getElementById('d_running').innerText=running;
  const prog=await j(API+'/samples/progress?'+dsQ());
  const btn=document.getElementById('d_resume_btn');
  if(prog.last_episode_id){
    btn.innerText='Continue: '+prog.last_episode_id;
    btn.dataset.ep=prog.last_episode_id;
  } else {
    btn.innerText='Start Review';
    btn.dataset.ep='';
  }
}

function resumeReview(){
  const ep=document.getElementById('d_resume_btn').dataset.ep;
  switchTab('samples');
  if(ep){
    setTimeout(()=>{
      const idx=sampleList.findIndex(s=>s.episode_id===ep);
      if(idx>=0) openDetail(idx);
    },300);
  }
}

// ---- sample list ----
async function loadSampleList(){
  const q=document.getElementById('s_search').value.trim();
  const lf=document.getElementById('s_filter').value||'all';
  const data=await j(API+'/samples/list?'+dsQ()+'&q='+encodeURIComponent(q)+'&lf='+encodeURIComponent(lf));
  sampleList=data.items||[];
  const meta=await j(API+'/samples/meta?'+dsQ());
  categories=meta.categories||[];
  renderFilterSelect();
  const box=document.getElementById('s_list');
  if(!sampleList.length){box.innerHTML='<div style="text-align:center;color:var(--fg2);padding:40px">No episodes found</div>';return;}
  box.innerHTML=sampleList.map((s,i)=>`
    <div class='ep-item' onclick='openDetail(${i})'>
      <div><div class='ep-id'>${s.episode_id}</div><div style='font-size:12px;color:var(--fg2)'>${s.camera_count} cameras</div></div>
      <div class='ep-tag ${s.label?"labeled":""}'>${s.label||'unlabeled'}</div>
    </div>`).join('');
}

function renderFilterSelect(){
  const sel=document.getElementById('s_filter');
  const prev=sel.value||'all';
  sel.innerHTML='<option value="all">All</option><option value="unlabeled">Unlabeled</option>';
  categories.forEach(c=>{sel.innerHTML+=`<option value="${c}">${c}</option>`;});
  sel.value=[...sel.options].some(o=>o.value===prev)?prev:'all';
}

// ---- sample detail ----
async function openDetail(idx){
  curEpIdx=idx;
  const ep=sampleList[idx];
  if(!ep) return;
  const data=await j(API+'/samples/episode/'+ep.episode_id+'?'+dsQ());
  if(!data.ok) return;

  document.getElementById('det_title').innerText=data.episode_id;
  document.getElementById('det_counter').innerText=`${idx+1}/${sampleList.length}`;
  curLabel=data.label||'';

  const vbox=document.getElementById('det_videos');
  vbox.innerHTML='';
  const sorted=data.videos.sort((a,b)=>{
    const order=['left_wrist','right_wrist','head'];
    const ai=order.findIndex(k=>a.camera.includes(k));
    const bi=order.findIndex(k=>b.camera.includes(k));
    return (ai<0?99:ai)-(bi<0?99:bi);
  });
  sorted.forEach((v,vi)=>{
    const auto=vi===0;
    vbox.innerHTML+=`<div class='cam-label'>${v.camera}${auto?' (auto)':''}</div>
      <video controls ${auto?'autoplay':''} muted playsinline preload='${auto?"metadata":"none"}' src='${v.url}'></video>`;
  });

  document.getElementById('det_meta').innerText=`Cameras: ${data.videos.length} | Label: ${curLabel||'none'}`;
  document.getElementById('det_note').value=data.note||'';
  renderLabelBtns();
  document.getElementById('detail_view').classList.add('open');

  j(API+'/samples/progress?'+dsQ(),{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({last_episode_id:ep.episode_id,filter_mode:document.getElementById('s_filter').value||'all'})});
}

function renderLabelBtns(){
  const box=document.getElementById('det_labels');
  box.innerHTML='';
  categories.forEach(c=>{
    box.innerHTML+=`<button class='${c===curLabel?"sel":""}' onclick='pickLabel("${c}")'>${c}</button>`;
  });
  box.innerHTML+=`<button onclick='pickLabel("")' style='color:var(--fg2)'>Clear</button>`;
}

function pickLabel(l){
  curLabel=l;
  renderLabelBtns();
}

async function saveLabel(){
  const ep=sampleList[curEpIdx];
  if(!ep) return;
  const note=document.getElementById('det_note').value;
  const data=await j(API+'/samples/label?'+dsQ(),{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({episode_id:ep.episode_id,label:curLabel,note:note})});
  if(data.ok){
    categories=data.categories||categories;
    sampleList[curEpIdx].label=curLabel;
    showToast('Saved: '+curLabel);
    const next=curEpIdx+1;
    if(next<sampleList.length){
      setTimeout(()=>openDetail(next),300);
    }
  }
}

function closeDetail(){
  document.getElementById('detail_view').classList.remove('open');
  document.querySelectorAll('#det_videos video').forEach(v=>{v.pause();v.src='';});
  loadSampleList();
}

function navEp(dir){
  document.querySelectorAll('#det_videos video').forEach(v=>{v.pause();v.src='';});
  const next=curEpIdx+dir;
  if(next<0||next>=sampleList.length){showToast(dir<0?'First one':'Last one');return;}
  openDetail(next);
}

// ---- training ----
async function loadTrainPage(){
  const cfgData=await j(API+'/train/configs');
  const box=document.getElementById('t_configs');
  const cfgs=cfgData.configs||[];
  box.innerHTML='';
  cfgs.forEach(c=>{
    const active=selectedConfig===c.file;
    box.innerHTML+=`<div class='cfg-card' style='${active?"border:2px solid var(--accent)":"border:2px solid transparent"};cursor:pointer'
      onclick='selectConfig("${c.file}","${c.exp_name||""}","${c.config_name}")'>
      <div class='cfg-name'>${c.config_name}</div>
      <div class='cfg-meta'>${c.file} | ${c.backend}</div>
    </div>`;
  });
  if(cfgs.length&&!selectedConfig){
    selectConfig(cfgs[0].file, cfgs[0].exp_name||'', cfgs[0].config_name);
  }
  await loadTasks();
  startTaskPoll();
}

function selectConfig(file,expName,configName){
  selectedConfig=file;
  document.getElementById('t_exp_name').value=expName;
  document.querySelectorAll('.cfg-card').forEach(el=>{
    el.style.border=el.querySelector('.cfg-name').innerText===configName?
      '2px solid var(--accent)':'2px solid transparent';
  });
}

async function launchTrain(){
  if(!selectedConfig){showToast('Select a config first');return;}
  const expName=document.getElementById('t_exp_name').value.trim();
  if(!expName){showToast('Enter experiment name');return;}
  if(!confirm('Launch training: '+selectedConfig+' / '+expName+'?')) return;
  const data=await j(API+'/train/launch',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({config_file:selectedConfig,exp_name:expName})});
  if(data.ok){showToast('Launched: '+data.task_id);}
  else{showToast(data.error||'Launch failed');}
  await loadTasks();
}

async function loadTasks(){
  const data=await j(API+'/train/tasks?limit=20');
  const box=document.getElementById('t_tasks');
  const tasks=data.tasks||[];
  if(!tasks.length){box.innerHTML='<div style="color:var(--fg2);text-align:center;padding:20px">No tasks yet</div>';return;}
  box.innerHTML=tasks.map(t=>`
    <div class='task-card' id='task_${t.task_id}'>
      <div class='task-header'>
        <span class='task-name'>${t.exp_name}</span>
        <span class='task-status ${t.status}'>${t.status}</span>
      </div>
      <div class='task-meta'>${t.config_name} | ${t.created_at}</div>
      <div style='display:flex;gap:6px;margin-top:6px'>
        <button style='font-size:12px;padding:6px 10px' onclick='toggleLogs("${t.task_id}")'>Logs</button>
        ${t.status==='running'?`<button style='font-size:12px;padding:6px 10px;color:var(--red)' onclick='killTask("${t.task_id}")'>Kill</button>`:''}
      </div>
      <div class='log-box ${expandedTask===t.task_id?"open":""}' id='logs_${t.task_id}'></div>
    </div>`).join('');
  if(expandedTask) fetchLogs(expandedTask);
}

async function toggleLogs(taskId){
  if(expandedTask===taskId){expandedTask=null;document.getElementById('logs_'+taskId).classList.remove('open');return;}
  expandedTask=taskId;
  document.querySelectorAll('.log-box').forEach(el=>el.classList.remove('open'));
  document.getElementById('logs_'+taskId).classList.add('open');
  await fetchLogs(taskId);
}

async function fetchLogs(taskId){
  const box=document.getElementById('logs_'+taskId);
  if(!box) return;
  const data=await j(API+'/train/tasks/'+taskId+'/logs?lines=80');
  if(data.ok){
    box.innerText=(data.lines||[]).join('\n');
    box.scrollTop=box.scrollHeight;
  }
}

async function killTask(taskId){
  if(!confirm('Kill task '+taskId+'?')) return;
  await j(API+'/train/tasks/'+taskId+'/kill',{method:'POST'});
  showToast('Killed');
  await loadTasks();
}

function startTaskPoll(){
  if(trainPollTimer) clearInterval(trainPollTimer);
  trainPollTimer=setInterval(async()=>{
    if(curTab!=='train') return;
    await loadTasks();
  },8000);
}

// ---- init ----
loadDashboard();
</script></body></html>"""
