#!/usr/bin/env python
from __future__ import annotations

PAGE = r"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'/>
<title>Episode Review - Grid</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:#111;color:#eee;font-family:'Segoe UI',Arial,sans-serif;font-size:14px}

.topbar{position:sticky;top:0;z-index:100;background:#1a1a2e;border-bottom:1px solid #333;padding:8px 16px;display:flex;flex-wrap:wrap;align-items:center;gap:8px}
.topbar .section{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.topbar .divider{width:1px;height:28px;background:#444;margin:0 4px}
input,select,button,textarea{background:#222;color:#eee;border:1px solid #555;padding:5px 8px;border-radius:3px;font-size:13px}
button{cursor:pointer;transition:background .15s} button:hover{background:#333}
.status-msg{font-size:12px;color:#8bc34a;min-width:120px}

.tabs{display:flex;gap:0;border-bottom:1px solid #333;background:#181828}
.tab{padding:8px 20px;cursor:pointer;border-bottom:2px solid transparent;color:#aaa;font-size:14px}
.tab.active{color:#eee;border-bottom-color:#4fc3f7}
.tab:hover{color:#fff}
.tab-content{display:none}
.tab-content.active{display:block}

.grid-container{padding:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px}
.ep-card{border:1px solid #333;border-radius:6px;background:#1a1a1a;overflow:hidden;transition:border-color .2s}
.ep-card.labeled{border-color:#2e7d32}
.ep-card .vid-area{position:relative;background:#000}
.ep-card video{width:100%;display:block;max-height:240px;object-fit:contain}
.ep-card .ep-info{padding:8px 10px;display:flex;flex-direction:column;gap:6px}
.ep-card .ep-title{font-size:13px;font-weight:600;display:flex;justify-content:space-between;align-items:center}
.ep-card .ep-title .label-badge{font-size:11px;padding:2px 8px;border-radius:10px;background:#2e7d32;color:#fff}
.ep-card .ep-title .label-badge.empty{background:#555;color:#999}
.ep-card .label-row{display:flex;flex-wrap:wrap;gap:4px}
.ep-card .label-row button{font-size:12px;padding:3px 10px;border-radius:12px;cursor:pointer;transition:all .15s}
.ep-card .label-row button.active{background:#1565c0;border-color:#1565c0;color:#fff}
.ep-card .label-row button:hover{background:#333}
.ep-card .label-row button.active:hover{background:#1976d2}
.page-controls{display:flex;justify-content:center;align-items:center;gap:12px;padding:16px;font-size:14px}
.page-controls button{padding:6px 16px}
.small{font-size:12px;color:#bbb}

.ds-panel{padding:16px 24px;max-width:1400px}
.ds-panel .info-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px;margin-bottom:16px}
.ds-panel .info-card{background:#1a1a2e;border:1px solid #333;border-radius:6px;padding:12px}
.ds-panel .info-card .title{font-size:12px;color:#888;margin-bottom:4px}
.ds-panel .info-card .value{font-size:15px;font-weight:600;word-break:break-all}
.ds-panel table{width:100%;border-collapse:collapse;margin-top:8px}
.ds-panel th,.ds-panel td{padding:8px 12px;text-align:left;border-bottom:1px solid #333}
.ds-panel th{color:#888;font-size:12px}
.ds-panel .prompt-input{width:100%;background:#1a1a2e;border:1px solid #444;padding:6px 8px;color:#eee;border-radius:3px}
.split-output{margin-top:12px;padding:10px;background:#0d0d1a;border:1px solid #333;border-radius:4px;font-family:monospace;font-size:12px;white-space:pre-wrap;max-height:400px;overflow:auto;color:#aaa}

.mode-selector{display:flex;gap:0;margin-bottom:16px}
.mode-btn{padding:8px 24px;cursor:pointer;border:1px solid #444;background:#1a1a1a;color:#aaa}
.mode-btn:first-child{border-radius:6px 0 0 6px}
.mode-btn:last-child{border-radius:0 6px 6px 0}
.mode-btn.active{background:#1565c0;color:#fff;border-color:#1565c0}

.source-table input[type=checkbox]{width:18px;height:18px;accent-color:#4fc3f7}

.add-ds-panel{margin:12px 0;border:1px dashed #444;border-radius:6px;padding:12px;background:#151520;display:none}
.add-ds-panel.open{display:block}
.add-ds-item{display:flex;align-items:center;gap:10px;padding:8px;border-bottom:1px solid #333}
.add-ds-item:last-child{border-bottom:none}
.add-ds-item .ds-name{font-weight:600;min-width:180px}
.add-ds-item .ds-meta{font-size:12px;color:#888;flex:1}
.add-ds-item button{background:#1565c0;color:#fff;padding:4px 12px;font-size:12px}

.modal-overlay{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.7);z-index:200;justify-content:center;align-items:center}
.modal-overlay.show{display:flex}
.modal{background:#222;border:1px solid #555;border-radius:8px;padding:20px;max-width:90vw;max-height:90vh;overflow:auto;min-width:600px}
.modal h3{margin:0 0 12px 0}
.modal .videos-detail{display:grid;grid-template-columns:repeat(auto-fill,minmax(400px,1fr));gap:10px}
.modal .cam-card{border:1px solid #333;padding:6px;border-radius:4px}
.modal .cam-card .cam-name{font-size:12px;color:#bbb;margin-bottom:4px}
.modal .cam-card video{width:100%;max-height:300px;background:#000}
.panel{border:1px solid #333;padding:8px;margin-top:10px;border-radius:4px}
.warn{color:#ffb74d}
.btn-remove{color:#ff8a80;border-color:#ff5252;font-size:11px;padding:3px 8px}
.btn-remove:hover{background:#3a1a1a}
</style></head><body>

<div class='topbar'>
  <div class='section'>
    <span style='font-weight:bold;font-size:15px'>Episode Review</span>
    <span id='meta_info' class='small'></span>
  </div>
  <div class='divider'></div>
  <div class='section'>
    <input id='q' placeholder='搜索 episode' style='width:140px'/>
    <select id='lf'></select>
    <button onclick='loadGrid()'>刷新</button>
  </div>
  <div class='divider'></div>
  <div class='section'>
    <label class='small'>速度:</label>
    <select id='playback_rate' onchange='setPlaybackRate(this.value)'>
      <option value='1'>1x</option><option value='1.5'>1.5x</option>
      <option value='2'>2x</option><option value='4' selected>4x</option>
      <option value='8'>8x</option>
    </select>
    <button onclick='toggleAllPlay()' title='空格键'>播放/暂停</button>
  </div>
  <div class='divider'></div>
  <div class='section'>
    <input id='new_category' placeholder='新类别' style='width:100px'/>
    <button onclick='addCategory()'>新增</button>
    <select id='delete_category_select'></select>
    <label class='small'><input id='purge_rows' type='checkbox'/> 连带删</label>
    <button onclick='deleteCategory()' style='color:#ff8a80'>删除</button>
  </div>
  <div class='divider'></div>
  <div class='section'>
    <label class='small'>每页:</label>
    <select id='page_size' onchange='changePageSize()'>
      <option value='8'>8</option><option value='12' selected>12</option>
      <option value='20'>20</option><option value='40'>40</option>
    </select>
  </div>
  <div class='status-msg' id='save_status'></div>
</div>

<div class='tabs'>
  <div class='tab active' onclick='switchTab("label")'>标注</div>
  <div class='tab' onclick='switchTab("dataset")'>数据集 & 分割/合并</div>
</div>

<!-- ===== TAB: 标注 ===== -->
<div id='tab_label' class='tab-content active'>
  <div class='grid-container'>
    <div class='grid' id='grid'></div>
    <div class='page-controls'>
      <button onclick='prevPage()'>上一页</button>
      <span id='page_info'>1/1</span>
      <button onclick='nextPage()'>下一页</button>
    </div>
  </div>
</div>

<!-- ===== TAB: 数据集信息 & 分割/合并 ===== -->
<div id='tab_dataset' class='tab-content'>
  <div class='ds-panel'>

    <h2 style='margin-top:0'>当前数据集信息</h2>
    <div class='info-grid' id='ds_info_grid'></div>

    <h3>标注统计（当前数据集）</h3>
    <table id='label_stats_table'>
      <thead><tr><th>类别</th><th>已标注数</th></tr></thead>
      <tbody></tbody>
    </table>

    <hr style='border-color:#333;margin:24px 0'/>

    <h2>分割 / 合并操作</h2>

    <div class='mode-selector'>
      <div class='mode-btn active' onclick='setSplitMode("split")'>单数据集分割</div>
      <div class='mode-btn' onclick='setSplitMode("merge")'>跨数据集合并</div>
    </div>

    <!-- Merge: source selection -->
    <div id='merge_section' style='display:none'>
      <h4>选择要合并的数据集 <span class='small'>(勾选参与合并的数据源)</span></h4>
      <table class='source-table' id='source_table'>
        <thead><tr>
          <th style='width:40px'><input type='checkbox' id='select_all_sources' onchange='toggleAllSources()' checked/></th>
          <th>数据集名</th><th>格式</th><th>Episode 数</th><th>FPS</th>
          <th>各类别标注数</th><th style='width:60px'>操作</th>
        </tr></thead>
        <tbody id='source_body'></tbody>
      </table>
      <div id='merge_summary' style='margin:12px 0' class='small'></div>

      <!-- Dynamic add dataset -->
      <div style='margin:8px 0;display:flex;gap:8px;align-items:center'>
        <button onclick='toggleAddPanel()' id='add_ds_toggle' style='background:#1565c0;color:#fff;padding:6px 16px'>+ 添加数据集</button>
        <span id='scan_root_info' class='small'></span>
      </div>
      <div class='add-ds-panel' id='add_ds_panel'>
        <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px'>
          <span style='font-weight:600'>可添加的数据集</span>
          <button onclick='scanDatasets()' style='font-size:12px;padding:4px 10px'>刷新扫描</button>
        </div>
        <div id='add_ds_list'>点击上方按钮扫描...</div>
      </div>
    </div>

    <div style='border:1px solid #444;border-radius:6px;padding:12px 16px;margin:16px 0;background:#151525'>
      <h4 style='margin-top:0'>类别统一改名 <span class='small'>(跨所有数据集批量修改标注类别名)</span></h4>
      <div style='display:flex;gap:8px;align-items:center;flex-wrap:wrap'>
        <select id='rename_old' style='min-width:200px'></select>
        <span style='color:#888'>-></span>
        <input id='rename_new' placeholder='新类别名' style='width:200px'/>
        <label class='small'><input type='checkbox' id='rename_all_sources' checked/> 应用到所有数据集</label>
        <button onclick='renameCategory()' style='background:#e65100;color:#fff;padding:6px 16px'>改名</button>
      </div>
      <div id='rename_status' class='small' style='margin-top:6px'></div>
    </div>

    <div style='margin:12px 0'>
      <label>输出目录: <span class='small'>(每个勾选的类别会在此目录下创建子文件夹)</span></label><br/>
      <input id='split_output_root' style='width:100%;margin-top:4px' placeholder='/path/to/output_root' oninput='this._userEdited=true'/>
    </div>

    <h4>选择要输出的类别 & Task Prompt <span class='small'>(勾选要包含的类别，填写训练任务描述)</span></h4>
    <table id='task_map_table'>
      <thead><tr>
        <th style='width:40px'><input type='checkbox' id='select_all_labels' checked onchange='toggleAllLabels()'/></th>
        <th>类别</th><th>总 Episode</th><th>Task Prompt</th>
      </tr></thead>
      <tbody id='task_map_body'></tbody>
    </table>

    <div id='output_preview' class='small' style='margin:12px 0;padding:8px;background:#0d0d1a;border:1px solid #333;border-radius:4px'></div>

    <div style='margin-top:12px;display:flex;gap:8px;align-items:center;flex-wrap:wrap'>
      <label class='small'><input type='checkbox' id='split_require_videos' checked/> 仅含完整视频</label>
      <label class='small'><input type='checkbox' id='split_overwrite' checked/> 覆盖已有</label>
      <button onclick='runJob()' id='run_job_btn' style='background:#1565c0;color:#fff;padding:8px 24px;font-size:14px'>一键分割</button>
      <span id='split_status' class='small'></span>
    </div>

    <div class='split-output' id='split_output' style='display:none'></div>
  </div>
</div>

<!-- ===== Detail modal ===== -->
<div class='modal-overlay' id='detail_modal'>
  <div class='modal'>
    <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:12px'>
      <h3 id='modal_title'>详情</h3>
      <button onclick='closeModal()'>关闭 (Esc)</button>
    </div>
    <div id='modal_videos' class='videos-detail'></div>
    <div style='margin-top:10px'>
      <textarea id='modal_note' rows='2' style='width:100%' placeholder='备注'></textarea>
      <button onclick='saveModalNote()' style='margin-top:6px'>保存备注</button>
    </div>
    <div class='panel'>
      <div style='font-weight:bold;margin-bottom:6px'>视频质量检查</div>
      <button onclick='runQualityCheck()' style='margin-bottom:6px'>运行质量检查</button>
      <div id='modal_quality' class='small'>未检查</div>
    </div>
    <div class='panel'>
      <div style='font-weight:bold;margin-bottom:6px'>关节角折线图 (agent_pos)</div>
      <canvas id='joint_chart' width='1200' height='260' style='width:100%;background:#0f0f0f;border:1px solid #333'></canvas>
      <div id='joint_legend' class='small' style='margin-top:6px'></div>
    </div>
  </div>
</div>

<script>
let allEps=[], filteredEps=[], categories=[], playbackRate=4.0;
let page=0, pageSize=12, modalEp=null;
const labelCache={}, jointsCache={}, qualityCache={};
let dsInfo=null, mergeSourcesData=null, splitMode='split';

async function j(u,o){const r=await fetch(u,o);return await r.json();}
function showStatus(msg,ok=true){
  const el=document.getElementById('save_status');
  el.style.color=ok?'#8bc34a':'#ff6b6b'; el.innerText=msg;
  if(ok) setTimeout(()=>{if(el.innerText===msg)el.innerText='';},3000);
}

/* --- Tab switching --- */
function switchTab(name){
  document.querySelectorAll('.tab').forEach((t,i)=>t.classList.toggle('active',
    (name==='label'&&i===0)||(name==='dataset'&&i===1)));
  document.getElementById('tab_label').classList.toggle('active',name==='label');
  document.getElementById('tab_dataset').classList.toggle('active',name==='dataset');
  if(name==='dataset'){loadDatasetInfo();loadMergeSources();}
}

function setSplitMode(mode){
  splitMode=mode;
  document.querySelectorAll('.mode-btn').forEach((b,i)=>b.classList.toggle('active',
    (mode==='split'&&i===0)||(mode==='merge'&&i===1)));
  document.getElementById('merge_section').style.display=mode==='merge'?'block':'none';
  document.getElementById('run_job_btn').innerText=mode==='merge'?'一键合并':'一键分割';
  document.getElementById('split_output_root')._userEdited=false;
  refreshTaskMapFromSources();
}

/* --- Playback --- */
function setPlaybackRate(v){
  playbackRate=Number(v)||1;
  document.getElementById('playback_rate').value=String(playbackRate);
  document.querySelectorAll('video').forEach(vid=>{vid.playbackRate=playbackRate;});
}
function toggleAllPlay(){
  const vids=document.querySelectorAll('.ep-card video');
  if(!vids.length)return;
  const any=[...vids].some(v=>!v.paused);
  vids.forEach(v=>any?v.pause():v.play().catch(()=>{}));
}

/* --- Meta & categories --- */
async function loadMeta(){
  const d=await j('/api/meta');
  categories=d.categories||[];
  document.getElementById('meta_info').innerText=`${d.episode_count}条 | ${categories.join(', ')}`;
  renderTopbarCategories();
}
function renderTopbarCategories(){
  const lf=document.getElementById('lf');
  const prev=lf.value||'all'; lf.innerHTML='';
  [{value:'all',text:'全部'},{value:'unlabeled',text:'未标注'},...categories.map(c=>({value:c,text:c}))].forEach(o=>{
    const op=document.createElement('option');op.value=o.value;op.text=o.text;lf.appendChild(op);
  });
  lf.value=[...lf.options].some(o=>o.value===prev)?prev:'all';
  const ds=document.getElementById('delete_category_select');ds.innerHTML='';
  categories.forEach(c=>{const op=document.createElement('option');op.value=c;op.text=c;ds.appendChild(op);});
}

/* --- Grid --- */
async function loadGrid(){
  const q=encodeURIComponent(document.getElementById('q').value.trim());
  const lf=encodeURIComponent(document.getElementById('lf').value);
  const d=await j(`/api/list?q=${q}&lf=${lf}`);
  allEps=d.items; filteredEps=allEps;
  for(const e of allEps) labelCache[e.episode_id]=e.label||'';
  page=0; renderPage();
}
function changePageSize(){pageSize=Number(document.getElementById('page_size').value)||12;page=0;renderPage();}
function totalPages(){return Math.max(1,Math.ceil(filteredEps.length/pageSize));}
function prevPage(){if(page>0){page--;renderPage();window.scrollTo(0,0);}}
function nextPage(){if(page<totalPages()-1){page++;renderPage();window.scrollTo(0,0);}}

function renderPage(){
  const grid=document.getElementById('grid'); grid.innerHTML='';
  const start=page*pageSize;
  const slice=filteredEps.slice(start,start+pageSize);
  document.getElementById('page_info').innerText=`${page+1}/${totalPages()} (${filteredEps.length}条)`;
  for(const ep of slice){
    const card=document.createElement('div');
    card.className='ep-card'+(ep.label?' labeled':'');
    card.id='card-'+ep.episode_id;
    const cl=labelCache[ep.episode_id]||'';
    card.innerHTML=`
      <div class='vid-area'>
        <video muted playsinline loop preload='metadata' src='${ep.thumb_url||""}'
               ondblclick='openModal("${ep.episode_id}")'></video>
      </div>
      <div class='ep-info'>
        <div class='ep-title'>
          <span>${ep.episode_id} <span class='small'>[${ep.camera_count}cam]</span></span>
          <span class='label-badge ${cl?"":"empty"}' id='badge-${ep.episode_id}'>${cl||'未标注'}</span>
        </div>
        <div class='label-row' id='btns-${ep.episode_id}'>
          ${categories.map(c=>`<button class='${cl===c?"active":""}' onclick='quickLabel("${ep.episode_id}","${c}")'>${c}</button>`).join('')}
          <button class='${cl===""?"active":""}' onclick='quickLabel("${ep.episode_id}","")' style='color:#999'>清空</button>
        </div>
      </div>`;
    grid.appendChild(card);
  }
  requestAnimationFrame(()=>{
    document.querySelectorAll('.ep-card video').forEach(v=>{
      v.playbackRate=playbackRate;
      const obs=new IntersectionObserver(entries=>{
        entries.forEach(e=>{if(e.isIntersecting)v.play().catch(()=>{});else v.pause();});
      },{threshold:0.3});
      obs.observe(v);
    });
  });
}

async function quickLabel(epId,label){
  const d=await j('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({episode_id:epId,label:label,note:''})});
  if(!d.ok){showStatus('保存失败',false);return;}
  labelCache[epId]=label;
  if(d.categories) categories=d.categories;
  const badge=document.getElementById('badge-'+epId);
  if(badge){badge.innerText=label||'未标注';badge.className='label-badge '+(label?'':'empty');}
  const card=document.getElementById('card-'+epId);
  if(card) card.className='ep-card'+(label?' labeled':'');
  const row=document.getElementById('btns-'+epId);
  if(row) row.querySelectorAll('button').forEach(b=>{
    b.className=((label&&b.innerText===label)||(label===''&&b.innerText==='清空'))?'active':'';
  });
  showStatus(`${epId} -> ${label||'(空)'}`);
  for(const e of filteredEps) if(e.episode_id===epId) e.label=label;
}

/* --- Modal --- */
function openModal(epId){
  modalEp=epId;
  document.getElementById('detail_modal').classList.add('show');
  document.getElementById('modal_title').innerText=epId;
  document.getElementById('modal_quality').innerText='未检查';
  loadModalVideos(epId);
  loadAndRenderJoints(epId);
}
async function loadModalVideos(epId){
  const d=await j(`/api/episode/${epId}`);
  if(!d.ok){alert(d.error);return;}
  document.getElementById('modal_note').value=d.note||'';
  const box=document.getElementById('modal_videos');box.innerHTML='';
  for(const v of d.videos){
    const c=document.createElement('div');c.className='cam-card';
    c.innerHTML=`<div class='cam-name'>${v.camera}</div><video controls muted playsinline autoplay loop src='${v.url}'></video>`;
    box.appendChild(c);
  }
  box.querySelectorAll('video').forEach(v=>{v.playbackRate=playbackRate;v.play().catch(()=>{});});
}
function closeModal(){document.getElementById('detail_modal').classList.remove('show');document.getElementById('modal_videos').innerHTML='';modalEp=null;}
async function saveModalNote(){
  if(!modalEp)return;
  const d=await j('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({episode_id:modalEp,label:labelCache[modalEp]||'',note:document.getElementById('modal_note').value})});
  if(d.ok) showStatus(`备注已保存: ${modalEp}`); else showStatus('失败',false);
}
async function runQualityCheck(){
  if(!modalEp)return;
  if(qualityCache[modalEp]){renderQuality(qualityCache[modalEp]);return;}
  document.getElementById('modal_quality').innerText='检查中...';
  const d=await j(`/api/episode/${modalEp}/quality`);
  if(!d.ok){document.getElementById('modal_quality').innerText=d.error||'失败';return;}
  qualityCache[modalEp]=d.checks;
  document.getElementById('modal_quality').innerHTML=d.checks.map(c=>{
    const w=(c.warnings&&c.warnings.length)?` [${c.warnings.join(',')}]`:'';
    return `<div class='${w?"warn":""}'>${c.camera}: ${c.width}x${c.height} fps=${Number(c.fps).toFixed(1)} frames=${c.frame_count}${w}</div>`;
  }).join('');
}

function renderJointChart(payload){
  const canvas=document.getElementById('joint_chart');
  const legend=document.getElementById('joint_legend');
  const ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if(!payload||!payload.ok){legend.innerText='关节数据读取失败';return;}
  const frames=payload.frame_index||[];
  const pos=payload.agent_pos||[];
  const names=payload.joint_names||[];
  if(!frames.length||!pos.length){legend.innerText='关节数据为空';return;}
  const dims=pos[0].length;
  let ymin=Infinity,ymax=-Infinity;
  for(const row of pos){for(let i=0;i<dims;i++){const v=row[i];if(v<ymin)ymin=v;if(v>ymax)ymax=v;}}
  if(ymax===ymin){ymax=ymin+1e-6;}
  const pad=24,w=canvas.width-pad*2,h=canvas.height-pad*2;
  ctx.strokeStyle='#555';ctx.strokeRect(pad,pad,w,h);
  const colors=['#ff5252','#ff9800','#ffeb3b','#4caf50','#00bcd4','#2196f3','#9c27b0','#e91e63','#8bc34a','#03a9f4','#ffc107','#795548','#f44336','#cddc39'];
  for(let d=0;d<dims;d++){
    ctx.beginPath();ctx.strokeStyle=colors[d%colors.length];ctx.lineWidth=1.1;
    for(let i=0;i<pos.length;i++){
      const x=pad+(i/(pos.length-1||1))*w;
      const y=pad+(1-(pos[i][d]-ymin)/(ymax-ymin))*h;
      if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
    }
    ctx.stroke();
  }
  const showNames=[];
  for(let i=0;i<Math.min(dims,14);i++){showNames.push(`${i}:${names[i]||('joint_'+i)}`);}
  legend.innerText=`范围[${ymin.toFixed(3)}, ${ymax.toFixed(3)}] | ${showNames.join(' | ')}`;
}
async function loadAndRenderJoints(ep){
  if(jointsCache[ep]){renderJointChart(jointsCache[ep]);return;}
  const jd=await j(`/api/episode/${ep}/joints`);
  jointsCache[ep]=jd;
  renderJointChart(jd);
}

/* --- Category management --- */
async function addCategory(){
  const input=document.getElementById('new_category');
  const name=(input.value||'').trim();
  if(!name){alert('空');return;}
  const d=await j(`/api/categories?name=${encodeURIComponent(name)}`,{method:'POST'});
  if(!d.ok){alert(d.error||'失败');return;}
  categories=d.categories||categories; input.value='';
  renderTopbarCategories(); renderPage(); await loadMeta();
}
async function deleteCategory(){
  const name=document.getElementById('delete_category_select').value;
  if(!name){alert('先选类别');return;}
  if(!confirm(`删除 "${name}"?`))return;
  const d=await j(`/api/categories/${encodeURIComponent(name)}?purge_labeled_rows=${document.getElementById('purge_rows').checked}`,{method:'DELETE'});
  if(!d.ok){alert(d.error||'失败');return;}
  categories=d.categories||categories;
  renderTopbarCategories(); await loadGrid();
}

/* ==== Rename category ==== */
async function renameCategory(){
  const oldName=document.getElementById('rename_old').value;
  const newName=document.getElementById('rename_new').value.trim();
  if(!oldName){alert('请选择要改名的类别');return;}
  if(!newName){alert('新名称不能为空');return;}
  if(oldName===newName){alert('新旧名称相同');return;}
  const applyAll=document.getElementById('rename_all_sources').checked;
  const scope=applyAll?'所有数据集':'仅当前数据集';
  if(!confirm(`将 "${oldName}" 改名为 "${newName}" (范围: ${scope})？`))return;

  const statusEl=document.getElementById('rename_status');
  statusEl.innerText='改名中...'; statusEl.style.color='#ffb74d';

  const d=await j('/api/rename-category',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({old_name:oldName,new_name:newName,apply_to_all_sources:applyAll})});
  if(!d.ok){statusEl.innerText='失败: '+(d.error||'');statusEl.style.color='#ff6b6b';return;}

  statusEl.style.color='#8bc34a';
  const details=d.details||[];
  const detailStr=details.map(x=>`${x.source}: ${x.affected}条`).join(', ');
  statusEl.innerText=`完成: "${oldName}" -> "${newName}", 共 ${d.total_affected} 条 (${detailStr})`;

  if(d.categories) categories=d.categories;
  renderTopbarCategories();
  document.getElementById('rename_new').value='';
  await loadMeta();
  await loadGrid();
  await loadMergeSources();
  await loadDatasetInfo();
}

/* ==== Dataset Info Tab ==== */
async function loadDatasetInfo(){
  const d=await j('/api/dataset-info'); dsInfo=d;
  const grid=document.getElementById('ds_info_grid'); grid.innerHTML='';
  [{t:'格式',v:d.codebase_version},{t:'路径',v:d.dataset_root},{t:'Episodes(meta)',v:d.total_episodes},
   {t:'有视频',v:d.total_episodes_with_video},{t:'帧数',v:d.total_frames},{t:'FPS',v:d.fps},
   {t:'已标注',v:d.labeled_count},{t:'未标注',v:d.unlabeled_count}].forEach(c=>{
    const div=document.createElement('div');div.className='info-card';
    div.innerHTML=`<div class='title'>${c.t}</div><div class='value'>${c.v}</div>`;
    grid.appendChild(div);
  });
  const tb=document.getElementById('label_stats_table').querySelector('tbody');tb.innerHTML='';
  const ls=d.label_stats||{};let tot=0;
  for(const[l,c]of Object.entries(ls).sort()){
    const tr=document.createElement('tr');tr.innerHTML=`<td>${l}</td><td>${c}</td>`;
    tb.appendChild(tr);tot+=c;
  }
  const tr2=document.createElement('tr');tr2.innerHTML=`<td style='font-weight:bold'>合计</td><td style='font-weight:bold'>${tot}</td>`;
  tb.appendChild(tr2);
}

/* ==== Merge sources ==== */
async function loadMergeSources(){
  const d=await j('/api/merge-sources');
  if(!d.ok)return;
  mergeSourcesData=d;
  const body=document.getElementById('source_body');body.innerHTML='';
  for(const s of d.sources){
    const tr=document.createElement('tr');
    const labelCols=Object.entries(s.label_stats||{}).sort().map(([l,c])=>`${l}:${c}`).join(', ')||'-';
    const isPrimary=s.name===d.primary_name;
    const removeBtn=isPrimary?'<span class="small">(主)</span>'
      :`<button class='btn-remove' onclick='removeSource("${s.name}")'>移除</button>`;
    tr.innerHTML=`
      <td><input type='checkbox' class='src-check' data-name='${s.name}' checked onchange='refreshTaskMapFromSources()'/></td>
      <td><strong>${s.name}</strong><br/><span class='small'>${s.dataset_root}</span></td>
      <td>${s.codebase_version}</td>
      <td>${s.total_episodes}</td>
      <td>${s.fps}</td>
      <td>${labelCols}</td>
      <td>${removeBtn}</td>`;
    body.appendChild(tr);
  }
  if(d.scan_root) document.getElementById('scan_root_info').innerText=`扫描目录: ${d.scan_root}`;
  refreshTaskMapFromSources();
  refreshRenameDropdown();
}

function refreshRenameDropdown(){
  const sel=document.getElementById('rename_old');
  const prev=sel.value;
  sel.innerHTML='';
  if(!mergeSourcesData)return;
  const allLabels=Object.keys(mergeSourcesData.all_labels||{}).sort();
  for(const l of allLabels){
    const op=document.createElement('option');op.value=l;op.text=`${l} (${mergeSourcesData.all_labels[l]}条)`;
    sel.appendChild(op);
  }
  if(allLabels.includes(prev)) sel.value=prev;
}

function toggleAllSources(){
  const checked=document.getElementById('select_all_sources').checked;
  document.querySelectorAll('.src-check').forEach(cb=>{cb.checked=checked;});
  refreshTaskMapFromSources();
}

function getSelectedSourceNames(){
  return [...document.querySelectorAll('.src-check:checked')].map(cb=>cb.dataset.name);
}

function toggleAllLabels(){
  const checked=document.getElementById('select_all_labels').checked;
  document.querySelectorAll('.label-check').forEach(cb=>{cb.checked=checked;});
  updateOutputPreview();
}

function refreshTaskMapFromSources(){
  if(!mergeSourcesData)return;
  const body=document.getElementById('task_map_body');
  const oldPrompts={}, oldChecked={};
  body.querySelectorAll('.prompt-input').forEach(inp=>{
    if(inp.value.trim()) oldPrompts[inp.dataset.label]=inp.value.trim();
  });
  body.querySelectorAll('.label-check').forEach(cb=>{
    oldChecked[cb.dataset.label]=cb.checked;
  });
  body.innerHTML='';

  let labelTotals={};
  if(splitMode==='merge'){
    const selected=new Set(getSelectedSourceNames());
    for(const s of mergeSourcesData.sources){
      if(!selected.has(s.name))continue;
      for(const[l,c]of Object.entries(s.label_stats||{})){
        labelTotals[l]=(labelTotals[l]||0)+c;
      }
    }
    const summary=document.getElementById('merge_summary');
    const totalEp=Object.values(labelTotals).reduce((a,b)=>a+b,0);
    summary.innerText=`已选 ${selected.size} 个数据集, 共 ${totalEp} 条标注 episode`;
  }else{
    labelTotals=dsInfo?.label_stats||{};
  }

  for(const[label,count]of Object.entries(labelTotals).sort()){
    const checked=(label in oldChecked)?oldChecked[label]:true;
    const tr=document.createElement('tr');
    tr.innerHTML=`
      <td><input type='checkbox' class='label-check' data-label='${label}' ${checked?'checked':''} onchange='updateOutputPreview()'/></td>
      <td><strong>${label}</strong></td>
      <td>${count}</td>
      <td><input class='prompt-input' data-label='${label}' value='${oldPrompts[label]||""}' placeholder='e.g. shirt open middle and catch' style='width:100%'/></td>`;
    body.appendChild(tr);
  }
  updateOutputPreview();

  const outEl=document.getElementById('split_output_root');
  if(!outEl._userEdited && mergeSourcesData){
    const dsRoot=mergeSourcesData.sources[0]?.dataset_root||'';
    const parent=dsRoot.replace(/\/[^/]+\/?$/,'');
    outEl.value=splitMode==='merge'?parent+'/merged_by_label':dsRoot+'_split';
  }
}

function updateOutputPreview(){
  const outRoot=document.getElementById('split_output_root').value.trim()||'<输出目录>';
  const checked=[...document.querySelectorAll('.label-check:checked')].map(cb=>cb.dataset.label);
  const preview=document.getElementById('output_preview');
  if(!checked.length){preview.innerText='未选择任何类别';return;}
  let lines=[`输出结构预览 (每个勾选的类别 = 一个独立 LeRobot 数据集):\n`];
  for(const l of checked){
    const prompt=document.querySelector(`.prompt-input[data-label="${l}"]`)?.value.trim();
    lines.push(`${outRoot}/${l}/`);
    lines.push(`  meta/info.json          <- 数据集元信息 (v2.1 格式)`);
    lines.push(`  meta/episodes.jsonl     <- episode 列表 (重新编号 0..N-1)`);
    lines.push(`  meta/tasks.jsonl        <- task: "${prompt||l}"`);
    lines.push(`  data/chunk-000/         <- parquet 数据`);
    lines.push(`  videos/...              <- 视频文件`);
    lines.push('');
  }
  preview.innerText=lines.join('\n');
}

/* ==== Dynamic add/remove dataset ==== */
function toggleAddPanel(){
  const panel=document.getElementById('add_ds_panel');
  panel.classList.toggle('open');
  if(panel.classList.contains('open')) scanDatasets();
}

async function scanDatasets(){
  const list=document.getElementById('add_ds_list');
  list.innerHTML='<div class="small" style="padding:8px">扫描中...</div>';
  const d=await j('/api/scan-datasets');
  if(!d.ok){list.innerHTML=`<div class="small" style="padding:8px;color:#ff6b6b">${d.error||'扫描失败'}</div>`;return;}
  if(!d.datasets.length){list.innerHTML='<div class="small" style="padding:8px">未发现新的数据集</div>';return;}
  list.innerHTML='';
  for(const ds of d.datasets){
    const item=document.createElement('div');
    item.className='add-ds-item';
    item.innerHTML=`
      <span class='ds-name'>${ds.name}</span>
      <span class='ds-meta'>${ds.codebase_version} | ${ds.total_episodes} eps | ${ds.fps} fps<br/>${ds.dataset_root}</span>
      <button onclick='addSource("${ds.dataset_root}","${ds.name}","${ds.label_csv}")'>添加</button>`;
    list.appendChild(item);
  }
}

async function addSource(root,name,labelCsv){
  const d=await j('/api/merge-sources/add',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({dataset_root:root,name:name,label_csv:labelCsv})});
  if(!d.ok){alert(d.error||'添加失败');return;}
  showStatus(`已添加: ${d.name}`);
  await loadMergeSources();
  await scanDatasets();
}

async function removeSource(name){
  if(!confirm(`移除数据源 "${name}"？`))return;
  const d=await j(`/api/merge-sources/${encodeURIComponent(name)}`,{method:'DELETE'});
  if(!d.ok){alert(d.error||'移除失败');return;}
  showStatus(`已移除: ${name}`);
  await loadMergeSources();
}

/* ==== Run split/merge ==== */
async function runJob(){
  const outputRoot=document.getElementById('split_output_root').value.trim();
  if(!outputRoot){alert('请填写输出目录');return;}

  const taskMap={};
  const targetLabels=[];
  document.querySelectorAll('.label-check:checked').forEach(cb=>{
    const label=cb.dataset.label;
    const prompt=document.querySelector(`.prompt-input[data-label="${label}"]`)?.value.trim()||'';
    if(label){targetLabels.push(label);if(prompt)taskMap[label]=prompt;}
  });
  if(!targetLabels.length){alert('请至少勾选一个类别');return;}

  const selectedSources=splitMode==='merge'?getSelectedSourceNames():[];
  if(splitMode==='merge'&&!selectedSources.length){alert('请至少选择一个数据集');return;}

  const actionName=splitMode==='merge'?'合并':'分割';
  const scopeDesc=splitMode==='merge'?`${selectedSources.length}个数据集`:'当前数据集';
  if(!confirm(`将${actionName} ${scopeDesc} 中 [${targetLabels.join(', ')}] 到\n${outputRoot}\n确认？`))return;

  const statusEl=document.getElementById('split_status');
  const outputEl=document.getElementById('split_output');
  statusEl.innerText=actionName+'中...'; statusEl.style.color='#ffb74d';
  outputEl.style.display='block'; outputEl.innerText='正在启动...';

  const d=await j('/api/split',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
    mode:splitMode, output_root:outputRoot, task_map:taskMap, labels:targetLabels,
    require_all_videos:document.getElementById('split_require_videos').checked,
    overwrite:document.getElementById('split_overwrite').checked,
    selected_sources:selectedSources,
  })});
  if(!d.ok){statusEl.innerText='启动失败: '+d.error;statusEl.style.color='#ff6b6b';return;}
  outputEl.innerText='命令: '+d.command+'\n\n等待结果...';

  const poll=setInterval(async()=>{
    const s=await j('/api/split/status');
    if(s.running){statusEl.innerText=actionName+'进行中...';}
    else{
      clearInterval(poll);
      if(s.error){statusEl.innerText=actionName+'失败';statusEl.style.color='#ff6b6b';outputEl.innerText+='\n\nERROR: '+s.error;}
      else if(s.result){
        const rc=s.result.returncode;
        statusEl.innerText=rc===0?actionName+'完成':actionName+'失败 (exit '+rc+')';
        statusEl.style.color=rc===0?'#8bc34a':'#ff6b6b';
        outputEl.innerText=`命令: ${d.command}\n\n--- stdout ---\n${s.result.stdout||'(空)'}\n\n--- stderr ---\n${s.result.stderr||'(空)'}`;
      }
    }
  },2000);
}

/* --- Keyboard --- */
document.getElementById('lf').onchange=loadGrid;
document.addEventListener('keydown',e=>{
  if(e.target&&['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName))return;
  if(e.key===' '){e.preventDefault();toggleAllPlay();}
  if(e.key==='Escape')closeModal();
  if(e.key==='ArrowLeft')prevPage();
  if(e.key==='ArrowRight')nextPage();
});

loadMeta().then(loadGrid);
</script></body></html>"""
