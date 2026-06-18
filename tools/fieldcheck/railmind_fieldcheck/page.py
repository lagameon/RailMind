"""Self-contained HTML/JS for the standalone Field Input Check (no external assets;
input-readiness check, no anomaly scoring)."""

PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>RailMind — Field Input Check (standalone)</title>
<style>
 :root{--bg:#0f1115;--card:#161a22;--bd:#262c38;--mut:#8b95a7;--fg:#e6e6e6}
 *{box-sizing:border-box} body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:var(--bg);color:var(--fg)}
 header{padding:16px 22px;background:var(--card);border-bottom:1px solid var(--bd)}
 h1{font-size:17px;margin:0} .sub{color:var(--mut);font-size:12px;margin-top:5px;max-width:780px}
 .badge{display:inline-block;margin-top:7px;font-size:11px;color:#9ad;border:1px solid #2d4a6b;border-radius:5px;padding:2px 7px}
 .wrap{display:grid;grid-template-columns:340px 1fr;gap:18px;padding:18px 22px;max-width:1140px}
 .panel{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:16px}
 label{display:block;font-size:12px;color:var(--mut);margin:10px 0 4px}
 select,input[type=text],input[type=number]{width:100%;padding:7px 9px;background:#0f1115;border:1px solid var(--bd);border-radius:6px;color:var(--fg);font-size:13px}
 button{margin-top:16px;width:100%;padding:10px;border:0;border-radius:7px;background:#2f6fed;color:#fff;font-size:14px;font-weight:600;cursor:pointer} button:disabled{opacity:.5;cursor:wait}
 .hide{display:none}
 .verdict{font-size:15px;font-weight:700;padding:10px 14px;border-radius:8px;margin-bottom:12px}
 .vPASS{background:#13351f;color:#51cf66;border:1px solid #2f7d4d}
 .vWARN{background:#3a3413;color:#f1c40f;border:1px solid #8a7b1e}
 .vFAIL{background:#3a1717;color:#ff6b6b;border:1px solid #8a2e2e}
 .check{display:flex;gap:10px;padding:9px 10px;border:1px solid var(--bd);border-radius:8px;margin-bottom:8px;align-items:flex-start}
 .cname{font-weight:600;font-size:13px} .cdet{color:var(--mut);font-size:12px;margin-top:2px}
 .pPASS{color:#51cf66}.pWARN{color:#f1c40f}.pFAIL{color:#ff6b6b}
 canvas{background:#0f1115;border:1px solid var(--bd);border-radius:8px;width:100%;margin-top:8px}
 .hint{color:var(--mut);font-size:11px;margin-top:3px}
</style></head><body>
<header><h1>RailMind — Field Input Check</h1>
<div class="sub">Confirm your on-site sensor data is well-formed and ready for monitoring. Pick a source, run the check, read the readiness report. Nothing leaves this machine.</div>
<div class="badge">standalone · input-readiness check · no anomaly scoring</div></header>
<div class="wrap">
 <div class="panel">
   <label>Input source</label>
   <select id="src" onchange="srcUI()">
     <option value="csv_text">CSV file (upload)</option>
     <option value="csv_path">CSV file (server path)</option>
     <option value="waveform_csv_text">Raw waveform CSV (1 channel) + FFT</option>
     <option value="mqtt">Live MQTT</option>
     <option value="opcua">Live OPC-UA</option>
   </select>
   <div id="g_csv_text" class="srcg">
     <label>CSV file</label><input type="file" id="csvfile" accept=".csv,.txt">
     <label>Has header row</label><select id="hdr"><option value="1">yes</option><option value="0">no</option></select>
     <div class="hint">Columns = sensor channels, rows = time. One numeric value per cell.</div>
   </div>
   <div id="g_csv_path" class="srcg hide"><label>Server-side CSV path</label><input type="text" id="path" placeholder="/data/site/stream.csv"></div>
   <div id="g_waveform_csv_text" class="srcg hide">
     <label>Waveform CSV (single column)</label><input type="file" id="wavfile" accept=".csv,.txt">
     <div class="hint">High-rate accelerometer/current; the FFT framer turns windows into spectra.</div>
   </div>
   <div id="g_mqtt" class="srcg hide">
     <label>Broker host</label><input type="text" id="mqtt_host" placeholder="192.168.1.10">
     <label>Topic</label><input type="text" id="mqtt_topic" placeholder="plant/line1/sensors">
     <label>Port</label><input type="number" id="mqtt_port" value="1883"></div>
   <div id="g_opcua" class="srcg hide">
     <label>Endpoint</label><input type="text" id="opc_ep" placeholder="opc.tcp://192.168.1.10:4840">
     <label>Node IDs (comma-separated)</label><input type="text" id="opc_nodes" placeholder="ns=2;i=2, ns=2;i=3">
     <label>Poll period (s)</label><input type="number" id="opc_period" value="1.0" step="0.1"></div>

   <label>Framer</label>
   <select id="framer" onchange="frUI()">
     <option value="passthrough">passthrough (already feature vectors)</option>
     <option value="window_stats">window_stats (raw → windowed stats)</option>
     <option value="window_fft">window_fft (raw waveform → spectrum)</option>
   </select>
   <div id="fr_win" class="hide"><label>window</label><input type="number" id="fr_window" value="64"><label>hop</label><input type="number" id="fr_hop" value="32"></div>
   <div id="fr_fft" class="hide"><label>n_fft</label><input type="number" id="fr_nfft" value="256"><label>hop</label><input type="number" id="fr_fhop" value="128"></div>

   <label>Warmup frames (healthy baseline)</label><input type="number" id="warmup" value="200">
   <button id="run" onclick="run()">Run check</button>
   <div class="hint" id="status"></div>
 </div>
 <div class="panel"><div id="out"><div class="sub">Results will appear here after you run a check.</div></div></div>
</div>
<script>
function srcUI(){const v=document.getElementById('src').value;
  document.querySelectorAll('.srcg').forEach(e=>e.classList.add('hide'));
  document.getElementById('g_'+v).classList.remove('hide');
  if(v==='waveform_csv_text'){document.getElementById('framer').value='window_fft';frUI();}}
function frUI(){const v=document.getElementById('framer').value;
  document.getElementById('fr_win').classList.toggle('hide',v!=='window_stats');
  document.getElementById('fr_fft').classList.toggle('hide',v!=='window_fft');}
function readFile(id){return new Promise(res=>{const f=document.getElementById(id).files[0];
  if(!f){res(null);return;}const r=new FileReader();r.onload=()=>res(r.result);r.readAsText(f);});}
async function buildCfg(){
  const st=document.getElementById('src').value, src={type:st};
  if(st==='csv_text'){src.text=await readFile('csvfile');src.has_header=document.getElementById('hdr').value==='1';}
  else if(st==='waveform_csv_text'){src.text=await readFile('wavfile');src.has_header=true;}
  else if(st==='csv_path'){src.path=document.getElementById('path').value;}
  else if(st==='mqtt'){src.host=document.getElementById('mqtt_host').value;src.topic=document.getElementById('mqtt_topic').value;src.port=+document.getElementById('mqtt_port').value;}
  else if(st==='opcua'){src.endpoint=document.getElementById('opc_ep').value;src.node_ids=document.getElementById('opc_nodes').value.split(',').map(s=>s.trim()).filter(Boolean);src.period_s=+document.getElementById('opc_period').value;}
  const fv=document.getElementById('framer').value, framer={type:fv};
  if(fv==='window_stats'){framer.window=+document.getElementById('fr_window').value;framer.hop=+document.getElementById('fr_hop').value;}
  if(fv==='window_fft'){framer.n_fft=+document.getElementById('fr_nfft').value;framer.hop=+document.getElementById('fr_fhop').value;}
  return {source:src,framer:framer,warmup_n:+document.getElementById('warmup').value};
}
async function run(){
  const btn=document.getElementById('run'),st=document.getElementById('status');btn.disabled=true;st.textContent='running…';
  try{const cfg=await buildCfg();
    if((cfg.source.type==='csv_text'||cfg.source.type==='waveform_csv_text')&&!cfg.source.text){st.textContent='pick a file first';btn.disabled=false;return;}
    const r=await fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(cfg)});
    render(await r.json());st.textContent='done';
  }catch(e){st.textContent='error: '+e;}
  btn.disabled=false;
}
function render(rep){
  const o=document.getElementById('out');let h='';let scree=null;
  h+=`<div class="verdict v${rep.verdict}">Readiness: ${rep.verdict} &nbsp;·&nbsp; ${rep.summary.pass} pass / ${rep.summary.warn} warn / ${rep.summary.fail} fail</div>`;
  for(const c of rep.checks){const p={PASS:'●',WARN:'▲',FAIL:'✕'}[c.status];
    h+=`<div class="check"><div class="p${c.status}">${p}</div><div><div class="cname">${c.name}</div><div class="cdet">${c.detail}</div></div></div>`;
    if(c.cumulative_variance)scree=c.cumulative_variance;}
  if(scree)h+=`<div class="cname" style="margin-top:12px">Cumulative explained variance (PCA)</div><canvas id="sc" height="180"></canvas>`;
  h+=`<div class="hint" style="margin-top:12px">${rep.note||''}</div>`;
  o.innerHTML=h; if(scree)drawScree(scree);
}
function drawScree(cv){
  const c=document.getElementById('sc'),x=c.getContext('2d');c.width=c.clientWidth;const W=c.width,H=c.height;
  const X=i=>10+(W-20)*(i/((cv.length-1)||1)),Y=v=>H-12-(H-24)*v;
  x.strokeStyle='#2d4a6b';x.setLineDash([4,4]);x.beginPath();x.moveTo(10,Y(0.8));x.lineTo(W-10,Y(0.8));x.stroke();x.setLineDash([]);
  x.fillStyle='#8b95a7';x.font='10px sans-serif';x.fillText('80%',12,Y(0.8)-3);
  x.strokeStyle='#4dabf7';x.lineWidth=1.8;x.beginPath();cv.forEach((v,i)=>{i?x.lineTo(X(i),Y(v)):x.moveTo(X(i),Y(v))});x.stroke();
  x.fillStyle='#4dabf7';cv.forEach((v,i)=>{x.beginPath();x.arc(X(i),Y(v),2.2,0,7);x.fill()});
}
srcUI();frUI();
</script></body></html>"""
