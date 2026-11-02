'use strict';
const $ = id => document.getElementById(id);
let activeId = null, busy = false;
function node(tag, text, className) { const e=document.createElement(tag); if(text!==undefined)e.textContent=text; if(className)e.className=className; return e; }
async function api(path, options={}) {
  const response=await fetch(path, options);
  const type=response.headers.get('content-type')||'';
  const body=type.includes('json')?await response.json():await response.text();
  if(!response.ok) throw new Error(typeof body.detail==='string'?body.detail:JSON.stringify(body.detail||body));
  return body;
}
function status(message, type='') { $('status').textContent=message; $('status').className=type; }
function updateCount() { $('input-count').textContent=`${$('discussion').value.length.toLocaleString()} characters`; }
$('discussion').addEventListener('input',updateCount);
$('example').onclick=async()=>{try{$('discussion').value=await api('/api/example');updateCount();}catch(e){status(e.message,'error');}};
$('file').onchange=async event=>{
  const file=event.target.files[0];if(!file)return;
  if(file.size>1_000_000){status('Choose a file smaller than 1 MB.','error');return;}
  const text=await file.text();
  if(text.length>250000){status('Input is limited to 250,000 characters.','error');return;}
  $('discussion').value=text;updateCount();status(`Loaded ${file.name}.`);
};
$('mode').onchange=()=>{$('mode-hint').textContent=$('mode').value==='ai'?'AI mode downloads the models on first use. CPU inference can take several minutes.':'Offline demo uses TF-IDF and extractive selection. It does not run the report’s neural pipeline.';};
function payload() {
  const raw=$('discussion').value.trim();if(!raw)throw new Error('Add a discussion first.');
  const value={options:{mode:$('mode').value,num_topics:Number($('num-topics').value),chunk_tokens:Number($('chunk-tokens').value),summary_tokens:Number($('summary-tokens').value),semantic_refinement:$('refinement').checked,second_pass:$('second-pass').checked,dedup_threshold:Number($('threshold').value)}};
  if(raw.startsWith('{')||/^\[\s*(?:\{|\[|"|\])/.test(raw)) {try{const parsed=JSON.parse(raw);if(typeof parsed!=='object'||parsed===null)throw new Error();value.discussion=parsed;}catch{throw new Error('Invalid JSON. Use a JSON object or array, or paste plain discussion text.');}}
  else value.text=raw;
  return value;
}
function setBusy(value){busy=value;$('analyze').disabled=value;$('analyze').replaceChildren(node('span',value?'Processing discussion…':'Analyze discussion'),node('span','↗'));}
$('analyze').onclick=async()=>{
  if(busy)return;
  try{const body=payload();setBusy(true);status('Submitting analysis…','busy');const job=await api('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});activeId=job.id;localStorage.setItem('insightflow-active',activeId);await poll(activeId);}
  catch(e){status(e.message,'error');setBusy(false);}
};
async function poll(id) {
  while(true){
    let job;
    try{job=await api(`/api/jobs/${id}`);}catch(e){status(`Connection interrupted: ${e.message}. Refresh history to reopen the job.`,'error');setBusy(false);return;}
    if(job.status==='completed'){render(job.result,id);status('Analysis complete. Inspect the source evidence or download your results.');break;}
    if(job.status==='failed'){status(job.error||'Analysis failed.','error');break;}
    status(job.stage,'busy');await new Promise(resolve=>setTimeout(resolve,1200));
  }
  localStorage.removeItem('insightflow-active');setBusy(false);await history();
}
function render(result,id) {
  $('results').replaceChildren();
  const bar=node('div',undefined,'summary-bar');
  for(const text of [`${result.topics.length} topics`,`${result.preprocessing.sentences.length} sentences`,`${result.elapsed_seconds.toFixed(1)}s measured runtime`,result.mode==='ai'?'MPNet + DistilBART':'Offline demo'])bar.append(node('b',text));
  $('results').append(bar,node('p',result.warnings.join(' '),'notice'));
  for(const topic of result.topics){
    const card=node('article',undefined,'topic');const top=node('div',undefined,'topic-top');top.append(node('span',`TOPIC ${String(topic.id).padStart(2,'0')}`),node('span',`${topic.sentence_count} SENTENCES`));
    card.append(top,node('h3',topic.label));const keys=node('div',undefined,'keywords');topic.keywords.forEach(word=>keys.append(node('span',word,'keyword')));card.append(keys,node('p',topic.summary));
    const evidence=node('details');evidence.append(node('summary',`Source evidence · ${topic.sources.length} sentences`));
    for(const source of topic.sources){const row=node('div',undefined,'evidence');row.append(node('small',`#${source.id+1} · ${source.selected_for_summary?'Selected':'Semantic duplicate excluded from summary'}`),node('div',source.text));evidence.append(row);}card.append(evidence);
    const intermediate=node('details');intermediate.append(node('summary',`Pipeline trace · ${topic.chunks.length} chunks · ${topic.compression_rounds} integration passes`));
    topic.chunks.forEach((chunk,index)=>{const item=node('div',undefined,'evidence');item.append(node('b',`Chunk ${index+1} — first-pass summary`),node('p',chunk.summary));const input=node('details');input.append(node('summary','Chunk input'),node('p',chunk.input));item.append(input);intermediate.append(item);});card.append(intermediate);$('results').append(card);
  }
  const timing=node('details',undefined,'timings');timing.append(node('summary','Measured pipeline timings and model details'),node('pre',`${result.models}\n${JSON.stringify(result.timings_seconds,null,2)}`));$('results').append(timing);
  $('export-actions').hidden=false;$('export-json').href=`/api/jobs/${id}/export?format=json`;$('export-text').href=`/api/jobs/${id}/export?format=txt`;
}
async function history(){
  try{
    const jobs=await api('/api/jobs');$('history').replaceChildren();
    if(!jobs.length){$('history').append(node('p','Your analyses will appear here.','hint'));return;}
    for(const job of jobs){
      const row=node('div',undefined,'history-row');const open=node('button',`${new Date(job.created).toLocaleString()} · ${job.id.slice(0,8)}`,'open');
      open.onclick=async()=>{if(busy){status('Wait for the active analysis to finish.');return;}try{const item=await api(`/api/jobs/${job.id}`);if(item.status==='completed'){render(item.result,item.id);status('Opened saved analysis.');}else if(item.status==='failed')status(item.error,'error');else{setBusy(true);await poll(item.id);}}catch(e){status(e.message,'error');setBusy(false);}};
      row.append(open,node('span',job.status,'badge'));
      if(!['running','queued'].includes(job.status)){const remove=node('button','Delete','text-button');remove.onclick=async()=>{try{await api(`/api/jobs/${job.id}`,{method:'DELETE'});await history();}catch(e){status(e.message,'error');}};row.append(remove);}$('history').append(row);
    }
  }catch(e){status(e.message,'error');}
}
$('refresh').onclick=history;
(async()=>{await history();try{const h=await api('/api/health');if(!h.ai_packages_installed)status('Server ready. Install requirements-ai.txt for AI mode, or choose Offline demo.');const previous=localStorage.getItem('insightflow-active');if(previous){setBusy(true);await poll(previous);}}catch(e){status(e.message,'error');}})();
