from __future__ import annotations

import argparse
import json
from pathlib import Path

from copal.construction_quality_validation import build_quality_disagreement_records
from copal.io import ensure_directory, read_jsonl, write_json


DEFAULT_SOURCE_RUN = Path("runs/experiments/construction_quality_validation_20260515")
DEFAULT_OUTPUT_DIR = Path("paper_final/annotation_adjudication/construction_quality_20260515")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Chinese static frontend for construction-quality adjudication.")
    parser.add_argument("--source-run-dir", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    samples = read_jsonl(args.source_run_dir / "construction_quality_samples.jsonl")
    annotations = read_jsonl(args.source_run_dir / "annotations.jsonl")
    records = build_quality_disagreement_records(
        samples=samples,
        annotations=annotations,
        source_run=str(args.source_run_dir),
    )
    write_quality_frontend(output_dir=args.output_dir, records=records, source_run=str(args.source_run_dir))
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "source_run": str(args.source_run_dir),
                "disagreement_record_count": len(records),
                "index_html": str(args.output_dir / "index.html"),
            },
            indent=2,
        )
    )


def write_quality_frontend(*, output_dir: Path, records: list[dict], source_run: str) -> None:
    ensure_directory(output_dir)
    payload = {
        "metadata": {
            "source_run": source_run,
            "record_count": len(records),
            "language": "zh-CN",
            "metrics": {
                "naturalness_valid": "自然性",
                "diagnosticity_valid": "诊断性",
            },
        },
        "records": records,
    }
    write_json(output_dir / "disagreements.json", payload)
    (output_dir / "disagreements.js").write_text(
        "window.COPAL_QUALITY_ADJUDICATION_DATA = "
        + json.dumps(payload, ensure_ascii=True, indent=2)
        + ";\n",
        encoding="utf-8",
    )
    (output_dir / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (output_dir / "styles.css").write_text(STYLES_CSS, encoding="utf-8")
    (output_dir / "app.js").write_text(APP_JS, encoding="utf-8")


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>COPAL 构造质量仲裁</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header class="topbar">
    <div>
      <h1>COPAL 构造质量仲裁</h1>
      <p id="runMeta"></p>
    </div>
    <div class="topActions">
      <button id="exportBtn" type="button">导出 JSON</button>
      <button id="copyBtn" type="button">复制 JSON</button>
      <button id="clearBtn" type="button" class="danger">清空本地结果</button>
    </div>
  </header>
  <main class="layout">
    <aside class="sidebar">
      <div class="progressBlock">
        <div id="progressText"></div>
        <div class="progressTrack"><div id="progressBar"></div></div>
      </div>
      <input id="searchBox" type="search" placeholder="搜索样本、问题、理由">
      <ol id="sampleList" class="sampleList"></ol>
    </aside>
    <section class="reviewPane">
      <nav class="itemNav">
        <button id="prevBtn" type="button">上一条</button>
        <div id="itemTitle"></div>
        <button id="nextBtn" type="button">下一条</button>
      </nav>
      <section class="panel">
        <h2>仲裁目标</h2>
        <div id="caseMeta" class="metaGrid"></div>
      </section>
      <section class="panel">
        <h2>用户问题</h2>
        <pre id="queryText" class="textBlock"></pre>
      </section>
      <section class="panel">
        <h2>策略证据</h2>
        <div id="clauses"></div>
      </section>
      <section class="panel">
        <h2>期望/禁止处理</h2>
        <div class="twoCol">
          <div>
            <h3>必须做</h3>
            <ul id="mustDo"></ul>
          </div>
          <div>
            <h3>不能做</h3>
            <ul id="mustNotDo"></ul>
          </div>
        </div>
        <h3>允许回答锚点</h3>
        <pre id="allowedAnchor" class="textBlock compact"></pre>
        <h3>禁止结果</h3>
        <pre id="forbiddenOutcome" class="textBlock compact"></pre>
        <h3>需要保留的门控/转交</h3>
        <pre id="requiredRoute" class="textBlock compact"></pre>
      </section>
      <section class="panel">
        <h2>GPT / Claude 标注分歧</h2>
        <div id="annotationCards" class="annotationGrid"></div>
      </section>
      <section class="panel">
        <h2>人工仲裁</h2>
        <div class="decisionGrid">
          <label>最终判定
            <select id="finalDecision">
              <option value="">未评审</option>
              <option value="pass">通过</option>
              <option value="fail">不通过</option>
              <option value="needs_discussion">需要讨论</option>
            </select>
          </label>
          <label>理由来源
            <select id="rationaleSource"></select>
          </label>
        </div>
        <label>备注
          <textarea id="notes" rows="5" placeholder="记录最终理由或讨论结论。"></textarea>
        </label>
        <div class="saveRow">
          <button id="saveBtn" type="button">保存判定</button>
          <span id="saveStatus"></span>
        </div>
      </section>
    </section>
  </main>
  <script src="disagreements.js"></script>
  <script src="app.js"></script>
</body>
</html>
"""


STYLES_CSS = """
:root { --bg:#f7f8fb; --panel:#fff; --text:#172033; --muted:#667085; --line:#d8dee9; --accent:#1f6feb; --bad:#b42318; --good:#067647; --warn:#9a6700; }
* { box-sizing: border-box; }
body { margin:0; color:var(--text); background:var(--bg); font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
button,input,select,textarea { font:inherit; }
button { min-height:34px; border:1px solid var(--line); border-radius:6px; background:#fff; color:var(--text); cursor:pointer; padding:6px 10px; }
button:hover { border-color:var(--accent); }
button.danger { color:var(--bad); }
.topbar { position:sticky; top:0; z-index:4; display:flex; align-items:center; justify-content:space-between; gap:16px; padding:14px 20px; border-bottom:1px solid var(--line); background:rgba(255,255,255,.96); }
h1,h2,h3,p { margin:0; }
h1 { font-size:20px; }
h2 { font-size:16px; margin-bottom:12px; }
h3 { font-size:13px; margin:12px 0 6px; color:var(--muted); }
.topActions { display:flex; flex-wrap:wrap; gap:8px; justify-content:flex-end; }
.layout { display:grid; grid-template-columns:340px minmax(0,1fr); gap:16px; padding:16px; }
.sidebar { position:sticky; top:78px; height:calc(100vh - 94px); display:flex; flex-direction:column; gap:12px; }
.progressBlock,.panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }
.progressTrack { height:8px; border-radius:999px; background:#eef2f7; overflow:hidden; margin-top:8px; }
#progressBar { width:0; height:100%; background:var(--accent); }
input,select,textarea { width:100%; border:1px solid var(--line); border-radius:6px; background:#fff; color:var(--text); padding:8px 10px; }
.sampleList { min-height:0; margin:0; padding:0; overflow:auto; list-style:none; border:1px solid var(--line); border-radius:8px; background:#fff; }
.sampleList li { border-bottom:1px solid var(--line); }
.sampleList button { width:100%; min-height:64px; border:0; border-radius:0; text-align:left; display:grid; gap:4px; }
.sampleList button.active { background:#eaf2ff; }
.sampleList button.done .listTitle::after { content:" 已完成"; color:var(--good); font-weight:600; }
.listTitle { font-weight:650; }
.listSub { color:var(--muted); font-size:12px; }
.reviewPane { display:grid; gap:14px; min-width:0; }
.itemNav { display:flex; align-items:center; justify-content:space-between; gap:12px; }
#itemTitle { color:var(--muted); text-align:center; overflow-wrap:anywhere; }
.metaGrid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:8px; }
.metaItem { border:1px solid var(--line); border-radius:6px; padding:8px; min-width:0; }
.metaKey { color:var(--muted); font-size:12px; }
.metaValue { overflow-wrap:anywhere; }
.textBlock { white-space:pre-wrap; overflow-wrap:anywhere; margin:0; border:1px solid var(--line); border-radius:6px; background:#fbfcfe; padding:10px; max-height:320px; overflow:auto; }
.textBlock.compact { max-height:150px; }
.twoCol,.annotationGrid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }
ul { margin:0; padding-left:18px; }
li { margin:4px 0; }
.clauseCard,.annotationCard { border:1px solid var(--line); border-radius:8px; padding:10px; background:#fbfcfe; margin-bottom:10px; }
.annotationHeader { display:flex; align-items:center; justify-content:space-between; gap:8px; margin-bottom:8px; }
.badge { display:inline-flex; align-items:center; border-radius:999px; padding:2px 8px; font-size:12px; border:1px solid var(--line); }
.badge.correct { color:var(--good); border-color:#abefc6; background:#ecfdf3; }
.badge.incorrect { color:var(--bad); border-color:#fecdca; background:#fef3f2; }
.badge.warn { color:var(--warn); border-color:#fedf89; background:#fffaeb; }
.decisionGrid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; margin-bottom:12px; }
label { display:grid; gap:6px; color:var(--muted); }
label > select,label > textarea { color:var(--text); }
.saveRow { display:flex; align-items:center; gap:10px; margin-top:12px; }
#saveStatus { color:var(--muted); }
@media (max-width:920px) { .topbar,.layout { display:block; } .topActions { justify-content:flex-start; margin-top:12px; } .sidebar { position:static; height:auto; margin-bottom:16px; } .sampleList { max-height:260px; } .twoCol,.annotationGrid,.decisionGrid { grid-template-columns:1fr; } }
"""


APP_JS = """
const DATA = window.COPAL_QUALITY_ADJUDICATION_DATA;
const records = DATA.records || [];
const metadata = DATA.metadata || {};
const storageKey = `copal-quality-adjudication:${metadata.source_run || "unknown"}:${metadata.record_count || records.length}`;
const metricLabels = metadata.metrics || {};
let decisions = loadDecisions();
let current = 0;
let filtered = records.slice();
const el = (id) => document.getElementById(id);
function loadDecisions(){ const raw=localStorage.getItem(storageKey); return raw ? JSON.parse(raw) : {}; }
function saveDecisions(){ localStorage.setItem(storageKey, JSON.stringify(decisions)); }
function text(value){ if(value===null||value===undefined) return ""; if(typeof value==="string") return value; return JSON.stringify(value,null,2); }
function short(value,max=90){ const s=text(value).replace(/\\s+/g," ").trim(); return s.length>max?`${s.slice(0,max-1)}...`:s; }
function boolLabel(value){ if(value===true) return "通过"; if(value===false) return "不通过"; return "未知"; }
function boolClass(value){ if(value===true) return "correct"; if(value===false) return "incorrect"; return "warn"; }
function setText(id,value){ el(id).textContent=value; }
function metricLabel(metric){ return metricLabels[metric] || metric; }
function render(){ applyFilters(); renderProgress(); renderSampleList(); if(!filtered.length){ setText("itemTitle","当前筛选条件下没有样本。"); return; } if(current>=filtered.length) current=filtered.length-1; if(current<0) current=0; renderRecord(filtered[current]); }
function applyFilters(){ const q=el("searchBox").value.toLowerCase().trim(); filtered=records.filter((record)=>!q || JSON.stringify(record).toLowerCase().includes(q)); }
function renderProgress(){ const done=records.filter((record)=>decisions[record.review_id] && decisions[record.review_id].final_decision).length; const total=records.length; setText("runMeta",`${total} 条 GPT/Claude 分歧需要仲裁，来源：${metadata.source_run || "未知运行"}`); setText("progressText",`${done}/${total} 已仲裁`); el("progressBar").style.width=total?`${Math.round(done/total*100)}%`:"0%"; }
function renderSampleList(){ const list=el("sampleList"); list.innerHTML=""; filtered.forEach((record,index)=>{ const item=document.createElement("li"); const btn=document.createElement("button"); btn.type="button"; const saved=decisions[record.review_id]; btn.className=`${index===current?"active":""} ${saved&&saved.final_decision?"done":""}`; btn.addEventListener("click",()=>{current=index; render();}); const title=document.createElement("div"); title.className="listTitle"; title.textContent=`${record.review_id} ${metricLabel(record.metric)}`; const sub=document.createElement("div"); sub.className="listSub"; sub.textContent=short(record.sample_id,110); btn.append(title,sub); item.appendChild(btn); list.appendChild(item); }); }
function metaItem(key,value){ const box=document.createElement("div"); box.className="metaItem"; const k=document.createElement("div"); k.className="metaKey"; k.textContent=key; const v=document.createElement("div"); v.className="metaValue"; v.textContent=text(value); box.append(k,v); return box; }
function renderRecord(record){ setText("itemTitle",`${current+1}/${filtered.length} ${record.sample_id}`); renderMeta(record); const input=record.input||{}; setText("queryText",input.query||""); renderClauses(input.active_clauses||[]); renderExpected(input.expected_handling||{}); renderAnnotations(record); renderForm(record); }
function renderMeta(record){ const c=el("caseMeta"); c.innerHTML=""; c.append(metaItem("仲裁指标",metricLabel(record.metric)), metaItem("关系模式",record.strata.relation_pattern), metaItem("目标 facet",record.strata.target_facet), metaItem("公司",record.strata.company_name), metaItem("行业",record.strata.industry)); }
function renderClauses(clauses){ const c=el("clauses"); c.innerHTML=""; clauses.forEach((clause)=>{ const card=document.createElement("div"); card.className="clauseCard"; const title=document.createElement("strong"); title.textContent=`${clause.clause_id || "clause"} ${clause.effect ? `(${clause.effect})` : ""}`; card.append(title); appendText(card,"子句文本",clause.clause_text||""); appendText(card,"Trigger",clause.trigger||{}); appendText(card,"Scope",clause.scope||{}); appendText(card,"来源证据",clause.source_span||""); c.appendChild(card); }); if(!clauses.length) c.textContent="没有记录 active clauses。"; }
function appendText(parent,titleText,value){ if(!text(value)) return; const h=document.createElement("h3"); h.textContent=titleText; const p=document.createElement("pre"); p.className="textBlock compact"; p.textContent=text(value); parent.append(h,p); }
function renderExpected(expected){ renderBulletList("mustDo", expected.must_do||[]); renderBulletList("mustNotDo", expected.must_not_do||[]); setText("allowedAnchor", expected.allowed_answer_anchor||""); setText("forbiddenOutcome", expected.forbidden_outcome||""); setText("requiredRoute", expected.required_gate_or_route||""); }
function renderBulletList(id,rows){ const list=el(id); list.innerHTML=""; rows.forEach((row)=>{ const li=document.createElement("li"); li.textContent=text(row); list.appendChild(li); }); }
function renderAnnotations(record){ const c=el("annotationCards"); c.innerHTML=""; Object.keys(record.annotations||{}).sort().forEach((model)=>{ const ann=record.annotations[model]; const value=ann[record.metric]; const card=document.createElement("div"); card.className="annotationCard"; const header=document.createElement("div"); header.className="annotationHeader"; const modelName=document.createElement("strong"); modelName.textContent=model; const badge=document.createElement("span"); badge.className=`badge ${boolClass(value)}`; badge.textContent=boolLabel(value); header.append(modelName,badge); card.append(header); appendText(card,"自然性理由",ann.naturalness_rationale||""); appendText(card,"诊断性理由",ann.diagnosticity_rationale||""); c.appendChild(card); }); }
function renderForm(record){ const saved=decisions[record.review_id]||{}; el("finalDecision").value=saved.final_decision||""; fillSources(record,saved.rationale_source||""); el("notes").value=saved.notes||""; setText("saveStatus",saved.saved_at?`已保存：${saved.saved_at}`:""); }
function fillSources(record,selected){ const s=el("rationaleSource"); s.innerHTML=""; ["custom",...Object.keys(record.annotations||{}).sort()].forEach((source)=>{ const opt=document.createElement("option"); opt.value=source; opt.textContent=source==="custom"?"自定义":source; if(source===selected) opt.selected=true; s.appendChild(opt); }); }
function currentRecord(){ return filtered[current]; }
function saveCurrent(){ const record=currentRecord(); if(!record) return; const d=el("finalDecision").value; decisions[record.review_id]={ review_id:record.review_id, sample_id:record.sample_id, metric:record.metric, decision_field:record.decision.field, final_decision:d, final_pass:d==="pass"?true:d==="fail"?false:null, rationale_source:el("rationaleSource").value, notes:el("notes").value, saved_at:new Date().toISOString() }; saveDecisions(); render(); }
function exportPayload(){ const rows=records.map((record)=>decisions[record.review_id]).filter(Boolean); return { metadata:{...metadata, exported_at:new Date().toISOString(), storage_key:storageKey, adjudicated_count:rows.filter((row)=>row.final_decision).length}, adjudications:rows}; }
function exportJson(){ const blob=new Blob([JSON.stringify(exportPayload(),null,2)+"\\n"],{type:"application/json"}); const url=URL.createObjectURL(blob); const a=document.createElement("a"); a.href=url; a.download="copal_construction_quality_adjudications.json"; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url); }
async function copyJson(){ await navigator.clipboard.writeText(JSON.stringify(exportPayload(),null,2)); setText("saveStatus","导出 JSON 已复制到剪贴板。"); }
function clearDecisions(){ if(!confirm("确定要清空这次运行的全部本地仲裁结果吗？")) return; decisions={}; saveDecisions(); render(); }
el("searchBox").addEventListener("input",()=>{current=0; render();});
el("prevBtn").addEventListener("click",()=>{current-=1; render();});
el("nextBtn").addEventListener("click",()=>{current+=1; render();});
el("saveBtn").addEventListener("click",saveCurrent);
el("exportBtn").addEventListener("click",exportJson);
el("copyBtn").addEventListener("click",copyJson);
el("clearBtn").addEventListener("click",clearDecisions);
render();
"""


if __name__ == "__main__":
    main()
