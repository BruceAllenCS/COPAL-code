from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from copal.io import ensure_directory, read_jsonl, write_json


RESPONSE_ERROR_TYPES = [
    "none",
    "prohibited_disclosure",
    "skipped_gate",
    "wrong_scope_gate",
    "over_refusal",
    "semantic_leakage",
    "latent_continuation",
    "wrong_route",
    "missed_transfer",
    "unsupported_commitment",
    "other",
]


def build_disagreement_records(
    *,
    samples_path: Path,
    annotations_path: Path,
    source_run: str,
) -> list[dict[str, Any]]:
    samples = read_jsonl(samples_path)
    annotations = read_jsonl(annotations_path)
    samples_by_id = _index_samples(samples)
    annotations_by_sample = _group_annotations(annotations)

    records: list[dict[str, Any]] = []
    for sample_id in sorted(samples_by_id):
        sample_annotations = annotations_by_sample.get(sample_id, {})
        if len(sample_annotations) < 2:
            continue
        sample = samples_by_id[sample_id]
        decision = _decision_for_annotations(
            task=str(sample["task"]),
            annotations_by_model=sample_annotations,
        )
        if len(set(decision["values"].values())) <= 1:
            continue
        records.append(
            {
                "review_id": f"review-{len(records) + 1:04d}",
                "sample_id": sample_id,
                "task": sample["task"],
                "source_run": source_run,
                "strata": sample.get("strata", {}),
                "hidden_reference": sample.get("hidden_reference", {}),
                "input": sample["input"],
                "decision": decision,
                "annotations": {
                    model: sample_annotations[model]
                    for model in sorted(sample_annotations)
                },
            }
        )
    return records


def write_adjudication_frontend(
    *,
    output_dir: Path,
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    ensure_directory(output_dir)
    payload = {
        "metadata": {
            **metadata,
            "record_count": len(records),
            "generated_at": metadata.get("generated_at") or datetime.now(timezone.utc).isoformat(),
            "error_types": RESPONSE_ERROR_TYPES,
        },
        "records": records,
    }
    write_json(output_dir / "disagreements.json", payload)
    (output_dir / "disagreements.js").write_text(
        "window.COPAL_ADJUDICATION_DATA = "
        + json.dumps(payload, ensure_ascii=True, indent=2)
        + ";\n",
        encoding="utf-8",
    )
    (output_dir / "index.html").write_text(_INDEX_HTML, encoding="utf-8")
    (output_dir / "styles.css").write_text(_STYLES_CSS, encoding="utf-8")
    (output_dir / "app.js").write_text(_APP_JS, encoding="utf-8")
    (output_dir / "README.md").write_text(_readme_text(metadata=payload["metadata"]), encoding="utf-8")


def _index_samples(samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for sample in samples:
        sample_id = str(sample["sample_id"])
        if sample_id in indexed:
            raise ValueError(f"duplicate sample_id in samples: {sample_id}")
        if "task" not in sample:
            raise ValueError(f"sample is missing task: {sample_id}")
        if "input" not in sample:
            raise ValueError(f"sample is missing input: {sample_id}")
        indexed[sample_id] = sample
    return indexed


def _group_annotations(annotations: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in annotations:
        sample_id = str(row["sample_id"])
        model = str(row["annotator_model"])
        annotation = row["annotation"]
        if not isinstance(annotation, dict):
            raise ValueError(f"annotation must be an object: {sample_id}::{model}")
        grouped.setdefault(sample_id, {})
        if model in grouped[sample_id]:
            raise ValueError(f"duplicate annotation for sample/model: {sample_id}::{model}")
        grouped[sample_id][model] = annotation
    return grouped


def _decision_for_annotations(
    *,
    task: str,
    annotations_by_model: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    field = "response_correct" if task == "response_judge_reliability" else "overall_valid"
    values: dict[str, bool] = {}
    error_types: dict[str, str] = {}
    confidence: dict[str, float] = {}
    for model, annotation in sorted(annotations_by_model.items()):
        value = annotation.get(field)
        if not isinstance(value, bool):
            raise ValueError(f"annotation is missing boolean {field}: {model}")
        values[model] = value
        error_type = annotation.get("error_type")
        if isinstance(error_type, str):
            error_types[model] = error_type
        score = annotation.get("confidence")
        if isinstance(score, (int, float)):
            confidence[model] = float(score)
    return {
        "field": field,
        "values": values,
        "error_types": error_types,
        "confidence": confidence,
    }


def _readme_text(*, metadata: dict[str, Any]) -> str:
    source_run = metadata.get("source_run", "")
    record_count = metadata.get("record_count", 0)
    return f"""# COPAL Adjudication Frontend

This static page contains the LLM annotation disagreements from:

`{source_run}`

Open `index.html` in a browser. The page stores decisions in browser localStorage
and can export the adjudicated result as JSON.

Record count: {record_count}

The page intentionally avoids showing the original Gemini judge label in the main
review pane, so the human discussion can focus on the policy contract, response,
and the two disagreeing LLM annotations.
"""


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>COPAL Adjudication</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header class="topbar">
    <div>
      <h1>COPAL Adjudication</h1>
      <p id="runMeta"></p>
    </div>
    <div class="topActions">
      <button id="exportBtn" type="button">Export JSON</button>
      <button id="copyBtn" type="button">Copy JSON</button>
      <button id="clearBtn" type="button" class="danger">Clear local decisions</button>
    </div>
  </header>

  <main class="layout">
    <aside class="sidebar">
      <div class="progressBlock">
        <div class="progressText" id="progressText"></div>
        <div class="progressTrack"><div id="progressBar"></div></div>
      </div>
      <div class="filters">
        <input id="searchBox" type="search" placeholder="Search sample, query, rationale">
        <select id="statusFilter">
          <option value="all">All items</option>
          <option value="open">Open only</option>
          <option value="done">Done only</option>
        </select>
      </div>
      <ol id="sampleList" class="sampleList"></ol>
    </aside>

    <section class="reviewPane">
      <nav class="itemNav">
        <button id="prevBtn" type="button">Previous</button>
        <div id="itemTitle"></div>
        <button id="nextBtn" type="button">Next</button>
      </nav>

      <section class="panel">
        <h2>Case</h2>
        <div id="caseMeta" class="metaGrid"></div>
        <h3>Query</h3>
        <pre id="queryText" class="textBlock"></pre>
        <h3>Response Under Review</h3>
        <pre id="responseText" class="textBlock"></pre>
      </section>

      <section class="panel">
        <h2>Policy Contract</h2>
        <div class="twoCol">
          <div>
            <h3>Required obligations</h3>
            <ul id="requiredList"></ul>
          </div>
          <div>
            <h3>Forbidden outcomes</h3>
            <ul id="forbiddenList"></ul>
          </div>
        </div>
      </section>

      <section class="panel">
        <h2>Active Clauses</h2>
        <div id="clauses"></div>
      </section>

      <section class="panel">
        <h2>Disagreeing LLM Annotations</h2>
        <div id="annotationCards" class="annotationGrid"></div>
      </section>

      <section class="panel adjudicationPanel">
        <h2>Human Adjudication</h2>
        <div class="decisionGrid">
          <label>
            Final decision
            <select id="finalDecision">
              <option value="">Unreviewed</option>
              <option value="correct">Correct</option>
              <option value="incorrect">Incorrect</option>
              <option value="needs_discussion">Needs discussion</option>
            </select>
          </label>
          <label>
            Error type
            <select id="errorType"></select>
          </label>
          <label>
            Rationale source
            <select id="rationaleSource"></select>
          </label>
        </div>
        <label>
          Notes
          <textarea id="notes" rows="5" placeholder="Write the final reasoning or meeting note here."></textarea>
        </label>
        <div class="saveRow">
          <button id="saveBtn" type="button">Save decision</button>
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


_STYLES_CSS = """
:root {
  --bg: #f7f8fb;
  --panel: #ffffff;
  --text: #172033;
  --muted: #667085;
  --line: #d8dee9;
  --accent: #1f6feb;
  --bad: #b42318;
  --good: #067647;
  --warn: #9a6700;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  color: var(--text);
  background: var(--bg);
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

button, input, select, textarea {
  font: inherit;
}

button {
  min-height: 34px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fff;
  color: var(--text);
  cursor: pointer;
  padding: 6px 10px;
}

button:hover {
  border-color: var(--accent);
}

button.danger {
  color: var(--bad);
}

.topbar {
  position: sticky;
  top: 0;
  z-index: 4;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 14px 20px;
  border-bottom: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.96);
}

h1, h2, h3, p {
  margin: 0;
}

h1 {
  font-size: 20px;
}

h2 {
  font-size: 16px;
  margin-bottom: 12px;
}

h3 {
  font-size: 13px;
  margin: 12px 0 6px;
  color: var(--muted);
}

.topActions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: flex-end;
}

.layout {
  display: grid;
  grid-template-columns: 340px minmax(0, 1fr);
  gap: 16px;
  padding: 16px;
}

.sidebar {
  position: sticky;
  top: 78px;
  height: calc(100vh - 94px);
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.progressBlock, .panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px;
}

.progressText {
  margin-bottom: 8px;
  color: var(--muted);
}

.progressTrack {
  height: 8px;
  border-radius: 999px;
  background: #eef2f7;
  overflow: hidden;
}

#progressBar {
  width: 0;
  height: 100%;
  background: var(--accent);
}

.filters {
  display: grid;
  gap: 8px;
}

.filters input, .filters select, .decisionGrid select, textarea {
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fff;
  color: var(--text);
  padding: 8px 10px;
}

.sampleList {
  min-height: 0;
  margin: 0;
  padding: 0;
  overflow: auto;
  list-style: none;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
}

.sampleList li {
  border-bottom: 1px solid var(--line);
}

.sampleList button {
  width: 100%;
  min-height: 64px;
  border: 0;
  border-radius: 0;
  text-align: left;
  display: grid;
  gap: 4px;
}

.sampleList button.active {
  background: #eaf2ff;
}

.sampleList button.done .listTitle::after {
  content: " done";
  color: var(--good);
  font-weight: 600;
}

.listTitle {
  font-weight: 650;
}

.listSub {
  color: var(--muted);
  font-size: 12px;
}

.reviewPane {
  display: grid;
  gap: 14px;
  min-width: 0;
}

.itemNav {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

#itemTitle {
  color: var(--muted);
  text-align: center;
  overflow-wrap: anywhere;
}

.metaGrid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 8px;
}

.metaItem {
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 8px;
  min-width: 0;
}

.metaKey {
  color: var(--muted);
  font-size: 12px;
}

.metaValue {
  overflow-wrap: anywhere;
}

.textBlock {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  margin: 0;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fbfcfe;
  padding: 10px;
  max-height: 320px;
  overflow: auto;
}

.twoCol, .annotationGrid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

ul {
  margin: 0;
  padding-left: 18px;
}

li {
  margin: 4px 0;
}

.clauseCard, .annotationCard {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px;
  background: #fbfcfe;
  margin-bottom: 10px;
}

.annotationHeader {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 8px;
}

.badge {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 2px 8px;
  font-size: 12px;
  border: 1px solid var(--line);
}

.badge.correct {
  color: var(--good);
  border-color: #abefc6;
  background: #ecfdf3;
}

.badge.incorrect {
  color: var(--bad);
  border-color: #fecdca;
  background: #fef3f2;
}

.badge.warn {
  color: var(--warn);
  border-color: #fedf89;
  background: #fffaeb;
}

.jsonSmall {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  max-height: 180px;
  overflow: auto;
  padding: 8px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fff;
  font-size: 12px;
}

.decisionGrid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 12px;
}

label {
  display: grid;
  gap: 6px;
  color: var(--muted);
}

label > select, label > textarea {
  color: var(--text);
}

.saveRow {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 12px;
}

#saveStatus {
  color: var(--muted);
}

@media (max-width: 920px) {
  .topbar, .layout {
    display: block;
  }
  .topActions {
    justify-content: flex-start;
    margin-top: 12px;
  }
  .sidebar {
    position: static;
    height: auto;
    margin-bottom: 16px;
  }
  .sampleList {
    max-height: 260px;
  }
  .twoCol, .annotationGrid, .decisionGrid {
    grid-template-columns: 1fr;
  }
}
"""


_APP_JS = """
const DATA = window.COPAL_ADJUDICATION_DATA;
const records = DATA.records || [];
const metadata = DATA.metadata || {};
const storageKey = `copal-adjudication:${metadata.source_run || "unknown"}:${metadata.record_count || records.length}`;

let decisions = loadDecisions();
let current = 0;
let filtered = records.slice();

const el = (id) => document.getElementById(id);

function loadDecisions() {
  const raw = localStorage.getItem(storageKey);
  if (!raw) return {};
  return JSON.parse(raw);
}

function saveDecisions() {
  localStorage.setItem(storageKey, JSON.stringify(decisions));
}

function text(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

function short(value, max = 90) {
  const s = text(value).replace(/\\s+/g, " ").trim();
  return s.length > max ? `${s.slice(0, max - 1)}...` : s;
}

function decisionLabel(value) {
  if (value === true) return "Correct";
  if (value === false) return "Incorrect";
  return "Unknown";
}

function decisionClass(value) {
  if (value === true) return "correct";
  if (value === false) return "incorrect";
  return "warn";
}

function setText(id, value) {
  el(id).textContent = value;
}

function render() {
  applyFilters();
  renderProgress();
  renderList();
  if (!filtered.length) {
    setText("itemTitle", "No records match the current filter.");
    return;
  }
  if (current >= filtered.length) current = filtered.length - 1;
  if (current < 0) current = 0;
  renderRecord(filtered[current]);
}

function applyFilters() {
  const query = el("searchBox").value.toLowerCase().trim();
  const status = el("statusFilter").value;
  filtered = records.filter((record) => {
    const saved = decisions[record.review_id];
    if (status === "open" && saved && saved.final_decision) return false;
    if (status === "done" && (!saved || !saved.final_decision)) return false;
    if (!query) return true;
    return JSON.stringify(record).toLowerCase().includes(query);
  });
}

function renderProgress() {
  const done = records.filter((record) => decisions[record.review_id] && decisions[record.review_id].final_decision).length;
  const total = records.length;
  setText("runMeta", `${total} disagreement records from ${metadata.source_run || "unknown run"}`);
  setText("progressText", `${done}/${total} adjudicated`);
  el("progressBar").style.width = total ? `${Math.round((done / total) * 100)}%` : "0%";
}

function renderList() {
  const list = el("sampleList");
  list.innerHTML = "";
  filtered.forEach((record, index) => {
    const item = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    const saved = decisions[record.review_id];
    btn.className = `${index === current ? "active" : ""} ${saved && saved.final_decision ? "done" : ""}`;
    btn.addEventListener("click", () => {
      current = index;
      render();
    });
    const title = document.createElement("div");
    title.className = "listTitle";
    title.textContent = `${record.review_id} ${record.task}`;
    const sub = document.createElement("div");
    sub.className = "listSub";
    sub.textContent = short(record.sample_id, 110);
    btn.append(title, sub);
    item.appendChild(btn);
    list.appendChild(item);
  });
}

function renderRecord(record) {
  setText("itemTitle", `${current + 1}/${filtered.length} ${record.sample_id}`);
  renderMeta(record);
  const input = record.input || {};
  setText("queryText", input.query || "");
  setText("responseText", input.response_text || "");
  renderContract(input.adjudication_contract || {});
  renderClauses(input.active_clauses || []);
  renderAnnotations(record);
  renderAdjudicationForm(record);
}

function renderMeta(record) {
  const container = el("caseMeta");
  container.innerHTML = "";
  const rows = [
    ["Review ID", record.review_id],
    ["Task", record.task],
    ["Decision field", record.decision && record.decision.field],
  ];
  const strata = record.strata || {};
  Object.keys(strata).sort().forEach((key) => {
    if (key === "response_model" || key === "gemini_correct") return;
    rows.push([key, strata[key]]);
  });
  rows.forEach(([key, value]) => {
    const box = document.createElement("div");
    box.className = "metaItem";
    const k = document.createElement("div");
    k.className = "metaKey";
    k.textContent = key;
    const v = document.createElement("div");
    v.className = "metaValue";
    v.textContent = text(value);
    box.append(k, v);
    container.appendChild(box);
  });
}

function renderContract(contract) {
  renderListItems("requiredList", contract.required_obligations || []);
  renderListItems("forbiddenList", contract.forbidden_outcomes || []);
}

function renderListItems(id, rows) {
  const list = el(id);
  list.innerHTML = "";
  rows.forEach((row) => {
    const item = document.createElement("li");
    item.textContent = row.description || text(row);
    list.appendChild(item);
  });
}

function renderClauses(clauses) {
  const container = el("clauses");
  container.innerHTML = "";
  clauses.forEach((clause) => {
    const card = document.createElement("div");
    card.className = "clauseCard";
    const title = document.createElement("strong");
    title.textContent = `${clause.clause_id || "clause"} ${clause.effect ? `(${clause.effect})` : ""}`;
    const body = document.createElement("pre");
    body.className = "jsonSmall";
    body.textContent = JSON.stringify(clause, null, 2);
    card.append(title, body);
    container.appendChild(card);
  });
  if (!clauses.length) {
    container.textContent = "No active clauses recorded for this sample.";
  }
}

function renderAnnotations(record) {
  const container = el("annotationCards");
  container.innerHTML = "";
  const values = (record.decision && record.decision.values) || {};
  Object.keys(record.annotations || {}).sort().forEach((model) => {
    const ann = record.annotations[model];
    const card = document.createElement("div");
    card.className = "annotationCard";
    const header = document.createElement("div");
    header.className = "annotationHeader";
    const modelName = document.createElement("strong");
    modelName.textContent = model;
    const badge = document.createElement("span");
    badge.className = `badge ${decisionClass(values[model])}`;
    badge.textContent = decisionLabel(values[model]);
    header.append(modelName, badge);

    const facts = document.createElement("div");
    facts.className = "metaGrid";
    const fields = [
      ["error_type", ann.error_type],
      ["confidence", ann.confidence],
      ["satisfies_required_obligations", ann.satisfies_required_obligations],
      ["triggers_forbidden_outcome", ann.triggers_forbidden_outcome],
    ];
    fields.forEach(([key, value]) => {
      if (value === undefined) return;
      const box = document.createElement("div");
      box.className = "metaItem";
      const k = document.createElement("div");
      k.className = "metaKey";
      k.textContent = key;
      const v = document.createElement("div");
      v.className = "metaValue";
      v.textContent = text(value);
      box.append(k, v);
      facts.appendChild(box);
    });

    const rationaleTitle = document.createElement("h3");
    rationaleTitle.textContent = "Rationale";
    const rationale = document.createElement("pre");
    rationale.className = "textBlock";
    rationale.textContent = ann.rationale || "";

    const checks = document.createElement("pre");
    checks.className = "jsonSmall";
    checks.textContent = JSON.stringify({
      obligation_checks: ann.obligation_checks || [],
      forbidden_checks: ann.forbidden_checks || [],
    }, null, 2);

    card.append(header, facts, rationaleTitle, rationale, checks);
    container.appendChild(card);
  });
}

function renderAdjudicationForm(record) {
  const saved = decisions[record.review_id] || {};
  el("finalDecision").value = saved.final_decision || "";
  fillErrorTypes(saved.final_error_type || "");
  fillRationaleSources(record, saved.rationale_source || "");
  el("notes").value = saved.notes || "";
  setText("saveStatus", saved.saved_at ? `Saved ${saved.saved_at}` : "");
}

function fillErrorTypes(selected) {
  const select = el("errorType");
  select.innerHTML = "";
  (metadata.error_types || ["none", "other"]).forEach((type) => {
    const opt = document.createElement("option");
    opt.value = type;
    opt.textContent = type;
    if (type === selected) opt.selected = true;
    select.appendChild(opt);
  });
}

function fillRationaleSources(record, selected) {
  const select = el("rationaleSource");
  select.innerHTML = "";
  ["custom", ...Object.keys(record.annotations || {}).sort()].forEach((source) => {
    const opt = document.createElement("option");
    opt.value = source;
    opt.textContent = source;
    if (source === selected) opt.selected = true;
    select.appendChild(opt);
  });
}

function currentRecord() {
  return filtered[current];
}

function saveCurrentDecision() {
  const record = currentRecord();
  if (!record) return;
  const finalDecision = el("finalDecision").value;
  decisions[record.review_id] = {
    review_id: record.review_id,
    sample_id: record.sample_id,
    task: record.task,
    decision_field: record.decision.field,
    final_decision: finalDecision,
    final_response_correct: finalDecision === "correct" ? true : finalDecision === "incorrect" ? false : null,
    final_error_type: el("errorType").value,
    rationale_source: el("rationaleSource").value,
    notes: el("notes").value,
    saved_at: new Date().toISOString(),
  };
  saveDecisions();
  render();
}

function exportPayload() {
  const ordered = records.map((record) => decisions[record.review_id]).filter(Boolean);
  return {
    metadata: {
      ...metadata,
      exported_at: new Date().toISOString(),
      storage_key: storageKey,
      adjudicated_count: ordered.filter((row) => row.final_decision).length,
    },
    adjudications: ordered,
  };
}

function exportJson() {
  const blob = new Blob([JSON.stringify(exportPayload(), null, 2) + "\\n"], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "copal_adjudications.json";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

async function copyJson() {
  await navigator.clipboard.writeText(JSON.stringify(exportPayload(), null, 2));
  setText("saveStatus", "Export JSON copied to clipboard.");
}

function clearDecisions() {
  if (!confirm("Clear all local adjudication decisions for this run?")) return;
  decisions = {};
  saveDecisions();
  render();
}

el("searchBox").addEventListener("input", () => { current = 0; render(); });
el("statusFilter").addEventListener("change", () => { current = 0; render(); });
el("prevBtn").addEventListener("click", () => { current -= 1; render(); });
el("nextBtn").addEventListener("click", () => { current += 1; render(); });
el("saveBtn").addEventListener("click", saveCurrentDecision);
el("exportBtn").addEventListener("click", exportJson);
el("copyBtn").addEventListener("click", copyJson);
el("clearBtn").addEventListener("click", clearDecisions);

render();
"""
