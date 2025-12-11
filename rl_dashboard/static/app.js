const state = {
  runs: [],
  selectedRun: null,
  traces: [],
  offset: 0,
  limit: 50,
  total: 0,
  mode: "",
  tail: true,
  metricTags: [],
  selectedTag: "",
};

const runsList = document.getElementById("runs-list");
const traceList = document.getElementById("trace-list");
const detailPanel = document.getElementById("trace-detail");
const detailTitle = document.getElementById("detail-title");
const traceMeta = document.getElementById("trace-meta");
const modeFilter = document.getElementById("mode-filter");
const pageSize = document.getElementById("page-size");
const tailToggle = document.getElementById("tail-toggle");
const metricTagsSel = document.getElementById("metric-tags");
const metricTail = document.getElementById("metric-tail");
const metricLimit = document.getElementById("metric-limit");
const metricRefresh = document.getElementById("metric-refresh");
const metricCanvas = document.getElementById("metric-canvas");
const metricMeta = document.getElementById("metric-meta");

async function fetchRuns() {
  runsList.innerHTML = `<div class="empty">Loading runs...</div>`;
  try {
    const res = await fetch("/api/runs");
    const data = await res.json();
    state.runs = data.runs || [];
    renderRuns();
  } catch (err) {
    runsList.innerHTML = `<div class="empty">Failed to load runs</div>`;
  }
}

function renderRuns() {
  if (!state.runs.length) {
    runsList.innerHTML = `<div class="empty">No runs with traces.jsonl found.</div>`;
    return;
  }
  runsList.innerHTML = "";
  state.runs.forEach((run) => {
    const card = document.createElement("div");
    card.className = "run-card" + (run.name === state.selectedRun ? " active" : "");
    const sizeMb = (run.size_bytes / (1024 * 1024)).toFixed(2);
    card.innerHTML = `
      <div class="title">${run.name}</div>
      <div class="hint">traces.jsonl • ${sizeMb} MB</div>
    `;
    card.onclick = () => {
      state.selectedRun = run.name;
      state.offset = 0;
      state.tail = true;
      renderRuns();
      loadTraces();
      loadMetricTags();
      loadRunConfig();
    };
    runsList.appendChild(card);
  });
}

async function loadTraces(direction = "none") {
  if (!state.selectedRun) {
    traceList.innerHTML = `<div class="empty">Select a run to view traces.</div>`;
    return;
  }

  const params = new URLSearchParams();
  params.set("limit", state.limit);
  params.set("offset", state.offset);
  if (state.mode) params.set("mode", state.mode);
  if (state.tail) params.set("tail", "true");

  traceList.innerHTML = `<div class="empty">Loading traces...</div>`;

  try {
    const res = await fetch(`/api/runs/${state.selectedRun}/traces?${params.toString()}`);
    const data = await res.json();
    state.traces = data.traces || [];
    state.offset = data.offset || 0;
    state.total = data.total || 0;
    state.tail = Boolean(data.tail);
    renderTraceList();
    traceMeta.textContent = `${state.total} traces • offset ${state.offset}`;
  } catch (err) {
    traceList.innerHTML = `<div class="empty">Failed to load traces</div>`;
  }
}

function renderTraceList() {
  if (!state.traces.length) {
    traceList.innerHTML = `<div class="empty">No traces found for this slice.</div>`;
    return;
  }
  traceList.innerHTML = "";
  state.traces.forEach((trace, idx) => {
    const card = document.createElement("div");
    card.className = "trace-card";
    const reward = trace.reward ?? "—";
    const correct = trace.eval_result?.correctness;
    const compiled = trace.eval_result?.compiled;
    const turn = trace.turn ?? trace?.metrics?.turn ?? 0;
    card.innerHTML = `
      <div class="title">
        <span>${trace.meta?.summary || "Trace"}</span>
        <span class="badge">${trace.mode || "?"}</span>
      </div>
      <div class="meta">
        <span class="badge ${compiled ? "good" : "bad"}">compiled ${compiled ? "yes" : "no"}</span>
        <span class="badge ${correct ? "good" : "bad"}">correct ${correct ? "yes" : "no"}</span>
        <span class="badge">turn ${turn}</span>
        <span class="badge">reward ${typeof reward === "number" ? reward.toFixed(3) : reward}</span>
        <span class="badge">line ${trace.meta?.line_index ?? "?"}</span>
      </div>
    `;
    card.onclick = () => renderDetail(trace);
    traceList.appendChild(card);
  });
}

function renderDetail(trace) {
  detailTitle.textContent = `L${trace.level} P${trace.problem_id} • ${trace.mode}`;
  detailPanel.innerHTML = "";

  const sections = [];

  const promptMessages = trace.prompt_messages || [];
  const promptText = promptMessages
    .map((m) => `# ${m.role}\n${m.content}`)
    .join("\n\n");
  sections.push(section("Prompt", `<pre>${escapeHtml(promptText || "n/a")}</pre>`));

  const responseText = [
    trace.response?.thought ? `<h4>Thought</h4><pre>${escapeHtml(trace.response.thought)}</pre>` : "",
    trace.response?.kernel ? `<h4>Kernel</h4><pre>${escapeHtml(trace.response.kernel)}</pre>` : "",
    trace.response?.raw ? `<h4>Raw</h4><pre>${escapeHtml(trace.response.raw)}</pre>` : "",
  ].join("");
  sections.push(section("Model Output", responseText || "<div class='hint'>No output captured.</div>"));

  const evalResult = trace.eval_result || {};
  const evalHtml = `
    <div class="meta">
      <span class="badge ${evalResult.compiled ? "good" : "bad"}">compiled ${evalResult.compiled ? "yes" : "no"}</span>
      <span class="badge ${evalResult.correctness ? "good" : "bad"}">correct ${evalResult.correctness ? "yes" : "no"}</span>
      <span class="badge">tests ${evalResult.tests_passed ?? 0}/${evalResult.tests_total ?? 0}</span>
      <span class="badge">speedup ${evalResult.speedup ? evalResult.speedup.toFixed(2) + "x" : "—"}</span>
    </div>
    ${evalResult.error_message ? `<pre>${escapeHtml(evalResult.error_message)}</pre>` : ""}
  `;
  sections.push(section("Evaluation", evalHtml));

  const rewardText = trace.reward_breakdown
    ? Object.entries(trace.reward_breakdown)
        .map(([k, v]) => `${k}: ${Number(v).toFixed(3)}`)
        .join("\n")
    : "n/a";
  sections.push(section("Reward", `<pre>${escapeHtml(rewardText)}</pre>`));

  const metrics = trace.metrics || {};
  const metricText = Object.entries(metrics)
    .map(([k, v]) => `${k}: ${v}`)
    .join("\n");
  sections.push(section("Metrics", `<pre>${escapeHtml(metricText)}</pre>`));

  sections.forEach((s) => detailPanel.appendChild(s));
}

function section(title, innerHtml) {
  const wrap = document.createElement("div");
  wrap.className = "section";
  wrap.innerHTML = `<h3>${title}</h3>${innerHtml}`;
  return wrap;
}

function escapeHtml(str) {
  return (str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

document.getElementById("refresh-btn").onclick = () => loadTraces();
document.getElementById("prev-btn").onclick = () => {
  state.tail = false;
  state.offset = Math.max(state.offset - state.limit, 0);
  loadTraces("prev");
};
document.getElementById("next-btn").onclick = () => {
  state.tail = false;
  const nextOffset = state.offset + state.limit;
  if (nextOffset < state.total) {
    state.offset = nextOffset;
    loadTraces("next");
  }
};

modeFilter.onchange = () => {
  state.mode = modeFilter.value;
  state.offset = 0;
  loadTraces();
};

pageSize.onchange = () => {
  state.limit = Number(pageSize.value);
  state.offset = 0;
  loadTraces();
};

tailToggle.onchange = () => {
  state.tail = tailToggle.checked;
  loadTraces();
};

async function loadRunConfig() {
  if (!state.selectedRun) {
    document.getElementById("run-hint").textContent = "Select a run with traces.jsonl";
    return;
  }
  try {
    const res = await fetch(`/api/runs/${state.selectedRun}/config`);
    if (!res.ok) throw new Error("no config");
    const data = await res.json();
    const cfg = data.config || {};
    const model = cfg.model_name || cfg.config?.model_name;
    const mode = cfg.mode || cfg.config?.mode;
    document.getElementById("run-hint").textContent = `Model: ${model || "?"} • Mode: ${mode || "?"}`;
  } catch {
    document.getElementById("run-hint").textContent = "Run config not found";
  }
}

metricRefresh.onclick = () => loadMetricSeries();
metricTail.onchange = () => loadMetricSeries();
metricLimit.onchange = () => loadMetricSeries();
metricTagsSel.onchange = () => {
  state.selectedTag = metricTagsSel.value;
  loadMetricSeries();
};

async function loadMetricTags() {
  if (!state.selectedRun) {
    metricTagsSel.innerHTML = `<option value="">Select a run first</option>`;
    state.metricTags = [];
    state.selectedTag = "";
    return;
  }
  metricTagsSel.innerHTML = `<option>Loading...</option>`;
  try {
    const res = await fetch(`/api/runs/${state.selectedRun}/tb/tags`);
    const data = await res.json();
    state.metricTags = data.tags || [];
    if (!state.metricTags.length) {
      metricTagsSel.innerHTML = `<option value="">No scalar tags</option>`;
      return;
    }
    metricTagsSel.innerHTML = state.metricTags
      .map((t, idx) => `<option value="${t}" ${idx === 0 ? "selected" : ""}>${t}</option>`)
      .join("");
    state.selectedTag = state.metricTags[0];
    loadMetricSeries();
  } catch (err) {
    metricTagsSel.innerHTML = `<option value="">Failed to load tags</option>`;
  }
}

async function loadMetricSeries() {
  if (!state.selectedRun || !state.selectedTag) {
    return;
  }
  const params = new URLSearchParams();
  params.set("tag", state.selectedTag);
  params.set("limit", metricLimit.value);
  if (metricTail.checked) params.set("tail", "true");

  metricMeta.textContent = "Loading metric...";
  try {
    const res = await fetch(`/api/runs/${state.selectedRun}/tb/scalars?${params.toString()}`);
    const data = await res.json();
    drawMetricChart(data.values || []);
    metricMeta.textContent = `${data.returned}/${data.total} points · start ${data.start_index}`;
  } catch (err) {
    metricMeta.textContent = "Failed to load metric";
  }
}

function drawMetricChart(values) {
  if (!metricCanvas) return;
  const ctx = metricCanvas.getContext("2d");
  const width = metricCanvas.width = metricCanvas.clientWidth;
  const height = metricCanvas.height = metricCanvas.clientHeight;
  ctx.clearRect(0, 0, width, height);
  if (!values.length) {
    ctx.fillStyle = "#7f8bb3";
    ctx.fillText("No data", 10, 20);
    return;
  }
  const xs = values.map((v) => v.step);
  const ys = values.map((v) => v.value);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const pad = 12;
  ctx.strokeStyle = "#2b3452";
  ctx.lineWidth = 1;
  ctx.strokeRect(0, 0, width, height);

  ctx.strokeStyle = "#7df9ff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  values.forEach((pt, idx) => {
    const x = scale(pt.step, minX, maxX || 1, pad, width - pad);
    const y = scale(pt.value, minY, maxY || 1, height - pad, pad);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function scale(val, min, max, outMin, outMax) {
  if (max === min) return (outMin + outMax) / 2;
  return outMin + ((val - min) / (max - min)) * (outMax - outMin);
}

fetchRuns();
