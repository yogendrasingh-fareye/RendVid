const $ = (s) => document.querySelector(s);
const COLORS = {
  producer: "#4b5bdc", writer: "#b5527f", director: "#c0392b", designer: "#8e6c1f", dop: "#2f8a4f",
  supervisor: "#6a4fb5", sound: "#1f7a8c", editor: "#a35a00", critic: "#555c66", system: "#888",
};
let roster = {};
let current = null;
let lastRender = "";
let timer = null;

async function api(path, init) {
  const r = await fetch(path, init);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body;
}
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const initials = (role) => role.split(/\s+/).map(w => w[0]).join("").slice(0, 2).toUpperCase();

async function init() {
  roster = await api("/api/crew/roster");
  $("#roster").innerHTML = Object.entries(roster).map(([k, m]) =>
    `<div class="member" data-k="${k}"><span class="avatar" style="background:${COLORS[k]}">${initials(m.role)}</span>
     <div>${esc(m.role)}<small>${esc(m.model)} · idle</small></div></div>`).join("");
  document.querySelectorAll(".examples button").forEach(b => b.onclick = () => { $("#input").value = b.textContent; $("#input").focus(); });
  const fromHash = location.hash.slice(1);
  await refreshList();
  if (fromHash) open(fromHash);
}

async function refreshList() {
  const list = await api("/api/crew/productions");
  $("#prods").innerHTML = list.map(p => `<div class="prod ${p.id === current ? "sel" : ""}" data-id="${p.id}">
      <b>${esc(p.title)}</b><span>${esc(stageLabel(p.stage))}${p.busy ? " · working…" : ""}</span></div>`).join("")
    || `<p class="hint">No films yet.</p>`;
  document.querySelectorAll(".prod").forEach(el => el.onclick = () => open(el.dataset.id));
}

const stageLabel = (s) => ({
  briefing: "briefing", awaiting_brief: "brief to approve", planning: "storyboarding",
  awaiting_storyboard: "storyboard to approve", rendering: "rendering", reviewing: "QA review", done: "ready",
}[s] || s);

function open(id) {
  current = id;
  location.hash = id;
  lastRender = "";
  $("#welcome").hidden = true;
  poll();
  refreshList();
}

$("#newBtn").onclick = () => { location.href = "/crew"; };

async function poll() {
  clearTimeout(timer);
  if (!current) return;
  let prod;
  try { prod = await api(`/api/crew/productions/${current}`); } catch { return; }
  render(prod);
  const live = prod.busy || ["rendering", "reviewing", "planning", "briefing"].includes(prod.stage);
  timer = setTimeout(poll, live ? 1500 : 6000);
}

function render(prod) {
  // Activity + roster highlight
  $("#activity").hidden = !prod.busy && !(prod.job && ["running", "queued"].includes(prod.job.status));
  const act = prod.activity || (prod.job && ["running", "queued"].includes(prod.job.status) ? `Rendering: ${prod.job.message || ""}` : "");
  $("#activityText").textContent = act;
  document.querySelectorAll(".member").forEach(el => {
    const on = act.startsWith(roster[el.dataset.k].role);
    el.classList.toggle("active", on);
    el.querySelector("small").textContent = `${roster[el.dataset.k].model} · ${on ? "working" : "idle"}`;
  });
  if (prod.cost) { $("#cost").hidden = false; $("#cost").textContent = `Claude usage ≈ $${prod.cost.toFixed(2)}`; }

  const key = JSON.stringify([prod.messages.length, prod.stage, prod.job?.progress, prod.job?.status]);
  if (key === lastRender) return;
  const atBottom = $("#thread").scrollHeight - $("#thread").scrollTop - $("#thread").clientHeight < 80;
  lastRender = key;
  const lastOf = (kind) => [...prod.messages].reverse().find(m => m.kind === kind)?.id;
  const lastBrief = lastOf("brief"), lastBoard = lastOf("storyboard"), lastProgress = lastOf("progress");
  $("#thread").innerHTML = "";
  for (const m of prod.messages) $("#thread").append(message(m, prod, { lastBrief, lastBoard, lastProgress }));
  if (atBottom || prod.messages.length < 3) $("#thread").scrollTop = $("#thread").scrollHeight;
}

function message(m, prod, last) {
  const el = document.createElement("div");
  const role = m.from === "user" ? "You" : m.from === "system" ? "RendVid" : roster[m.from]?.role || m.from;
  el.className = `msg ${m.from === "user" ? "user" : ""} ${m.kind === "error" ? "error" : ""}`;
  el.innerHTML = m.from === "user" ? `<div class="body"></div>` :
    `<span class="avatar" style="background:${COLORS[m.from] || "#888"}">${initials(role)}</span>
     <div><div class="who">${esc(role)}</div><div class="body"></div></div>`;
  const body = el.querySelector(".body");
  body.textContent = m.text;
  const card = cardFor(m, prod, last);
  if (card) body.append(card);
  return el;
}

function cardFor(m, prod, last) {
  const c = document.createElement("div");
  c.className = "card";
  if (m.kind === "brief") {
    const b = m.data.brief || {};
    const rows = [["Title", b.title], ["Format", b.format], ["Language", b.language], ["Source", b.source],
      ["Length", `${b.minutes} min`], ["Aspect", b.aspect], ["Quality", b.quality], ["Style", b.style],
      ["Audience", b.audience], ["Notes", b.notes], ["Idea", b.idea], ["Script", b.script && (b.script.length > 300 ? b.script.slice(0, 300) + "…" : b.script)]];
    c.innerHTML = `<table>${rows.filter(r => r[1]).map(([k, v]) => `<tr><th>${k}</th><td>${esc(v)}</td></tr>`).join("")}</table>`;
    if (m.id === last.lastBrief && prod.stage === "awaiting_brief") addActions(c, [["Approve brief", () => approve("brief"), true]]);
    return c;
  }
  if (m.kind === "storyboard") {
    const rows = m.data.rows || [];
    c.innerHTML = `<div class="sb"><table>${rows.map(r => `<tr><td class="n">${r.line}</td><td>
        <div class="narr">${esc(r.narration || "(silent) " + (r.beat || ""))}</div>
        <div class="vis">🎥 ${esc(r.visual)}${r.characters.length ? ` · 👤 ${esc(r.characters.join(", "))}` : ""}</div></td></tr>`).join("")}</table></div>`;
    if (m.id === last.lastBoard && prod.stage === "awaiting_storyboard") {
      addActions(c, [["Approve & render", () => approve("storyboard"), true]]);
      c.querySelector(".actions").insertAdjacentHTML("beforeend", `<span class="hint">or reply with changes, e.g. “line 3 should be a close-up”</span>`);
    }
    return c;
  }
  if (m.kind === "script") {
    const lines = m.data.lines || [];
    c.innerHTML = `<details><summary>Read the script (${lines.length} lines)</summary><div class="sb"><table>${
      lines.map((l, i) => `<tr><td class="n">${i + 1}</td><td class="narr">${esc(l.narration || "(silent) " + (l.beat || ""))}</td></tr>`).join("")}</table></div></details>`;
    return c;
  }
  if (m.kind === "progress" && m.id === last.lastProgress && prod.job && prod.stage === "rendering") {
    c.innerHTML = `<div>${esc(prod.job.message || prod.job.status)}</div><div class="bar"><i style="width:${(100 * (prod.job.progress || 0)).toFixed(1)}%"></i></div>`;
    return c;
  }
  if (m.kind === "video") {
    c.innerHTML = `<video controls playsinline src="${esc(m.data.url)}"></video>
      <div class="actions"><a href="/edit/${esc(m.data.job_id)}"><button>✎ Open in editor</button></a>
      <a href="${esc(m.data.url.split("?")[0])}" download><button>Download</button></a></div>`;
    return c;
  }
  if (m.kind === "report" && m.data.bad) {
    c.innerHTML = `<table>${m.data.bad.map(b => `<tr><th>Line ${parseInt(b.shot.match(/\d+/))}</th><td>${esc(b.problem)}</td></tr>`).join("")}</table>`;
    if (prod.stage === "done") addActions(c, [["Reshoot these", () => send("Please reshoot the shots the QA Critic flagged, fixing exactly those problems."), true]]);
    return c;
  }
  return null;
}

function addActions(card, buttons) {
  const a = document.createElement("div");
  a.className = "actions";
  for (const [label, fn, primary] of buttons) {
    const b = document.createElement("button");
    b.textContent = label;
    if (primary) b.className = "primary";
    b.onclick = async () => { b.disabled = true; try { await fn(); } catch (e) { alert(e.message); b.disabled = false; } };
    a.append(b);
  }
  card.append(a);
}

async function approve(what) {
  await post(`/api/crew/productions/${current}/approve`, { what });
  lastRender = "";
  poll();
}

async function send(text) {
  text = (text ?? $("#input").value).trim();
  if (!text) return;
  $("#sendBtn").disabled = true;
  try {
    if (!current) {
      const prod = await post("/api/crew/productions", { text });
      open(prod.id);
    } else {
      await post(`/api/crew/productions/${current}/message`, { text });
      lastRender = "";
      poll();
    }
    $("#input").value = "";
  } catch (e) {
    alert(e.message);
  } finally {
    $("#sendBtn").disabled = false;
  }
}

$("#composer").onsubmit = (e) => { e.preventDefault(); send(); };
$("#input").onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } };
setInterval(() => refreshList().catch(() => {}), 8000);
init();
