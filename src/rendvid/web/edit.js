const $ = (s) => document.querySelector(s);
const JOB = location.pathname.split("/").pop();
const TRANS_LABEL = { cut: "cut", fade: "fade", dissolve: "dissolve", wipeleft: "wipe", slideleft: "slide", circleopen: "circle", fadeblack: "dip to black" };

let data = null;   // GET /edit payload
let edit = null;   // working copy
let sel = -1;      // selected clip index
let dirty = false;
let busy = false;
let pollTimer = null;

async function api(path, init) {
  const r = await fetch(path, init);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body;
}
const json = (method, body) => ({ method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const fmt = (s) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;
const scene = (n) => data.scenes[n - 1];

function segmented(el, values, initial, onChange, labels = {}) {
  el.innerHTML = "";
  for (const v of values) {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = labels[v] || v; b.dataset.v = v;
    b.onclick = () => { setSeg(el, v); onChange && onChange(v); };
    el.append(b);
  }
  setSeg(el, initial);
}
const setSeg = (el, v) => { el.dataset.value = v; el.querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.v === v)); };

function markDirty() { dirty = true; $("#dirty").hidden = false; renderTimeline(); }

// --- load ------------------------------------------------------------------------
async function load() {
  data = await api(`/api/jobs/${JOB}/edit`);
  edit = data.edit;
  document.body.dataset.mode = data.mode === "song" ? "song" : "story";
  const job = await api(`/api/jobs/${JOB}`);
  $("#jobTitle").textContent = job.spec.title || JOB;
  document.title = `Edit · ${job.spec.title || JOB}`;
  buildStatic();
  renderAll();
  setPreview("final");
  watchJob(job);
}

function buildStatic() {
  segmented($("#tabs"), ["Clip", "Text", "Audio", "Look", "Export"], "Clip", showTab);
  showTab("Clip");
  segmented($("#previewMode"), ["final", "clip"], "final", setPreview, { final: "Exported video", clip: "Selected clip" });
  $("#transition").innerHTML = Object.keys(TRANS_LABEL).filter(t => data.transitions.includes(t))
    .map(t => `<option value="${t}">${TRANS_LABEL[t]}</option>`).join("");
  $("#voice").innerHTML = `<option value="">None</option>` + Object.entries(data.voices).map(([k, v]) => `<option value="${k}">${v}</option>`).join("");
  segmented($("#aspect"), ["16:9", "9:16", "1:1"], edit.export.aspect, v => { edit.export.aspect = v; markDirty(); });
  $("#filters").innerHTML = "";
  for (const f of data.filters) {
    const b = document.createElement("button");
    b.textContent = f; b.dataset.v = f;
    b.onclick = () => { edit.look.filter = f; renderLook(); markDirty(); };
    $("#filters").append(b);
  }
}

function showTab(t) { document.querySelectorAll(".tab").forEach(el => el.classList.toggle("on", el.dataset.tab === t)); setSeg($("#tabs"), t); }

function renderAll() {
  renderTimeline(); renderClip(); renderTexts(); renderLyrics(); renderAudio(); renderLook(); renderExport(); renderVersions();
}

// --- timeline ----------------------------------------------------------------------
function clipLen(c) {
  const src = scene(c.scene).duration || 0;
  const out = c.trim_out ?? src;
  return Math.max(0, out - c.trim_in) / c.speed;
}

function renderTimeline() {
  const tl = $("#timeline");
  tl.innerHTML = "";
  let total = 0;
  edit.clips.forEach((c, i) => {
    const sc = scene(c.scene);
    const len = clipLen(c);
    total += len - (i > 0 && edit.clips[i - 1].transition !== "cut" ? edit.clips[i - 1].transition_dur : 0);
    const el = document.createElement("div");
    el.className = "tclip" + (i === sel ? " sel" : "") + (sc.clip ? "" : " missing");
    el.draggable = true;
    el.dataset.i = i;
    el.innerHTML = `${sc.clip ? `<video muted preload="metadata" src="${sc.clip}#t=${(c.trim_in + 0.5).toFixed(1)}"></video>` : `<video></video>`}
      <div class="cap"><b>Scene ${c.scene}</b><span>${len.toFixed(1)}s${c.speed !== 1 ? ` · ${c.speed}×` : ""}</span></div>`;
    el.onclick = () => { sel = i; renderTimeline(); renderClip(); showTab("Clip"); if ($("#previewMode").dataset.value === "clip") setPreview("clip"); };
    el.ondragstart = (e) => { e.dataTransfer.setData("text/plain", i); el.classList.add("dragging"); };
    el.ondragend = () => el.classList.remove("dragging");
    el.ondragover = (e) => e.preventDefault();
    el.ondrop = (e) => {
      e.preventDefault();
      const from = +e.dataTransfer.getData("text/plain");
      if (from === i) return;
      const [moved] = edit.clips.splice(from, 1);
      edit.clips.splice(i, 0, moved);
      sel = i;
      markDirty(); renderClip();
    };
    tl.append(el);
    if (i < edit.clips.length - 1) {
      const t = document.createElement("div");
      t.className = "trans" + (c.transition !== "cut" ? " on" : "");
      t.textContent = c.transition === "cut" ? "|" : "⨯";
      t.title = TRANS_LABEL[c.transition];
      tl.append(t);
    }
  });
  $("#timelineInfo").textContent = `${edit.clips.length} clips · ${fmt(total)}`;
}

// --- clip inspector -----------------------------------------------------------------
function renderClip() {
  const c = edit.clips[sel];
  $("#noClip").hidden = !!c;
  $("#clipPane").hidden = !c;
  if (!c) return;
  const sc = scene(c.scene);
  $("#clipTitle").textContent = `Clip ${sel + 1} · Scene ${c.scene}${sc.new_shot ? " · new shot" : ""} · source ${sc.duration.toFixed(1)}s`;
  $("#trimIn").value = c.trim_in;
  $("#trimIn").max = sc.duration;
  $("#trimOut").value = (c.trim_out ?? sc.duration).toFixed(2);
  $("#trimOut").max = sc.duration;
  $("#speed").value = String(c.speed);
  $("#transition").value = c.transition;
  $("#transDur").value = c.transition_dur;
  $("#scenePrompt").value = sc.prompt;
  $("#sceneSeed").value = sc.seed ?? "";
  $("#narration").value = edit.narrations[c.scene - 1] || "";
}

const onClip = (id, fn) => $(id).addEventListener("change", () => { const c = edit.clips[sel]; if (c) { fn(c); markDirty(); renderClip(); } });
onClip("#trimIn", c => c.trim_in = Math.max(0, +$("#trimIn").value));
onClip("#trimOut", c => { const v = +$("#trimOut").value; c.trim_out = v >= scene(c.scene).duration - 0.01 ? null : v; });
onClip("#speed", c => c.speed = +$("#speed").value);
onClip("#transition", c => c.transition = $("#transition").value);
onClip("#transDur", c => c.transition_dur = +$("#transDur").value);
onClip("#narration", c => edit.narrations[c.scene - 1] = $("#narration").value);

$("#moveLeft").onclick = () => { if (sel > 0) { [edit.clips[sel - 1], edit.clips[sel]] = [edit.clips[sel], edit.clips[sel - 1]]; sel--; markDirty(); renderClip(); } };
$("#moveRight").onclick = () => { if (sel < edit.clips.length - 1) { [edit.clips[sel + 1], edit.clips[sel]] = [edit.clips[sel], edit.clips[sel + 1]]; sel++; markDirty(); renderClip(); } };
$("#dupClip").onclick = () => { if (sel >= 0) { edit.clips.splice(sel + 1, 0, { ...edit.clips[sel] }); sel++; markDirty(); renderClip(); } };
$("#delClip").onclick = () => {
  if (sel < 0) return;
  if (edit.clips.length === 1) return alert("The timeline needs at least one clip.");
  edit.clips.splice(sel, 1);
  sel = Math.min(sel, edit.clips.length - 1);
  markDirty(); renderClip();
};
$("#playClip").onclick = () => setPreview("clip", true);

async function redo(newVariation) {
  const c = edit.clips[sel];
  if (!c || busy) return;
  const seedVal = $("#sceneSeed").value;
  const body = {
    scenes: [{ scene: c.scene, prompt: $("#scenePrompt").value, seed: seedVal === "" ? null : +seedVal, new_variation: newVariation }],
    cascade: $("#cascade").checked,
    edit,
  };
  try {
    const job = await api(`/api/jobs/${JOB}/redo`, json("POST", body));
    dirty = false; $("#dirty").hidden = true;
    $("#taskMsg").textContent = `Re-rendering scene${job.redo_scenes.length > 1 ? "s" : ""} ${job.redo_scenes.join(", ")}…`;
    watchJob(job);
  } catch (e) { alert(e.message); }
}
$("#redoBtn").onclick = () => redo(false);
$("#variationBtn").onclick = () => redo(true);

// --- preview --------------------------------------------------------------------------
function setPreview(mode, play = false) {
  setSeg($("#previewMode"), mode);
  const v = $("#preview");
  if (mode === "clip" && edit.clips[sel] && scene(edit.clips[sel].scene).clip) {
    const c = edit.clips[sel];
    const out = c.trim_out ?? scene(c.scene).duration;
    v.src = `${scene(c.scene).clip.split("?")[0]}?${Date.now()}#t=${c.trim_in},${out}`;
    v.playbackRate = c.speed;
    $("#previewInfo").textContent = `Scene ${c.scene} · ${c.trim_in.toFixed(1)}–${out.toFixed(1)}s (raw clip, no audio)`;
  } else {
    v.src = `/outputs/${JOB}/final.mp4?${Date.now()}`;
    v.playbackRate = 1;
    $("#previewInfo").textContent = "Latest export";
    if (mode === "clip") $("#previewInfo").textContent = "Select a clip first";
  }
  applyPreviewLook();
  if (play) v.play().catch(() => {});
}

function applyPreviewLook() {
  const l = edit.look;
  const extra = { bw: " grayscale(1)", vintage: " sepia(0.35)", warm: " sepia(0.15) saturate(1.1)", cool: " hue-rotate(-8deg)", vivid: " saturate(1.35) contrast(1.08)", film: " contrast(1.1)", dream: " blur(0.6px) brightness(1.04)" }[l.filter] || "";
  // Only the raw clip gets the CSS approximation; the export already has the grade baked in.
  $("#preview").style.filter = $("#previewMode").dataset.value === "clip"
    ? `brightness(${1 + l.brightness}) contrast(${l.contrast}) saturate(${l.saturation})${extra}` : "";
}

// --- text ----------------------------------------------------------------------------------
const playhead = () => +($("#preview").currentTime || 0).toFixed(1);

function renderTexts() {
  const box = $("#texts");
  box.innerHTML = edit.texts.length ? "" : `<p class="hint">No titles yet.</p>`;
  edit.texts.forEach((t, i) => {
    const row = document.createElement("div");
    row.className = "row text";
    row.innerHTML = `<input type="text" data-k="text" placeholder="Text">
      <input type="number" data-k="start" step="0.1" title="Start (s)"><input type="number" data-k="end" step="0.1" title="End (s)">
      <select data-k="position"><option>top</option><option>center</option><option>bottom</option></select>
      <select data-k="size"><option value="s">S</option><option value="m">M</option><option value="l">L</option><option value="xl">XL</option></select>
      <select data-k="style"><option value="title">Title box</option><option value="caption">Outline</option></select>
      <button title="Remove">✕</button>`;
    row.querySelectorAll("[data-k]").forEach(el => {
      el.value = t[el.dataset.k];
      el.onchange = () => { t[el.dataset.k] = el.type === "number" ? +el.value : el.value; markDirty(); };
    });
    row.querySelector("button").onclick = () => { edit.texts.splice(i, 1); markDirty(); renderTexts(); };
    box.append(row);
  });
  $("#subsOn").checked = edit.subtitles_on;
}
$("#addText").onclick = () => {
  const s = playhead();
  edit.texts.push({ text: "", start: s, end: +(s + 3).toFixed(1), position: "center", size: "l", style: "title" });
  markDirty(); renderTexts();
  $("#texts .row:last-child input").focus();
};
$("#subsOn").onchange = () => { edit.subtitles_on = $("#subsOn").checked; markDirty(); };

function renderLyrics() {
  const box = $("#lyrics");
  box.innerHTML = edit.lyrics.length ? "" : `<p class="hint">No lyrics.</p>`;
  edit.lyrics.forEach((l, i) => {
    const row = document.createElement("div");
    row.className = "row lyric";
    row.innerHTML = `<input type="number" step="0.1"><input type="number" step="0.1"><input type="text"><button title="Remove">✕</button>`;
    const [a, b, t] = row.querySelectorAll("input");
    a.value = l[0]; b.value = l[1]; t.value = l[2];
    a.onchange = () => { l[0] = +a.value; markDirty(); };
    b.onchange = () => { l[1] = +b.value; markDirty(); };
    t.onchange = () => { l[2] = t.value; markDirty(); };
    a.ondblclick = () => { a.value = playhead(); a.onchange(); };
    row.querySelector("button").onclick = () => { edit.lyrics.splice(i, 1); markDirty(); renderLyrics(); };
    box.append(row);
  });
}
$("#addLyric").onclick = () => {
  const s = playhead();
  edit.lyrics.push([s, +(s + 2.5).toFixed(1), ""]);
  edit.lyrics.sort((x, y) => x[0] - y[0]);
  markDirty(); renderLyrics();
};

// --- audio -----------------------------------------------------------------------------
function renderAudio() {
  const sel = $("#musicFile");
  sel.innerHTML = `<option value="">None</option>` + data.audio_files.map(f => `<option>${f}</option>`).join("");
  sel.value = edit.audio.music_file || "";
  $("#musicVol").value = edit.audio.music_volume;
  $("#voice").value = edit.audio.voice || "";
  $("#voiceVol").value = edit.audio.voice_volume;
}
$("#musicFile").onchange = () => { edit.audio.music_file = $("#musicFile").value || null; markDirty(); };
$("#musicVol").onchange = () => { edit.audio.music_volume = +$("#musicVol").value; markDirty(); };
$("#voice").onchange = () => { edit.audio.voice = $("#voice").value || null; markDirty(); };
$("#voiceVol").onchange = () => { edit.audio.voice_volume = +$("#voiceVol").value; markDirty(); };
$("#musicUpload").onchange = async () => {
  const f = $("#musicUpload").files[0];
  if (!f) return;
  const fd = new FormData();
  fd.append("audio", f);
  try {
    const r = await api(`/api/jobs/${JOB}/audio`, { method: "POST", body: fd });
    data.audio_files.push(r.music_file);
    edit.audio.music_file = r.music_file;
    renderAudio(); markDirty();
  } catch (e) { alert(e.message); }
  $("#musicUpload").value = "";
};

// --- look & export ---------------------------------------------------------------------
function renderLook() {
  $("#filters").querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.v === edit.look.filter));
  $("#bright").value = edit.look.brightness;
  $("#contrast").value = edit.look.contrast;
  $("#sat").value = edit.look.saturation;
  applyPreviewLook();
}
for (const [id, k] of [["#bright", "brightness"], ["#contrast", "contrast"], ["#sat", "saturation"]]) {
  $(id).oninput = () => { edit.look[k] = +$(id).value; applyPreviewLook(); };
  $(id).onchange = markDirty;
}

function renderExport() {
  setSeg($("#aspect"), edit.export.aspect);
  $("#fit").value = edit.export.fit;
  $("#resolution").value = edit.export.resolution;
  $("#smooth").checked = edit.export.smooth;
  $("#matchSong").checked = edit.export.match_song;
}
$("#fit").onchange = () => { edit.export.fit = $("#fit").value; markDirty(); };
$("#resolution").onchange = () => { edit.export.resolution = $("#resolution").value; markDirty(); };
$("#smooth").onchange = () => { edit.export.smooth = $("#smooth").checked; markDirty(); };
$("#matchSong").onchange = () => { edit.export.match_song = $("#matchSong").checked; markDirty(); };

// --- versions ------------------------------------------------------------------------------
function renderVersions() {
  const s = $("#versions");
  s.innerHTML = `<option value="">Current version</option>` +
    data.versions.map(v => `<option value="${v}">${v.split("/").pop().replace(".mp4", "")}</option>`).join("");
  $("#restoreBtn").hidden = !data.versions.length;
  s.hidden = !data.versions.length;
}
$("#versions").onchange = () => {
  const v = $("#versions").value;
  $("#preview").src = (v || `/outputs/${JOB}/final.mp4`) + `?${Date.now()}`;
  setSeg($("#previewMode"), "final");
  $("#previewInfo").textContent = v ? "Earlier version" : "Latest export";
};
$("#restoreBtn").onclick = async () => {
  const v = $("#versions").value;
  if (!v) return;
  if (!confirm("Make this earlier version the current video? (The current one is kept as a version.)")) return;
  try { await api(`/api/jobs/${JOB}/restore`, json("POST", { version: v })); await reload(); } catch (e) { alert(e.message); }
};

// --- save / export / reset ----------------------------------------------------------------
$("#saveBtn").onclick = async () => {
  try {
    edit = await api(`/api/jobs/${JOB}/edit`, json("PUT", edit));
    dirty = false; $("#dirty").hidden = true; renderAll();
  } catch (e) { alert(e.message); }
};
$("#exportBtn").onclick = async () => {
  try {
    const job = await api(`/api/jobs/${JOB}/export`, json("POST", edit));
    dirty = false; $("#dirty").hidden = true;
    watchJob(job);
  } catch (e) { alert(e.message); }
};
$("#resetBtn").onclick = async () => {
  if (!confirm("Discard all edits and go back to the original cut? (Exports so far stay in versions.)")) return;
  try { await api(`/api/jobs/${JOB}/edit`, { method: "DELETE" }); await reload(); } catch (e) { alert(e.message); }
};

async function reload() {
  const keep = sel;
  data = await api(`/api/jobs/${JOB}/edit`);
  edit = data.edit;
  sel = Math.min(keep, edit.clips.length - 1);
  dirty = false; $("#dirty").hidden = true;
  renderAll();
  setPreview("final");
}

// --- task status ----------------------------------------------------------------------------
function watchJob(job) {
  clearTimeout(pollTimer);
  const active = job.status === "running" || job.status === "queued";
  const wasBusy = busy;
  busy = active;
  $("#jobState").textContent = job.status;
  $("#jobState").className = "chip " + (job.status === "done" ? "ok" : "no");
  $("#progress").hidden = !active;
  $("#progress i").style.width = `${(job.progress * 100).toFixed(1)}%`;
  for (const id of ["#exportBtn", "#redoBtn", "#variationBtn", "#saveBtn", "#resetBtn", "#restoreBtn"]) $(id).disabled = active;
  if (active) {
    $("#taskMsg").textContent = job.message;
    pollTimer = setTimeout(async () => watchJob(await api(`/api/jobs/${JOB}`)), 1500);
  } else {
    if (job.status === "failed") $("#taskMsg").textContent = "Failed: " + job.message;
    else if (wasBusy) { $("#taskMsg").textContent = job.status === "done" ? "Export finished." : job.message; reload(); }
  }
}

window.addEventListener("beforeunload", (e) => { if (dirty) { e.preventDefault(); e.returnValue = ""; } });
load().catch(e => { $("#jobState").textContent = "error"; alert(e.message); });
