const $ = (s) => document.querySelector(s);
const PRESET_INFO = {
  fast: "Fast: one pass, small size, 24 fps. Good for testing ideas.",
  balanced: "Balanced: one pass at medium size, upscaled to 720p.",
  max: "Max: low-res pass → 2× AI latent upscale → refine pass, then 1080p and smoothed to 48 fps. Slowest, sharpest.",
};
// Rough seconds per scene on an M5 / 16 GB (max measured; others estimated).
const SCENE_SECS = { fast: 30, balanced: 45, max: 90 };
const PLACEHOLDER = {
  story: "A lone astronaut explores an abandoned city on Mars and discovers a glowing garden",
  song: "e.g. A girl dances alone through neon-lit Mumbai streets at night, rain on the pavement, ending at sunrise on Marine Drive",
};
let mode = "story";
let songPlan = null;  // {lyrics, duration} from the last song plan
let opts = null;
let selectedJob = null;
let pollTimer = null;

async function api(path, init) {
  const r = await fetch(path, init);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body;
}

function segmented(el, values, initial, onChange) {
  el.innerHTML = "";
  for (const v of values) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = v;
    b.dataset.v = v;
    b.onclick = () => { setSeg(el, v); onChange && onChange(v); };
    el.append(b);
  }
  setSeg(el, initial);
}
const setSeg = (el, v) => { el.dataset.value = v; el.querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.v === v)); };

async function init() {
  opts = await api("/api/options");
  $("#style").innerHTML = opts.styles.map(s => `<option value="${s}">${s}</option>`).join("");
  $("#style").value = "cinematic";
  for (const [id, name] of Object.entries(opts.voices)) $("#voice").add(new Option(name, id));
  $("#voice").value = "af_heart";
  $("#negative").value = opts.default_negative;
  segmented($("#mode"), ["Story", "Music video", "Studio"], "Story", v => setMode({ Story: "story", "Music video": "song", Studio: "studio" }[v]));
  setMode("story");
  segmented($("#aspect"), opts.aspects, "16:9");
  segmented($("#preset"), opts.presets, "max", v => { $("#presetHint").textContent = PRESET_INFO[v]; renumber(); });
  $("#presetHint").textContent = PRESET_INFO.max;
  renderStatus(opts.models);
  refreshJobs();
}

function renderStatus(m) {
  const chip = (ok, label) => `<span class="chip ${ok ? "ok" : "no"}" title="${ok ? "ready" : "not ready"}">${label}</span>`;
  $("#status").innerHTML =
    chip(m.video, "Video model") + chip(m.text_encoder, "Text encoder") +
    chip(m.upscaler, "Upscaler") + chip(m.llm, `Brain: ${m.brain || "none"}`);
}

function setMode(m) {
  mode = m;
  document.body.dataset.mode = m;
  setSeg($("#mode"), { story: "Story", song: "Music video", studio: "Studio" }[m]);
  if (PLACEHOLDER[m]) $("#idea").placeholder = PLACEHOLDER[m];
  if (m === "studio" && window.studioInit) window.studioInit();
  $("#planBtn").textContent = m === "song" ? "Analyze song & write scenes" : "Write scenes";
}

const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;

$("#songFile").onchange = () => {
  songPlan = null;
  const f = $("#songFile").files[0];
  if (!f) return;
  const a = new Audio(URL.createObjectURL(f));
  a.onloadedmetadata = () => {
    const n = Math.max(1, 1 + Math.ceil((a.duration * 24 - 121) / 112));
    const over = a.duration > opts.song_max_seconds;
    $("#songInfo").textContent = over
      ? `${mmss(a.duration)} — too long (max ${opts.song_max_seconds / 60} min).`
      : `${mmss(a.duration)} → ${n} scenes · render ≈ ${fmtTime(n * SCENE_SECS[$("#preset").dataset.value])} on ${$("#preset").dataset.value}.`;
  };
};

// --- sound / extras ---------------------------------------------------------
$("#musicMode").onchange = () => {
  $("#musicUploadRow").hidden = $("#musicMode").value !== "upload";
  $("#musicGenRow").hidden = $("#musicMode").value !== "generate";
};
$("#image").onchange = () => {
  const f = $("#image").files[0];
  $("#imagePreview").hidden = !f;
  if (f) $("#imagePreview").src = URL.createObjectURL(f);
};

// --- scenes -----------------------------------------------------------------
function addScene(s = { prompt: "", narration: "", new_shot: false }) {
  const li = $("#sceneTpl").content.firstElementChild.cloneNode(true);
  li.querySelector(".prompt").value = s.prompt;
  li.querySelector(".narration").value = s.narration || "";
  li.querySelector(".newShot").checked = !!s.new_shot;
  li.querySelector(".up").onclick = () => { li.previousElementSibling && li.parentNode.insertBefore(li, li.previousElementSibling); renumber(); };
  li.querySelector(".down").onclick = () => { li.nextElementSibling && li.parentNode.insertBefore(li.nextElementSibling, li); renumber(); };
  li.querySelector(".del").onclick = () => { li.remove(); renumber(); };
  $("#scenes").append(li);
  renumber();
}

function renumber() {
  const items = [...$("#scenes").children];
  items.forEach((li, i) => li.querySelector(".num").textContent = `Scene ${i + 1}`);
  const secs = items.length ? Math.round(items.length * opts.scene_seconds + 0.2) : 0;
  const len = mode === "song" && songPlan ? `song ${mmss(songPlan.duration)}` : `≈ ${secs} s`;
  const est = fmtTime(items.length * SCENE_SECS[$("#preset").dataset.value || "max"]);
  $("#sceneSummary").textContent = `${items.length} scenes · ${len} · render ≈ ${est}`;
  $("#scenesWrap").hidden = items.length === 0;
}

function readScenes() {
  return [...$("#scenes").children].map(li => ({
    prompt: li.querySelector(".prompt").value.trim(),
    narration: li.querySelector(".narration").value.trim(),
    new_shot: li.querySelector(".newShot").checked,
  })).filter(s => s.prompt);
}

$("#addScene").onclick = () => addScene();

async function planSong() {
  const f = $("#songFile").files[0];
  if (!f) throw new Error("Choose your song file first");
  const fd = new FormData();
  fd.append("audio", f);
  fd.append("context", $("#idea").value);
  fd.append("style", $("#style").value);
  fd.append("use_lyrics", $("#useLyrics").checked);
  $("#planInfo").textContent = ($("#useLyrics").checked ? "Transcribing lyrics, then " : "") + "writing scenes in batches… (a few minutes for a full song)";
  const res = await api("/api/plan_song", { method: "POST", body: fd });
  songPlan = { lyrics: res.lyrics, duration: res.duration };
  return res;
}

$("#planBtn").onclick = async () => {
  const btn = $("#planBtn");
  btn.disabled = true;
  $("#planInfo").textContent = "Writing scenes… (about a minute)";
  try {
    const voice = $("#voice").value;
    const res = mode === "song" ? await planSong() : await api("/api/plan", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        idea: $("#idea").value, duration: +$("#duration").value,
        narration: !!voice, language: voice.startsWith("h") ? "Hindi" : "English",
        style: $("#style").value,
      }),
    });
    $("#scenes").innerHTML = "";
    $("#title").value = res.title || "";
    $("#characters").textContent = res.characters ? `Characters: ${res.characters}` : "";
    res.scenes.forEach(addScene);
    const lyr = mode === "song" ? (res.lyrics.length ? ` Found ${res.lyrics.length} lyric lines (${res.language}).` : " No lyrics found; scenes follow your description.") : "";
    $("#planInfo").textContent = (res.warning || "Scenes written by the AI brain. Edit anything before rendering.") + lyr;
  } catch (e) {
    $("#planInfo").textContent = "Error: " + e.message;
  } finally {
    btn.disabled = false;
  }
};

$("#renderBtn").onclick = async () => {
  const scenes = readScenes();
  if (!scenes.length) return alert("Add at least one scene.");
  const musicMode = $("#musicMode").value;
  const song = mode === "song";
  if (song && !$("#songFile").files[0]) return alert("Choose your song file.");
  if (!song && musicMode === "upload" && !$("#musicFile").files[0]) return alert("Choose an audio file or change Music.");
  const spec = {
    title: $("#title").value || $("#idea").value.slice(0, 60),
    idea: $("#idea").value,
    aspect: $("#aspect").dataset.value,
    preset: $("#preset").dataset.value,
    style: $("#style").value,
    negative: $("#negative").value,
    seed: +$("#seed").value,
    guidance: +$("#guidance").value,
    voice: song ? null : ($("#voice").value || null),
    subtitles: $("#subtitles").checked,
    music: song ? { mode: "upload", volume: 1 } : { mode: musicMode, prompt: $("#musicPrompt").value, volume: +$("#musicVol").value },
    scenes,
  };
  if (song) Object.assign(spec, {
    mode: "song",
    lyrics: songPlan ? songPlan.lyrics : [],
    lyric_subtitles: $("#lyricSubs").checked && !!songPlan,
  });
  const fd = new FormData();
  fd.append("spec", JSON.stringify(spec));
  if ($("#image").files[0]) fd.append("image", $("#image").files[0]);
  if (song) fd.append("music", $("#songFile").files[0]);
  else if (musicMode === "upload") fd.append("music", $("#musicFile").files[0]);
  $("#renderBtn").disabled = true;
  try {
    const job = await api("/api/jobs", { method: "POST", body: fd });
    selectedJob = job.id;
    refreshJobs();
  } catch (e) {
    alert(e.message);
  } finally {
    $("#renderBtn").disabled = false;
  }
};

// --- jobs -------------------------------------------------------------------
const fmtTime = (s) => s >= 3600 ? `${Math.floor(s / 3600)}h ${Math.round(s % 3600 / 60)}m` : s >= 60 ? `${Math.round(s / 60)} min` : `${Math.round(s)} s`;

function eta(job) {
  if (job.status !== "running" || !job.started || job.progress < 0.08) return "";
  const elapsed = Date.now() / 1000 - job.started;
  return ` · ~${fmtTime(elapsed / job.progress - elapsed)} left`;
}

async function refreshJobs() {
  clearTimeout(pollTimer);
  let list = [];
  try { list = await api("/api/jobs"); } catch { /* server restarting */ }
  const el = $("#jobs");
  if (!list.length) el.innerHTML = `<div class="empty">No videos yet.</div>`;
  else el.innerHTML = "";
  for (const job of list) el.append(jobCard(job));
  if (selectedJob) showJob(selectedJob);
  const active = list.some(j => j.status === "running" || j.status === "queued");
  pollTimer = setTimeout(refreshJobs, active ? 2000 : 15000);
}

function jobCard(job) {
  const d = document.createElement("div");
  d.className = "job";
  const t = Date.now();
  const thumb = `/outputs/${job.id}/thumb.jpg`;
  d.innerHTML = `
    <div>${job.progress > 0.06 || job.status === "done" ? `<img src="${thumb}?${Math.floor(t / 20000)}" onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'ph'}))">` : `<div class="ph"></div>`}</div>
    <div>
      <h3></h3>
      <div class="meta"><span class="st-${job.status}">${job.status}</span>${job.spec.mode === "song" ? " · ♪ music video" : job.spec.mode === "studio" ? " · 🎙 studio" : ""} · ${job.spec.preset} · ${job.spec.aspect} · ${job.spec.mode === "studio" ? `${job.spec.lines.length} lines` : `${job.spec.scenes.length} scenes`}${eta(job)}</div>
      <div class="meta msg"></div>
      ${job.status === "running" || job.status === "queued" ? `<div class="bar"><i style="width:${(job.progress * 100).toFixed(1)}%"></i></div>` : ""}
      <div class="btns"></div>
    </div>`;
  d.querySelector("h3").textContent = job.spec.title || job.id;
  d.querySelector(".msg").textContent = job.status === "done" && job.timings?.total ? `Rendered in ${fmtTime(job.timings.total)}` : job.message || "";
  const btns = d.querySelector(".btns");
  const btn = (label, fn) => { const b = document.createElement("button"); b.textContent = label; b.onclick = (e) => { e.stopPropagation(); fn(); }; btns.append(b); };
  if (job.status === "running" || job.status === "queued") btn("Cancel", () => api(`/api/jobs/${job.id}/cancel`, { method: "POST" }).then(refreshJobs));
  if (["cancelled", "failed", "interrupted"].includes(job.status)) btn("Resume", () => api(`/api/jobs/${job.id}/resume`, { method: "POST" }).then(refreshJobs));
  if (job.status === "done") {
    const a = document.createElement("a");
    a.href = job.final; a.download = `${(job.spec.title || job.id).replace(/[^\w-]+/g, "_")}.mp4`; a.textContent = "Download";
    a.onclick = (e) => e.stopPropagation();
    btns.append(a);
  }
  if (job.status === "done" || (job.task && job.task !== "render" && job.status !== "running" && job.status !== "queued")) btn("✎ Edit", () => location.href = `/edit/${job.id}`);
  btn("Reuse settings", () => loadSpec(job.spec));
  if (!(job.status === "running" || job.status === "queued")) btn("Delete", () => {
    if (confirm("Delete this video and its files?")) api(`/api/jobs/${job.id}`, { method: "DELETE" }).then(() => { if (selectedJob === job.id) { selectedJob = null; $("#viewer").hidden = true; } refreshJobs(); });
  });
  d.onclick = () => { selectedJob = job.id; showJob(job.id); };
  return d;
}

let shownKey = "";
async function showJob(id) {
  let job;
  try { job = await api(`/api/jobs/${id}`); } catch { return; }
  const key = `${id}:${job.status}:${job.clips.length}`;
  if (key === shownKey) return;  // don't reset the player while it plays
  shownKey = key;
  $("#viewer").hidden = false;
  const player = $("#player");
  const info = $("#viewerInfo");
  info.innerHTML = "";
  if (job.status === "done") {
    player.hidden = false;
    player.src = job.final + "?" + job.updated;
    const t = job.timings || {};
    info.textContent = `${job.spec.title} · rendered in ${fmtTime(t.total || 0)}` + (t.per_scene_avg ? ` · ${fmtTime(t.per_scene_avg)} per scene` : "");
  } else {
    player.hidden = true;
    player.removeAttribute("src");
    info.textContent = job.status === "failed" ? `Failed: ${job.message}` : `${job.spec.title}: ${job.clips.length}/${job.spec.scenes.length} scenes rendered`;
    if (job.clips.length) {
      const grid = document.createElement("div");
      grid.className = "clips";
      for (const c of job.clips) {
        const v = document.createElement("video");
        v.src = c; v.muted = true; v.loop = true; v.playsInline = true;
        v.onmouseenter = () => v.play(); v.onmouseleave = () => v.pause();
        grid.append(v);
      }
      info.append(grid);
    }
  }
}

function loadSpec(spec) {
  if (spec.mode === "studio") { setMode("studio"); window.studioLoad && window.studioLoad(spec); window.scrollTo({ top: 0, behavior: "smooth" }); return; }
  setMode(spec.mode === "song" ? "song" : "story");
  if (spec.mode === "song") songPlan = { lyrics: spec.lyrics || [], duration: 0 };
  $("#idea").value = spec.idea || "";
  $("#title").value = spec.title || "";
  $("#style").value = spec.style;
  setSeg($("#aspect"), spec.aspect);
  setSeg($("#preset"), spec.preset);
  $("#presetHint").textContent = PRESET_INFO[spec.preset];
  $("#voice").value = spec.voice || "";
  $("#subtitles").checked = spec.subtitles !== false;
  $("#seed").value = spec.seed;
  $("#guidance").value = spec.guidance;
  $("#negative").value = spec.negative || opts.default_negative;
  const m = spec.music || {};
  $("#musicMode").value = m.mode === "upload" ? "none" : (m.mode || "none");  // files can't be re-attached
  $("#musicPrompt").value = m.prompt || "";
  $("#musicVol").value = m.volume || 0.35;
  $("#musicMode").onchange();
  $("#scenes").innerHTML = "";
  spec.scenes.forEach(addScene);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

init();
