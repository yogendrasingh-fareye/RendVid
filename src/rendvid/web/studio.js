// Studio mode: narrated story film (script or idea -> storyboard -> render).
// Shares helpers ($, api, segmented, setSeg, opts, fmtTime, refreshJobs) with app.js.
(() => {
  const SAMPLE = "एक बार की बात है, एक कौवे को बहुत तेज़ प्यास लगी थी। ...";
  // Seconds per shot on an M5 / 16 GB, including its keyframe (rough).
  const SHOT_SECS = { fast: 45, balanced: 65, max: 105 };
  let ready = false;
  let plan = null;

  function init() {
    if (ready) return;
    ready = true;
    segmented($("#studioSource"), ["script", "idea"], "script", setSource, { script: "My script", idea: "Write from an idea" });
    setSource("script");
    const voices = opts.studio_voices || {};
    $("#studioVoice").innerHTML = Object.entries(voices).map(([k, v]) => `<option value="${k}">${v}</option>`).join("");
    const preferred = ["parler:Divya:storyteller", "hf_alpha"].find(v => v in voices);
    if (preferred) $("#studioVoice").value = preferred;
    if (!opts.models.hindi_voice) $("#studioInfo").textContent = "Hindi HQ voice not installed yet: using Kokoro.";
    if (!opts.models.keyframes) $("#studioInfo").textContent = "Image model still downloading: rendering will wait for it.";
  }
  window.studioInit = init;

  function setSource(src) {
    setSeg($("#studioSource"), src);
    $("#minutesRow").hidden = src !== "idea";
    $("#studioScriptHint").hidden = src !== "script";
    $("#studioText").placeholder = src === "script"
      ? `Paste your story here.\n\n${SAMPLE}`
      : "e.g. A clever crow in a drought-hit village who teaches the other birds to work together";
    $("#studioText").rows = src === "script" ? 9 : 3;
  }

  $("#studioMusic").onchange = () => { $("#studioMusicFileRow").hidden = $("#studioMusic").value !== "upload"; };
  $("#studioLang").onchange = () => {
    const v = $("#studioVoice");
    if ($("#studioLang").value === "English" && v.value.startsWith("h")) v.value = "af_heart";
  };

  // --- characters -------------------------------------------------------------
  function addChar(c = { name: "", look: "" }) {
    const row = document.createElement("div");
    row.className = "charRow";
    row.innerHTML = `<input type="text" class="cname" placeholder="name"><input type="text" class="clook" placeholder="fixed look in English: species/age, colors, clothing"><button title="Remove">✕</button>`;
    row.querySelector(".cname").value = c.name;
    row.querySelector(".clook").value = c.look;
    row.querySelector("button").onclick = () => row.remove();
    $("#studioChars").append(row);
  }
  $("#addChar").onclick = () => addChar();
  const readChars = () => [...$("#studioChars").children].map(r => ({
    name: r.querySelector(".cname").value.trim(), look: r.querySelector(".clook").value.trim(),
  })).filter(c => c.name && c.look);

  // --- lines --------------------------------------------------------------------
  function lineEl(ln) {
    const li = $("#lineTpl").content.firstElementChild.cloneNode(true);
    li.querySelector(".narr").value = ln.narration || "";
    li.querySelector(".visual").value = ln.visual || "";
    li.querySelector(".who").value = (ln.characters || []).join(", ");
    li.querySelector(".newShot").checked = ln.new_shot !== false;
    li.dataset.chapter = ln.chapter || 0;
    li.querySelector(".del").onclick = () => { li.remove(); renumber(); };
    li.querySelector(".add").onclick = () => {
      li.after(lineEl({ narration: "", visual: "", characters: [], new_shot: true, chapter: li.dataset.chapter }));
      renumber();
    };
    li.querySelector(".narr").onchange = renumber;
    return li;
  }
  $("#addLine").onclick = () => {
    const last = $("#studioLines").lastElementChild;
    $("#studioLines").append(lineEl({ chapter: last ? last.dataset.chapter : 0 }));
    renumber();
  };

  function readLines() {
    return [...$("#studioLines").children].map(li => ({
      narration: li.querySelector(".narr").value.trim(),
      visual: li.querySelector(".visual").value.trim(),
      characters: li.querySelector(".who").value.split(",").map(s => s.trim()).filter(Boolean),
      new_shot: li.querySelector(".newShot").checked,
      chapter: +li.dataset.chapter || 0,
    })).filter(l => l.narration);
  }

  function renumber() {
    const items = [...$("#studioLines").children];
    let prevCh = null;
    items.forEach((li, i) => {
      li.querySelector(".num").textContent = `${i + 1}`;
      const ch = li.dataset.chapter;
      li.querySelector(".chapterTag").textContent = ch !== prevCh ? `· part ${+ch + 1}` : "";
      prevCh = ch;
    });
    const lines = readLines();
    const words = lines.reduce((n, l) => n + l.narration.split(/\s+/).length, 0);
    const secs = words / ($("#studioLang").value === "Hindi" ? 125 : 140) * 60 + lines.length * 0.6;
    const shots = lines.reduce((n, l) => n + Math.max(1, Math.ceil((l.narration.split(/\s+/).length / 2.1 + 0.5) / 4.8)), 0);
    const preset = $("#preset").dataset.value || "balanced";
    $("#studioSummary").textContent = `${lines.length} lines · ≈ ${fmtTime(secs)} film · ~${shots} shots · render ≈ ${fmtTime(shots * SHOT_SECS[preset] + 120)} on ${preset}`;
    $("#studioWrap").hidden = items.length === 0;
  }

  function show(p) {
    plan = p;
    $("#studioTitleText").value = p.title || "";
    $("#studioChars").innerHTML = "";
    (p.characters || []).forEach(addChar);
    $("#studioLines").innerHTML = "";
    (p.lines || []).forEach(l => $("#studioLines").append(lineEl(l)));
    renumber();
  }

  $("#studioPlanBtn").onclick = async () => {
    const btn = $("#studioPlanBtn");
    const source = $("#studioSource").dataset.value;
    btn.disabled = true;
    $("#studioInfo").textContent = source === "idea"
      ? "Writing the story chapter by chapter… (a few minutes)"
      : "Designing characters and shots… (about a minute)";
    try {
      const p = await api("/api/plan_studio", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source, text: $("#studioText").value, minutes: +$("#studioMinutes").value,
          language: $("#studioLang").value, style: $("#style").value,
        }),
      });
      show(p);
      $("#studioInfo").textContent = p.warning || "Storyboard ready. Review the characters and shots, then render.";
    } catch (e) {
      $("#studioInfo").textContent = "Error: " + e.message;
    } finally {
      btn.disabled = false;
    }
  };

  $("#studioRenderBtn").onclick = async () => {
    const lines = readLines();
    if (!lines.length) return alert("Add at least one narration line.");
    const music = $("#studioMusic").value;
    if (music === "upload" && !$("#studioMusicFile").files[0]) return alert("Choose a music file or change Background music.");
    const spec = {
      mode: "studio",
      title: $("#studioTitleText").value.trim(),
      idea: $("#studioText").value.slice(0, 2000),
      language: $("#studioLang").value,
      voice: $("#studioVoice").value,
      aspect: $("#aspect").dataset.value,
      preset: $("#preset").dataset.value,
      style: $("#style").value,
      seed: +$("#seed").value,
      characters: readChars(),
      setting: plan?.setting || "",
      look: plan?.look || "",
      lines,
      subtitles: $("#studioSubs").checked,
      title_card: $("#studioTitle").checked,
      music: { mode: music, volume: +$("#studioMusicVol").value },
    };
    const fd = new FormData();
    fd.append("spec", JSON.stringify(spec));
    if (music === "upload") fd.append("music", $("#studioMusicFile").files[0]);
    $("#studioRenderBtn").disabled = true;
    try {
      const job = await api("/api/jobs", { method: "POST", body: fd });
      selectedJob = job.id;
      refreshJobs();
      $("#studioInfo").textContent = "Rendering. Follow progress on the right.";
    } catch (e) {
      alert(e.message);
    } finally {
      $("#studioRenderBtn").disabled = false;
    }
  };

  window.studioLoad = (spec) => {
    init();
    $("#studioLang").value = spec.language || "Hindi";
    if (spec.voice) $("#studioVoice").value = spec.voice;
    $("#studioSubs").checked = spec.subtitles !== false;
    $("#studioTitle").checked = spec.title_card !== false;
    $("#studioMusic").value = (spec.music || {}).mode === "upload" ? "none" : ((spec.music || {}).mode || "none");
    $("#studioMusicVol").value = (spec.music || {}).volume || 0.18;
    show({ title: spec.title, characters: spec.characters, lines: spec.lines, setting: spec.setting, look: spec.look });
  };
})();
