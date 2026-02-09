const $ = (sel) => document.querySelector(sel);

const listEl = $("#taskList");
const overlay = $("#modalOverlay");
const modal = $("#taskModal");
const closeModalBtn = $("#closeModal");
const refreshBtn = $("#refreshBtn");
const copyBtn = $("#copyBtn");
const completeBtn = $("#completeBtn");

const promptHost = $("#promptEditor");
const answerHost = $("#answerEditor");

let currentTaskId = null;
let currentPromptText = "";

let monacoLoaded = false;
let promptEditor = null;
let answerEditor = null;

// Relative base: when opened at /manual, "api/..." resolves to /manual/api/...
const API_BASE = `${window.location.pathname.replace(/\/$/, "")}/api`;

/* ---------------- Monaco loader ---------------- */
function loadMonaco() {
  return new Promise((resolve, reject) => {
    if (monacoLoaded && window.monaco) return resolve();
    if (!window.require) return reject(new Error("Monaco AMD loader not found"));
    window.require.config({
      paths: { vs: "https://cdn.jsdelivr.net/npm/monaco-editor@0.52.0/min/vs" },
    });
    window.require(["vs/editor/editor.main"], () => {
      monacoLoaded = true;
      resolve();
    });
  });
}

/* ---------------- Helpers ---------------- */
async function fetchJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

function taskCardHtml(t) {
  const short = t.id.slice(0, 8);
  return `
    <div class="task-card" data-id="${t.id}">
      <div class="task-meta">
        <div class="task-model">${t.model || "model: n/a"}</div>
        <div class="task-id">#${short}</div>
        <div class="task-created">${t.created_at || ""}</div>
      </div>
      <div><button class="primary provide-btn">Provide answer</button></div>
    </div>
  `;
}

async function loadTasks() {
  const data = await fetchJSON(`${API_BASE}/tasks`);
  listEl.innerHTML = data.tasks.length
    ? data.tasks.map(taskCardHtml).join("")
    : `<div class="task-card"><div>No pending tasks.</div></div>`;

  document.querySelectorAll(".provide-btn").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      const card = e.target.closest(".task-card");
      const taskId = card.getAttribute("data-id");
      await openTask(taskId);
    });
  });
}

function getEditorText(editor) {
  return editor ? editor.getValue() : "";
}

function createEditors() {
  const optsCommon = {
    language: "plaintext",
    wordWrap: "on",
    scrollBeyondLastLine: false,
    minimap: { enabled: false },
    fontLigatures: false,
    fontSize: 14,
    lineNumbers: "on",
    renderLineHighlight: "line",
    automaticLayout: true,
  };

  promptEditor = monaco.editor.create(promptHost, {
    ...optsCommon,
    readOnly: true,
    theme: "vs-dark",
    value: currentPromptText || "",
  });

  answerEditor = monaco.editor.create(answerHost, {
    ...optsCommon,
    readOnly: false,
    theme: "hc-black",
    value: "",
  });
}

function disposeEditors() {
  if (promptEditor) { promptEditor.dispose(); promptEditor = null; }
  if (answerEditor) { answerEditor.dispose(); answerEditor = null; }
}

/* ---------------- Modal open/close ---------------- */
async function openTask(taskId) {
  const data = await fetchJSON(`${API_BASE}/tasks/${taskId}`);
  currentTaskId = data.id;
  currentPromptText = data.display_prompt || "";

  overlay.classList.remove("hidden");
  modal.classList.remove("hidden");

  await loadMonaco();
  createEditors();

  copyBtn.onclick = async () => {
    try {
      await navigator.clipboard.writeText(currentPromptText);
      copyBtn.textContent = "Copied";
      setTimeout(() => (copyBtn.textContent = "Copy"), 900);
    } catch (e) {
      alert("Copy failed: " + e.message);
    }
  };

  setTimeout(() => { answerEditor && answerEditor.focus(); }, 0);
}

function closeModal() {
  disposeEditors();
  currentTaskId = null;
  currentPromptText = "";
  overlay.classList.add("hidden");
  modal.classList.add("hidden");
}

closeModalBtn.addEventListener("click", closeModal);
overlay.addEventListener("click", closeModal);
refreshBtn.addEventListener("click", loadTasks);

/* ---------------- Submit ---------------- */
completeBtn.addEventListener("click", async () => {
  const answer = getEditorText(answerEditor).trim();
  if (!answer) {
    alert("Answer is empty.");
    return;
  }
  const form = new FormData();
  form.append("answer", answer);

  const r = await fetch(`${API_BASE}/tasks/${currentTaskId}/answer`, {
    method: "POST",
    body: form,
  });
  if (!r.ok) {
    alert("Submit failed: " + (await r.text()));
    return;
  }
  closeModal();
  await loadTasks();
});

/* ---------------- Init ---------------- */
window.addEventListener("DOMContentLoaded", loadTasks);
