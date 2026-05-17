/**
 * AS Code — Document Upload UI
 * Integrado en el input area (estilo Gemini/Claude/ChatGPT).
 * Vanilla JS, zero framework overhead.
 */

let documentSessionId = null;

// ── Sesión ─────────────────────────────────────────────────────
async function initDocumentSession() {
  try {
    const res = await fetch('/api/documents/session', { method: 'POST' });
    const data = await res.json();
    documentSessionId = data.session_id;
    return documentSessionId;
  } catch (e) {
    console.error('Error creando sesión:', e);
  }
}

// ── Upload ─────────────────────────────────────────────────────
async function uploadDocument(file) {
  if (!documentSessionId) return;

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(
      `/api/documents/upload?session_id=${documentSessionId}`,
      { method: 'POST', body: formData }
    );
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail);
    }
    const data = await res.json();
    showToast(`📎 ${data.filename} listo`);
    await refreshDocumentList();
  } catch (e) {
    showToast(`✗ ${e.message}`, 'error');
  }
}

// ── Chips ──────────────────────────────────────────────────────
async function refreshDocumentList() {
  if (!documentSessionId) return;

  try {
    const res = await fetch(`/api/documents/${documentSessionId}`);
    const data = await res.json();
    const chips = document.getElementById('docChips');
    if (!chips) return;

    if (data.document_count === 0) {
      chips.innerHTML = '';
      chips.style.display = 'none';
    } else {
      chips.style.display = 'flex';
      chips.innerHTML = data.documents.map(d => `
        <span style="
          display:inline-flex; align-items:center; gap:0.3em;
          background:rgba(110,168,254,0.10); border:1px solid rgba(110,168,254,0.25);
          border-radius:99px; padding:0.18em 0.65em;
          font-size:0.76em; color:#6ea8fe; white-space:nowrap;
          cursor:default;
        " title="${d.filename} · ${d.char_count} chars">
          📄 ${d.filename}
          <span onclick="clearDocuments()" title="Limpiar docs"
            style="cursor:pointer; opacity:0.55; font-size:1.1em; line-height:1; margin-left:0.15em;">×</span>
        </span>
      `).join('');
    }
  } catch (e) {
    console.error('Error refrescando chips:', e);
  }
}

// ── Limpiar ────────────────────────────────────────────────────
async function clearDocuments() {
  if (!documentSessionId) return;
  try {
    await fetch(`/api/documents/${documentSessionId}`, { method: 'DELETE' });
    showToast('Documentos eliminados');
    await refreshDocumentList();
  } catch (e) {
    showToast('Error limpiando', 'error');
  }
}

// ── Drag & Drop sobre el input container ───────────────────────
function setupDocumentUpload() {
  const fileInput = document.getElementById('documentInput');
  const container = document.getElementById('inputContainer');

  if (!container || !fileInput) return;

  // Drag-drop sobre el input area completo
  container.addEventListener('dragover', (e) => {
    e.preventDefault();
    container.style.boxShadow = '0 0 0 2px #6ea8fe55';
  });

  container.addEventListener('dragleave', () => {
    container.style.boxShadow = '';
  });

  container.addEventListener('drop', async (e) => {
    e.preventDefault();
    container.style.boxShadow = '';
    for (const file of e.dataTransfer.files) {
      if (file.name.match(/\.(txt|pdf|docx)$/i)) {
        await uploadDocument(file);
      }
    }
  });

  // File input → upload
  fileInput.addEventListener('change', async (e) => {
    for (const file of e.target.files) {
      await uploadDocument(file);
    }
    fileInput.value = '';
  });
}

// ── Toast ligero ───────────────────────────────────────────────
function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.style.cssText = `
    position:fixed; bottom:80px; right:20px;
    padding:10px 18px;
    background:${type === 'error' ? '#c53030' : '#2563eb'};
    color:white; border-radius:8px; font-size:0.85em;
    z-index:9999; box-shadow:0 4px 12px rgba(0,0,0,0.3);
    animation:fadeIn 0.2s ease;
  `;
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}

// ── Init ───────────────────────────────────────────────────────
initDocumentSession().then(() => {
  setupDocumentUpload();
});
