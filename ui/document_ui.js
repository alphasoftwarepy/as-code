/**
 * AS Code — Document Upload UI (RAG NotebookLM mode)
 *
 * Upload flow:
 *   1. User selects / drops file → attach button shows uploading state
 *   2. POST /api/rag/documents/upload → returns 200 immediately
 *   3. Chip appears instantly with ⏳ Indexing... badge
 *   4. Background ingest runs server-side (chunk → embed → FAISS)
 *   5. Poller checks /api/rag/documents every 3s
 *   6. When chunk_count > 0 → badge updates to ✅ Ready, poll stops
 *
 * Upload != ingestion. UI is responsive immediately after upload.
 */

// ── Ingest Poller ──────────────────────────────────────────────
// Tracks doc_ids that are still "indexing" so we know when to stop polling.
let _indexingDocIds = new Set();
let _pollTimer = null;

function _startPolling() {
    if (_pollTimer) return;  // already running
    _pollTimer = setInterval(async () => {
        if (_indexingDocIds.size === 0) {
            _stopPolling();
            return;
        }
        await refreshDocumentList();
    }, 3000);
}

function _stopPolling() {
    if (_pollTimer) {
        clearInterval(_pollTimer);
        _pollTimer = null;
    }
}

// ── Upload ──────────────────────────────────────────────────────
async function uploadDocument(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('pipeline', 'chat');

    // Visual feedback on attach button during upload fetch
    const attachBtn = document.getElementById('attachBtn');
    const originalTitle = attachBtn?.title;
    if (attachBtn) {
        attachBtn.disabled = true;
        attachBtn.title = 'Uploading…';
        attachBtn.style.opacity = '0.5';
    }

    try {
        const res = await fetch('/api/rag/documents/upload', {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
            throw new Error(err.detail || 'Upload failed');
        }

        const data = await res.json();

        // Track this doc as "indexing" so poller knows to watch it
        _indexingDocIds.add(data.document_id);

        // Refresh list immediately — chip will appear with ⏳ Indexing...
        await refreshDocumentList();

        // Start polling for ingest completion
        _startPolling();

        showToast(`📎 ${data.filename} uploaded — indexing in background`);
    } catch (e) {
        showToast(`✗ ${e.message}`, 'error');
    } finally {
        // Restore attach button immediately so chat remains usable
        if (attachBtn) {
            attachBtn.disabled = false;
            attachBtn.title = originalTitle;
            attachBtn.style.opacity = '';
        }
    }
}

// ── Chips ───────────────────────────────────────────────────────
async function refreshDocumentList() {
    try {
        const res = await fetch('/api/rag/documents');
        if (!res.ok) return;
        const data = await res.json();

        const chips = document.getElementById('docChips');
        if (!chips) return;

        if (!data.documents || data.documents.length === 0) {
            chips.innerHTML = '';
            chips.style.display = 'none';
            _indexingDocIds.clear();
            _stopPolling();
            return;
        }

        chips.style.display = 'flex';

        // Update polled state — remove doc_ids that are now ready
        let stillIndexing = false;
        data.documents.forEach(d => {
            if (d.ingest_status === 'ready' && _indexingDocIds.has(d.id)) {
                _indexingDocIds.delete(d.id);
            }
            if (d.ingest_status === 'indexing') {
                stillIndexing = true;
            }
        });

        if (!stillIndexing) {
            _stopPolling();
        }

        chips.innerHTML = data.documents.map(d => {
            const statusBadge = _renderStatusBadge(d.ingest_status);
            const chunkLabel = d.chunk_count > 0
                ? `${d.chunk_count} chunk${d.chunk_count !== 1 ? 's' : ''}`
                : '';
            const tooltip = chunkLabel
                ? `${d.filename} · ${chunkLabel}`
                : `${d.filename} · indexing…`;

            return `
                <span style="
                    display:inline-flex; align-items:center; gap:0.35em;
                    background:rgba(110,168,254,0.08); border:1px solid rgba(110,168,254,0.20);
                    border-radius:99px; padding:0.2em 0.55em 0.2em 0.65em;
                    font-size:0.76em; color:#6ea8fe; white-space:nowrap;
                    cursor:default; transition:background 0.2s ease;
                " title="${tooltip}">
                    📄 ${d.filename}
                    ${statusBadge}
                    <span
                        onclick="deleteDocument('${d.id}')"
                        title="Remove document"
                        style="cursor:pointer; opacity:0.45; font-size:1.1em; line-height:1; margin-left:0.1em;"
                    >×</span>
                </span>
            `;
        }).join('');

    } catch (e) {
        console.error('[document_ui] refreshDocumentList error:', e);
    }
}

function _renderStatusBadge(status) {
    if (status === 'ready') {
        return `<span style="
            font-size:0.68em; color:#34d399;
            background:rgba(52,211,153,0.10);
            border:1px solid rgba(52,211,153,0.20);
            border-radius:4px; padding:0.05em 0.35em;
            font-weight:600; letter-spacing:0.02em;
        ">✅ Ready</span>`;
    }
    // indexing (default)
    return `<span style="
        font-size:0.68em; color:#fbbf24;
        background:rgba(251,191,36,0.10);
        border:1px solid rgba(251,191,36,0.20);
        border-radius:4px; padding:0.05em 0.35em;
        font-weight:600; letter-spacing:0.02em;
        animation: pulse-badge 1.8s ease-in-out infinite;
    ">⏳ Indexing…</span>`;
}

// ── Delete all ──────────────────────────────────────────────────
async function clearDocuments() {
    try {
        const res = await fetch('/api/rag/documents', { method: 'DELETE' });
        if (!res.ok) throw new Error('Error clearing documents');
        _indexingDocIds.clear();
        _stopPolling();
        showToast('All documents removed');
        await refreshDocumentList();
    } catch (e) {
        showToast(`✗ ${e.message}`, 'error');
    }
}

// ── Delete individual ───────────────────────────────────────────
async function deleteDocument(docId) {
    try {
        const res = await fetch(`/api/rag/documents/${docId}`, { method: 'DELETE' });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: 'Delete failed' }));
            throw new Error(err.detail || 'Delete failed');
        }
        _indexingDocIds.delete(docId);
        showToast('Document removed');
        await refreshDocumentList();
    } catch (e) {
        showToast(`✗ ${e.message}`, 'error');
    }
}

// ── Drag & Drop ─────────────────────────────────────────────────
function setupDocumentUpload() {
    const fileInput = document.getElementById('documentInput');
    const container = document.getElementById('inputContainer');

    if (!container || !fileInput) return;

    container.addEventListener('dragover', (e) => {
        e.preventDefault();
        container.style.boxShadow = '0 0 0 2px rgba(110,168,254,0.4)';
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

    fileInput.addEventListener('change', async (e) => {
        for (const file of e.target.files) {
            await uploadDocument(file);
        }
        fileInput.value = '';
    });
}

// ── Toast ────────────────────────────────────────────────────────
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.style.cssText = `
        position:fixed; bottom:80px; right:20px;
        padding:10px 18px;
        background:${type === 'error' ? '#c53030' : '#1e40af'};
        color:white; border-radius:8px; font-size:0.83em;
        z-index:9999; box-shadow:0 4px 16px rgba(0,0,0,0.35);
        animation:fadeIn 0.2s ease;
        max-width:320px; line-height:1.4;
    `;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3500);
}

// ── Pulse animation for Indexing badge ──────────────────────────
(function injectPulseKeyframe() {
    const style = document.createElement('style');
    style.textContent = `
        @keyframes pulse-badge {
            0%, 100% { opacity: 1; }
            50%       { opacity: 0.55; }
        }
    `;
    document.head.appendChild(style);
})();

// ── Init ─────────────────────────────────────────────────────────
setupDocumentUpload();
refreshDocumentList();
