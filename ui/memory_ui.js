/**
 * AS Code — Working Memory UI  (v1.0.0)
 *
 * Event-driven. NO polling — Working Memory changes infrequently.
 * Refresh triggers: drawer open, after any write/delete/reset.
 *
 * Public API (used by app.js):
 *   window.memoryUI.getSessionId()   → string
 *   window.memoryUI.setSessionId(id) → void
 *   window.memoryUI.addObservation(content, source) → Promise
 */

console.log('[MEMORY] module loaded ✓');

// ── Source config ─────────────────────────────────────────────
const SOURCE_COLORS = {
    user:       { bg: 'rgba(99,179,237,0.12)',  border: 'rgba(99,179,237,0.3)',  text: '#63b3ed' },
    system:     { bg: 'rgba(154,117,243,0.12)', border: 'rgba(154,117,243,0.3)', text: '#9a75f3' },
    rag:        { bg: 'rgba(72,187,120,0.12)',  border: 'rgba(72,187,120,0.3)',  text: '#48bb78' },
    capability: { bg: 'rgba(237,137,54,0.12)',  border: 'rgba(237,137,54,0.3)',  text: '#ed8936' },
};

const STATUS_CONFIG = {
    pending:     { icon: '○', class: 'task-pending',     label: 'Pending' },
    in_progress: { icon: '◑', class: 'task-in-progress', label: 'In Progress' },
    completed:   { icon: '●', class: 'task-completed',   label: 'Done' },
    failed:      { icon: '✕', class: 'task-failed',      label: 'Failed' },
};

const STATUS_CYCLE = ['pending', 'in_progress', 'completed', 'failed'];

// ─────────────────────────────────────────────────────────────
class MemoryUI {
    constructor() {
        this._sessionId = 'default_session';
        this._data = { variables: [], tasks: [], observations: [] };

        // DOM refs — resolved in init()
        this.drawer     = null;
        this.overlay    = null;
        this.toggleBtn  = null;
        this.closeBtn   = null;
        this.content    = null;

        console.log('[MEMORY] MemoryUI instance created');
    }

    // ── Initialization ──────────────────────────────────────────
    init() {
        console.log('[MEMORY] init() called — resolving DOM elements...');

        this.drawer    = document.getElementById('memoryDrawer');
        this.overlay   = document.getElementById('memoryOverlay');
        this.toggleBtn = document.getElementById('toggleMemory');
        this.closeBtn  = document.getElementById('closeMemory');
        this.content   = document.getElementById('memoryContent');

        if (!this.drawer || !this.toggleBtn) {
            console.error('[MEMORY] ❌ #memoryDrawer or #toggleMemory not found — init aborted');
            return;
        }

        this.toggleBtn.addEventListener('click', () => this.toggleDrawer());
        this.closeBtn?.addEventListener('click', () => this.closeDrawer());
        this.overlay?.addEventListener('click', () => this.closeDrawer());

        window.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.drawer && !this.drawer.classList.contains('hidden')) {
                this.closeDrawer();
            }
        });

        console.log('[MEMORY] initialized ✓');
    }

    // ── Session ─────────────────────────────────────────────────
    getSessionId() { return this._sessionId; }
    setSessionId(id) {
        this._sessionId = id || 'default_session';
        if (this.drawer && !this.drawer.classList.contains('hidden')) {
            this.refresh(); // Reload for the new session
        }
    }

    // ── Drawer ──────────────────────────────────────────────────
    toggleDrawer() {
        const isOpen = this.drawer && !this.drawer.classList.contains('hidden');
        isOpen ? this.closeDrawer() : this.openDrawer();
    }

    openDrawer() {
        if (!this.drawer) return;
        this.drawer.classList.remove('hidden');
        this.toggleBtn?.classList.add('active');
        this.refresh(); // Single fetch on open — not repeated
    }

    closeDrawer() {
        if (!this.drawer) return;
        this.drawer.classList.add('hidden');
        this.toggleBtn?.classList.remove('active');
    }

    // ── Fetch ───────────────────────────────────────────────────
    async refresh() {
        if (!this.content) return;
        this.content.innerHTML = '<div class="loading-spinner">Loading memory...</div>';

        try {
            const res = await fetch(`/v1/memory?session_id=${encodeURIComponent(this._sessionId)}`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            this._data = await res.json();
            this._render();
        } catch (err) {
            console.error('[MEMORY] fetch failed:', err);
            this.content.innerHTML = `
                <div style="padding:1.5rem;text-align:center;color:rgba(225,229,235,0.35);font-size:0.8rem;">
                    <div style="font-size:1.5rem;margin-bottom:0.5rem;">⚠️</div>
                    Failed to load working memory<br>
                    <span style="font-size:0.7rem;opacity:0.6;">${err.message}</span>
                </div>`;
        }
    }

    // ── Render ──────────────────────────────────────────────────
    _render() {
        if (!this.content) return;
        const { variables, tasks, observations } = this._data;

        this.content.innerHTML = '';
        this.content.appendChild(this._buildVariablesSection(variables));
        this.content.appendChild(this._buildTasksSection(tasks));
        this.content.appendChild(this._buildObservationsSection(observations));
        this.content.appendChild(this._buildResetSection());
    }

    // ── Variables Section ───────────────────────────────────────
    _buildVariablesSection(variables) {
        const wrap = document.createElement('div');
        wrap.className = 'memory-section';

        const header = this._sectionHeader('Variables', `${variables.length}`);
        wrap.appendChild(header);

        if (variables.length > 0) {
            const list = document.createElement('div');
            list.className = 'memory-variable-list';
            variables.forEach(v => list.appendChild(this._variableChip(v)));
            wrap.appendChild(list);
        }

        // Add variable form
        const form = document.createElement('div');
        form.className = 'memory-add-row';
        form.innerHTML = `
            <input id="mem-var-key" class="memory-input" type="text" placeholder="key" style="flex:1;min-width:0;">
            <input id="mem-var-val" class="memory-input" type="text" placeholder="value" style="flex:2;min-width:0;">
            <button class="memory-add-btn" onclick="window.memoryUI._addVariable()">+ Set</button>`;
        wrap.appendChild(form);

        return wrap;
    }

    _variableChip(v) {
        const chip = document.createElement('div');
        chip.className = 'memory-variable-chip';
        chip.innerHTML = `
            <span class="mem-var-key">${this._esc(v.key)}</span>
            <span class="mem-var-sep">→</span>
            <span class="mem-var-val">${this._esc(v.value)}</span>
            <button class="mem-delete-btn" onclick="window.memoryUI._deleteVariable('${this._esc(v.key)}')" title="Remove">×</button>`;
        return chip;
    }

    async _addVariable() {
        const key = document.getElementById('mem-var-key')?.value.trim();
        const val = document.getElementById('mem-var-val')?.value.trim();
        if (!key) return;
        await this._post('/v1/memory/variables', {
            session_id: this._sessionId, key, value: val || ''
        });
        document.getElementById('mem-var-key').value = '';
        document.getElementById('mem-var-val').value = '';
        await this.refresh(); // Event-driven refresh
    }

    async _deleteVariable(key) {
        await fetch(
            `/v1/memory/variables/${encodeURIComponent(key)}?session_id=${encodeURIComponent(this._sessionId)}`,
            { method: 'DELETE' }
        );
        await this.refresh();
    }

    // ── Tasks Section ───────────────────────────────────────────
    _buildTasksSection(tasks) {
        const wrap = document.createElement('div');
        wrap.className = 'memory-section';

        const active = tasks.filter(t => t.status !== 'completed' && t.status !== 'failed');
        const done   = tasks.filter(t => t.status === 'completed' || t.status === 'failed');

        wrap.appendChild(this._sectionHeader('Tasks', `${active.length} active`));

        const list = document.createElement('div');
        list.className = 'memory-task-list';

        [...active, ...done].forEach(t => list.appendChild(this._taskRow(t)));
        wrap.appendChild(list);

        // Add task form
        const form = document.createElement('div');
        form.className = 'memory-add-row';
        form.innerHTML = `
            <input id="mem-task-title" class="memory-input" type="text" placeholder="New task..." style="flex:3;min-width:0;">
            <input id="mem-task-prio" class="memory-input" type="number" placeholder="P" min="0" value="0" style="flex:0 0 50px;text-align:center;">
            <button class="memory-add-btn" onclick="window.memoryUI._addTask()">+ Add</button>`;
        wrap.appendChild(form);

        return wrap;
    }

    _taskRow(t) {
        const cfg  = STATUS_CONFIG[t.status] || STATUS_CONFIG.pending;
        const prio = t.priority > 0 ? `<span class="mem-prio-badge">P${t.priority}</span>` : '';

        const row = document.createElement('div');
        row.className = `memory-task-row ${cfg.class}`;
        row.innerHTML = `
            <button class="mem-status-btn" title="Cycle status"
                onclick="window.memoryUI._cycleTaskStatus('${t.id}', '${t.status}')">
                ${cfg.icon}
            </button>
            <span class="mem-task-title">${this._esc(t.title)}</span>
            ${prio}
            <button class="mem-delete-btn" onclick="window.memoryUI._deleteTask('${t.id}')" title="Remove">×</button>`;
        return row;
    }

    async _addTask() {
        const title = document.getElementById('mem-task-title')?.value.trim();
        const prio  = parseInt(document.getElementById('mem-task-prio')?.value || '0', 10);
        if (!title) return;
        await this._post('/v1/memory/tasks', {
            session_id: this._sessionId, title, priority: prio
        });
        document.getElementById('mem-task-title').value = '';
        document.getElementById('mem-task-prio').value  = '0';
        await this.refresh();
    }

    async _cycleTaskStatus(taskId, currentStatus) {
        const idx  = STATUS_CYCLE.indexOf(currentStatus);
        const next = STATUS_CYCLE[(idx + 1) % STATUS_CYCLE.length];
        await fetch(`/v1/memory/tasks/${taskId}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status: next }),
        });
        await this.refresh();
    }

    async _deleteTask(taskId) {
        await fetch(`/v1/memory/tasks/${taskId}`, { method: 'DELETE' });
        await this.refresh();
    }

    // ── Observations Section ────────────────────────────────────
    _buildObservationsSection(observations) {
        const wrap = document.createElement('div');
        wrap.className = 'memory-section';
        wrap.appendChild(this._sectionHeader('Observations', `${observations.length}`));

        const list = document.createElement('div');
        list.className = 'memory-obs-list';
        observations.forEach(o => list.appendChild(this._obsRow(o)));
        wrap.appendChild(list);

        // Add observation form
        const form = document.createElement('div');
        form.className = 'memory-add-col';
        form.innerHTML = `
            <div class="memory-add-row">
                <input id="mem-obs-text" class="memory-input" type="text"
                    placeholder="Add observation..." style="flex:1;min-width:0;">
                <select id="mem-obs-src" class="memory-select">
                    <option value="user">user</option>
                    <option value="system">system</option>
                    <option value="rag">rag</option>
                    <option value="capability">capability</option>
                </select>
                <button class="memory-add-btn" onclick="window.memoryUI._addObservation()">+ Add</button>
            </div>`;
        wrap.appendChild(form);

        return wrap;
    }

    _obsRow(o) {
        const colors = SOURCE_COLORS[o.source] || SOURCE_COLORS.user;
        const row = document.createElement('div');
        row.className = 'memory-obs-row';
        row.style.cssText = `
            background:${colors.bg};border:1px solid ${colors.border};
            border-radius:6px;padding:0.4rem 0.6rem;margin-bottom:0.3rem;
            display:flex;align-items:flex-start;gap:0.5rem;`;
        row.innerHTML = `
            <span style="color:${colors.text};font-size:0.65rem;font-family:'JetBrains Mono',monospace;
                white-space:nowrap;padding-top:2px;">[${this._esc(o.source)}]</span>
            <span style="flex:1;font-size:0.75rem;color:rgba(225,229,235,0.8);
                line-height:1.4;">${this._esc(o.content)}</span>
            <button class="mem-delete-btn"
                onclick="window.memoryUI._deleteObservation('${o.id}')" title="Remove">×</button>`;
        return row;
    }

    async addObservation(content, source = 'user') {
        // Public API — called by app.js or other modules
        await this._post('/v1/memory/observations', {
            session_id: this._sessionId, content, source
        });
        if (this.drawer && !this.drawer.classList.contains('hidden')) {
            await this.refresh();
        }
    }

    async _addObservation() {
        const content = document.getElementById('mem-obs-text')?.value.trim();
        const source  = document.getElementById('mem-obs-src')?.value || 'user';
        if (!content) return;
        await this.addObservation(content, source);
        document.getElementById('mem-obs-text').value = '';
    }

    async _deleteObservation(obsId) {
        await fetch(`/v1/memory/observations/${obsId}`, { method: 'DELETE' });
        await this.refresh();
    }

    // ── Reset Section ───────────────────────────────────────────
    _buildResetSection() {
        const wrap = document.createElement('div');
        wrap.className = 'memory-section memory-section--reset';
        wrap.innerHTML = `
            <button class="memory-reset-btn"
                onclick="window.memoryUI._confirmReset()">
                🗑 Clear Memory (${this._sessionId})
            </button>`;
        return wrap;
    }

    async _confirmReset() {
        if (!confirm(`Clear all working memory for session "${this._sessionId}"?`)) return;
        await this._post('/v1/memory/reset', { session_id: this._sessionId });
        await this.refresh();
    }

    // ── Helpers ─────────────────────────────────────────────────
    _sectionHeader(title, badge) {
        const hdr = document.createElement('div');
        hdr.className = 'memory-section-header';
        hdr.innerHTML = `
            <span class="memory-section-title">${title}</span>
            <span class="memory-section-badge">${badge}</span>`;
        return hdr;
    }

    async _post(url, body) {
        try {
            const res = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: res.statusText }));
                console.error(`[MEMORY] POST ${url} failed:`, err.detail);
            }
        } catch (err) {
            console.error(`[MEMORY] POST ${url} error:`, err);
        }
    }

    _esc(str) {
        return String(str ?? '')
            .replace(/&/g,'&amp;').replace(/</g,'&lt;')
            .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
    }
}

// ── Self-initialize ────────────────────────────────────────────
window.memoryUI = new MemoryUI();

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        console.log('[MEMORY] DOMContentLoaded fired — calling init()');
        window.memoryUI.init();
    });
} else {
    console.log('[MEMORY] DOM already ready — calling init() immediately');
    window.memoryUI.init();
}
