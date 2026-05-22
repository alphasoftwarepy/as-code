/**
 * AS Code — Skills UI  (v1.2.0)
 *
 * Self-initializing: attaches to DOM via its own DOMContentLoaded listener.
 * Does NOT depend on app.js calling .init().
 *
 * Public API (used by app.js):
 *   window.skillsUI.getActiveSkillId()  → string | null
 *   window.skillsUI.analyzeContext(msg, docNames)
 */

console.log('[SKILLS] module loaded ✓');

// ── Skill Icons ────────────────────────────────────────────────
const SKILL_ICONS = {
    marketing:       '📈',
    sales:           '📞',
    business:        '💼',
    legal:           '⚖️',
    content_creator: '🎬',
};
const SKILL_ICON_DEFAULT = '✨';

// ── Suggestion Rules ───────────────────────────────────────────
const SUGGESTION_RULES = [
    {
        skill: 'sales',
        keywords: ['ventas','venta','clientes','cerrar','cierre','convertir','prospecto',
                   'follow-up','whatsapp','objecion','objeción','propuesta','cotización',
                   'sales','close deal','prospect','lead'],
    },
    {
        skill: 'marketing',
        keywords: ['campaña','marketing','publicidad','anuncio','instagram','redes sociales',
                   'branding','estrategia de marca','ads','facebook ads','google ads',
                   'campaign','advertisement','brand'],
    },
    {
        skill: 'legal',
        keywords: ['contrato','legal','cláusula','acuerdo','términos','compliance',
                   'abogado','ley','contratación','nda','privacy policy','terms',
                   'contract','clause','agreement','liability'],
    },
    {
        skill: 'business',
        keywords: ['negocio','estrategia','precio','operaciones','crecimiento','productividad',
                   'plan de negocio','flujo de caja','rentabilidad','emprendimiento',
                   'business','strategy','pricing','growth','operations'],
    },
    {
        skill: 'content_creator',
        keywords: ['reels','reel','caption','hook','contenido','video','tiktok','viral',
                   'post','hashtag','instagram','stories','content','creator','influencer'],
    },
];

const FILENAME_RULES = [
    { skill: 'legal',           patterns: ['contrato','contract','nda','acuerdo','terms','legal','policy'] },
    { skill: 'marketing',       patterns: ['marketing','campaign','campaña','brand','ads','publicidad'] },
    { skill: 'sales',           patterns: ['sales','ventas','prospecto','leads','pipeline'] },
    { skill: 'business',        patterns: ['business','negocio','plan','strategy','finance','finanzas'] },
    { skill: 'content_creator', patterns: ['content','contenido','reels','posts','calendar','calendario'] },
];

// ─────────────────────────────────────────────────────────────
class SkillsUI {
    constructor() {
        // DOM refs — resolved in init()
        this.drawer        = null;
        this.overlay       = null;
        this.listBody      = null;
        this.toggleBtn     = null;
        this.closeBtn      = null;
        this.pill          = null;
        this.pillIcon      = null;
        this.pillName      = null;
        this.deactivateBtn = null;
        this.suggBar       = null;
        this.hiddenSelect  = null;

        // State
        this.activeSkillId   = null;
        this.loadedSkills    = {};
        this.dismissedSuggs  = new Set();
        this.lastSuggestions = [];

        console.log('[SKILLS] SkillsUI instance created');
    }

    // ── Initialization ──────────────────────────────────────────
    init() {
        console.log('[SKILLS] init() called — resolving DOM elements...');

        this.drawer        = document.getElementById('skillsDrawer');
        this.overlay       = document.getElementById('skillsOverlay');
        this.listBody      = document.getElementById('skillsList');
        this.toggleBtn     = document.getElementById('toggleSkills');
        this.closeBtn      = document.getElementById('closeSkills');
        this.pill          = document.getElementById('activeSkillPill');
        this.pillIcon      = document.getElementById('activeSkillIcon');
        this.pillName      = document.getElementById('activeSkillName');
        this.deactivateBtn = document.getElementById('deactivateSkillBtn');
        this.suggBar       = document.getElementById('skillSuggestionBar');
        this.hiddenSelect  = document.getElementById('skillSelect');

        // Diagnostic log — confirm what was found
        console.log('[SKILLS] DOM check:', {
            skillsDrawer:        !!this.drawer,
            skillsOverlay:       !!this.overlay,
            skillsList:          !!this.listBody,
            toggleSkills:        !!this.toggleBtn,
            closeSkills:         !!this.closeBtn,
            activeSkillPill:     !!this.pill,
            activeSkillIcon:     !!this.pillIcon,
            activeSkillName:     !!this.pillName,
            deactivateSkillBtn:  !!this.deactivateBtn,
            skillSuggestionBar:  !!this.suggBar,
            skillSelect:         !!this.hiddenSelect,
        });

        if (!this.drawer) {
            console.error('[SKILLS] ❌ #skillsDrawer not found in DOM — init aborted');
            return;
        }
        if (!this.toggleBtn) {
            console.error('[SKILLS] ❌ #toggleSkills button not found — init aborted');
            return;
        }

        // Attach toggle button listener
        this.toggleBtn.addEventListener('click', () => {
            console.log('[SKILLS] ✨ toggle button clicked');
            this.toggleDrawer();
        });

        // Attach close listeners
        this.closeBtn?.addEventListener('click', () => this.closeDrawer());
        this.overlay?.addEventListener('click',  () => this.closeDrawer());

        // Escape key
        window.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.drawer && !this.drawer.classList.contains('hidden')) {
                this.closeDrawer();
            }
        });

        // Deactivate pill
        this.deactivateBtn?.addEventListener('click', () => this.deactivateSkill());

        console.log('[SKILLS] drawer initialized ✓ — ready');
    }

    // ── Drawer ─────────────────────────────────────────────────
    toggleDrawer() {
        const isOpen = this.drawer && !this.drawer.classList.contains('hidden');
        console.log('[SKILLS] toggleDrawer — currently open:', isOpen);
        isOpen ? this.closeDrawer() : this.openDrawer();
    }

    openDrawer() {
        console.log('[SKILLS] openDrawer()');
        if (!this.drawer) return;
        this.drawer.classList.remove('hidden');
        this.toggleBtn?.classList.add('active');
        this.loadSkills();
    }

    closeDrawer() {
        if (!this.drawer) return;
        this.drawer.classList.add('hidden');
        this.toggleBtn?.classList.remove('active');
    }

    // ── Fetch + Render Skills ───────────────────────────────────
    async loadSkills() {
        if (!this.listBody) {
            console.error('[SKILLS] #skillsList not found — cannot render');
            return;
        }
        this.listBody.innerHTML = '<div class="loading-spinner">Loading skills...</div>';

        try {
            console.log('[SKILLS] fetching /v1/skills...');
            const res = await fetch('/v1/skills');
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const skills = await res.json();
            console.log('[SKILLS] fetched skills ✓', skills);

            this.loadedSkills = skills;
            this._renderSkillCards(skills);
            this._syncHiddenSelect(skills);
        } catch (err) {
            console.error('[SKILLS] failed to load skills:', err);
            this.listBody.innerHTML = `
                <div style="padding:1.5rem;text-align:center;color:rgba(225,229,235,0.35);font-size:0.8rem;">
                    <div style="font-size:1.5rem;margin-bottom:0.5rem;">⚠️</div>
                    Failed to load skills<br>
                    <span style="font-size:0.7rem;opacity:0.6;">${err.message}</span>
                </div>`;
        }
    }

    _renderSkillCards(skills) {
        if (!this.listBody) return;
        this.listBody.innerHTML = '';

        const ids = Object.keys(skills);
        if (ids.length === 0) {
            this.listBody.innerHTML = '<div class="loading-spinner">No skills installed</div>';
            return;
        }

        const compatible   = ids.filter(id => skills[id].compatible);
        const incompatible = ids.filter(id => !skills[id].compatible);

        [...compatible, ...incompatible].forEach((id, index) => {
            const card = this._createSkillCard(id, skills[id]);
            card.style.animationDelay = `${index * 45}ms`;
            this.listBody.appendChild(card);
        });
    }

    _createSkillCard(id, skill) {
        const isActive     = this.activeSkillId === id;
        const isCompatible = skill.compatible;
        const icon         = SKILL_ICONS[id] || SKILL_ICON_DEFAULT;

        const card = document.createElement('div');
        card.className = [
            'skill-card',
            isActive      ? 'skill-card--active'       : '',
            !isCompatible ? 'skill-card--incompatible' : '',
        ].filter(Boolean).join(' ');
        card.id = `skill-card-${id}`;

        const statusHtml = isCompatible
            ? `<span class="skill-status-badge badge-healthy">✅ Compatible</span>`
            : `<span class="skill-status-badge badge-unavailable">⚠ Incompatible</span>`;

        let btnLabel = 'Activate';
        let btnClass = 'skill-activate-btn';
        let btnExtra = '';
        if (!isCompatible) {
            btnLabel = '⛔ Unavailable';
            btnExtra = 'disabled';
        } else if (isActive) {
            btnLabel = '✓ Active';
            btnClass = 'skill-activate-btn skill-activate-btn--active';
        }

        const reasonHtml = (!isCompatible && skill.reason)
            ? `<div style="margin-top:0.5rem;font-size:0.65rem;color:rgba(248,113,113,0.7);
                font-family:'JetBrains Mono',monospace;background:rgba(248,113,113,0.05);
                border:1px solid rgba(248,113,113,0.12);border-radius:6px;padding:0.3rem 0.5rem;">
                ${skill.reason}</div>`
            : '';

        const scopesHtml = this._buildScopesHtml(skill);

        card.innerHTML = `
            <div class="skill-card-header">
                <div class="skill-card-title">
                    <span class="skill-icon">${icon}</span>
                    <span class="skill-name">${skill.name}</span>
                </div>
                ${statusHtml}
            </div>
            <p class="skill-description">${skill.description || ''}</p>
            ${scopesHtml}
            ${reasonHtml}
            <div class="skill-card-footer">
                <span style="font-size:0.6rem;color:rgba(225,229,235,0.2);font-family:'JetBrains Mono',monospace;">${id}</span>
                <button class="${btnClass}" ${btnExtra}
                    onclick="window.skillsUI._handleActivateClick('${id}')">
                    ${btnLabel}
                </button>
            </div>`;

        return card;
    }

    _buildScopesHtml(skill) {
        if (!skill.compatible && skill.reason) {
            const missing = skill.reason.replace('Missing required scopes: ', '').split(', ');
            return `<div class="skill-scopes">
                ${missing.map(s => `<span class="skill-scope-tag skill-scope-tag--missing">❌ ${s.trim()}</span>`).join('')}
            </div>`;
        }
        return `<div class="skill-scopes">
            <span class="skill-scope-tag">✅ documents.read</span>
            <span class="skill-scope-tag">✅ rag.retrieve</span>
        </div>`;
    }

    // ── Activation ──────────────────────────────────────────────
    _handleActivateClick(id) {
        console.log('[SKILLS] activate clicked:', id);
        if (this.activeSkillId === id) {
            this.deactivateSkill();
        } else {
            this.activateSkill(id);
        }
    }

    activateSkill(id) {
        const skill = this.loadedSkills[id];
        if (!skill) { console.warn('[SKILLS] activateSkill: skill not found:', id); return; }
        if (!skill.compatible) { console.warn('[SKILLS] activateSkill: skill not compatible:', id); return; }

        this.activeSkillId = id;
        this._syncHiddenSelect(this.loadedSkills);
        this._updatePill(id, skill);
        this._refreshCardStates();
        this.dismissedSuggs.add(id);
        this._removeSuggestionChip(id);

        console.log(`[SKILLS] ✅ Activated: ${id} (${skill.name})`);
    }

    deactivateSkill() {
        console.log('[SKILLS] deactivateSkill()');
        this.activeSkillId = null;
        this._syncHiddenSelect(this.loadedSkills);
        this._hidePill();
        this._refreshCardStates();
    }

    getActiveSkillId() {
        return this.activeSkillId;
    }

    // ── Pill ────────────────────────────────────────────────────
    _updatePill(id, skill) {
        if (!this.pill) { console.warn('[SKILLS] pill element not found'); return; }
        const icon = SKILL_ICONS[id] || SKILL_ICON_DEFAULT;
        if (this.pillIcon) this.pillIcon.textContent = icon;
        if (this.pillName) this.pillName.textContent = skill.name;
        this.pill.classList.remove('hidden');
        console.log('[SKILLS] pill shown:', skill.name);
    }

    _hidePill() {
        this.pill?.classList.add('hidden');
    }

    // ── Hidden select sync ──────────────────────────────────────
    _syncHiddenSelect(skills) {
        if (!this.hiddenSelect) return;
        this.hiddenSelect.innerHTML = '<option value="none">none</option>';
        Object.keys(skills).forEach(id => {
            const opt = document.createElement('option');
            opt.value = id;
            opt.textContent = skills[id].name;
            this.hiddenSelect.appendChild(opt);
        });
        this.hiddenSelect.value = this.activeSkillId || 'none';
    }

    // ── Card state refresh ──────────────────────────────────────
    _refreshCardStates() {
        Object.keys(this.loadedSkills).forEach(id => {
            const card = document.getElementById(`skill-card-${id}`);
            if (!card) return;
            const isActive = this.activeSkillId === id;
            card.classList.toggle('skill-card--active', isActive);

            const btn = card.querySelector('.skill-activate-btn');
            if (!btn || !this.loadedSkills[id]?.compatible) return;

            if (isActive) {
                btn.textContent = '✓ Active';
                btn.classList.add('skill-activate-btn--active');
            } else {
                btn.textContent = 'Activate';
                btn.classList.remove('skill-activate-btn--active');
            }
        });
    }

    // ── Smart Suggestion Engine ─────────────────────────────────
    analyzeContext(userMessage = '', docNames = []) {
        if (this.activeSkillId) return;  // skill already active

        const text     = userMessage.toLowerCase();
        const fileText = docNames.join(' ').toLowerCase();
        const matched  = new Set();

        for (const rule of SUGGESTION_RULES) {
            if (this.dismissedSuggs.has(rule.skill)) continue;
            for (const kw of rule.keywords) {
                if (text.includes(kw)) { matched.add(rule.skill); break; }
            }
        }

        for (const rule of FILENAME_RULES) {
            if (this.dismissedSuggs.has(rule.skill)) continue;
            for (const pat of rule.patterns) {
                if (fileText.includes(pat)) { matched.add(rule.skill); break; }
            }
        }

        const suggestions = [...matched].filter(id => this.loadedSkills[id]?.compatible);
        if (suggestions.length > 0) {
            console.log('[SKILLS] suggestions:', suggestions);
            this._showSuggestions(suggestions);
        }
    }

    _showSuggestions(skillIds) {
        if (!this.suggBar) return;
        const key = [...skillIds].sort().join(',');
        if (key === [...this.lastSuggestions].sort().join(',')) return;
        this.lastSuggestions = skillIds;

        this.suggBar.classList.remove('hidden');
        this.suggBar.innerHTML = `
            <span class="suggestion-label">✨ Suggested</span>
            ${skillIds.map(id => {
                const skill = this.loadedSkills[id];
                if (!skill) return '';
                const icon = SKILL_ICONS[id] || SKILL_ICON_DEFAULT;
                return `<button class="suggestion-chip"
                    onclick="window.skillsUI._acceptSuggestion('${id}')">
                    ${icon} ${skill.name}
                </button>`;
            }).join('')}
            <button class="suggestion-dismiss"
                onclick="window.skillsUI._dismissAllSuggestions()" title="Dismiss">×</button>`;
    }

    _acceptSuggestion(id) {
        this.activateSkill(id);
        this._hideSuggestionBar();
    }

    _dismissAllSuggestions() {
        this.lastSuggestions.forEach(id => this.dismissedSuggs.add(id));
        this._hideSuggestionBar();
    }

    _removeSuggestionChip(id) {
        const remaining = this.lastSuggestions.filter(s => s !== id);
        remaining.length === 0 ? this._hideSuggestionBar() : this._showSuggestions(remaining);
    }

    _hideSuggestionBar() {
        if (!this.suggBar) return;
        this.suggBar.classList.add('hidden');
        this.suggBar.innerHTML = '';
        this.lastSuggestions = [];
    }

    onDocumentsUpdated(docNames = []) {
        this.analyzeContext('', docNames);
    }
}

// ── Self-initialize ────────────────────────────────────────────
// Create instance immediately (no DOM needed for constructor).
// Call init() once DOM is ready — works whether the script runs
// before or after DOMContentLoaded fires.
window.skillsUI = new SkillsUI();

if (document.readyState === 'loading') {
    // DOM not ready yet — wait for it
    document.addEventListener('DOMContentLoaded', () => {
        console.log('[SKILLS] DOMContentLoaded fired — calling init()');
        window.skillsUI.init();
    });
} else {
    // DOM already ready (script loaded late / deferred)
    console.log('[SKILLS] DOM already ready — calling init() immediately');
    window.skillsUI.init();
}
