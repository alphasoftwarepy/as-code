/**
 * AS Code — Skills UI & Experimental Skills Lab (v3.0.0)
 *
 * Full Experimental Lifecycle:
 * CREATE → READY → TEST → RESULT → EDIT & RETEST → VERSION HISTORY → VIEW PROPOSAL → DELETE
 *
 * Public API (used by app.js):
 *   window.skillsUI.getActiveSkillId()  → string | null
 *   window.skillsUI.analyzeContext(msg, docNames)
 *   window.skillsUI.loadSkills()
 */

console.log('[SKILLS] module v3.0.0 (Lab Lifecycle Edition) loaded ✓');

// ── Skill Icons ────────────────────────────────────────────────
const SKILL_ICONS = {
    marketing:          '📈',
    sales:              '📞',
    business:           '💼',
    legal:              '⚖️',
    content_creator:    '🎬',
    csv_data_extractor: '📊',
    json_validator_lab: '📐',
};
const SKILL_ICON_DEFAULT = '✨';
const SKILL_ICON_EXP_DEFAULT = '🧪';

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
        // DOM refs
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

        // Tabs & Counts
        this.tabOfficial    = null;
        this.tabExp         = null;
        this.officialCount  = null;
        this.expCount       = null;

        // Modals
        this.createModal    = null;
        this.editModal      = null;
        this.deleteModal    = null;
        this.proposalModal  = null;

        // State
        this.activeTab          = 'official'; // 'official' | 'experimental'
        this.activeSkillId      = null;
        this.loadedOfficial     = {};
        this.loadedExperimental = [];
        this.dismissedSuggs     = new Set();
        this.lastSuggestions    = [];
        this.testingSkillIds    = new Set();
        this.deletingSkillId    = null;

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

        // Navigation Tabs
        this.tabOfficial   = document.getElementById('tabOfficialSkills');
        this.tabExp        = document.getElementById('tabExperimentalSkills');
        this.officialCount = document.getElementById('officialSkillsCount');
        this.expCount      = document.getElementById('experimentalSkillsCount');

        // Modals
        this.createModal   = document.getElementById('createSkillModal');
        this.editModal     = document.getElementById('editSkillModal');
        this.deleteModal   = document.getElementById('deleteSkillModal');
        this.proposalModal = document.getElementById('proposalModal');

        if (!this.drawer || !this.toggleBtn) {
            console.error('[SKILLS] ❌ #skillsDrawer or #toggleSkills not found — init aborted');
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

        // Escape key handling
        window.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                if (this.editModal && !this.editModal.classList.contains('hidden')) {
                    this.closeEditModal();
                } else if (this.deleteModal && !this.deleteModal.classList.contains('hidden')) {
                    this.closeDeleteModal();
                } else if (this.createModal && !this.createModal.classList.contains('hidden')) {
                    this.closeCreateModal();
                } else if (this.proposalModal && !this.proposalModal.classList.contains('hidden')) {
                    this.closeProposalModal();
                } else if (this.drawer && !this.drawer.classList.contains('hidden')) {
                    this.closeDrawer();
                }
            }
        });

        // Deactivate pill
        this.deactivateBtn?.addEventListener('click', () => this.deactivateSkill());

        // Tab Navigation
        this.tabOfficial?.addEventListener('click', () => this.switchTab('official'));
        this.tabExp?.addEventListener('click', () => this.switchTab('experimental'));

        // Setup Modals
        this._initCreateModal();
        this._initEditModal();
        this._initDeleteModal();
        this._initProposalModal();

        console.log('[SKILLS] drawer and lab initialized ✓');
    }

    _initCreateModal() {
        document.getElementById('closeCreateSkillModal')?.addEventListener('click', () => this.closeCreateModal());
        document.getElementById('cancelCreateSkillBtn')?.addEventListener('click', () => this.closeCreateModal());
        document.getElementById('submitCreateSkillBtn')?.addEventListener('click', () => this.submitCreateSkill());

        const nameInput = document.getElementById('expSkillName');
        const idInput = document.getElementById('expSkillId');
        nameInput?.addEventListener('input', () => {
            if (idInput && (!idInput.dataset.touched || idInput.dataset.touched === 'false')) {
                idInput.value = nameInput.value.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
            }
        });
        idInput?.addEventListener('input', () => {
            idInput.dataset.touched = 'true';
        });
    }

    _initEditModal() {
        document.getElementById('closeEditSkillModal')?.addEventListener('click', () => this.closeEditModal());
        document.getElementById('cancelEditSkillBtn')?.addEventListener('click', () => this.closeEditModal());
        document.getElementById('submitEditAndTestBtn')?.addEventListener('click', () => this.submitEditAndTest());
    }

    _initDeleteModal() {
        document.getElementById('cancelDeleteSkillBtn')?.addEventListener('click', () => this.closeDeleteModal());
        document.getElementById('confirmDeleteSkillBtn')?.addEventListener('click', () => this.confirmDelete());
    }

    _initProposalModal() {
        document.getElementById('closeProposalModal')?.addEventListener('click', () => this.closeProposalModal());
        document.getElementById('dismissProposalBtn')?.addEventListener('click', () => this.closeProposalModal());
        document.getElementById('copyProposalBtn')?.addEventListener('click', () => {
            const content = document.getElementById('proposalContentRender')?.textContent || '';
            navigator.clipboard.writeText(content).then(() => {
                const btn = document.getElementById('copyProposalBtn');
                if (btn) {
                    const original = btn.innerHTML;
                    btn.innerHTML = '<span>✅ Copied!</span>';
                    setTimeout(() => { btn.innerHTML = original; }, 2000);
                }
            });
        });
    }

    // ── Tabs ───────────────────────────────────────────────────
    switchTab(tabName) {
        this.activeTab = tabName;
        if (this.tabOfficial) {
            this.tabOfficial.classList.toggle('active', tabName === 'official');
            this.tabOfficial.classList.toggle('border-accent-400', tabName === 'official');
            this.tabOfficial.classList.toggle('text-accent-400', tabName === 'official');
            this.tabOfficial.classList.toggle('border-transparent', tabName !== 'official');
            this.tabOfficial.classList.toggle('text-surface-400', tabName !== 'official');
        }
        if (this.tabExp) {
            this.tabExp.classList.toggle('active', tabName === 'experimental');
            this.tabExp.classList.toggle('border-amber-400', tabName === 'experimental');
            this.tabExp.classList.toggle('text-amber-300', tabName === 'experimental');
            this.tabExp.classList.toggle('border-transparent', tabName !== 'experimental');
            this.tabExp.classList.toggle('text-surface-400', tabName !== 'experimental');
        }
        this.renderCurrentView();
    }

    // ── Drawer ─────────────────────────────────────────────────
    toggleDrawer() {
        const isOpen = this.drawer && !this.drawer.classList.contains('hidden');
        isOpen ? this.closeDrawer() : this.openDrawer();
    }

    openDrawer() {
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

    // ── Fetch + Render ─────────────────────────────────────────
    async loadSkills() {
        if (!this.listBody) return;
        this.listBody.innerHTML = '<div class="loading-spinner">Loading skills & experimental lab...</div>';

        try {
            const [officialRes, expRes] = await Promise.allSettled([
                fetch('/v1/skills'),
                fetch('/v1/skills/experimental')
            ]);

            if (officialRes.status === 'fulfilled' && officialRes.value.ok) {
                this.loadedOfficial = await officialRes.value.json();
            } else {
                this.loadedOfficial = {};
            }

            if (expRes.status === 'fulfilled' && expRes.value.ok) {
                this.loadedExperimental = await expRes.value.json();
            } else {
                this.loadedExperimental = [];
            }

            // Update tab badges
            const officialCount = Object.keys(this.loadedOfficial).length;
            const expCount = this.loadedExperimental.length;
            if (this.officialCount) this.officialCount.textContent = officialCount;
            if (this.expCount) this.expCount.textContent = expCount;

            this._syncHiddenSelect(this.loadedOfficial);
            this.renderCurrentView();
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

    renderCurrentView() {
        if (this.activeTab === 'experimental') {
            this._renderExperimentalSkills(this.loadedExperimental);
        } else {
            this._renderOfficialSkills(this.loadedOfficial);
        }
    }

    // ── Official Skills View ───────────────────────────────────
    _renderOfficialSkills(skills) {
        if (!this.listBody) return;
        this.listBody.innerHTML = '';

        const ids = Object.keys(skills);
        if (ids.length === 0) {
            this.listBody.innerHTML = `
                <div style="padding:2rem;text-align:center;color:rgba(225,229,235,0.4);font-size:0.8rem;">
                    <span style="font-size:1.5rem;display:block;margin-bottom:0.5rem;">✨</span>
                    No official skills installed in <code>skills/</code>
                </div>`;
            return;
        }

        const headerNotice = document.createElement('div');
        headerNotice.style.cssText = 'padding:0.6rem 0.8rem;margin-bottom:0.8rem;border-radius:8px;background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);font-size:0.7rem;color:rgba(225,229,235,0.6);display:flex;align-items:center;justify-content:space-between;';
        headerNotice.innerHTML = `
            <span>✨ <strong>Official Skills</strong> (${ids.length}) — Readily available in coordinator</span>
            <span style="font-size:0.65rem;color:#6ea8fe;background:rgba(110,168,254,0.1);padding:0.15rem 0.4rem;border-radius:4px;">PRODUCTION</span>`;
        this.listBody.appendChild(headerNotice);

        const compatible   = ids.filter(id => skills[id].compatible);
        const incompatible = ids.filter(id => !skills[id].compatible);

        [...compatible, ...incompatible].forEach((id, index) => {
            const card = this._createOfficialSkillCard(id, skills[id]);
            card.style.animationDelay = `${index * 40}ms`;
            this.listBody.appendChild(card);
        });
    }

    _createOfficialSkillCard(id, skill) {
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
                <span style="font-size:0.6rem;color:rgba(225,229,235,0.2);font-family:'JetBrains Mono',monospace;">skills/${id}</span>
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

    // ── Experimental Skills Lab View ───────────────────────────
    _renderExperimentalSkills(expSkills) {
        if (!this.listBody) return;
        this.listBody.innerHTML = '';

        // Lab Control Header Banner
        const labHeader = document.createElement('div');
        labHeader.style.cssText = 'padding:0.75rem 0.9rem;margin-bottom:0.9rem;border-radius:8px;background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.2);display:flex;flex-direction:column;gap:0.5rem;';
        labHeader.innerHTML = `
            <div style="display:flex;align-items:center;justify-content:space-between;">
                <div style="display:flex;align-items:center;gap:0.4rem;">
                    <span style="font-size:1.1rem;">🧪</span>
                    <span style="font-size:0.8rem;font-weight:600;color:#fcd34d;">Experimental Skills Lab</span>
                </div>
                <button id="btnOpenCreateModal" class="modal-btn modal-btn-primary" style="font-size:0.72rem;padding:0.3rem 0.65rem;background:#f59e0b;color:#000;font-weight:600;display:flex;align-items:center;gap:0.25rem;">
                    <span>+ Create Experimental Skill</span>
                </button>
            </div>
            <p style="margin:0;font-size:0.7rem;color:rgba(225,229,235,0.65);line-height:1.4;">
                Controlled development cycle: <strong>Create → Test → Edit & Retest → Version History → Proposal</strong>. Safe sandbox isolation under <code>temp_skills/</code>.
            </p>`;
        this.listBody.appendChild(labHeader);

        labHeader.querySelector('#btnOpenCreateModal')?.addEventListener('click', () => this.openCreateModal());

        if (!expSkills || expSkills.length === 0) {
            const emptyNotice = document.createElement('div');
            emptyNotice.style.cssText = 'padding:2.5rem 1rem;text-align:center;color:rgba(225,229,235,0.35);font-size:0.8rem;border:1px dashed rgba(255,255,255,0.08);border-radius:8px;';
            emptyNotice.innerHTML = `
                <div style="font-size:2rem;margin-bottom:0.6rem;">🧪</div>
                <div style="font-weight:600;color:rgba(225,229,235,0.6);margin-bottom:0.3rem;">No Experimental Skills in Sandbox</div>
                <p style="font-size:0.72rem;margin:0 0 1rem 0;color:rgba(225,229,235,0.4);">Create a sandboxed temporary skill to test automated tools and generate reproducible proposals.</p>
                <button class="modal-btn modal-btn-primary" style="font-size:0.75rem;padding:0.4rem 0.8rem;background:#f59e0b;color:#000;" onclick="window.skillsUI.openCreateModal()">
                    + Create First Experimental Skill
                </button>`;
            this.listBody.appendChild(emptyNotice);
            return;
        }

        expSkills.forEach((skill, index) => {
            const card = this._createExperimentalSkillCard(skill);
            card.style.animationDelay = `${index * 40}ms`;
            this.listBody.appendChild(card);
        });
    }

    _createExperimentalSkillCard(skill) {
        const id = skill.skill_id;
        const icon = SKILL_ICONS[id] || SKILL_ICON_EXP_DEFAULT;
        const lifecycle = skill.lifecycle || 'READY';
        const isTesting = this.testingSkillIds.has(id);
        const version = skill.version || 1;
        const history = skill.history || [];

        let statusBadgeClass = 'badge-healthy';
        let statusText = `● ${lifecycle}`;
        if (isTesting) {
            statusBadgeClass = 'badge-unavailable';
            statusText = '⏳ TESTING...';
        } else if (lifecycle === 'PASSED') {
            statusBadgeClass = 'badge-healthy';
            statusText = '✅ PASSED';
        } else if (lifecycle === 'FAILED') {
            statusBadgeClass = 'badge-unavailable';
            statusText = '❌ FAILED';
        } else if (lifecycle === 'READY') {
            statusBadgeClass = 'badge-healthy';
            statusText = '● READY';
        }

        // Recommendation badge
        let recBadge = '';
        if (skill.recommendation === 'APPROVE') {
            recBadge = `<span style="font-size:0.65rem;background:rgba(16,185,129,0.15);color:#34d399;border:1px solid rgba(16,185,129,0.3);padding:0.1rem 0.4rem;border-radius:4px;font-weight:600;">Recommendation: APPROVE</span>`;
        } else if (skill.recommendation === 'NEEDS_REFINEMENT') {
            recBadge = `<span style="font-size:0.65rem;background:rgba(245,158,11,0.15);color:#fbbf24;border:1px solid rgba(245,158,11,0.3);padding:0.1rem 0.4rem;border-radius:4px;font-weight:600;">Recommendation: NEEDS_REFINEMENT</span>`;
        } else if (skill.recommendation === 'REJECT') {
            recBadge = `<span style="font-size:0.65rem;background:rgba(239,68,68,0.15);color:#f87171;border:1px solid rgba(239,68,68,0.3);padding:0.1rem 0.4rem;border-radius:4px;font-weight:600;">Recommendation: REJECT</span>`;
        }

        // Version History Transition Pill (e.g. Previous: v1 - FAILED, Current: v2 - PASSED)
        let historyPill = '';
        if (history.length > 1) {
            const prev = history[history.length - 2];
            const prevStatus = prev.test_passed ? 'PASSED' : (prev.test_passed === false ? 'FAILED' : 'READY');
            const prevColor = prev.test_passed ? '#34d399' : (prev.test_passed === false ? '#f87171' : '#94a3b8');
            historyPill = `
                <div style="font-size:0.65rem;color:rgba(225,229,235,0.6);margin-bottom:0.35rem;display:flex;align-items:center;gap:0.4rem;">
                    <span>Previous: <strong style="color:${prevColor};">v${prev.version} — ${prevStatus}</strong></span>
                    <span>→</span>
                    <span>Current: <strong style="color:#fcd34d;">v${version} — ${lifecycle}</strong></span>
                </div>`;
        }

        const scopes = (skill.required_scopes || []).map(s => `<span class="skill-scope-tag">🧩 ${s}</span>`).join(' ');

        // Metrics / test info box if available
        let metricsHtml = '';
        if (skill.metrics && skill.metrics.duration_total_ms !== undefined) {
            metricsHtml = `
                <div style="margin-top:0.5rem;padding:0.4rem 0.6rem;background:rgba(0,0,0,0.3);border:1px solid rgba(255,255,255,0.06);border-radius:6px;font-size:0.68rem;display:flex;align-items:center;justify-content:space-between;color:rgba(225,229,235,0.7);font-family:'JetBrains Mono',monospace;">
                    <span>⏱️ Duration: <strong>${skill.metrics.duration_total_ms}ms</strong></span>
                    <span>Security: <strong style="color:#34d399;">PASS</strong></span>
                    <span>Isolation: <strong style="color:#34d399;">PASS</strong></span>
                </div>`;
        }

        const isSelected = (this.activeSkillId === id);

        const card = document.createElement('div');
        card.className = 'skill-card' + (isSelected ? ' skill-card--active' : '');
        card.id = `exp-skill-card-${id}`;
        card.style.cssText = 'border-color: rgba(245,158,11,0.25); background: rgba(21, 26, 38, 0.7);';

        card.innerHTML = `
            <div class="skill-card-header">
                <div class="skill-card-title">
                    <span class="skill-icon">${icon}</span>
                    <span class="skill-name">${skill.name}</span>
                    <span style="font-size:0.6rem;background:rgba(245,158,11,0.15);color:#fbbf24;border:1px solid rgba(245,158,11,0.3);padding:0.1rem 0.35rem;border-radius:4px;">EXPERIMENTAL</span>
                    <span style="font-size:0.6rem;background:rgba(110,168,254,0.15);color:#93c5fd;border:1px solid rgba(110,168,254,0.3);padding:0.1rem 0.35rem;border-radius:4px;font-weight:600;">v${version}</span>
                </div>
                <span class="skill-status-badge ${statusBadgeClass}">${statusText}</span>
            </div>
            
            ${historyPill}

            <p class="skill-description">${skill.description || ''}</p>
            ${skill.objective ? `<div style="font-size:0.7rem;color:rgba(225,229,235,0.7);margin-bottom:0.4rem;"><strong>Objective:</strong> ${skill.objective}</div>` : ''}
            
            <div style="display:flex;flex-wrap:wrap;gap:0.3rem;margin-top:0.3rem;">
                <span class="skill-scope-tag" style="background:rgba(110,168,254,0.08);color:#6ea8fe;">🤖 Model: ${skill.recommended_model || 'auto'}</span>
                ${scopes}
            </div>

            <div style="margin-top:0.4rem;display:flex;align-items:center;justify-content:space-between;">
                ${recBadge}
                <span style="font-size:0.6rem;color:rgba(225,229,235,0.3);font-family:'JetBrains Mono',monospace;">temp_skills/${id}</span>
            </div>

            ${metricsHtml}

            <!-- Card Actions -->
            <div class="skill-card-footer" style="margin-top:0.75rem;padding-top:0.6rem;border-top:1px solid rgba(255,255,255,0.06);display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:0.4rem;">
                <div style="display:flex;align-items:center;gap:0.3rem;">
                    <button class="skill-activate-btn" style="background:rgba(255,255,255,0.08);color:#cbd5e1;font-size:0.7rem;padding:0.25rem 0.5rem;"
                        onclick="window.skillsUI._selectExperimentalSkill('${id}', '${skill.name}')">
                        ${isSelected ? '✓ Selected' : 'Select'}
                    </button>
                    <button class="skill-activate-btn" style="background:rgba(245,158,11,0.85);color:#000;font-weight:600;font-size:0.7rem;padding:0.25rem 0.5rem;"
                        onclick="window.skillsUI.testExperimentalSkill('${id}')" ${isTesting ? 'disabled' : ''}>
                        ${isTesting ? '⏳ Testing...' : '⚡ Test'}
                    </button>
                    <button class="skill-activate-btn" style="background:rgba(245,158,11,0.15);color:#fbbf24;border:1px solid rgba(245,158,11,0.3);font-size:0.7rem;padding:0.25rem 0.5rem;"
                        onclick="window.skillsUI.openEditModal('${id}')" ${isTesting ? 'disabled' : ''}>
                        ✏️ Edit & Retest
                    </button>
                </div>
                <div style="display:flex;align-items:center;gap:0.3rem;">
                    <button class="skill-activate-btn" style="background:rgba(110,168,254,0.15);color:#93c5fd;border:1px solid rgba(110,168,254,0.3);font-size:0.7rem;padding:0.25rem 0.5rem;"
                        onclick="window.skillsUI.viewProposal('${id}')" ${skill.has_proposal ? '' : 'disabled title="Run test first to generate proposal"'}>
                        📄 View Proposal
                    </button>
                    <button class="icon-btn" style="color:rgba(239,68,68,0.7);font-size:0.85rem;padding:0.25rem 0.4rem;"
                        title="Delete experimental skill" onclick="window.skillsUI.openDeleteModal('${id}')" ${isTesting ? 'disabled' : ''}>
                        🗑️
                    </button>
                </div>
            </div>`;

        return card;
    }

    _selectExperimentalSkill(id, name) {
        if (this.activeSkillId === id) {
            this.deactivateSkill();
        } else {
            this.activeSkillId = id;
            this._updatePill(id, { name: `${name} (EXP)` });
            this.renderCurrentView();
            console.log(`[SKILLS-LAB] Selected Experimental Skill: ${id}`);
        }
    }

    // ── Test Experimental Skill ────────────────────────────────
    async testExperimentalSkill(skillId) {
        if (this.testingSkillIds.has(skillId)) {
            console.warn(`[SKILLS-LAB] Test already running for ${skillId}`);
            return;
        }

        console.log(`[SKILLS-LAB] Testing experimental skill: ${skillId}`);
        this.testingSkillIds.add(skillId);
        this.renderCurrentView();

        try {
            const res = await fetch(`/v1/skills/experimental/${encodeURIComponent(skillId)}/test`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.detail || `HTTP ${res.status}`);
            }

            const testResult = await res.json();
            console.log(`[SKILLS-LAB] Test result for ${skillId} v${testResult.version}:`, testResult);

            // Refresh loaded list
            await this.loadSkills();

            // Auto-prompt to view generated proposal if test passed
            if (testResult.passed) {
                this.viewProposal(skillId);
            }
        } catch (err) {
            console.error(`[SKILLS-LAB] Test failed for ${skillId}:`, err);
            alert(`Test execution failed: ${err.message}`);
        } finally {
            this.testingSkillIds.delete(skillId);
            this.renderCurrentView();
        }
    }

    // ── Edit & Retest Flow ─────────────────────────────────────
    async openEditModal(skillId) {
        if (!this.editModal) return;
        const errEl = document.getElementById('editSkillError');
        if (errEl) errEl.style.display = 'none';

        try {
            const res = await fetch(`/v1/skills/experimental/${encodeURIComponent(skillId)}`);
            if (!res.ok) throw new Error('Failed to load skill details');
            const skill = await res.json();

            document.getElementById('editSkillId').value = skill.skill_id;
            document.getElementById('editSkillName').value = skill.name || '';
            document.getElementById('editSkillVersionPreview').value = `v${skill.version} → v${skill.version + 1}`;
            document.getElementById('editSkillChangesDesc').value = '';
            document.getElementById('editSkillDesc').value = skill.description || '';
            document.getElementById('editSkillObjective').value = skill.objective || '';
            document.getElementById('editSkillModel').value = skill.recommended_model || 'code';
            document.getElementById('editSkillCapabilities').value = (skill.requested_capabilities || []).join(', ');
            document.getElementById('editSkillUsesCaps').checked = skill.uses_capabilities ?? true;
            document.getElementById('editSkillInstructions').value = skill.instructions || '';
            document.getElementById('editSkillRecipe').value = skill.recipe || '';

            const titleEl = document.getElementById('editSkillModalTitle');
            if (titleEl) titleEl.textContent = `Edit Experimental Skill — ${skill.name}`;

            this.editModal.classList.remove('hidden');
            document.getElementById('editSkillChangesDesc')?.focus();
        } catch (err) {
            console.error('[SKILLS-LAB] Could not open edit modal:', err);
            alert(`Could not load skill details: ${err.message}`);
        }
    }

    closeEditModal() {
        this.editModal?.classList.add('hidden');
    }

    async submitEditAndTest() {
        const skillId = document.getElementById('editSkillId')?.value.trim();
        const name = document.getElementById('editSkillName')?.value.trim();
        const changes_description = document.getElementById('editSkillChangesDesc')?.value.trim() || 'Prompt and parameters updated';
        const description = document.getElementById('editSkillDesc')?.value.trim();
        const objective = document.getElementById('editSkillObjective')?.value.trim();
        const model = document.getElementById('editSkillModel')?.value || 'code';
        const capsStr = document.getElementById('editSkillCapabilities')?.value.trim() || '';
        const uses_capabilities = document.getElementById('editSkillUsesCaps')?.checked ?? true;
        const instructions = document.getElementById('editSkillInstructions')?.value.trim();
        const recipe = document.getElementById('editSkillRecipe')?.value.trim();
        const errEl = document.getElementById('editSkillError');

        if (!skillId || !name || !description || !objective || !instructions) {
            if (errEl) {
                errEl.textContent = 'Please fill out all required fields marked with *';
                errEl.style.display = 'block';
            }
            return;
        }

        const requested_capabilities = capsStr ? capsStr.split(',').map(s => s.trim()).filter(Boolean) : [];

        const payload = {
            name,
            description,
            objective,
            instructions,
            recipe,
            requested_capabilities,
            recommended_model: model,
            prompt_family: model === 'code' ? 'SOFTWARE_PROMPT' : 'GENERAL_PROMPT',
            uses_capabilities,
            changes_description,
        };

        const submitBtn = document.getElementById('submitEditAndTestBtn');
        if (submitBtn) submitBtn.disabled = true;

        try {
            console.log(`[SKILLS-LAB] Saving edit for ${skillId}:`, payload);
            const res = await fetch(`/v1/skills/experimental/${encodeURIComponent(skillId)}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.detail || `HTTP ${res.status}`);
            }

            const data = await res.json();
            console.log('[SKILLS-LAB] Updated successfully:', data);

            this.closeEditModal();

            // Immediately trigger test on the new version (SAVE & TEST)
            await this.testExperimentalSkill(skillId);
        } catch (err) {
            console.error('[SKILLS-LAB] Edit failed:', err);
            if (errEl) {
                errEl.textContent = `Error: ${err.message}`;
                errEl.style.display = 'block';
            }
        } finally {
            if (submitBtn) submitBtn.disabled = false;
        }
    }

    // ── Delete Flow ────────────────────────────────────────────
    openDeleteModal(skillId) {
        if (!this.deleteModal) return;
        this.deletingSkillId = skillId;
        const msgEl = document.getElementById('deleteSkillModalMessage');
        const dirEl = document.getElementById('deleteSkillDir');
        if (dirEl) dirEl.textContent = `temp_skills/${skillId}/`;
        this.deleteModal.classList.remove('hidden');
    }

    closeDeleteModal() {
        this.deletingSkillId = null;
        this.deleteModal?.classList.add('hidden');
    }

    async confirmDelete() {
        const skillId = this.deletingSkillId;
        if (!skillId) return;

        const confirmBtn = document.getElementById('confirmDeleteSkillBtn');
        if (confirmBtn) confirmBtn.disabled = true;

        try {
            console.log(`[SKILLS-LAB] Deleting experimental skill sandbox: ${skillId}`);
            const res = await fetch(`/v1/skills/experimental/${encodeURIComponent(skillId)}`, {
                method: 'DELETE',
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.detail || `HTTP ${res.status}`);
            }

            const data = await res.json();
            console.log('[SKILLS-LAB] Deleted:', data);

            if (this.activeSkillId === skillId) {
                this.deactivateSkill();
            }

            this.closeDeleteModal();
            await this.loadSkills();
        } catch (err) {
            console.error('[SKILLS-LAB] Deletion failed:', err);
            alert(`Deletion failed: ${err.message}`);
        } finally {
            if (confirmBtn) confirmBtn.disabled = false;
        }
    }

    // ── View Proposal ──────────────────────────────────────────
    async viewProposal(skillId) {
        console.log(`[SKILLS-LAB] Viewing proposal for: ${skillId}`);
        if (!this.proposalModal) return;

        const titleEl = document.getElementById('proposalModalTitle');
        const fileEl = document.getElementById('proposalModalFile');
        const contentEl = document.getElementById('proposalContentRender');

        if (titleEl) titleEl.textContent = `Skill Proposal — ${skillId}`;
        if (fileEl) fileEl.textContent = `dev-notes/skill-proposals/${skillId}.md`;
        if (contentEl) contentEl.textContent = 'Fetching proposal and version history...';

        this.proposalModal.classList.remove('hidden');

        try {
            const res = await fetch(`/v1/skills/experimental/${encodeURIComponent(skillId)}/proposal`);
            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.detail || 'Proposal not found');
            }

            const data = await res.json();
            if (fileEl) fileEl.textContent = data.filename || data.filepath || `dev-notes/skill-proposals/${skillId}.md`;
            if (contentEl) contentEl.textContent = data.content;
        } catch (err) {
            if (contentEl) contentEl.textContent = `⚠️ Could not load proposal: ${err.message}\n\nPlease run [⚡ Test] on this skill first to generate its reproducible proposal.`;
        }
    }

    closeProposalModal() {
        this.proposalModal?.classList.add('hidden');
    }

    // ── Create Modal ───────────────────────────────────────────
    openCreateModal() {
        if (!this.createModal) return;
        const errEl = document.getElementById('createSkillError');
        if (errEl) errEl.style.display = 'none';

        const nameInput = document.getElementById('expSkillName');
        const idInput = document.getElementById('expSkillId');
        const descInput = document.getElementById('expSkillDesc');
        const objInput = document.getElementById('expSkillObjective');
        const instInput = document.getElementById('expSkillInstructions');
        const recipeInput = document.getElementById('expSkillRecipe');

        if (nameInput) nameInput.value = '';
        if (idInput) {
            idInput.value = '';
            idInput.dataset.touched = 'false';
        }
        if (descInput) descInput.value = '';
        if (objInput) objInput.value = '';
        if (instInput) {
            instInput.value = 'MODO: extractor_analitico\nPOSTURA: especializada\nREGLAS: Validar entradas y estructurar salida determinista.';
        }
        if (recipeInput) {
            recipeInput.value = '1. Create sandbox\n2. Run deterministic validation\n3. Verify zero official contamination';
        }

        this.createModal.classList.remove('hidden');
        nameInput?.focus();
    }

    closeCreateModal() {
        this.createModal?.classList.add('hidden');
    }

    async submitCreateSkill() {
        const name = document.getElementById('expSkillName')?.value.trim();
        const skill_id = document.getElementById('expSkillId')?.value.trim();
        const description = document.getElementById('expSkillDesc')?.value.trim();
        const objective = document.getElementById('expSkillObjective')?.value.trim();
        const model = document.getElementById('expSkillModel')?.value || 'code';
        const capsStr = document.getElementById('expSkillCapabilities')?.value.trim() || '';
        const uses_capabilities = document.getElementById('expSkillUsesCaps')?.checked ?? true;
        const instructions = document.getElementById('expSkillInstructions')?.value.trim();
        const recipe = document.getElementById('expSkillRecipe')?.value.trim();
        const errEl = document.getElementById('createSkillError');

        if (!name || !skill_id || !description || !objective || !instructions) {
            if (errEl) {
                errEl.textContent = 'Please fill out all required fields marked with *';
                errEl.style.display = 'block';
            }
            return;
        }

        const requested_capabilities = capsStr ? capsStr.split(',').map(s => s.trim()).filter(Boolean) : [];

        const payload = {
            name,
            skill_id,
            description,
            objective,
            recipe,
            instructions,
            requested_capabilities,
            recommended_model: model,
            prompt_family: model === 'code' ? 'SOFTWARE_PROMPT' : 'GENERAL_PROMPT',
            uses_capabilities,
        };

        const submitBtn = document.getElementById('submitCreateSkillBtn');
        if (submitBtn) submitBtn.disabled = true;

        try {
            console.log('[SKILLS-LAB] Creating experimental skill:', payload);
            const res = await fetch('/v1/skills/experimental', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.detail || `HTTP ${res.status}`);
            }

            const data = await res.json();
            console.log('[SKILLS-LAB] Created:', data);

            this.closeCreateModal();
            this.activeTab = 'experimental';
            await this.loadSkills();
        } catch (err) {
            console.error('[SKILLS-LAB] Creation failed:', err);
            if (errEl) {
                errEl.textContent = `Error: ${err.message}`;
                errEl.style.display = 'block';
            }
        } finally {
            if (submitBtn) submitBtn.disabled = false;
        }
    }

    // ── Official Skill Activation ──────────────────────────────
    _handleActivateClick(id) {
        if (this.activeSkillId === id) {
            this.deactivateSkill();
        } else {
            this.activateSkill(id);
        }
    }

    activateSkill(id) {
        const skill = this.loadedOfficial[id];
        if (!skill) return;
        if (!skill.compatible) return;

        this.activeSkillId = id;
        this._syncHiddenSelect(this.loadedOfficial);
        this._updatePill(id, skill);
        this._refreshCardStates();
        this.dismissedSuggs.add(id);
        this._removeSuggestionChip(id);

        console.log(`[SKILLS] ✅ Activated Official Skill: ${id} (${skill.name})`);
    }

    deactivateSkill() {
        this.activeSkillId = null;
        this._syncHiddenSelect(this.loadedOfficial);
        this._hidePill();
        this._refreshCardStates();
        this.renderCurrentView();
    }

    getActiveSkillId() {
        return this.activeSkillId;
    }

    // ── Pill ────────────────────────────────────────────────────
    _updatePill(id, skill) {
        if (!this.pill) return;
        const icon = SKILL_ICONS[id] || SKILL_ICON_DEFAULT;
        if (this.pillIcon) this.pillIcon.textContent = icon;
        if (this.pillName) this.pillName.textContent = skill.name;
        this.pill.classList.remove('hidden');
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

    _refreshCardStates() {
        Object.keys(this.loadedOfficial).forEach(id => {
            const card = document.getElementById(`skill-card-${id}`);
            if (!card) return;
            const isActive = this.activeSkillId === id;
            card.classList.toggle('skill-card--active', isActive);

            const btn = card.querySelector('.skill-activate-btn');
            if (!btn || !this.loadedOfficial[id]?.compatible) return;

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
        if (this.activeSkillId) return;

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

        const suggestions = [...matched].filter(id => this.loadedOfficial[id]?.compatible);
        if (suggestions.length > 0) {
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
                const skill = this.loadedOfficial[id];
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
window.skillsUI = new SkillsUI();

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.skillsUI.init();
    });
} else {
    window.skillsUI.init();
}
