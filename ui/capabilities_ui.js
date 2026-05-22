/**
 * AS Code — Runtime Capabilities & Skills UI Module
 * Dynamically fetches and visualizes system capabilities and skills.
 */

class CapabilitiesUI {
    constructor() {
        this.drawer = document.getElementById('capabilitiesDrawer');
        this.overlay = document.getElementById('capabilitiesOverlay');
        this.listContainer = document.getElementById('capabilitiesList');
        
        this.toggleBtn = document.getElementById('toggleCapabilities');
        this.closeBtn = document.getElementById('closeCapabilities');

        this.categoryIcons = {
            'core': '🧠',
            'documents': '📄',
            'tools': '🛠',
            'developer': '💻',
            'multimodal': '🎨',
            'network': '🌐'
        };
    }

    init() {
        if (!this.drawer || !this.toggleBtn) return;

        // Toggle drawer on button click
        this.toggleBtn.addEventListener('click', () => this.toggleDrawer());

        // Close on close button or overlay click
        if (this.closeBtn) {
            this.closeBtn.addEventListener('click', () => this.closeDrawer());
        }
        if (this.overlay) {
            this.overlay.addEventListener('click', () => this.closeDrawer());
        }

        // Close on Escape key press
        window.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && !this.drawer.classList.contains('hidden')) {
                this.closeDrawer();
            }
        });
    }

    async toggleDrawer() {
        const isOpen = !this.drawer.classList.contains('hidden');
        if (isOpen) {
            this.closeDrawer();
        } else {
            this.drawer.classList.remove('hidden');
            this.toggleBtn.classList.add('active');
            await this.loadCapabilities();
        }
    }

    closeDrawer() {
        this.drawer.classList.add('hidden');
        this.toggleBtn.classList.remove('active');
    }

    async loadCapabilities() {
        this.listContainer.innerHTML = '<div class="loading-spinner">Loading capabilities...</div>';

        try {
            const response = await fetch('/v1/capabilities');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const capabilities = await response.json();
            this.renderCapabilities(capabilities);
        } catch (error) {
            console.error('Failed to fetch capabilities:', error);
            this.listContainer.innerHTML = `
                <div class="p-4 text-center text-sm text-red-400">
                    <p class="font-semibold">Failed to load capabilities</p>
                    <p class="text-xs text-surface-200/50 mt-1">${error.message}</p>
                </div>
            `;
        }
    }

    renderCapabilities(capabilities) {
        this.listContainer.innerHTML = '';
        const capIds = Object.keys(capabilities);

        if (capIds.length === 0) {
            this.listContainer.innerHTML = '<div class="text-center text-sm text-surface-200/40 p-4">No capabilities found</div>';
            return;
        }

        capIds.forEach(id => {
            const cap = capabilities[id];
            const card = this.createCapabilityCard(cap);
            this.listContainer.appendChild(card);
        });
    }

    createCapabilityCard(cap) {
        const card = document.createElement('div');
        card.className = 'capability-card';

        // Icon based on category
        const icon = this.categoryIcons[cap.category] || '🧩';

        // Determine status styling and text based on status rules
        let statusClass = 'badge-unavailable';
        let statusText = '⛔ Unavailable';

        if (cap.available) {
            if (cap.enabled) {
                if (cap.status === 'healthy') {
                    statusClass = 'badge-healthy';
                    statusText = '✅ Healthy';
                } else if (cap.status === 'degraded') {
                    statusClass = 'badge-degraded';
                    statusText = '⚠ Degraded';
                } else {
                    statusClass = 'badge-disabled';
                    statusText = '❌ Disabled';
                }
            } else {
                statusClass = 'badge-disabled';
                statusText = '❌ Disabled';
            }
        }

        const scopesList = cap.scopes && cap.scopes.length > 0 
            ? `<div class="mt-2 text-[10px] text-surface-200/40"><span class="font-semibold">Scopes:</span> ${cap.scopes.join(', ')}</div>`
            : '';

        const reasonHtml = cap.reason 
            ? `<div class="mt-2 text-xs text-red-400/80 bg-red-950/20 border border-red-950/40 rounded px-2 py-1 font-mono">${cap.reason}</div>`
            : '';

        const providerText = cap.provider ? cap.provider : 'none';

        card.innerHTML = `
            <div class="capability-card-header">
                <div class="flex items-center gap-2">
                    <span class="text-base">${icon}</span>
                    <h3 class="text-sm font-medium text-surface-100">${cap.name}</h3>
                </div>
                <span class="capability-badge ${statusClass}">${statusText}</span>
            </div>
            <p class="text-xs text-surface-200/60 mt-1">${cap.description}</p>
            <div class="capability-meta">
                <div><span class="text-surface-200/30">Provider:</span> ${providerText}</div>
                <div><span class="text-surface-200/30">Version:</span> ${cap.version || '1.0.0'}</div>
            </div>
            ${scopesList}
            ${reasonHtml}
        `;

        return card;
    }
}

// Instantiate and expose globally
window.capabilitiesUI = new CapabilitiesUI();
// NOTE: window.skillsUI is set by /static/skills_ui.js — do NOT set it here.
