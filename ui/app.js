/**
 * AS Code — Minimal UI Application Logic
 * Vanilla JS only, zero framework overhead.
 * Handles SSE streaming, state management, basic rendering, and telemetry.
 */

document.addEventListener('DOMContentLoaded', () => {
    // ── DOM Elements ──────────────────────────────────────────
    const elements = {
        messageInput: document.getElementById('messageInput'),
        sendBtn: document.getElementById('sendBtn'),
        stopBtn: document.getElementById('stopBtn'),
        clearBtn: document.getElementById('clearBtn'),
        messagesContainer: document.getElementById('messagesContainer'),
        welcomeScreen: document.getElementById('welcomeScreen'),
        modelSelect: document.getElementById('modelSelect'),
        presetSelect: document.getElementById('presetSelect'),
        temperatureSlider: document.getElementById('temperatureSlider'),
        tempValue: document.getElementById('tempValue'),
        maxTokensInput: document.getElementById('maxTokensInput'),
        statusIndicator: document.getElementById('statusIndicator'),
        routingIndicator: document.getElementById('routingIndicator'),
        toggleSettings: document.getElementById('toggleSettings'),
        settingsPanel: document.getElementById('settingsPanel'),
        toggleTelemetry: document.getElementById('toggleTelemetry'),
        telemetryBar: document.getElementById('telemetryBar'),

        // Telemetry values
        telRam: document.getElementById('telRam'),
        telVram: document.getElementById('telVram'),
        telTps: document.getElementById('telTps'),
        telModel: document.getElementById('telModel'),
        telProvider: document.getElementById('telProvider'),
    };

    // ── State ─────────────────────────────────────────────────
    let state = {
        isGenerating: false,
        chatHistory: [], // Array of { role, content }
        abortController: null,
        currentRequestId: null,
        statusPollInterval: null
    };

    // ── Event Listeners ───────────────────────────────────────

    // Input & Send
    elements.sendBtn.addEventListener('click', handleSend);
    elements.messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });

    // Auto-resize textarea
    elements.messageInput.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        if (this.value === '') {
            this.style.height = 'auto';
        }
    });

    // Welcome screen chips
    document.querySelectorAll('.welcome-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            elements.messageInput.value = chip.dataset.prompt;
            elements.messageInput.style.height = 'auto';
            handleSend();
        });
    });

    // Actions
    elements.stopBtn.addEventListener('click', stopGeneration);
    elements.clearBtn.addEventListener('click', clearChat);

    // Toggles
    elements.toggleSettings.addEventListener('click', () => {
        elements.settingsPanel.classList.toggle('hidden');
        elements.toggleSettings.classList.toggle('active');
    });

    elements.toggleTelemetry.addEventListener('click', () => {
        elements.telemetryBar.classList.toggle('hidden');
        elements.toggleTelemetry.classList.toggle('active');
    });

    // Settings
    elements.temperatureSlider.addEventListener('input', (e) => {
        elements.tempValue.textContent = e.target.value;
    });

    elements.presetSelect?.addEventListener('change', (e) => {
        const presetMap = {
            PRECISE: { temp: 0.1, tokens: 2048 },
            BALANCED: { temp: 0.5, tokens: 4096 },
            CREATIVE: { temp: 0.8, tokens: 5120 }
        };
        const config = presetMap[e.target.value];
        if (config) {
            if (elements.temperatureSlider) {
                elements.temperatureSlider.value = config.temp;
                elements.tempValue.textContent = config.temp;
            }
            if (elements.maxTokensInput) {
                elements.maxTokensInput.value = config.tokens;
            }
        }
    });

    // Global Hotkeys
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && state.isGenerating) {
            stopGeneration();
        }
        if (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === 'k') {
            clearChat();
        }
    });

    // ── Core Logic ────────────────────────────────────────────

    async function handleSend() {
        if (state.isGenerating) return;

        const text = elements.messageInput.value.trim();
        if (!text) return;

        // UI Updates
        elements.messageInput.value = '';
        elements.messageInput.style.height = 'auto';
        elements.welcomeScreen.classList.add('hidden');
        elements.messagesContainer.classList.remove('hidden');
        elements.routingIndicator.classList.add('hidden');

        // Add user message
        appendMessage('user', text);
        state.chatHistory.push({ role: 'user', content: text });

        // Run skill suggestion engine on each user message
        if (window.skillsUI) {
            const docChips = document.getElementById('docChips');
            const docNames = docChips
                ? [...docChips.querySelectorAll('span[title]')]
                    .map(el => el.getAttribute('title') || '')
                : [];
            window.skillsUI.analyzeContext(text, docNames);
        }

        await startGeneration();
    }

    async function startGeneration() {
        state.isGenerating = true;
        elements.sendBtn.classList.add('hidden');
        elements.stopBtn.classList.remove('hidden');
        elements.statusIndicator.className = 'status-dot status-busy';
        elements.telTps.textContent = '—';

        state.abortController = new AbortController();

        // Create assistant message bubble
        const msgId = 'msg-' + Date.now();
        const contentNode = appendMessage('assistant', '', msgId);
        contentNode.innerHTML = '<span class="typing-cursor"></span>';

        let fullText = '';
        const startTime = performance.now();
        let tokenCount = 0;
        let wasUserAborted = false;  // true ONLY when user clicked Stop or pressed Escape

        const requestBody = {
            model: elements.modelSelect.value,
            messages: state.chatHistory,
            temperature: parseFloat(elements.temperatureSlider.value),
            max_tokens: parseInt(elements.maxTokensInput.value, 10),
            stream: true
        };

        const headers = {
            'Content-Type': 'application/json'
        };
        if (elements.presetSelect?.value) {
            headers['X-Runtime-Preset'] = elements.presetSelect.value;
        }
        // Read active skill from SkillsUI (authoritative state);
        // falls back to hidden #skillSelect for compatibility.
        const activeSkillId = window.skillsUI?.getActiveSkillId()
            || (() => {
                const sel = document.getElementById('skillSelect');
                return (sel && sel.value && sel.value !== 'none') ? sel.value : null;
            })();
        if (activeSkillId) {
            headers['X-Skill'] = activeSkillId;
        }
        if (window.memoryUI) {
            headers['X-Session-Id'] = window.memoryUI.getSessionId();
        }

        try {
            const response = await fetch('/v1/chat/completions', {
                method: 'POST',
                headers: headers,
                body: JSON.stringify(requestBody),
                signal: state.abortController.signal
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            state.currentRequestId = response.headers.get('X-Request-ID');

            // Handle SSE Stream
            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';
            let modelUsed = null;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop(); // Keep incomplete line in buffer

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.substring(6).trim();
                        if (dataStr === '[DONE]') continue;

                        try {
                            const data = JSON.parse(dataStr);
                            const delta = data.choices[0]?.delta?.content || '';

                            if (fullText === '') {
                                contentNode.innerHTML = ''; // Remove typing cursor on first token
                            }

                            if (data.model && !modelUsed) {
                                modelUsed = data.model;
                                updateRoutingIndicator(modelUsed);
                                elements.telModel.textContent = modelUsed;
                                elements.telProvider.textContent = data.provider || 'litert_cli';
                            }

                            fullText += delta;

                            // Basic token counting heuristic
                            tokenCount += delta.split(/\s+/).filter(x => x).length;

                            // Calculate TPS every ~10 tokens to avoid UI thrashing
                            if (tokenCount % 10 === 0) {
                                const elapsedSec = (performance.now() - startTime) / 1000;
                                if (elapsedSec > 0) {
                                    elements.telTps.textContent = (tokenCount / elapsedSec).toFixed(1);
                                }
                            }

                            contentNode.innerHTML = formatMarkdown(fullText) + '<span class="typing-cursor"></span>';
                            elements.messagesContainer.scrollTop = elements.messagesContainer.scrollHeight;

                        } catch (e) {
                            console.warn('Error parsing SSE chunk:', e, dataStr);
                        }
                    }
                }
            }
        } catch (e) {
            if (e.name === 'AbortError') {
                wasUserAborted = true;  // User explicitly stopped — mark for cancel call
                console.log('Generation stopped by user');
            } else {
                console.error('Generation error:', e);
                fullText += `\n\n**[Error: ${e.message}]**`;
            }
        } finally {
            state.isGenerating = false;
            state.abortController = null;

            // Finalize HTML
            contentNode.innerHTML = formatMarkdown(fullText);

            // Final TPS
            const elapsedSec = (performance.now() - startTime) / 1000;
            if (elapsedSec > 0) {
                elements.telTps.textContent = (tokenCount / elapsedSec).toFixed(1);
            }

            state.chatHistory.push({ role: 'assistant', content: fullText });

            elements.sendBtn.classList.remove('hidden');
            elements.stopBtn.classList.add('hidden');
            elements.statusIndicator.className = 'status-dot status-ready';

            // ONLY call /v1/cancel when the user explicitly aborted.
            // Do NOT cancel after successful completion — the stream already ended cleanly.
            if (wasUserAborted && state.currentRequestId) {
                const modelId = elements.modelSelect.value;
                fetch(`/v1/cancel?request_id=${state.currentRequestId}&model_id=${modelId}`, { method: 'POST' }).catch(() => { });
            }
            state.currentRequestId = null;

            // Check if we need to refresh the title of the active chat
            const activeSessionId = localStorage.getItem('as_active_session_id');
            const activeProjId = localStorage.getItem('as_active_project_id');
            const activeProj = projectsList.find(p => p.id === activeProjId);
            const currentChat = activeProj?.chats.find(c => c.session_id === activeSessionId);
            if (currentChat && (currentChat.title === 'Nuevo Chat' || currentChat.title.startsWith('Chat '))) {
                setTimeout(async () => {
                    try {
                        const res = await fetch(`/v1/projects/${activeProjId}/chats/${activeSessionId}`);
                        if (res.ok) {
                            const updatedChat = await res.json();
                            if (updatedChat.title !== currentChat.title) {
                                currentChat.title = updatedChat.title;
                                renderSidebar();
                            }
                        }
                    } catch (err) {
                        console.error('Error updating chat title:', err);
                    }
                }, 500);
            }
        }
    }


    async function stopGeneration() {
        if (!state.isGenerating) return;

        if (state.abortController) {
            state.abortController.abort(); // Cancels fetch
        }

        elements.statusIndicator.className = 'status-dot status-ready';
    }

    function clearChat() {
        const newChatBtn = document.getElementById('newChatBtn');
        if (newChatBtn) {
            newChatBtn.click();
            return;
        }

        if (state.isGenerating) stopGeneration();

        state.chatHistory = [];
        elements.messagesContainer.innerHTML = '';
        elements.messagesContainer.classList.add('hidden');
        elements.welcomeScreen.classList.remove('hidden');
        elements.routingIndicator.classList.add('hidden');
        elements.telModel.textContent = '—';
        elements.telTps.textContent = '—';

        // Reset skills UI state
        if (window.skillsUI) {
            window.skillsUI.deactivateSkill();
            window.skillsUI.dismissedSuggs?.clear();
            window.skillsUI._hideSuggestionBar();
        }

        // Generate new session ID to start completely clean
        const newSessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substring(2, 11);
        if (window.memoryUI) {
            window.memoryUI.setSessionId(newSessionId);
            window.memoryUI.refresh();
        }

        // Refresh document list to reflect new session ID scope (chips should disappear)
        if (window.refreshDocumentList) {
            window.refreshDocumentList();
        }

        elements.messageInput.focus();
    }

    // Expose clearChat globally so document_ui.js can invoke it
    window.clearChat = clearChat;

    function appendMessage(role, text, msgId = null) {
        const div = document.createElement('div');
        div.className = `message message-${role}`;

        const inner = document.createElement('div');
        inner.className = 'message-inner';

        const avatar = document.createElement('div');
        avatar.className = `message-avatar avatar-${role}`;
        avatar.textContent = role === 'user' ? 'U' : 'AI';

        const content = document.createElement('div');
        content.className = 'message-content';
        if (msgId) content.id = msgId;

        if (text) {
            content.innerHTML = formatMarkdown(text);
        }

        inner.appendChild(avatar);
        inner.appendChild(content);
        div.appendChild(inner);

        elements.messagesContainer.appendChild(div);
        elements.messagesContainer.scrollTop = elements.messagesContainer.scrollHeight;

        return content;
    }

    function updateRoutingIndicator(modelId) {
        elements.routingIndicator.classList.remove('hidden');
        elements.routingIndicator.textContent = modelId;

        if (modelId === 'chat') {
            elements.routingIndicator.className = 'routing-indicator routing-reasoning';
        } else if (modelId === 'code') {
            elements.routingIndicator.className = 'routing-indicator routing-coding';
        } else {
            elements.routingIndicator.className = 'routing-indicator';
            elements.routingIndicator.style.background = 'rgba(225, 229, 235, 0.1)';
        }
    }

    // ── Ultra-lightweight Markdown Formatter ──────────────────
    function formatMarkdown(text) {
        if (!text) return '';

        // Escape HTML
        let html = text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

        // Code blocks: ```language\n code \n```
        html = html.replace(/```([a-z0-9]*)\n([\s\S]*?)```/gi, (match, lang, code) => {
            return `<pre><code class="language-${lang}">${code}</code></pre>`;
        });

        // Inline code: `code`
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Bold: **text**
        html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

        // Line breaks
        html = html.replace(/\n/g, '<br>');

        return html;
    }

    // ── Telemetry Polling ─────────────────────────────────────
    async function pollStatus() {
        try {
            const res = await fetch('/v1/status');
            if (res.ok) {
                const data = await res.json();
                if (data.ram_available_mb) {
                    elements.telRam.textContent = `${Math.round(data.ram_available_mb / 1024 * 10) / 10} GB free`;
                }
                if (data.gpu && data.gpu.vram_free_mb !== undefined) {
                    elements.telVram.textContent = `${Math.round(data.gpu.vram_free_mb / 1024 * 10) / 10} GB free`;
                }
                if (data.provider && data.provider.status) {
                    if (state.isGenerating) {
                        elements.statusIndicator.className = 'status-dot status-busy';
                    } else if (data.provider.status === 'ready') {
                        elements.statusIndicator.className = 'status-dot status-ready';
                    } else {
                        elements.statusIndicator.className = 'status-dot status-error';
                    }
                }
            }
        } catch (e) {
            // Silently fail telemetry on connection error
            elements.statusIndicator.className = 'status-dot status-error';
        }
    }

    // Start polling
    setInterval(pollStatus, 5000);
    pollStatus();

    // Initialize Capabilities UI if loaded
    if (window.capabilitiesUI) {
        window.capabilitiesUI.init();
    }

    // NOTE: SkillsUI self-initializes via its own DOMContentLoaded in skills_ui.js.
    // No explicit .init() call needed here.
    if (!window.skillsUI) {
        console.warn('[app.js] window.skillsUI not found — ensure /static/skills_ui.js loads before app.js');
    }

    // ── Projects and Chats Management (Fase 5.1 Refinement) ───────
    const newProjBtn = document.getElementById('newProjectBtn');
    const newChatBtn = document.getElementById('newChatBtn');

    let projectsList = []; // stores all projects with their chats
    let expandedProjects = new Set(); // stores IDs of expanded projects (in memory)

    async function loadChatHistory(projectId, sessionId) {
        try {
            const res = await fetch(`/v1/projects/${projectId}/chats/${sessionId}/messages`);
            if (res.ok) {
                const messages = await res.json();
                state.chatHistory = [];
                elements.messagesContainer.innerHTML = '';
                if (messages && messages.length > 0) {
                    elements.welcomeScreen.classList.add('hidden');
                    elements.messagesContainer.classList.remove('hidden');
                    messages.forEach(msg => {
                        appendMessage(msg.role, msg.content);
                        state.chatHistory.push({ role: msg.role, content: msg.content });
                    });
                    elements.messagesContainer.scrollTop = elements.messagesContainer.scrollHeight;
                } else {
                    elements.welcomeScreen.classList.remove('hidden');
                    elements.messagesContainer.classList.add('hidden');
                }
            }
        } catch (err) {
            console.error('Error loading chat history:', err);
        }
    }

    async function initProjectsSystem() {
        const sidebarContent = document.getElementById('sidebarContent');
        if (!sidebarContent) return;

        // Load projects from API
        let projects = [];
        try {
            const res = await fetch('/v1/projects');
            if (res.ok) {
                projects = await res.json();
            }
        } catch (err) {
            console.error('Error loading projects:', err);
        }

        // Fetch chats for each project in parallel
        projectsList = await Promise.all(projects.map(async (p) => {
            let chats = [];
            try {
                const res = await fetch(`/v1/projects/${p.id}/chats`);
                if (res.ok) {
                    chats = await res.json();
                }
            } catch (err) {
                console.error(`Error loading chats for project ${p.id}:`, err);
            }
            return { ...p, chats };
        }));

        // Resolve active project
        let activeProjId = localStorage.getItem('as_active_project_id');
        const generalProj = projectsList.find(p => p.slug === 'general');
        if (!activeProjId || !projectsList.some(p => p.id === activeProjId)) {
            activeProjId = generalProj ? generalProj.id : (projectsList[0]?.id || '');
        }
        if (activeProjId) {
            localStorage.setItem('as_active_project_id', activeProjId);
            // By default, expand the active project
            expandedProjects.add(activeProjId);
        }

        // Resolve active chat (session_id)
        let activeSessionId = localStorage.getItem('as_active_session_id');
        const activeProj = projectsList.find(p => p.id === activeProjId);
        
        if (activeProj) {
            if (!activeSessionId || !activeProj.chats.some(c => c.session_id === activeSessionId)) {
                if (activeProj.chats.length > 0) {
                    activeSessionId = activeProj.chats[0].session_id;
                } else {
                    // Create a new chat on backend
                    const newSid = 'session_' + Date.now() + '_' + Math.random().toString(36).substring(2, 11);
                    const title = 'Chat ' + new Date().toLocaleTimeString();
                    try {
                        const res = await fetch(`/v1/projects/${activeProjId}/chats`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ session_id: newSid, title: title })
                        });
                        if (res.ok) {
                            const newChat = await res.json();
                            activeSessionId = newChat.session_id;
                            activeProj.chats.push(newChat);
                        } else {
                            activeSessionId = newSid;
                        }
                    } catch (err) {
                        console.error('Error creating chat:', err);
                        activeSessionId = newSid;
                    }
                }
            }
        }

        if (activeSessionId && activeProjId) {
            localStorage.setItem('as_active_session_id', activeSessionId);
            if (window.memoryUI) {
                window.memoryUI.setSessionId(activeSessionId);
            }
            if (window.refreshDocumentList) {
                window.refreshDocumentList();
            }
            await loadChatHistory(activeProjId, activeSessionId);
        }

        renderSidebar();
    }

    function renderSidebar() {
        const sidebarContent = document.getElementById('sidebarContent');
        if (!sidebarContent) return;

        const activeProjId = localStorage.getItem('as_active_project_id');
        const activeSessionId = localStorage.getItem('as_active_session_id');

        sidebarContent.innerHTML = '';

        if (projectsList.length === 0) {
            sidebarContent.innerHTML = '<div class="text-xs text-surface-400/60 text-center py-4">Sin proyectos.</div>';
            return;
        }

        projectsList.forEach(proj => {
            const isExpanded = expandedProjects.has(proj.id);
            const isActiveProj = proj.id === activeProjId;

            // Project container
            const container = document.createElement('div');
            container.className = `proj-container ${isExpanded ? 'expanded' : ''} ${isActiveProj ? 'active-proj' : ''}`;

            // Project header (accordion toggle)
            const header = document.createElement('div');
            header.className = 'proj-header';
            
            const titleWrapper = document.createElement('div');
            titleWrapper.className = 'proj-title-wrapper';
            
            const folderIcon = document.createElement('span');
            folderIcon.textContent = proj.slug === 'general' ? '🏠' : '📁';
            
            const titleSpan = document.createElement('span');
            titleSpan.className = 'truncate max-w-[120px]';
            titleSpan.textContent = proj.name;
            titleSpan.title = proj.name;

            titleWrapper.appendChild(folderIcon);
            titleWrapper.appendChild(titleSpan);

            const arrow = document.createElement('span');
            arrow.className = 'proj-arrow';
            arrow.textContent = '▶';

            const rightSide = document.createElement('div');
            rightSide.className = 'proj-right-side';

            // Project actions (only for non-default projects)
            if (proj.slug !== 'general') {
                const projActions = document.createElement('div');
                projActions.className = 'proj-actions';

                const projRenameBtn = document.createElement('button');
                projRenameBtn.className = 'chat-action-btn';
                projRenameBtn.innerHTML = '✏️';
                projRenameBtn.title = 'Renombrar proyecto';
                projRenameBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    showInlineModal({
                        title: 'Renombrar Proyecto',
                        label: 'Nuevo nombre:',
                        placeholder: proj.name,
                        defaultValue: proj.name,
                        confirmText: 'Guardar',
                        onConfirm: async (newName) => {
                            try {
                                const res = await fetch(`/v1/projects/${proj.id}`, {
                                    method: 'PATCH',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({ title: newName.trim() })
                                });
                                if (res.ok) {
                                    proj.name = newName.trim();
                                    renderSidebar();
                                }
                            } catch (err) {
                                console.error('Error renombrando proyecto:', err);
                            }
                        }
                    });
                });

                const projDeleteBtn = document.createElement('button');
                projDeleteBtn.className = 'chat-action-btn';
                projDeleteBtn.innerHTML = '🗑️';
                projDeleteBtn.title = 'Eliminar proyecto';
                projDeleteBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    showInlineModal({
                        title: 'Eliminar Proyecto',
                        message: `¿Eliminar "${proj.name}" y todos sus chats? Esta acción no se puede deshacer.`,
                        confirmText: 'Eliminar',
                        confirmClass: 'modal-btn-danger',
                        onConfirm: async () => {
                            try {
                                const res = await fetch(`/v1/projects/${proj.id}`, { method: 'DELETE' });
                                if (res.ok) {
                                    // If active project was deleted, fall back to general
                                    if (proj.id === localStorage.getItem('as_active_project_id')) {
                                        const general = projectsList.find(p => p.slug === 'general');
                                        if (general) {
                                            localStorage.setItem('as_active_project_id', general.id);
                                            localStorage.removeItem('as_active_session_id');
                                        }
                                        clearChatUI();
                                    }
                                    projectsList = projectsList.filter(p => p.id !== proj.id);
                                    expandedProjects.delete(proj.id);
                                    await initProjectsSystem();
                                }
                            } catch (err) {
                                console.error('Error eliminando proyecto:', err);
                            }
                        }
                    });
                });

                projActions.appendChild(projRenameBtn);
                projActions.appendChild(projDeleteBtn);
                rightSide.appendChild(projActions);
            }

            rightSide.appendChild(arrow);
            header.appendChild(titleWrapper);
            header.appendChild(rightSide);

            // Toggle expansion on click header
            header.addEventListener('click', (e) => {
                if (isExpanded) {
                    expandedProjects.delete(proj.id);
                } else {
                    expandedProjects.add(proj.id);
                }
                renderSidebar();
            });


            // Chat list container
            const chatList = document.createElement('div');
            chatList.className = 'chat-list';

            if (proj.chats && proj.chats.length > 0) {
                proj.chats.forEach(chat => {
                    const isActiveChat = chat.session_id === activeSessionId;
                    
                    const chatItem = document.createElement('div');
                    chatItem.className = `chat-item ${isActiveChat ? 'active-chat' : ''}`;
                    
                    const chatTitle = document.createElement('span');
                    chatTitle.className = 'chat-title-text truncate max-w-[120px]';
                    chatTitle.textContent = chat.title || 'Chat sin título';
                    chatTitle.title = chat.title;

                    chatItem.appendChild(chatTitle);

                    // Actions Container
                    const actionsDiv = document.createElement('div');
                    actionsDiv.className = 'chat-actions';

                    const renameBtn = document.createElement('button');
                    renameBtn.className = 'chat-action-btn rename-chat-btn';
                    renameBtn.innerHTML = '✏️';
                    renameBtn.title = 'Renombrar';
                    renameBtn.addEventListener('click', (e) => {
                        e.stopPropagation();
                        e.preventDefault();
                        showInlineModal({
                            title: 'Renombrar Chat',
                            label: 'Nuevo nombre:',
                            placeholder: chat.title,
                            defaultValue: chat.title,
                            confirmText: 'Guardar',
                            onConfirm: async (newTitle) => {
                                try {
                                    const res = await fetch(`/v1/projects/${proj.id}/chats/${chat.session_id}`, {
                                        method: 'PATCH',
                                        headers: { 'Content-Type': 'application/json' },
                                        body: JSON.stringify({ title: newTitle.trim() })
                                    });
                                    if (res.ok) {
                                        chat.title = newTitle.trim();
                                        renderSidebar();
                                    }
                                } catch (err) {
                                    console.error('Error renombrando chat:', err);
                                }
                            }
                        });
                    });

                    const deleteBtn = document.createElement('button');
                    deleteBtn.className = 'chat-action-btn delete-chat-btn';
                    deleteBtn.innerHTML = '🗑️';
                    deleteBtn.title = 'Eliminar';
                    deleteBtn.addEventListener('click', (e) => {
                        e.stopPropagation();
                        e.preventDefault();
                        showInlineModal({
                            title: 'Eliminar Chat',
                            message: `¿Eliminar "${chat.title}"? Esta acción no se puede deshacer.`,
                            confirmText: 'Eliminar',
                            confirmClass: 'modal-btn-danger',
                            onConfirm: async () => {
                                try {
                                    const res = await fetch(`/v1/projects/${proj.id}/chats/${chat.session_id}`, {
                                        method: 'DELETE'
                                    });
                                    if (res.ok) {
                                        if (chat.session_id === activeSessionId) {
                                            localStorage.removeItem('as_active_session_id');
                                            clearChatUI();
                                        }
                                        proj.chats = proj.chats.filter(c => c.session_id !== chat.session_id);
                                        const newActiveSid = localStorage.getItem('as_active_session_id');
                                        if (!newActiveSid) {
                                            const ap = projectsList.find(p => p.id === localStorage.getItem('as_active_project_id'));
                                            if (ap && ap.chats.length > 0) {
                                                const sid = ap.chats[0].session_id;
                                                localStorage.setItem('as_active_session_id', sid);
                                                await loadChatHistory(ap.id, sid);
                                            }
                                        }
                                        renderSidebar();
                                    }
                                } catch (err) {
                                    console.error('Error eliminando chat:', err);
                                }
                            }
                        });
                    });

                    actionsDiv.appendChild(renameBtn);
                    actionsDiv.appendChild(deleteBtn);
                    chatItem.appendChild(actionsDiv);

                    // Switch chat on click
                    chatItem.addEventListener('click', (e) => {
                        e.stopPropagation();
                        if (chat.session_id === activeSessionId && isActiveProj) return;

                        localStorage.setItem('as_active_project_id', proj.id);
                        localStorage.setItem('as_active_session_id', chat.session_id);
                        
                        clearChatUI();
                        
                        if (window.memoryUI) {
                            window.memoryUI.setSessionId(chat.session_id);
                            window.memoryUI.refresh();
                        }
                        if (window.refreshDocumentList) {
                            window.refreshDocumentList();
                        }

                        loadChatHistory(proj.id, chat.session_id);
                        renderSidebar();
                    });

                    chatList.appendChild(chatItem);
                });
            } else {
                const emptyItem = document.createElement('div');
                emptyItem.className = 'text-[10px] text-surface-500 italic px-2 py-1';
                emptyItem.textContent = 'Sin chats';
                chatList.appendChild(emptyItem);
            }

            container.appendChild(header);
            container.appendChild(chatList);
            sidebarContent.appendChild(container);
        });
    }

    newProjBtn?.addEventListener('click', () => {
        showInlineModal({
            title: 'Nuevo Proyecto',
            label: 'Nombre:',
            placeholder: 'Ej: Marketing',
            confirmText: 'Crear',
            onConfirm: async (name) => {
                const slug = name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');
                if (!slug) return;
                try {
                    const res = await fetch('/v1/projects', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ name, slug })
                    });
                    if (res.ok) {
                        const newProj = await res.json();
                        localStorage.setItem('as_active_project_id', newProj.id);
                        localStorage.removeItem('as_active_session_id');
                        clearChatUI();
                        await initProjectsSystem();
                    } else {
                        const err = await res.json().catch(() => ({ detail: 'Error desconocido' }));
                        console.error('Error al crear proyecto:', err.detail);
                    }
                } catch (err) {
                    console.error('Error de conexión:', err.message);
                }
            }
        });
    });

    newChatBtn?.addEventListener('click', async () => {
        const projId = localStorage.getItem('as_active_project_id');
        if (!projId) {
            alert('Selecciona o crea un proyecto primero.');
            return;
        }

        const newSid = 'session_' + Date.now() + '_' + Math.random().toString(36).substring(2, 11);
        const title = 'Chat ' + new Date().toLocaleTimeString();

        try {
            const res = await fetch(`/v1/projects/${projId}/chats`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: newSid, title: title })
            });
            if (res.ok) {
                const newChat = await res.json();
                localStorage.setItem('as_active_session_id', newChat.session_id);
                clearChatUI();
                await initProjectsSystem();
            } else {
                alert('Error al crear chat');
            }
        } catch (err) {
            alert('Error: ' + err.message);
        }
    });

    function clearChatUI() {
        if (state.isGenerating) stopGeneration();
        state.chatHistory = [];
        elements.messagesContainer.innerHTML = '';
        elements.messagesContainer.classList.add('hidden');
        elements.welcomeScreen.classList.remove('hidden');
        elements.routingIndicator.classList.add('hidden');
        elements.telModel.textContent = '—';
        elements.telTps.textContent = '—';
        if (window.skillsUI) {
            window.skillsUI.deactivateSkill();
            window.skillsUI.dismissedSuggs?.clear();
            window.skillsUI._hideSuggestionBar();
        }
        elements.messageInput.focus();
    }

    // ── Inline Modal (replaces native prompt/confirm) ────────────
    function showInlineModal({ title = '', label = '', placeholder = '', defaultValue = '', message = '', confirmText = 'Confirmar', confirmClass = 'modal-btn-primary', onConfirm }) {
        // Remove any existing modal
        document.getElementById('as-inline-modal')?.remove();

        const isConfirm = !label; // no label = confirm-only dialog

        const overlay = document.createElement('div');
        overlay.id = 'as-inline-modal';
        overlay.className = 'modal-overlay';

        const box = document.createElement('div');
        box.className = 'modal-box';

        const titleEl = document.createElement('div');
        titleEl.className = 'modal-title';
        titleEl.textContent = title;
        box.appendChild(titleEl);

        let input = null;
        if (label) {
            const labelEl = document.createElement('label');
            labelEl.className = 'modal-label';
            labelEl.textContent = label;
            box.appendChild(labelEl);

            input = document.createElement('input');
            input.type = 'text';
            input.className = 'modal-input';
            input.placeholder = placeholder;
            input.value = defaultValue;
            box.appendChild(input);
        } else if (message) {
            const msgEl = document.createElement('p');
            msgEl.className = 'modal-message';
            msgEl.textContent = message;
            box.appendChild(msgEl);
        }

        const actions = document.createElement('div');
        actions.className = 'modal-actions';

        const cancelBtn = document.createElement('button');
        cancelBtn.className = 'modal-btn modal-btn-cancel';
        cancelBtn.textContent = 'Cancelar';
        cancelBtn.addEventListener('click', () => overlay.remove());

        const confirmBtn = document.createElement('button');
        confirmBtn.className = `modal-btn ${confirmClass}`;
        confirmBtn.textContent = confirmText;
        confirmBtn.addEventListener('click', async () => {
            const value = input ? input.value.trim() : '';
            if (input && !value) {
                input.focus();
                input.classList.add('modal-input-error');
                return;
            }
            overlay.remove();
            await onConfirm(value);
        });

        actions.appendChild(cancelBtn);
        actions.appendChild(confirmBtn);
        box.appendChild(actions);
        overlay.appendChild(box);
        document.body.appendChild(overlay);

        // Close on overlay click
        overlay.addEventListener('click', (e) => { if (e.target === overlay) overlay.remove(); });

        // Focus input or confirm button
        setTimeout(() => (input || confirmBtn).focus(), 50);

        // Enter key submits
        if (input) {
            input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') confirmBtn.click();
                if (e.key === 'Escape') overlay.remove();
            });
        }
    }

    // Run Projects Initialization
    initProjectsSystem();

    // Initial focus
    elements.messageInput.focus();
});
