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
    elements.messageInput.addEventListener('input', function() {
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
                fetch(`/v1/cancel?request_id=${state.currentRequestId}&model_id=${modelId}`, { method: 'POST' }).catch(() => {});
            }
            state.currentRequestId = null;
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
        if (state.isGenerating) stopGeneration();
        
        state.chatHistory = [];
        elements.messagesContainer.innerHTML = '';
        elements.messagesContainer.classList.add('hidden');
        elements.welcomeScreen.classList.remove('hidden');
        elements.routingIndicator.classList.add('hidden');
        elements.telModel.textContent = '—';
        elements.telTps.textContent = '—';
        elements.messageInput.focus();
    }

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
    
    // Initial focus
    elements.messageInput.focus();
});
