const API_BASE = 'http://localhost:8000';

/**
 * Start a new chat session
 */
export async function startChat() {
    const res = await fetch(`${API_BASE}/api/chat/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
    });
    if (!res.ok) throw new Error('Failed to start chat');
    return res.json();
}

/**
 * Send a message and consume SSE stream
 * @param {string} threadId
 * @param {string} message
 * @param {object} callbacks - { onChunk, onInterrupt, onStageChange, onTasks, onPhases, onDone, onError }
 */
export async function sendMessage(threadId, message, callbacks) {
    const res = await fetch(`${API_BASE}/api/chat/message`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ thread_id: threadId, message }),
    });

    return consumeSSE(res, callbacks);
}

/**
 * Resume from an interrupt and consume SSE stream
 * @param {string} threadId
 * @param {string} response
 * @param {object} callbacks
 */
export async function resumeChat(threadId, response, callbacks) {
    const res = await fetch(`${API_BASE}/api/chat/resume`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ thread_id: threadId, response }),
    });

    return consumeSSE(res, callbacks);
}

/**
 * Get current workflow state
 */
export async function getState(threadId) {
    const res = await fetch(`${API_BASE}/api/chat/state?thread_id=${threadId}`);
    if (!res.ok) throw new Error('Failed to get state');
    return res.json();
}

/**
 * Consume SSE stream from a fetch response
 */
async function consumeSSE(response, callbacks) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        let eventType = '';
        for (const line of lines) {
            if (line.startsWith('event: ')) {
                eventType = line.slice(7).trim();
            } else if (line.startsWith('data: ')) {
                const dataStr = line.slice(6);
                try {
                    const data = JSON.parse(dataStr);
                    switch (eventType) {
                        case 'message':
                            callbacks.onChunk?.(data.chunk);
                            break;
                        case 'interrupt':
                            callbacks.onInterrupt?.(data.value);
                            break;
                        case 'stage_change':
                            callbacks.onStageChange?.(data.stage);
                            break;
                        case 'tasks':
                            callbacks.onTasks?.(data.tasks);
                            break;
                        case 'phases':
                            callbacks.onPhases?.(data.phases);
                            break;
                        case 'done':
                            callbacks.onDone?.();
                            break;
                        case 'error':
                            callbacks.onError?.(data.message);
                            break;
                    }
                } catch (e) {
                    // ignore parse errors for incomplete chunks
                }
            }
        }
    }
}
