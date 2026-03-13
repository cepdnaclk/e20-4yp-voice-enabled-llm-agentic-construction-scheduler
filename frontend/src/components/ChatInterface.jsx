import { useRef, useEffect } from 'react';
import MessageBubble from './MessageBubble';
import useVoiceInput from './useVoiceInput';
import './ChatInterface.css';

function ChatInterface({ messages, onSend, isLoading, isInitializing, pendingInterrupt, currentStage, hasStarted }) {
    const messagesEndRef = useRef(null);
    const inputRef = useRef(null);
    const { isListening, isSupported, toggleListening } = useVoiceInput(inputRef);

    // Auto-scroll to bottom
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    // Focus input
    useEffect(() => {
        if (!isLoading && !isInitializing) {
            inputRef.current?.focus();
        }
    }, [isLoading, isInitializing]);

    const handleSubmit = (e) => {
        e.preventDefault();
        const text = inputRef.current?.value?.trim();
        if (text) {
            onSend(text);
            inputRef.current.value = '';
            if (inputRef.current) {
                inputRef.current.style.height = 'auto';
            }
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
        }
    };

    const inputPlaceholder = isInitializing
        ? 'Connecting...'
        : pendingInterrupt
            ? 'Type or speak your response...'
            : hasStarted
                ? isSupported ? 'Type or 🎤 speak a message...' : 'Type a message...'
                : isSupported ? 'Describe your project (or click 🎤 to speak)...' : 'Describe your construction project...';

    // ── HERO / WELCOME SCREEN (before first message) ──────────────────────────
    if (!hasStarted) {
        return (
            <div className="chat-interface hero-mode">
                <div className="hero-body">
                    {isInitializing ? (
                        <div className="initializing">
                            <div className="init-spinner"></div>
                            <p>Connecting to AI Assistant...</p>
                        </div>
                    ) : (
                        <>
                            <div className="hero-icon">🏗️</div>
                            <h2 className="hero-title">How can I help you today?</h2>
                            <p className="hero-subtitle">
                                Describe your construction project and I'll generate a full schedule for you.
                            </p>
                        </>
                    )}

                    {/* Centered input */}
                    <div className="hero-input-wrap">
                        <form onSubmit={handleSubmit} className="input-form">
                            <div className="input-wrapper">
                                {isSupported && (
                                    <button
                                        type="button"
                                        onClick={toggleListening}
                                        disabled={isLoading || isInitializing}
                                        className={`mic-btn${isListening ? ' listening' : ''}`}
                                        title={isListening ? 'Stop recording' : 'Start voice input'}
                                        aria-label={isListening ? 'Stop recording' : 'Start voice input'}
                                    >
                                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                                            <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                                            <line x1="12" y1="19" x2="12" y2="23"/>
                                            <line x1="8" y1="23" x2="16" y2="23"/>
                                        </svg>
                                    </button>
                                )}
                                <textarea
                                    ref={inputRef}
                                    rows={1}
                                    placeholder={inputPlaceholder}
                                    disabled={isLoading || isInitializing}
                                    onKeyDown={handleKeyDown}
                                    onInput={(e) => {
                                        e.target.style.height = 'auto';
                                        e.target.style.height = Math.min(e.target.scrollHeight, 150) + 'px';
                                    }}
                                    className="chat-input hero-input"
                                />
                                <button
                                    type="submit"
                                    disabled={isLoading || isInitializing}
                                    className="send-btn"
                                >
                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                        <line x1="22" y1="2" x2="11" y2="13"></line>
                                        <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                                    </svg>
                                </button>
                            </div>
                            <div className="input-hint">
                                <span className="stage-badge">{currentStage}</span>
                            </div>
                        </form>
                    </div>
                </div>
            </div>
        );
    }

    // ── NORMAL CHAT LAYOUT (after first message) ───────────────────────────────
    return (
        <div className="chat-interface">
            {/* Messages Area */}
            <div className="messages-container">
                {messages.map((msg, idx) => (
                    <MessageBubble
                        key={idx}
                        role={msg.role}
                        content={msg.content}
                        streaming={msg.streaming}
                    />
                ))}

                {isLoading && !messages.some(m => m.streaming) && (
                    <div className="typing-indicator">
                        <div className="typing-dots">
                            <span></span>
                            <span></span>
                            <span></span>
                        </div>
                    </div>
                )}

                {pendingInterrupt && (
                    <div className="interrupt-banner">
                        <div className="interrupt-icon">⚡</div>
                        <p>Awaiting your response to continue...</p>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="input-area">
                <form onSubmit={handleSubmit} className="input-form">
                    <div className="input-wrapper">
                        {isSupported && (
                            <button
                                type="button"
                                onClick={toggleListening}
                                disabled={isLoading || isInitializing}
                                className={`mic-btn${isListening ? ' listening' : ''}`}
                                title={isListening ? 'Stop recording' : 'Start voice input'}
                                aria-label={isListening ? 'Stop recording' : 'Start voice input'}
                            >
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                                    <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                                    <line x1="12" y1="19" x2="12" y2="23"/>
                                    <line x1="8" y1="23" x2="16" y2="23"/>
                                </svg>
                            </button>
                        )}
                        <textarea
                            ref={inputRef}
                            rows={1}
                            placeholder={inputPlaceholder}
                            disabled={isLoading || isInitializing}
                            onKeyDown={handleKeyDown}
                            onInput={(e) => {
                                e.target.style.height = 'auto';
                                e.target.style.height = Math.min(e.target.scrollHeight, 150) + 'px';
                            }}
                            className="chat-input"
                        />
                        <button
                            type="submit"
                            disabled={isLoading || isInitializing}
                            className="send-btn"
                        >
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <line x1="22" y1="2" x2="11" y2="13"></line>
                                <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                            </svg>
                        </button>
                    </div>
                    <div className="input-hint">
                        <span className="stage-badge">{currentStage}</span>
                        {pendingInterrupt && <span className="interrupt-hint">↑ Respond to the prompt above</span>}
                    </div>
                </form>
            </div>
        </div>
    );
}

export default ChatInterface;
