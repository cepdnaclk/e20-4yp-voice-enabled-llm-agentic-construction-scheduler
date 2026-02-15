import './MessageBubble.css';

function MessageBubble({ role, content, streaming }) {
    const isUser = role === 'user';

    // Parse content for formatting (bold, lists, etc.)
    const formatContent = (text) => {
        if (!text) return '';

        // Split into lines for better rendering
        const lines = text.split('\n');
        return lines.map((line, i) => {
            // Bold text: **text**
            let processed = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

            // Bullet points
            if (line.trim().startsWith('•') || line.trim().startsWith('-') || line.trim().startsWith('*')) {
                processed = `<span class="list-item">${processed}</span>`;
            }

            // Checkmark items
            if (line.includes('✅') || line.includes('✓')) {
                processed = `<span class="check-item">${processed}</span>`;
            }

            return processed;
        }).join('<br/>');
    };

    return (
        <div className={`message-row ${isUser ? 'user' : 'ai'}`}>
            {!isUser && (
                <div className="avatar ai-avatar">
                    <span>🤖</span>
                </div>
            )}
            <div className={`message-bubble ${isUser ? 'user-bubble' : 'ai-bubble'} ${streaming ? 'streaming' : ''}`}>
                <div
                    className="message-text"
                    dangerouslySetInnerHTML={{ __html: formatContent(content) }}
                />
                {streaming && (
                    <span className="cursor-blink">▊</span>
                )}
            </div>
            {isUser && (
                <div className="avatar user-avatar">
                    <span>👤</span>
                </div>
            )}
        </div>
    );
}

export default MessageBubble;
