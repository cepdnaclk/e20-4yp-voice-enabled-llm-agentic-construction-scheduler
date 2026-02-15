import './TaskCard.css';

function TaskCard({ task, index }) {
    return (
        <div className="task-card" style={{ animationDelay: `${index * 0.08}s` }}>
            <div className="task-card-header">
                <span className="task-name">{task.name}</span>
                <span className="task-duration">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="12" cy="12" r="10"></circle>
                        <polyline points="12 6 12 12 16 14"></polyline>
                    </svg>
                    {task.duration_days}d
                </span>
            </div>

            {/* Dependencies */}
            {task.dependencies && task.dependencies.length > 0 && (
                <div className="task-section">
                    <span className="section-label">Dependencies</span>
                    <div className="dep-tags">
                        {task.dependencies.map((dep, i) => (
                            <span key={i} className="dep-tag">
                                {Array.isArray(dep) ? dep[0] : dep}
                                {Array.isArray(dep) && dep[1] && (
                                    <span className="dep-type">{dep[1]}</span>
                                )}
                            </span>
                        ))}
                    </div>
                </div>
            )}

            {/* Resources */}
            {task.resources && task.resources.length > 0 && (
                <div className="task-section">
                    <span className="section-label">Resources</span>
                    <div className="resource-tags">
                        {task.resources.map((res, i) => (
                            <span key={i} className="resource-tag">
                                {Array.isArray(res) ? `${res[0]} (${res[1]})` : res}
                            </span>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

export default TaskCard;
