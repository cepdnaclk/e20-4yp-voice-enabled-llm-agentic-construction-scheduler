import './StageTracker.css';

const stages = [
    { id: 'intent', label: 'Intent', icon: '🎯', description: 'Gathering project requirements' },
    { id: 'phases', label: 'Phases', icon: '📋', description: 'Identifying construction phases' },
    { id: 'details', label: 'Details', icon: '🔧', description: 'Generating detailed tasks' },
    { id: 'scheduling', label: 'Schedule', icon: '📅', description: 'Finalizing schedule' },
];

function StageTracker({ currentStage }) {
    const currentIdx = stages.findIndex(s => s.id === currentStage);

    return (
        <div className="stage-tracker">
            {stages.map((stage, idx) => {
                const isActive = stage.id === currentStage;
                const isCompleted = idx < currentIdx;
                const isPending = idx > currentIdx;

                return (
                    <div key={stage.id} className="stage-item-wrapper">
                        {idx > 0 && (
                            <div className={`stage-connector ${isCompleted ? 'completed' : ''}`} />
                        )}
                        <div
                            className={`stage-item ${isActive ? 'active' : ''} ${isCompleted ? 'completed' : ''} ${isPending ? 'pending' : ''}`}
                            title={stage.description}
                        >
                            <div className="stage-icon-circle">
                                {isCompleted ? (
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                                        <polyline points="20 6 9 17 4 12"></polyline>
                                    </svg>
                                ) : (
                                    <span className="stage-emoji">{stage.icon}</span>
                                )}
                            </div>
                            <span className="stage-label">{stage.label}</span>
                        </div>
                    </div>
                );
            })}
        </div>
    );
}

export default StageTracker;
