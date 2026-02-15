import TaskCard from './TaskCard';
import './TaskBoard.css';

function TaskBoard({ phases, generatedTasks }) {
    const hasAnyTasks = Object.keys(generatedTasks).length > 0;

    return (
        <div className="task-board">
            <div className="task-board-header">
                <h2>📋 Task Board</h2>
                <span className="task-count">
                    {Object.values(generatedTasks).flat().length} tasks
                </span>
            </div>

            <div className="task-board-content">
                {phases.length === 0 && !hasAnyTasks && (
                    <div className="empty-state">
                        <div className="empty-icon">🔧</div>
                        <p>Tasks will appear here as they are generated...</p>
                    </div>
                )}

                {phases.map((phase, phaseIdx) => {
                    const phaseTasks = generatedTasks[phase] || [];
                    const isProcessing = !generatedTasks[phase] && phaseIdx <= Object.keys(generatedTasks).length;

                    return (
                        <div key={phase} className="phase-column" style={{ animationDelay: `${phaseIdx * 0.1}s` }}>
                            <div className="phase-header">
                                <div className="phase-number">{phaseIdx + 1}</div>
                                <h3 className="phase-title">{phase}</h3>
                                <span className={`phase-status ${phaseTasks.length > 0 ? 'done' : isProcessing ? 'processing' : 'pending'}`}>
                                    {phaseTasks.length > 0
                                        ? `${phaseTasks.length} tasks`
                                        : isProcessing
                                            ? 'Generating...'
                                            : 'Pending'}
                                </span>
                            </div>

                            <div className="phase-tasks">
                                {isProcessing && phaseTasks.length === 0 && (
                                    <div className="task-placeholder">
                                        <div className="shimmer-line" style={{ width: '80%' }}></div>
                                        <div className="shimmer-line" style={{ width: '60%' }}></div>
                                        <div className="shimmer-line" style={{ width: '70%' }}></div>
                                    </div>
                                )}

                                {phaseTasks.map((task, taskIdx) => (
                                    <TaskCard
                                        key={`${phase}-${taskIdx}`}
                                        task={task}
                                        index={taskIdx}
                                    />
                                ))}
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}

export default TaskBoard;
