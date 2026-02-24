import { useState, useMemo, useRef } from 'react';
import './GanttChart.css';

function GanttChart({ scheduleData }) {
    const [tooltip, setTooltip] = useState(null);
    const chartRef = useRef(null);

    // Process data
    const { phases, phaseIndices, totalDays, minDate, dayWidth } = useMemo(() => {
        if (!scheduleData || scheduleData.length === 0) {
            return { phases: [], phaseIndices: {}, totalDays: 0, minDate: null, dayWidth: 0 };
        }

        // Get unique phases in order
        const seenPhases = [];
        const phaseIdx = {};
        for (const task of scheduleData) {
            if (!phaseIdx.hasOwnProperty(task.phase)) {
                phaseIdx[task.phase] = seenPhases.length;
                seenPhases.push(task.phase);
            }
        }

        const maxDay = Math.max(...scheduleData.map(t => t.end_day));
        const minD = new Date(scheduleData[0].start_date);

        // Calculate day width: aim for ~18px per day, min 12px
        const dw = Math.max(12, Math.min(24, 600 / (maxDay || 1)));

        return {
            phases: seenPhases,
            phaseIndices: phaseIdx,
            totalDays: maxDay,
            minDate: minD,
            dayWidth: dw,
        };
    }, [scheduleData]);

    if (!scheduleData || scheduleData.length === 0) {
        return (
            <div className="gantt-container">
                <div className="gantt-header">
                    <h2>📅 Gantt Chart</h2>
                </div>
                <div className="gantt-empty">
                    <div className="gantt-empty-icon">📊</div>
                    <p>Schedule will appear here after optimisation...</p>
                </div>
            </div>
        );
    }

    // Generate timeline labels
    const timelineCells = [];
    for (let d = 0; d <= totalDays; d += 1) {
        const date = new Date(minDate);
        date.setDate(date.getDate() + d);
        const isMonday = date.getDay() === 1;

        // Show label every 7 days or on Mondays
        const showLabel = d === 0 || d % 7 === 0;

        timelineCells.push(
            <div
                key={d}
                className={`gantt-timeline-cell ${isMonday ? 'is-monday' : ''}`}
                style={{ width: dayWidth, minWidth: dayWidth }}
            >
                {showLabel && (
                    <span>{date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}</span>
                )}
            </div>
        );
    }

    // Group tasks by phase
    const tasksByPhase = {};
    for (const task of scheduleData) {
        if (!tasksByPhase[task.phase]) tasksByPhase[task.phase] = [];
        tasksByPhase[task.phase].push(task);
    }

    const handleBarHover = (e, task) => {
        const rect = e.currentTarget.getBoundingClientRect();
        setTooltip({
            task,
            x: rect.left + rect.width / 2,
            y: rect.top - 10,
        });
    };

    const handleBarLeave = () => {
        setTooltip(null);
    };

    // Stats
    const totalTasks = scheduleData.length;
    const startDate = scheduleData.length > 0
        ? new Date(Math.min(...scheduleData.map(t => new Date(t.start_date)))).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
        : '';
    const endDate = scheduleData.length > 0
        ? new Date(Math.max(...scheduleData.map(t => new Date(t.end_date)))).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
        : '';

    return (
        <div className="gantt-container">
            <div className="gantt-header">
                <h2>📅 Gantt Chart</h2>
                <div className="gantt-stats">
                    <div className="gantt-stat">
                        <span className="gantt-stat-value">{totalTasks}</span>
                        <span className="gantt-stat-label">Tasks</span>
                    </div>
                    <div className="gantt-stat">
                        <span className="gantt-stat-value">{totalDays}d</span>
                        <span className="gantt-stat-label">Duration</span>
                    </div>
                </div>
            </div>

            <div className="gantt-chart-area" ref={chartRef}>
                <div className="gantt-chart-wrapper">
                    {/* Timeline header */}
                    <div className="gantt-timeline">
                        {timelineCells}
                    </div>

                    {/* Phase groups */}
                    {phases.map((phase, phaseIdx) => (
                        <div key={phase} className="gantt-phase-group">
                            <div className="gantt-phase-label">
                                <span className="gantt-phase-dot" />
                                {phase}
                                <span style={{ opacity: 0.6, fontWeight: 400, marginLeft: 'auto' }}>
                                    {(tasksByPhase[phase] || []).length} tasks
                                </span>
                            </div>

                            {(tasksByPhase[phase] || []).map((task, taskIdx) => (
                                <div key={task.id} className="gantt-row">
                                    <div className="gantt-task-label" title={task.name}>
                                        {task.name}
                                    </div>
                                    <div className="gantt-bar-area">
                                        <div
                                            className={`gantt-bar phase-${phaseIdx % 8}`}
                                            style={{
                                                left: task.start_day * dayWidth,
                                                width: Math.max(task.duration_days * dayWidth, 4),
                                                animationDelay: `${(phaseIdx * 0.1) + (taskIdx * 0.04)}s`,
                                            }}
                                            onMouseEnter={(e) => handleBarHover(e, task)}
                                            onMouseLeave={handleBarLeave}
                                        >
                                            <div className="gantt-bar-fill" />
                                            {task.duration_days * dayWidth > 50 && (
                                                <span className="gantt-bar-text">
                                                    {task.duration_days}d
                                                </span>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    ))}
                </div>
            </div>

            {/* Tooltip */}
            {tooltip && (
                <div
                    className="gantt-tooltip"
                    style={{
                        left: tooltip.x,
                        top: tooltip.y,
                        transform: 'translate(-50%, -100%)',
                    }}
                >
                    <div className="gantt-tooltip-title">{tooltip.task.name}</div>
                    <div className="gantt-tooltip-row">
                        <span>Phase</span>
                        <span>{tooltip.task.phase}</span>
                    </div>
                    <div className="gantt-tooltip-row">
                        <span>Start</span>
                        <span>{tooltip.task.start_date}</span>
                    </div>
                    <div className="gantt-tooltip-row">
                        <span>End</span>
                        <span>{tooltip.task.end_date}</span>
                    </div>
                    <div className="gantt-tooltip-row">
                        <span>Duration</span>
                        <span>{tooltip.task.duration_days} days</span>
                    </div>
                    {tooltip.task.dependencies.length > 0 && (
                        <div className="gantt-tooltip-row">
                            <span>Depends on</span>
                            <span>{tooltip.task.dependencies.join(', ')}</span>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

export default GanttChart;
