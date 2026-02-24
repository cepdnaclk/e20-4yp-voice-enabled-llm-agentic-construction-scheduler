"""
OR-Tools CP-SAT based construction schedule optimizer.
Takes generated tasks (by phase) and produces an optimal schedule
minimizing overall project duration (makespan).
"""

from datetime import datetime, timedelta
from ortools.sat.python import cp_model


def _safe_int(value, default: int = 0) -> int:
    """Safely convert a value to int, returning default for empty/invalid values."""
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def solve_schedule(
    generated_tasks: dict, project_start_date: str | None = None
) -> list:
    """
    Solve the scheduling problem using CP-SAT constraint programming.

    Args:
        generated_tasks: dict of {phase_name: [task_dicts]} where each task has
            name, duration_days, dependencies (list of [prev_task, relationship, lag]),
            resources (list of [resource_name, amount])
        project_start_date: ISO date string (YYYY-MM-DD). Defaults to today.

    Returns:
        List of scheduled task dicts with start_date, end_date, etc.
    """
    if not project_start_date:
        project_start_date = datetime.now().strftime("%Y-%m-%d")

    start_date = datetime.strptime(project_start_date, "%Y-%m-%d")

    # ── 1. Flatten all tasks with phase context ──
    all_tasks = []
    phase_order = []
    for phase_name, tasks in generated_tasks.items():
        phase_order.append(phase_name)
        for task in tasks:
            all_tasks.append(
                {
                    **task,
                    "phase": phase_name,
                }
            )

    if not all_tasks:
        return []

    # Build a name → index lookup
    task_index = {}
    for i, t in enumerate(all_tasks):
        task_index[t["name"]] = i

    # ── 2. Build CP-SAT Model ──
    model = cp_model.CpModel()

    # Compute an upper bound for the horizon (sum of all durations + lags)
    total_duration = sum(t["duration_days"] for t in all_tasks)
    total_lag = 0
    for t in all_tasks:
        for dep in t.get("dependencies", []):
            if len(dep) > 2:
                total_lag += abs(_safe_int(dep[2]))
    horizon = total_duration + total_lag + 30  # generous buffer

    # Create variables for each task
    starts = []
    ends = []
    intervals = []

    for i, task in enumerate(all_tasks):
        duration = task["duration_days"]
        start_var = model.new_int_var(0, horizon, f"start_{i}")
        end_var = model.new_int_var(0, horizon, f"end_{i}")
        interval_var = model.new_interval_var(
            start_var, duration, end_var, f"interval_{i}"
        )

        starts.append(start_var)
        ends.append(end_var)
        intervals.append(interval_var)

    # ── 3. Add dependency constraints ──
    for i, task in enumerate(all_tasks):
        for dep in task.get("dependencies", []):
            prev_task_name = dep[0]
            relationship = dep[1].upper() if len(dep) > 1 else "FS"
            lag = _safe_int(dep[2]) if len(dep) > 2 else 0

            if prev_task_name not in task_index:
                print(f"  ⚠️ Dependency '{prev_task_name}' not found, skipping")
                continue

            j = task_index[prev_task_name]  # predecessor index

            if relationship == "FS":
                # Finish-to-Start: task i starts after task j finishes + lag
                model.add(starts[i] >= ends[j] + lag)
            elif relationship == "SS":
                # Start-to-Start: task i starts after task j starts + lag
                model.add(starts[i] >= starts[j] + lag)
            elif relationship == "FF":
                # Finish-to-Finish: task i finishes after task j finishes + lag
                model.add(ends[i] >= ends[j] + lag)
            elif relationship == "SF":
                # Start-to-Finish: task i finishes after task j starts + lag
                model.add(ends[i] >= starts[j] + lag)
            else:
                # Default to FS
                model.add(starts[i] >= ends[j] + lag)

    # ── 4. Add phase ordering constraints ──
    # First task of phase N+1 must start after last task of phase N
    phase_task_indices: dict[str, list[int]] = {}
    for i, task in enumerate(all_tasks):
        phase = task["phase"]
        if phase not in phase_task_indices:
            phase_task_indices[phase] = []
        phase_task_indices[phase].append(i)

    for p_idx in range(len(phase_order) - 1):
        current_phase = phase_order[p_idx]
        next_phase = phase_order[p_idx + 1]

        if current_phase in phase_task_indices and next_phase in phase_task_indices:
            current_indices = phase_task_indices[current_phase]
            next_indices = phase_task_indices[next_phase]

            if current_indices and next_indices:
                # Max end of all tasks in current phase
                max_end = model.new_int_var(0, horizon, f"phase_end_{p_idx}")
                model.add_max_equality(max_end, [ends[k] for k in current_indices])

                # All tasks in next phase start after max end of current phase
                for k in next_indices:
                    model.add(starts[k] >= max_end)

    # ── 5. Objective: minimize makespan ──
    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, ends)
    model.minimize(makespan)

    # ── 6. Solve ──
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    status = solver.solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"  ❌ Solver status: {solver.status_name(status)}")
        # Return a naive sequential fallback
        return _fallback_schedule(all_tasks, start_date)

    print(
        f"  ✅ Solver status: {solver.status_name(status)}, Makespan: {solver.value(makespan)} days"
    )

    # ── 7. Build result ──
    schedule = []
    for i, task in enumerate(all_tasks):
        task_start_day = solver.value(starts[i])
        task_end_day = solver.value(ends[i])
        task_start_date = start_date + timedelta(days=task_start_day)
        task_end_date = start_date + timedelta(days=task_end_day)

        dep_names = [d[0] for d in task.get("dependencies", []) if d[0] in task_index]

        schedule.append(
            {
                "id": f"task_{i}",
                "name": task["name"],
                "phase": task["phase"],
                "duration_days": task["duration_days"],
                "start_day": task_start_day,
                "end_day": task_end_day,
                "start_date": task_start_date.strftime("%Y-%m-%d"),
                "end_date": task_end_date.strftime("%Y-%m-%d"),
                "dependencies": dep_names,
            }
        )

    return schedule


def _fallback_schedule(all_tasks: list, start_date: datetime) -> list:
    """Simple sequential schedule as fallback when solver fails."""
    schedule = []
    current_day = 0

    for i, task in enumerate(all_tasks):
        task_start = current_day
        task_end = current_day + task["duration_days"]

        schedule.append(
            {
                "id": f"task_{i}",
                "name": task["name"],
                "phase": task["phase"],
                "duration_days": task["duration_days"],
                "start_day": task_start,
                "end_day": task_end,
                "start_date": (start_date + timedelta(days=task_start)).strftime(
                    "%Y-%m-%d"
                ),
                "end_date": (start_date + timedelta(days=task_end)).strftime(
                    "%Y-%m-%d"
                ),
                "dependencies": [d[0] for d in task.get("dependencies", [])],
            }
        )

        current_day = task_end

    return schedule
