"""
OR-Tools based construction scheduler
"""

from matplotlib import pyplot as plt
from ortools.sat.python import cp_model


class Task:
    _id_counter = 1

    def __init__(self, name, duration, dependencies=None):
        self.id = Task._id_counter
        Task._id_counter += 1
        self.name = name
        self.duration = duration
        self.dependencies = dependencies or []
        self.start = None
        self.end = None


class ConstructionScheduler:
    def __init__(self):
        self.model = cp_model.CpModel()
        self.tasks = []
        self.task_starts = {}
        self.task_ends = {}
        self.horizon = 50  # upper bound


    def add_task(self, name, duration, dependencies=None):
        task = Task(name, duration, dependencies)
        self.tasks.append(task)
        return task

    def create_variables(self):
        # Create variables
        for task in self.tasks:
            self.task_starts[task.name] = self.model.NewIntVar(
                0, self.horizon, f"start_{task.name}"
            )
            self.task_ends[task.name] = self.model.NewIntVar(0, self.horizon, f"end_{task.name}")
            self.model.Add(
                self.task_ends[task.name] == self.task_starts[task.name] + task.duration
            )

    def add_dependencies(self):
        # Add dependencies (finish-to-start)
        for task in self.tasks:
            for dep, link, lag_or_delay in task.dependencies:
                match link:
                    case "FS":
                        self.model.Add(self.task_starts[task.name] >= self.task_ends[dep] + lag_or_delay)

                    case "SS":
                        self.model.Add(self.task_starts[task.name] >= self.task_starts[dep] + lag_or_delay)

                    case "FF":
                        self.model.Add(self.task_ends[task.name] >= self.task_ends[dep] + lag_or_delay)

                    case "SF":
                        self.model.Add(self.task_ends[task.name] >= self.task_starts[dep] + lag_or_delay)

        # Minimize overall project duration
        makespan = self.model.NewIntVar(0, self.horizon, "makespan")
        self.model.AddMaxEquality(makespan, [self.task_ends[t.name] for t in self.tasks])
        self.model.Minimize(makespan)

        # Solve
        solver = cp_model.CpSolver()
        status = solver.Solve(self.model)

        schdeuled_task = []

        if status == cp_model.OPTIMAL:
            print("\n=== CONSTRUCTION SCHEDULE ===")
            sorted_tasks = sorted(
                self.tasks, key=lambda t: solver.Value(self.task_starts[t.name])
            )
            for task in sorted_tasks:
                start = solver.Value(self.task_starts[task.name])
                end = solver.Value(self.task_ends[task.name])

                schdeuled_task.append({"name": task.name, "start": start, "end": end})
                print(
                    f"{task.name:12} : Day {start:2d} to {end:2d} (Duration: {task.duration} days)"
                )

            print(f"\nTotal project duration: {solver.Value(makespan)} days")

        else:
            print("No optimal solution found.")

        return schdeuled_task, solver.Value(makespan)

    def display_gantt_chart(self, schdeuled_task):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot each task as a horizontal bar
        for i, task in enumerate(schdeuled_task["tasks"]):
            ax.barh(
                task["name"],
                task["end"] - task["start"],
                left=task["start"],
                height=0.4,
                color="skyblue",
            )
            ax.text(
                task["start"] + 0.1,
                i,
                f'{task["start"]}-{task["end"]}',
                va="center",
                ha="left",
            )

        ax.invert_yaxis()
        ax.set_xlabel("Days")
        ax.set_ylabel("Tasks")
        ax.set_title("Construction Project Gantt Chart")
        ax.grid(True, axis="x", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()