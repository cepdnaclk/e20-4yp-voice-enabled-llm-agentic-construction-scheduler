"""
Subtask Consistency Test
========================
Tests whether the LLM generates a consistent set of WBS subtasks
when given the exact same construction intent multiple times.

Workflow covered:
  1. Start with initial state (bypasses intent agent)
  2. Invoke → phase_node builds the WBS and hits interrupt()
  3. Resume with "confirm" → phase_node stores the WBS in
     state["project_wbs"] and transitions to details_agent
  4. details_agent hits its first interrupt (asking for variables)
     — we do NOT answer; we just read the state.
  5. Extract per-phase task names from state["project_wbs"] and
     report Jaccard similarity across NUM_RUNS.
"""

import uuid
import pytest

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from src.model import (
    AgenticSchedulerModel,
    ConstructionIntent,
    WorkflowStage,
    AgentState,
)


@pytest.fixture(scope="module")
def model():
    """Create a single AgenticSchedulerModel instance shared across all tests.
    scope=module so we only pay the Neo4j/vector-index init cost once."""
    return AgenticSchedulerModel()


RESIDENTIAL_INTENT = ConstructionIntent(
    project_type="residential",
    building_category="single-family home",
    size={"value": 20, "unit": "m2"},
    floors=1,
    location="Kandy, Sri Lanka",
    special_requirements=[],
    timeline_preference="12 months",
    budget_range={},
    other_details={"phase_agent": "limit the scheduling upto foundation phase"},
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_RUNS = 5  # How many times to repeat the same intent


def _make_initial_state(intent: ConstructionIntent) -> dict:
    """Build the state that the intent node would produce after user confirms."""
    return {
        "messages": [],
        "sender": "intent_agent",
        "current_stage": WorkflowStage.PHASES.value,
        "phases": [],
        "user_intent": intent.model_dump(),
        "current_phase_index": None,
        "generated_tasks": {},
        "schedule_result": None,
        "interrupt": False,
        "cache": {},
    }


def _run_once_and_get_subtasks(
    model: AgenticSchedulerModel, intent: ConstructionIntent
) -> dict[str, list[str]]:
    """
    Run the workflow through phase confirmation and into details_agent,
    then extract the per-phase task names from the adapted WBS.

    Returns:
        dict mapping phase_name -> list of task names (str)

    Workflow steps:
      1. invoke(initial_state)  — phase_node builds WBS, hits interrupt()
      2. invoke(Command("confirm"))  — phase confirmed → routes to details_agent
         details_agent will hit its first interrupt (variable questions); we let
         it sit there and read the state without answering.
    """
    thread_id = str(uuid.uuid4())
    config = RunnableConfig(
        configurable={"thread_id": thread_id},
        recursion_limit=100,
    )

    initial_state = _make_initial_state(intent)

    # ── Step 1: invoke until phase_node hits interrupt (WBS proposal) ──
    model.workflow.invoke(initial_state, config=config)  # type: ignore

    # ── Step 2: confirm the WBS; workflow proceeds to details_agent ──
    #    details_agent will stop at its first interrupt (variable questions)
    model.workflow.invoke(
        Command(resume="confirm"),  # type: ignore
        config=config,  # type: ignore
    )

    # ── Step 3: read state and extract project_wbs ──
    snapshot = model.workflow.get_state(config)
    state_values = snapshot.values

    project_wbs: dict = state_values.get("project_wbs") or {}
    phases: list[dict] = project_wbs.get("phases", [])

    # Build per-phase task name sets
    tasks_by_phase: dict[str, list[str]] = {}
    for phase in phases:
        phase_name = phase["name"]
        task_names = []
        for pkg in phase.get("packages", []):
            for task in pkg.get("tasks", []):
                task_names.append(task["name"])
        tasks_by_phase[phase_name] = task_names

    return tasks_by_phase


def _average_pairwise_jaccard(sets: list[set]) -> float:
    """Average pairwise Jaccard similarity across multiple sets.
    This is much less sensitive to single-run outliers than intersection/union.
    """
    if len(sets) < 2:
        return 1.0  # only one set → trivially consistent

    import itertools

    pairs = list(itertools.combinations(sets, 2))

    total_score = 0.0
    for s1, s2 in pairs:
        intersection = set.intersection(s1, s2)
        union = set.union(s1, s2)
        score = len(intersection) / len(union) if union else 1.0
        total_score += score

    return total_score / len(pairs)


def _print_consistency_report(name: str, all_runs: list[dict[str, list[str]]]) -> None:
    """Print a detailed per-phase consistency report."""
    print(f"\n{'='*60}")
    print(f"  CONSISTENCY REPORT: {name}")
    print(f"{'='*60}")

    # Collect all phase names seen across runs
    all_phase_names: set[str] = set()
    for run in all_runs:
        all_phase_names.update(run.keys())

    phase_scores: dict[str, float] = {}
    for phase in sorted(all_phase_names):
        sets = [set(run.get(phase, [])) for run in all_runs]

        # Print task lists per run
        print(f"\n  📌 Phase: {phase}")
        for i, s in enumerate(sets):
            print(f"    Run {i+1}: {sorted(s)}")

        score = _average_pairwise_jaccard(sets)
        phase_scores[phase] = score
        print(f"    → Avg Pairwise Jaccard: {score:.3f}")

        # Print task frequencies
        from collections import Counter

        task_counts = Counter()
        for s in sets:
            for task in s:
                task_counts[task] += 1

        print("    → Task Frequencies:")
        for task, count in task_counts.most_common():
            # Highlight consistent vs inconsistent tasks
            indicator = "✅" if count == len(sets) else "⚠️"
            print(f"        {indicator} {task}: {count}/{len(sets)} runs")

    # Overall score = mean of per-phase scores
    if phase_scores:
        overall = sum(phase_scores.values()) / len(phase_scores)
        print(f"\n  🏆 Overall Mean Pairwise Jaccard: {overall:.3f}")
    else:
        print("\n  ⚠️  No phases found in any run.")

    print()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    """Run directly with: python -m tests.test_phase_consistency"""
    print("🚀 Subtask Consistency Test (standalone mode)")
    print("=" * 60)

    mdl = AgenticSchedulerModel()

    intents = {
        # Basic test cases
        "Residential (2500 sq_ft, 2-story)": RESIDENTIAL_INTENT,
        # Complex test cases (uncomment to enable)
        # "Luxury Residential (4500 sq_ft, 3-story villa with pool)": LUXURY_RESIDENTIAL_INTENT,
        # "Mixed-Use (75000 sq_ft, 8-story retail+office+residential)": MIXED_USE_INTENT,
        # "Industrial (100000 sq_ft, 2-story manufacturing plant)": INDUSTRIAL_INTENT,
        # "Historic Renovation (3200 sq_ft, 2-story with addition)": RENOVATION_ADDITION_INTENT,
        # "Healthcare (45000 sq_ft, 4-story medical+surgery)": HEALTHCARE_INTENT,
        # "High-Rise (350000 sq_ft, 32-story residential tower)": HIGH_RISE_INTENT,
        # "Educational (120000 sq_ft, 3-story K-12 campus)": EDUCATIONAL_INTENT,
        # "Transit-Oriented (200000 sq_ft, 12-story over subway)": INFRASTRUCTURE_INTENT,
        # "Off-Grid (1800 sq_ft, 1-story extreme climate)": EXTREME_CLIMATE_INTENT,
        # "Data Center (250000 sq_ft, 2-story phased campus)": DATA_CENTER_INTENT,
    }

    for name, intent in intents.items():
        print(f"\n\n🔄 Testing: {name}")
        all_runs: list[dict[str, list[str]]] = []

        for i in range(NUM_RUNS):
            print(f"\n--- Run {i + 1}/{NUM_RUNS} ---")
            try:
                tasks_by_phase = _run_once_and_get_subtasks(mdl, intent)
                for phase, tasks in tasks_by_phase.items():
                    print(f"  [{phase}] → {tasks}")
                all_runs.append(tasks_by_phase)
            except Exception as e:
                print(f"  ⚠️ Run {i + 1} failed: {e}")
                print(
                    f"  Stopping further runs for '{name}' and proceeding to consistency report with {len(all_runs)} successful runs..."
                )
                break

        _print_consistency_report(name, all_runs)

    print("\n✅ Done!")
