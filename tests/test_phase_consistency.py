"""
Phase Agent Consistency Test
=============================
Tests whether the phase agent identifies a consistent set of phases
when given the exact same construction intent multiple times.

Bypasses the intent node entirely by injecting pre-populated state
directly into the graph with current_stage="phases".

Requirements:
  - Valid .env with OPENAI_API_KEY, MODEL, NEO4J_* credentials
  - Neo4j database running
  - Each run makes real LLM API calls

Usage:
  conda activate fyp_env
  python -m pytest tests/test_phase_consistency.py -v -s
"""

import uuid
import pytest
from collections import Counter

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from src.model import (
    AgenticSchedulerModel,
    ConstructionIntent,
    WorkflowStage,
    AgentState,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model():
    """Create a single AgenticSchedulerModel instance shared across all tests.
    scope=module so we only pay the Neo4j/vector-index init cost once."""
    return AgenticSchedulerModel()


# ---------------------------------------------------------------------------
# Test intents — add more dicts here to test different project types
# ---------------------------------------------------------------------------

RESIDENTIAL_INTENT = ConstructionIntent(
    project_type="residential",
    building_category="single-family home",
    size={"value": 2500, "unit": "sq_ft"},
    floors=2,
    location="Austin, Texas",
    special_requirements=[],
    timeline_preference="12 months",
    budget_range={"min": 300000, "max": 500000, "currency": "USD"},
    other_details=None,
)

COMMERCIAL_INTENT = ConstructionIntent(
    project_type="commercial",
    building_category="office building",
    size={"value": 15000, "unit": "sq_ft"},
    floors=5,
    location="Chicago, Illinois",
    special_requirements=[],
    timeline_preference="18 months",
    budget_range={"min": 5000000, "max": 8000000, "currency": "USD"},
    other_details=None,
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


def _run_phase_node_once(
    model: AgenticSchedulerModel, intent: ConstructionIntent
) -> list[str]:
    """
    Invoke the graph starting at the phase node and return the
    proposed phases list.

    Mirrors the server.py flow:
      1. invoke() with initial state
      2. Check get_state().tasks for interrupts
      3. If interrupted → phases are in cache["phases_data"]
      4. If NOT interrupted (LLM didn't call the tool) → re-invoke
         with a nudge message so the LLM calls confirm_phases
    """
    thread_id = str(uuid.uuid4())
    config = RunnableConfig(
        configurable={"thread_id": thread_id},
        recursion_limit=50,
    )

    initial_state = _make_initial_state(intent)

    # Step 1: Initial invoke — runs until interrupt or END
    model.workflow.invoke(initial_state, config=config)  # type: ignore

    model.workflow.invoke(
        {
        "messages": [
                HumanMessage(
                    content="Please confirm the phases by calling the confirm_phases tool."
                )
            ]
        }, # type: ignore
        config=config,  # type: ignore
    )

    # Step 3: Read phases from state
    snapshot = model.workflow.get_state(config)
    state_values = snapshot.values
    cache = state_values.get("cache", {})
    phases = cache.get("phases_data", [])

    return phases


def _print_consistency_report(intent_name: str, all_runs: list[list[str]]):
    """Print a formatted consistency report for a set of runs."""
    print(f"\n{'='*60}")
    print(f"  CONSISTENCY REPORT: {intent_name}")
    print(f"{'='*60}")

    for i, phases in enumerate(all_runs):
        print(f"\n  Run {i+1}: {phases}")

    # Normalize phase lists to sorted tuples for comparison
    normalized = [tuple(sorted(p.lower().strip() for p in run)) for run in all_runs]
    unique_results = set(normalized)

    # Count frequency of each unique result
    counter = Counter(normalized)
    most_common, most_common_count = counter.most_common(1)[0]

    consistency_pct = (most_common_count / len(all_runs)) * 100

    print(f"\n  {'─'*50}")
    print(f"  Total runs:        {len(all_runs)}")
    print(f"  Unique results:    {len(unique_results)}")
    print(
        f"  Consistency:       {consistency_pct:.0f}% ({most_common_count}/{len(all_runs)} identical)"
    )
    print(f"  Most common set:   {list(most_common)}")

    if len(unique_results) > 1:
        print(f"\n  ⚠️  INCONSISTENCY DETECTED — different phase sets across runs:")
        for phases_tuple, count in counter.most_common():
            print(f"     {count}x → {list(phases_tuple)}")

    print(f"{'='*60}\n")

    return consistency_pct, unique_results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPhaseConsistency:
    """Run the phase agent multiple times with the same intent and check
    whether it produces the same phases each time."""

    def test_residential_consistency(self, model):
        """Test phase consistency for a residential project."""
        all_runs = []
        for i in range(NUM_RUNS):
            print(f"\n--- Residential Run {i+1}/{NUM_RUNS} ---")
            phases = _run_phase_node_once(model, RESIDENTIAL_INTENT)
            assert phases, f"Run {i+1} returned empty phases!"
            all_runs.append(phases)

        consistency_pct, unique_results = _print_consistency_report(
            "Residential (2500 sq_ft, 2-story home)", all_runs
        )

        # Assert 100% consistency — all runs should produce the same phases
        assert len(unique_results) == 1, (
            f"Phase agent produced {len(unique_results)} different phase sets "
            f"across {NUM_RUNS} runs. Expected identical results.\n"
            f"Results: {all_runs}"
        )

    def test_commercial_consistency(self, model):
        """Test phase consistency for a commercial project."""
        all_runs = []
        for i in range(NUM_RUNS):
            print(f"\n--- Commercial Run {i+1}/{NUM_RUNS} ---")
            phases = _run_phase_node_once(model, COMMERCIAL_INTENT)
            assert phases, f"Run {i+1} returned empty phases!"
            all_runs.append(phases)

        consistency_pct, unique_results = _print_consistency_report(
            "Commercial (15000 sq_ft, 5-story office)", all_runs
        )

        assert len(unique_results) == 1, (
            f"Phase agent produced {len(unique_results)} different phase sets "
            f"across {NUM_RUNS} runs. Expected identical results.\n"
            f"Results: {all_runs}"
        )


# ---------------------------------------------------------------------------
# Standalone runner (no pytest needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Run directly with: python -m tests.test_phase_consistency"""
    print("🚀 Phase Agent Consistency Test (standalone mode)")
    print("=" * 60)

    mdl = AgenticSchedulerModel()

    intents = {
        "Residential (2500 sq_ft, 2-story)": RESIDENTIAL_INTENT,
        # "Commercial (15000 sq_ft, 5-story office)": COMMERCIAL_INTENT,
    }

    for name, intent in intents.items():
        print(f"\n\n🔄 Testing: {name}")
        all_runs = []
        for i in range(NUM_RUNS):
            print(f"\n--- Run {i+1}/{NUM_RUNS} ---")
            phases = _run_phase_node_once(mdl, intent)
            print(f"  Phases: {phases}")
            all_runs.append(phases)

        _print_consistency_report(name, all_runs)

    print("\n✅ Done!")
