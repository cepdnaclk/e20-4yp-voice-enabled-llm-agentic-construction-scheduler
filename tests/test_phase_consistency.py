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
    other_details={"phase_agent": "limit the scheduling upto structural framing"},
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

LUXURY_RESIDENTIAL_INTENT = ConstructionIntent(
    project_type="residential",
    building_category="luxury villa",
    size={"value": 4500, "unit": "sq_ft"},
    floors=3,
    location="Miami, Florida",
    special_requirements=[
        "hurricane-resistant windows",
        "smart home automation system",
        "elevator installation",
        "home theater room",
        "infinity pool with deck",
    ],
    timeline_preference="24 months",
    budget_range={"min": 1200000, "max": 1800000, "currency": "USD"},
    other_details={
        "phase_agent": "Include pool construction as part of site work phase",
    },
)

MIXED_USE_INTENT = ConstructionIntent(
    project_type="commercial",
    building_category="mixed-use complex",
    size={"value": 75000, "unit": "sq_ft"},
    floors=8,
    location="Seattle, Washington",
    special_requirements=[
        "retail spaces on ground floor",
        "office spaces floors 2-5",
        "residential apartments floors 6-8",
        "underground parking for 150 cars",
        "green roof with solar panels",
        "separate HVAC zones for each use type",
    ],
    timeline_preference="30 months",
    budget_range={"min": 15000000, "max": 20000000, "currency": "USD"},
    other_details={
        "phase_agent": "Coordinate MEP works across different floor types",
        "phasing": "Retail portion to open 6 months before residential",
    },
)

INDUSTRIAL_INTENT = ConstructionIntent(
    project_type="industrial",
    building_category="manufacturing plant",
    size={"value": 100000, "unit": "sq_ft"},
    floors=2,
    location="Houston, Texas",
    special_requirements=[
        "heavy-duty flooring for machinery",
        "overhead crane system",
        "clean room facilities (Class 100,000)",
        "hazardous material storage area",
        "high-capacity electrical system (2000A)",
        "compressed air distribution system",
        "loading docks with levelers (6 docks)",
    ],
    timeline_preference="36 months",
    budget_range={"min": 25000000, "max": 35000000, "currency": "USD"},
    other_details={
        "phase_agent": "Include equipment installation lead times in scheduling",
    },
)

RENOVATION_ADDITION_INTENT = ConstructionIntent(
    project_type="residential",
    building_category="historic house renovation with addition",
    size={"value": 3200, "unit": "sq_ft"},
    floors=2,
    location="Charleston, South Carolina",
    special_requirements=[
        "preserve historic facade",
        "match existing architectural details",
        "new 800 sq ft master suite addition",
        "updated electrical and plumbing throughout",
        "earthquake retrofitting",
        "original hardwood floor restoration",
    ],
    timeline_preference="18 months",
    budget_range={"min": 600000, "max": 850000, "currency": "USD"},
    other_details={
        "phase_agent": "Schedule demolition carefully to protect historic portions",
    },
)

HEALTHCARE_INTENT = ConstructionIntent(
    project_type="commercial",
    building_category="medical office building with surgery center",
    size={"value": 45000, "unit": "sq_ft"},
    floors=4,
    location="Rochester, Minnesota",
    special_requirements=[
        "operating rooms with laminar airflow",
        "medical gas systems",
        "radiation shielding in imaging suites",
        "emergency power backup (generators + UPS)",
        "infection control measures during construction",
        "handicap accessibility beyond code",
        "separate patient and service elevators",
    ],
    timeline_preference="28 months",
    budget_range={"min": 18000000, "max": 22000000, "currency": "USD"},
    other_details={
        "phase_agent": "Coordinate inspections with health department",
    },
)

HIGH_RISE_INTENT = ConstructionIntent(
    project_type="commercial",
    building_category="high-rise residential tower",
    size={"value": 350000, "unit": "sq_ft"},
    floors=32,
    location="San Francisco, California",
    special_requirements=[
        "deep foundation with caissons",
        "shoring and underpinning for adjacent buildings",
        "curtain wall facade system",
        "sky lobby on floor 20",
        "rooftop amenity deck with pool",
        "four basement levels for parking",
        "seismic isolation system",
    ],
    timeline_preference="42 months",
    budget_range={"min": 85000000, "max": 110000000, "currency": "USD"},
    other_details={
        "phase_agent": "Floor-by-floor scheduling for MEP rough-ins",
    },
)

EDUCATIONAL_INTENT = ConstructionIntent(
    project_type="commercial",
    building_category="K-12 school campus",
    size={"value": 120000, "unit": "sq_ft"},
    floors=3,
    location="Denver, Colorado",
    special_requirements=[
        "classroom buildings (3 interconnected)",
        "gymnasium with retractable seating",
        "auditorium with theater systems",
        "science labs with fume hoods",
        "kitchen and cafeteria facilities",
        "sports fields and outdoor courts",
        "safe room/storm shelter",
    ],
    timeline_preference="24 months",
    budget_range={"min": 45000000, "max": 55000000, "currency": "USD"},
    other_details={
        "phase_agent": "Complete exterior shell before winter",
    },
)

INFRASTRUCTURE_INTENT = ConstructionIntent(
    project_type="commercial",
    building_category="transit-oriented development",
    size={"value": 200000, "unit": "sq_ft"},
    floors=12,
    location="New York, New York",
    special_requirements=[
        "direct connection to subway station",
        "above-track platform construction",
        "retail podium (floors 1-4)",
        "office tower above (floors 5-12)",
        "public plaza with green space",
        "bicycle storage and shower facilities",
        "MTA utility relocations",
    ],
    timeline_preference="48 months",
    budget_range={"min": 120000000, "max": 150000000, "currency": "USD"},
    other_details={
        "phase_agent": "Coordinate track outages with transit authority",
    },
)

EXTREME_CLIMATE_INTENT = ConstructionIntent(
    project_type="residential",
    building_category="off-grid sustainable home",
    size={"value": 1800, "unit": "sq_ft"},
    floors=1,
    location="Fairbanks, Alaska",
    special_requirements=[
        "permafrost-adapted foundation",
        "super-insulated envelope (R-60 walls)",
        "triple-pane windows",
        "solar panels with battery storage",
        "passive house certification",
        "rainwater collection system",
        "composting toilets",
    ],
    timeline_preference="16 months",
    budget_range={"min": 550000, "max": 700000, "currency": "USD"},
    other_details={
        "phase_agent": "Complete weather-tight envelope before winter",
    },
)

DATA_CENTER_INTENT = ConstructionIntent(
    project_type="commercial",
    building_category="data center campus",
    size={"value": 250000, "unit": "sq_ft"},
    floors=2,
    location="Northern Virginia",
    special_requirements=[
        "three data halls (Phase 1: 1 hall, Phase 2: 2 halls)",
        "redundant power feeds (2N configuration)",
        "backup generators (12 total)",
        "chilled water cooling system",
        "raised floors with precision cooling",
        "blast-resistant construction",
        "physical security per Tier IV standards",
    ],
    timeline_preference="36 months",
    budget_range={"min": 200000000, "max": 250000000, "currency": "USD"},
    other_details={
        "phase_agent": "Phase 1 operational while Phase 2 under construction",
    },
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
        },  # type: ignore
        config=config,  # type: ignore
    )

    # Step 3: Read phases from state
    snapshot = model.workflow.get_state(config)
    state_values = snapshot.values
    cache = state_values.get("cache", {})
    phases = cache.get("phases_data", [])

    return phases


def _generalized_jaccard(sets):
    """Jaccard similarity for multiple sets"""
    print(f"\n{'='*60}")
    print(f"  JACCARD SIMILARITY ")
    print(f"{'='*60}")
    if len(sets) < 2:
        return 0

    intersection = set.intersection(*sets)
    union = set.union(*sets)

    return len(intersection) / len(union) if union else 0


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

if __name__ == "__main__":
    """Run directly with: python -m tests.test_phase_consistency"""
    print("🚀 Phase Agent Consistency Test (standalone mode)")
    print("=" * 60)

    mdl = AgenticSchedulerModel()

    intents = {
        # Basic test cases
        # "Residential (2500 sq_ft, 2-story)": RESIDENTIAL_INTENT,
        "Commercial (15000 sq_ft, 5-story office)": COMMERCIAL_INTENT,
        # Complex test cases
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
        all_runs = []
        for i in range(NUM_RUNS):
            print(f"\n--- Run {i+1}/{NUM_RUNS} ---")
            phases = _run_phase_node_once(mdl, intent)
            print(f"  Phases: {phases}")
            all_runs.append(set(phases))

        print(_generalized_jaccard(all_runs))
        # _print_consistency_report(name, all_runs)

    print("\n✅ Done!")
