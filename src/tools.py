from langchain.tools import tool
from typing import Literal
from langchain_core.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from enum import Enum

agentic_model = None


class WorkflowStage(Enum):
    INTENT = "intent"
    PHASES = "phases"
    DETAILS = "details"
    SCHEDULING = "scheduling"


def setup_tools(model_instance):
    """Set the global model instance for tools to access"""
    from src.model import AgenticSchedulerModel

    model: AgenticSchedulerModel = model_instance
    global agentic_model
    agentic_model = model


#### Intent Tools ####


@tool
def submit_construction_intent(
    project_type: Literal[
        "residential", "commercial", "industrial", "infrastructure", "other"
    ],
    building_category: str,
    size_value: float,
    size_unit: str,
    floors: int | None = None,
    location: str | None = None,
    special_requirements: list[str] | None = None,
    timeline_preference: str | None = None,
    budget_min: float | None = None,
    budget_max: float | None = None,
    budget_currency: str = "USD",
    other_details: dict[str, str] | None = None,
) -> str:
    """
    Submit the finalized construction project intent after gathering all necessary information.

    Only call this tool when you have collected sufficient information about:
    - Project type and category
    - Size with proper units
    - Other relevant details the user has provided
    - Store other detail in other_details. the key should be phases_agent,
    details_agent or scheduling_agent. depedending on who requires that data

    Do NOT call this for initial greetings or when you still need more information.
    """
    return "Intent submitted successfully for user confirmation"


#### Phases Tools ####


@tool
def confirm_phases(phases_list: str) -> list:
    """
    Call this tool once you have identified the major construction phases for the project.
    Input should be a comma-separated list of phases (e.g., "Foundation, Framing, Roofing").
    """
    # Parse the list
    phases = [p.strip() for p in phases_list.split(",")]
    if not phases:
        return []

    return phases


intent_tools = [submit_construction_intent]

phase_tools = [confirm_phases]

details_tools = []
