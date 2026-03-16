from pydantic.fields import Field
from typing import Any, Dict, Optional
import os
import json
import re
import math
import operator
import uuid
import time
import logging

logger = logging.getLogger(__name__)
from pydantic.main import BaseModel
from langgraph.types import Command, interrupt
from enum import Enum
from typing import Sequence, Annotated, TypedDict, Literal, List
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent
from src.tools import setup_tools, phase_tools, details_tools, intent_tools
from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_neo4j.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_openai import OpenAIEmbeddings
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.scheduler import solve_schedule


class _CompatAgent:
    def __init__(self, runnable):
        self._runnable = runnable

    def invoke(self, state):
        messages = state.get("messages", []) if isinstance(state, dict) else state
        return self._runnable.invoke({"messages": messages})


def create_agent(*, system_prompt, tools, model, response_format=None):
    runnable = create_react_agent(model=model, tools=tools, prompt=system_prompt)
    return _CompatAgent(runnable)

# load the env variables from the .env file
load_dotenv(".env")


class WorkflowStage(Enum):
    INTENT = "intent"
    PHASES = "phases"
    DETAILS = "details"
    SCHEDULING = "scheduling"


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: Annotated[str, "The sender of last message"]
    current_stage: str
    phases: list
    project_wbs: Optional[
        dict
    ]  # Full WBS from phase_node (FullProjectWBS.model_dump())
    user_intent: Optional[str | dict]
    current_phase_index: Optional[int]
    generated_tasks: Optional[dict]  # Store tasks by phase name
    schedule_result: Optional[list]  # Optimised schedule from OR-Tools
    interrupt: bool
    cache: Optional[dict]


class ConstructionIntent(BaseModel):
    """Structured construction project intent"""

    project_type: Literal[
        "residential", "commercial", "industrial", "infrastructure", "other"
    ] = Field(description="Type of construction project")
    building_category: str = Field(
        description="Specific building type (e.g., 'single-family home', 'office building', 'warehouse')"
    )
    size: dict = Field(
        default=...,
        description="Project size with units",
        examples=[{"value": 5000, "unit": "sq_ft"}],
    )
    floors: int | None = Field(None, description="Number of floors/stories")
    location: str | None = Field(None, description="Project location if specified")
    special_requirements: list[str] | None = Field(
        None,
        description="Special requirements like sustainable design, specific materials, etc.",
    )
    timeline_preference: str | None = Field(
        None, description="Desired project timeline if mentioned"
    )
    budget_range: dict | None = Field(
        None, description="Budget information if provided"
    )
    other_details: dict[str, str] | None = Field(
        None,
        description="Additional instructions keyed by target agent (phase_agent, details_agent, scheduling_agent)",
    )

    def to_summary(self) -> str:
        """Convert to human-readable summary"""
        summary = (
            f"Project Type: {self.project_type.title()} - {self.building_category}\n"
        )
        summary += f"Size: {self.size['value']} {self.size['unit']}"
        if self.floors:
            summary += f"\nFloors: {self.floors}"
        if self.location:
            summary += f"\nLocation: {self.location}"
        if self.special_requirements:
            summary += f"\nSpecial Requirements: {', '.join(self.special_requirements)}"
        if self.timeline_preference:
            summary += f"\nTimeline: {self.timeline_preference}"
        if self.budget_range:
            summary += f"\nBudget: {self.budget_range}"
        if self.other_details:
            for key, val in self.other_details.items():
                summary += f"\nNote for {key}: {val}"
        return summary


class Dependency(BaseModel):
    """Represents a task dependency with relationship and lag"""

    previous_task: str = Field(..., description="Name of the previous task")
    relationship: str = Field(
        ..., description="Relationship type (e.g., 'FS', 'SS', 'FF', 'SF')"
    )
    lag_days: int = Field(default=0, description="Lag time in days")

    @classmethod
    def from_list(cls, dep_list: List) -> "Dependency":
        """Create from [previous_task, relationship, lag] format"""
        return cls(
            previous_task=dep_list[0],
            relationship=dep_list[1],
            lag_days=dep_list[2] if len(dep_list) > 2 else 0,
        )


class SelectedDependency(BaseModel):
    """LLM-selected dependency when graph traversal yields ambiguous results"""

    predecessor: str = Field(..., description="Name of the selected predecessor task")
    relationship_type: str = Field(
        ..., description="Relationship type: FS, FF, SS, or SF"
    )
    lag: int = Field(default=0, description="Lag in days")
    reasoning: str = Field(..., description="Why this predecessor was selected")


class Resource(BaseModel):
    """Represents a required resource"""

    name: str = Field(..., description="Resource name")
    amount: float = Field(..., description="Required amount/quantity")

    @classmethod
    def from_list(cls, res_list: List) -> "Resource":
        """Create from [resource_name, amount] format"""
        return cls(name=res_list[0], amount=res_list[1])


class VariableEntry(BaseModel):
    """A single variable name-value pair"""

    variable_name: str = Field(..., description="Name of the variable")
    value: float = Field(..., description="Numeric value of the variable")


class TaskVariableValue(BaseModel):
    """A single variable value for a specific task"""

    task_name: str = Field(..., description="Exact name of the task")
    variable_entries: list[VariableEntry] = Field(
        ..., description="List of variable name-value pairs for this task"
    )


class TaskVariableValues(BaseModel):
    """Extracted variable values for all tasks from user response"""

    task_values: list[TaskVariableValue] = Field(
        ..., description="List of per-task variable values"
    )


class Task(BaseModel):
    """Represents a single task in the project schedule"""

    name: str = Field(..., description="Task name")
    duration_days: int = Field(..., gt=0, description="Duration in days")
    dependencies: List[List[str]] = Field(
        default_factory=list,
        description="List of dependencies as [previous_task, relationship, lag]",
    )
    resources: List[List[str]] = Field(
        default_factory=list, description="List of resources as [resource_name, amount]"
    )

    def get_dependencies(self) -> List[Dependency]:
        """Parse dependencies into Dependency objects"""
        return [Dependency.from_list(dep) for dep in self.dependencies]

    def get_resources(self) -> List[Resource]:
        """Parse resources into Resource objects"""
        return [Resource.from_list(res) for res in self.resources]


class TaskList(BaseModel):
    """Collection of tasks for a project phase"""

    tasks: List[Task] = Field(..., description="List of tasks")


class WBS_Phases(BaseModel):
    """Collection of WBS Major phase for the current project"""

    phases: List[str] = Field(..., description="List of WBS phases for current project")


class WBSTask(BaseModel):
    """A single task within a work package"""

    name: str = Field(..., description="Task name")
    description: str | None = Field(None, description="Brief task description")


class WBSPackage(BaseModel):
    """A work package containing tasks"""

    name: str = Field(..., description="Work package name")
    tasks: list[WBSTask] = Field(
        default_factory=list, description="Tasks in this package"
    )


class WBSPhase(BaseModel):
    """A project phase containing work packages"""

    name: str = Field(..., description="Phase name")
    packages: list[WBSPackage] = Field(
        default_factory=list, description="Work packages in this phase"
    )


class FullProjectWBS(BaseModel):
    """Complete Work Breakdown Structure adapted for the user's project"""

    project_name: str = Field(..., description="Name of the reference project template")
    phases: list[WBSPhase] = Field(
        ..., description="Adapted phases with packages and tasks"
    )


class SelectedProject(BaseModel):
    project: str


class AgenticSchedulerModel:

    # ── Neo4j retry configuration ──────────────────────────────────────────
    NEO4J_MAX_RETRIES = 5  # total attempts
    NEO4J_INITIAL_DELAY = 2.0  # seconds before first retry
    NEO4J_BACKOFF_FACTOR = 2.0  # multiply delay by this on each failure

    @staticmethod
    def _connect_neo4j(
        max_retries: int = 5,
        initial_delay: float = 2.0,
        backoff_factor: float = 2.0,
    ) -> "Neo4jGraph":
        """Create a Neo4jGraph connection with exponential-backoff retries.

        Raises:
            ConnectionError: if all attempts fail, with the last exception
                             chained so callers (and the SSE error event) can
                             surface a meaningful message to the user.
        """
        url = os.getenv("NEO4J_URI") or ""
        username = os.getenv("NEO4J_USERNAME") or ""
        password = os.getenv("NEO4J_PASSWORD") or ""
        database = os.getenv("NEO4J_DATABASE") or ""

        delay: float = float(initial_delay)
        last_exc: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    "Neo4j connection attempt %d/%d to %s …",
                    attempt,
                    max_retries,
                    url,
                )
                graph = Neo4jGraph(
                    url=url,
                    username=username,
                    password=password,
                    database=database,
                )
                logger.info("Neo4j connected successfully on attempt %d.", attempt)
                return graph
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Neo4j connection attempt %d/%d failed: %s",
                    attempt,
                    max_retries,
                    exc,
                )
                if attempt < max_retries:
                    logger.info("Retrying in %.1f second(s)…", delay)
                    sleep_secs: float = float(delay)
                    time.sleep(sleep_secs)
                    delay = float(sleep_secs) * float(backoff_factor)

        assert last_exc is not None  # always set — loop ran at least once
        raise ConnectionError(
            f"Could not connect to Neo4j at '{url}' after {max_retries} attempts. "
            f"Last error: {last_exc}"
        ) from last_exc

    def __init__(self):
        setup_tools(self)
        print(os.getenv("NEO4J_URI"))
        self.llm = ChatOpenAI(
            model=os.getenv("MODEL") or "",
            api_key=SecretStr(os.getenv("OPENAI_API_KEY") or ""),
            timeout=60.0,
            max_retries=1,
        )

        self.graph = self._connect_neo4j(
            max_retries=self.NEO4J_MAX_RETRIES,
            initial_delay=self.NEO4J_INITIAL_DELAY,
            backoff_factor=self.NEO4J_BACKOFF_FACTOR,
        )

        # self.vector_index = Neo4jVector.from_existing_graph(
        #     OpenAIEmbeddings(),
        #     url=os.getenv("NEO4J_URI") or "",
        #     username=os.getenv("NEO4J_USERNAME") or "",
        #     password=os.getenv("NEO4J_PASSWORD") or "",
        #     index_name="task",
        #     node_label="Subtask",
        #     text_node_properties=["name", "description"],
        #     embedding_node_property="embedding",
        # )

        # TODO find better alternative to allow_dangerous_requests=True
        self.cypher_chain = GraphCypherQAChain.from_llm(
            llm=self.llm, graph=self.graph, verbose=True, allow_dangerous_requests=True
        )

        # TODO vectorizing the graph
        # retriever = self.vector_index.as_retriever()

        llm = ChatOpenAI(temperature=0)

        prompt = ChatPromptTemplate.from_template(
            """
        Answer the question based only on the following context:
        {context}

        Question: {input}

        Answer the question and provide relevant details from the context:
        """
        )

        document_chain = create_stuff_documents_chain(llm, prompt)

        # self.chain = create_retrieval_chain(retriever, document_chain)

        self.workflow = self._build_workflow()
        self._visualize_graph()

    def _build_workflow(self):
        # Create Intent Agent
        INTENT_AGENT_SYSTEM_PROMPT = """
        You are an expert construction project analyst helping to gather project requirements.

        Your goal is to have a natural, friendly conversation to collect:
        - Project type (residential/commercial/industrial/infrastructure)
        - Building category (e.g., 'office building', 'apartment complex', 'single-family home')
        - Size with units (sq_ft, sq_m, acres, etc.)
        - Number of floors (if applicable)
        - Location
        - Special requirements
        - Timeline preferences
        - Budget range

        **IMPORTANT INSTRUCTIONS:**
        1. Start with a friendly greeting if the user says "hi" or "hello"
        2. Ask questions naturally, one or two at a time - don't overwhelm the user
        3. Listen to what the user provides and adapt your questions. Also think like a planner. For example (If house is large and has multiple floors a lift maybe useful otherwise not)
        4. Only call the submit_construction_intent tool when you have enough information
        5. If the user gives vague answers, ask for clarification before submitting
        6. Store other details in other_details. the key should be phase_agent,details_agent or scheduling_agent. depedending on who requires that data
        7. Do not try to explain the details or ask for more if it is not required for intent phase
        8. When user wants to limit the project upto a certain level. limit it upto the matching work package

        **When you have sufficient information**, call the submit_construction_intent tool with all the details.

        **Example conversation flow:**
        User: "Hi"
        You: "Hello! I'm here to help you plan your construction project. What type of building are you looking to construct?"

        User: "I want to build a house"
        You: "Great! A residential home. Could you tell me approximately how large you'd like it to be? For example, in square feet or square meters?"

        User: "Around 2500 square feet, 2 floors"
        You: "Perfect! 2,500 sq ft with 2 floors. Do you have a location in mind for this project?"

        ...continue until you have enough details, then call the tool.

        Be conversational, helpful, and thorough."""

        intent_agent = create_agent(
            system_prompt=INTENT_AGENT_SYSTEM_PROMPT,
            tools=intent_tools,
            model=self.llm,
        )

        def agent_router(
            state: AgentState,
        ) -> Literal["intent_agent", "phase_agent", "details_agent"]:

            current_stage = state.get("current_stage", WorkflowStage.INTENT.value)

            if current_stage == WorkflowStage.INTENT.value:
                return "intent_agent"
            elif current_stage == WorkflowStage.PHASES.value:
                return "phase_agent"
            elif current_stage == WorkflowStage.DETAILS.value:
                return "details_agent"
            else:
                return "intent_agent"

        def extract_toolcall(messages, toolname: str):

            tool_calls = []

            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for i, tc in enumerate(msg.tool_calls):
                        if tc["name"] == toolname:
                            tool_calls.append({"tool_call": tc, "messages": msg})

            return tool_calls

        def intent_node(state: AgentState) -> AgentState | Command:

            print("\n===== INTENT NODE =====\n")

            # To check if the graph is interuppted
            interrupted = state.get("interrupt", False)

            if not interrupted:
                # Normal flow - invoke the agent
                result = intent_agent.invoke(state)  # type: ignore
                messages = result["messages"]
                last_message = messages[-1]

                intent_phase_tool_calls = extract_toolcall(
                    messages, "submit_construction_intent"
                )

                if (
                    last_message
                    and hasattr(last_message, "content")
                    and last_message.content
                    and (not intent_phase_tool_calls)
                ):
                    print(f"\n🤖 Assistant: {last_message.content}")

                if intent_phase_tool_calls:
                    tool_data = intent_phase_tool_calls[-1]  # Get the last tool call
                    tool_call = tool_data["tool_call"]
                    tool_messages = tool_data["messages"]
                    user_response_lower = None
                    structured_intent = None  # type: ignore

                    if tool_call["name"] == "submit_construction_intent":

                        args = tool_call["args"]

                        structured_intent = ConstructionIntent(
                            project_type=args["project_type"],
                            building_category=args["building_category"],
                            size={
                                "value": args["size_value"],
                                "unit": args["size_unit"],
                            },
                            floors=args.get("floors"),
                            location=args.get("location"),
                            special_requirements=args.get("special_requirements", []),
                            timeline_preference=args.get("timeline_preference"),
                            budget_range=(
                                {
                                    "min": args.get("budget_min"),
                                    "max": args.get("budget_max"),
                                    "currency": args.get("budget_currency", "USD"),
                                }
                                if args.get("budget_min") or args.get("budget_max")
                                else None
                            ),
                            other_details=args.get("other_details"),
                        )

                        return Command(
                            update=AgentState(
                                {
                                    **state,
                                    "messages": [],
                                    "cache": {
                                        "intent_data": structured_intent,
                                    },
                                    "interrupt": True,
                                }
                            ),
                            goto="intent_agent",
                        )

                else:
                    return AgentState({**state, "messages": [last_message]})

            else:
                state_cache = state["cache"]

                if state_cache:
                    structured_intent: ConstructionIntent = state_cache.get("intent_data")  # type: ignore
                    intent_summary = structured_intent.to_summary()

                    user_response = interrupt(
                        f"📋 Project Summary:\n{intent_summary}\n\n"
                        "Please confirm:\n"
                        "• Type 'yes' or 'confirm' to proceed\n"
                        "• Type corrections if something needs to be changed\n"
                        "• Type 'cancel' to start over"
                    )

                    user_response_lower = (
                        user_response.lower().strip() if user_response else ""
                    )

                    if user_response_lower in ["yes", "confirm"]:
                        # ✅ User confirmed — proceed to phase agent
                        print(f"✓ Intent confirmed by user")

                        user_intent = None
                        if structured_intent:
                            user_intent = structured_intent.model_dump()

                        return Command(
                            update=AgentState(
                                {
                                    **state,
                                    "messages": [],
                                    "sender": "intent_agent",
                                    "current_stage": WorkflowStage.PHASES.value,
                                    "user_intent": user_intent,
                                    "cache": {},
                                    "interrupt": False,
                                }
                            ),
                            goto="phase_agent",
                        )

                    elif user_response_lower in ["cancel", "start over", "reset"]:
                        # ❌ User cancelled — restart intent gathering
                        print("User cancelled - restarting intent gathering")

                        restart_msg = AIMessage(
                            content="No problem! Let's start fresh. What type of construction project would you like to plan?"
                        )

                        return Command(
                            update=AgentState(
                                {
                                    **state,
                                    "messages": [restart_msg],
                                    "sender": "user",
                                    "current_stage": WorkflowStage.INTENT.value,
                                    "user_intent": None,
                                    "phases": [],
                                    "current_phase_index": None,
                                    "generated_tasks": {},
                                    "interrupt": False,
                                    "cache": {},
                                }
                            ),
                            goto=END,
                        )

                    else:
                        # ✏️ User wants corrections — feed feedback back to intent agent
                        print(f"User requested corrections: {user_response_lower}")

                        correction_msg = HumanMessage(
                            content=f"Please update the project details based on this feedback: {user_response}"
                        )

                        return Command(
                            update=AgentState(
                                {
                                    **state,
                                    "messages": [correction_msg],
                                    "sender": "user",
                                    "current_stage": WorkflowStage.INTENT.value,
                                    "interrupt": False,
                                    "cache": {},
                                }
                            ),
                            goto="intent_agent",
                        )
                else:
                    return AgentState(
                        {**state, "messages": [], "interrupt": False, "cache": {}}
                    )

            return AgentState(
                {**state, "messages": [], "interrupt": False, "cache": {}}
            )

        def _fetch_full_template_tree(project_name: str) -> dict:
            """
            One recursive Cypher query to fetch the full project template tree
            (project → phases → packages → tasks) and build a nested dict.
            """
            query = """
            MATCH path = (p:WorkTemplate {name: $project_name})-[:HAS_CHILD*1..3]->(n:WorkTemplate)
            WITH n,
                 length(path) AS depth,
                 [rel IN relationships(path) | startNode(rel).name] AS ancestors
            RETURN n.name AS name,
                   n.description AS description,
                   depth,
                   ancestors
            ORDER BY depth
            """
            records = self.graph.query(query, {"project_name": project_name})

            # Build nested dict: phases → packages → tasks
            phases: Dict[str, Dict[str, list]] = {}  # phase -> {package -> [task, ...]}

            for rec in records:
                name = rec["name"]
                desc = rec.get("description")
                depth = rec["depth"]
                ancestors = rec[
                    "ancestors"
                ]  # e.g. [project, phase] or [project, phase, package]

                if depth == 1:
                    # Phase level
                    phases[name] = {}
                elif depth == 2:
                    # Package level — parent is the phase (ancestors[-1])
                    phase_name = ancestors[-1]
                    if phase_name in phases:
                        phases[phase_name][name] = []
                elif depth == 3:
                    # Task level — grandparent is phase, parent is package
                    phase_name = ancestors[-2] if len(ancestors) >= 2 else None
                    package_name = ancestors[-1]
                    if (
                        phase_name
                        and phase_name in phases
                        and package_name in phases[phase_name]
                    ):
                        phases[phase_name][package_name].append(
                            {"name": name, "description": desc}
                        )

            # Convert to JSON-friendly format
            template = {
                "project_name": project_name,
                "phases": [
                    {
                        "name": phase,
                        "packages": [
                            {
                                "name": pkg,
                                "tasks": tasks,
                            }
                            for pkg, tasks in packages.items()
                        ],
                    }
                    for phase, packages in phases.items()
                ],
            }
            return template

        def _fetch_required_task_and_task_details(
            task_list: list,
        ) -> list[dict]:
            """
            Fetch task details from Neo4j and extract required task_duration formula variables.

            Returns:
                - task_records: list of dicts with keys from the Neo4j 't' node
            """
            query = """
            MATCH (:WorkTemplate)-[:HAS_CHILD]->(t:WorkTemplate)
            WHERE t.name IN $task_names
            RETURN t
            """

            records = self.graph.query(query, {"task_names": task_list})

            # Extract the 't' node properties from each record
            task_records = [rec["t"] for rec in records if "t" in rec]

            required_variables: dict[str, list[str]] = {}

            for t in task_records:
                task_duration = t.get("task_duration", "")
                task_name = t.get("name", "")
                # Extract variable placeholders like {volume}, {area}
                # Exclude {productivity} since it comes from the record itself
                variables = re.findall(r"\{(\w+)\}", str(task_duration))
                for var in variables:
                    if var != "productivity":
                        if var not in required_variables:
                            required_variables[var] = []
                        required_variables[var].append(task_name)

            print(f"  Fetched {len(task_records)} task details from Neo4j")
            print(f"  Required variables: {required_variables}")

            return task_records

        def _calculate_task_durations(
            task_records: list[dict],
            per_task_values: dict[str, dict[str, float]],
        ) -> list[dict]:
            """
            Calculate duration_days for each task by evaluating its formula
            with per-task variable values and the task's own productivity.

            Args:
                task_records: list of task dicts from Neo4j
                per_task_values: dict mapping task_name -> {variable_name: value}

            Returns a list of task dicts with computed duration_days.
            """
            computed_tasks = []

            for t in task_records:
                task_name = t.get("name", "")
                task_duration = t.get("task_duration", "")
                productivity = t.get("productivity", 1)
                unit = t.get("unit", "")
                optional = t.get("optional", False)

                # Get this task's specific variable values
                task_vars = per_task_values.get(task_name, {})
                # Add productivity from the Neo4j record
                eval_context = {**task_vars, "productivity": float(productivity)}

                try:
                    # Replace {var} placeholders with actual values for eval
                    expression = task_duration
                    for var_name, var_value in eval_context.items():
                        expression = expression.replace(
                            f"{{{var_name}}}", str(var_value)
                        )

                    # Evaluate the arithmetic expression safely
                    duration = eval(expression)  # e.g. "1000.0/80.0" -> 12.5
                    duration_days = math.ceil(duration)
                    print(
                        f"  {task_name}: {task_duration} -> {expression} = {duration:.2f} -> {duration_days} days"
                    )
                except Exception as e:
                    print(f"  ⚠️ Failed to calculate duration for {task_name}: {e}")
                    duration_days = 1  # Fallback

                computed_tasks.append(
                    {
                        "name": task_name,
                        "duration_days": duration_days,
                        "unit": unit,
                        "productivity": productivity,
                        "task_duration": task_duration,
                        "optional": optional,
                        "dependencies": [],
                        "resources": [],
                    }
                )

            # ── Resolve dependencies from knowledge graph ──
            task_names = [t["name"] for t in computed_tasks]
            try:
                dependency_map = _resolve_task_dependencies(task_names)
                for task in computed_tasks:
                    deps = dependency_map.get(task["name"], [])
                    task["dependencies"] = deps
                    if deps:
                        print(
                            f"  📎 {task['name']} depends on: "
                            + ", ".join(f"{d[0]} ({d[1]}, lag={d[2]})" for d in deps)
                        )
            except Exception as e:
                print(f"  ⚠️ Dependency resolution failed: {e}")
                # Tasks will keep their empty dependencies — scheduler
                # still works (just no inter-task constraints)

            return computed_tasks

        def _fetch_task_dependencies(
            task_names: list[str],
        ) -> dict[str, list]:
            """
            Query direct PRECEDES relationships between tasks that are
            both present in the user's task list.

            Returns:
                dict mapping successor_name -> list of
                [predecessor_name, rel_type, lag]
            """
            query = """
            MATCH (a:WorkTemplate)-[r:PRECEDES]->(b:WorkTemplate)
            WHERE a.name IN $task_names AND b.name IN $task_names
            RETURN a.name AS predecessor, b.name AS successor,
                   r.type AS rel_type, r.lag AS lag
            """
            records = self.graph.query(query, {"task_names": task_names})

            dep_map: dict[str, list] = {}
            for rec in records:
                successor = rec["successor"]
                pred = rec["predecessor"]
                rel_type = rec.get("rel_type", "FS") or "FS"
                lag = int(rec.get("lag", 0) or 0)
                dep_map.setdefault(successor, []).append([pred, rel_type, lag])

            print(
                f"  Found {sum(len(v) for v in dep_map.values())} direct "
                f"PRECEDES relationships among {len(task_names)} tasks"
            )
            return dep_map

        def _resolve_missing_dependencies(
            task_names: list[str],
            direct_dep_map: dict[str, list],
        ) -> dict[str, list]:
            """
            For tasks whose graph predecessors are NOT in task_names,
            traverse backward through PRECEDES chains to find the
            nearest ancestor that IS in the user's task list.

            Uses an LLM call when multiple candidates exist at the
            same hop distance.

            Returns:
                dict mapping successor_name -> list of
                [predecessor_name, rel_type, lag]  (resolved entries only)
            """
            # 1. Find tasks that have a PRECEDES predecessor in the graph
            #    but that predecessor is NOT in our task list.
            query_missing = """
            MATCH (a:WorkTemplate)-[r:PRECEDES]->(b:WorkTemplate)
            WHERE b.name IN $task_names AND NOT a.name IN $task_names
            RETURN b.name AS successor, a.name AS missing_pred,
                   r.type AS rel_type, r.lag AS lag
            """
            records = self.graph.query(query_missing, {"task_names": task_names})

            if not records:
                return {}

            # Collect which successors have missing predecessors
            missing_successors = set()
            for rec in records:
                # Only treat as "missing" if the successor doesn't
                # already have a direct (present) predecessor
                successor = rec["successor"]
                if successor not in direct_dep_map:
                    missing_successors.add(successor)

            if not missing_successors:
                return {}

            print(
                f"  🔍 {len(missing_successors)} task(s) have predecessors "
                f"not in the current list — traversing graph..."
            )

            # 2. For each missing-predecessor task, traverse backward
            resolved: dict[str, list] = {}
            task_names_set = set(task_names)

            for successor in missing_successors:
                query_traverse = """
                MATCH path = (ancestor:WorkTemplate)
                              -[:PRECEDES*1..5]->
                              (target:WorkTemplate {name: $task_name})
                WHERE ancestor.name IN $task_names
                RETURN ancestor.name AS found_predecessor,
                       length(path)   AS hops,
                       [r IN relationships(path) |
                           {type: r.type, lag: r.lag}] AS chain
                ORDER BY hops ASC
                """
                candidates = self.graph.query(
                    query_traverse,
                    {
                        "task_name": successor,
                        "task_names": task_names,
                    },
                )

                if not candidates:
                    print(
                        f"    ⚠️ No reachable predecessor found for "
                        f"'{successor}' within 5 hops"
                    )
                    continue

                # Take the shortest-hop candidates
                min_hops = candidates[0]["hops"]
                best = [c for c in candidates if c["hops"] == min_hops]

                if len(best) == 1:
                    # Unambiguous — use the chain's first relationship
                    # type and cumulative lag
                    chain = best[0]["chain"]
                    rel_type = chain[-1].get("type", "FS") or "FS"
                    total_lag = sum(int(link.get("lag", 0) or 0) for link in chain)
                    resolved.setdefault(successor, []).append(
                        [best[0]["found_predecessor"], rel_type, total_lag]
                    )
                    print(
                        f"    ✔ '{successor}' → resolved to "
                        f"'{best[0]['found_predecessor']}' "
                        f"({rel_type}, lag={total_lag}, "
                        f"{min_hops} hop(s))"
                    )
                else:
                    # Ambiguous — ask LLM to choose
                    selected = _llm_select_dependency(successor, best)
                    resolved.setdefault(successor, []).append(
                        [
                            selected.predecessor,
                            selected.relationship_type,
                            selected.lag,
                        ]
                    )
                    print(
                        f"    🤖 '{successor}' → LLM selected "
                        f"'{selected.predecessor}' "
                        f"({selected.relationship_type}, "
                        f"lag={selected.lag}) — "
                        f"{selected.reasoning}"
                    )

            return resolved

        def _llm_select_dependency(
            task_name: str,
            candidates: list[dict],
        ) -> SelectedDependency:
            """
            Use the LLM to select the best predecessor when graph
            traversal finds multiple candidates at the same distance.
            """
            candidate_descriptions = []
            for c in candidates:
                chain = c["chain"]
                rel_type = chain[-1].get("type", "FS") or "FS"
                total_lag = sum(int(link.get("lag", 0) or 0) for link in chain)
                candidate_descriptions.append(
                    f"- {c['found_predecessor']} "
                    f"(relationship: {rel_type}, "
                    f"cumulative lag: {total_lag} days, "
                    f"hops: {c['hops']})"
                )

            prompt = HumanMessage(
                content=f"""You are a construction scheduling expert.

                The task "{task_name}" needs a predecessor dependency, but its
                immediate predecessor was removed from the schedule.

                After traversing the knowledge graph, the following candidate
                predecessors were found at the same distance:

                {chr(10).join(candidate_descriptions)}

                Select the BEST predecessor for "{task_name}" based on
                construction sequencing logic. Consider which activity must
                logically complete (or start) before "{task_name}" can proceed.

                Return your selection with the relationship type and lag."""
            )

            structured_llm = self.llm.with_structured_output(SelectedDependency)
            result: SelectedDependency = structured_llm.invoke([prompt])  # type: ignore
            return result

        def _resolve_task_dependencies(
            task_names: list[str],
        ) -> dict[str, list]:
            """
            Master function: fetch direct dependencies, then resolve
            any missing predecessors via graph traversal + LLM.

            Returns:
                dict mapping task_name -> list of
                [predecessor, rel_type, lag]
            """
            # Step 1: Direct dependencies (both tasks present)
            direct = _fetch_task_dependencies(task_names)

            # Step 2: Resolve missing predecessors
            resolved = _resolve_missing_dependencies(task_names, direct)

            # Merge resolved into direct
            for successor, deps in resolved.items():
                direct.setdefault(successor, []).extend(deps)

            return direct

        def _adapt_wbs_with_llm(template: dict, user_intent) -> FullProjectWBS:
            """
            Single LLM call: given a reference template tree + user intent,
            produce an adapted FullProjectWBS.
            """
            adapt_msg = HumanMessage(
                content=f"""You are a construction WBS expert.

            Here is a reference Work Breakdown Structure (WBS) template from a similar project:
            {json.dumps(template, indent=2)}

            The user's project details:
            {json.dumps(user_intent, indent=2) if isinstance(user_intent, dict) else user_intent}

            Adapt this WBS for the user's specific project:
            - Remove phases, packages, or tasks that don't apply to this project
            - Keep phases, packages, or tasks that are relevant
            - You may slightly rename tasks if needed to better fit the project type
            - Preserve the hierarchical structure: phases → packages → tasks
            - Pay attention to any special instructions in the user's 'other_details' field (especially those addressed to 'phase_agent')
            - Keep the project_name from the template as-is

            Return the adapted WBS."""
            )

            structured_llm = self.llm.with_structured_output(FullProjectWBS)
            result = structured_llm.invoke([adapt_msg])
            return result  # type: ignore

        def _format_wbs_for_display(wbs_data: dict) -> str:
            """Format a WBS dict (from FullProjectWBS.model_dump()) for user display."""
            lines = [f"📋 Proposed WBS (based on: {wbs_data['project_name']}):"]
            for phase in wbs_data["phases"]:
                lines.append(f"\n  📌 {phase['name']}")
                for pkg in phase.get("packages", []):
                    lines.append(f"    📦 {pkg['name']}")
                    for task in pkg.get("tasks", []):
                        desc = (
                            f" — {task['description']}"
                            if task.get("description")
                            else ""
                        )
                        lines.append(f"      • {task['name']}{desc}")
            return "\n".join(lines)

        def phase_node(state: AgentState) -> AgentState | Command:

            print("\n===== PHASE NODE =====\n")

            interrupted = state["interrupt"]

            if not interrupted:

                sender = state["sender"]

                if sender == "intent_agent":
                    user_intent = state["user_intent"]

                    # ── Step 1: Pick the most similar project (1 LLM call) ──
                    print("Retrieving similar projects...")
                    query1 = (
                        """MATCH (p:WorkTemplate {level: "Project"}) RETURN p.name"""
                    )
                    projects = self.graph.query(query1)

                    similar_project_msg = HumanMessage(
                        content=f"Select the most similar project from {projects} that matches {user_intent}"
                    )
                    structured_llm = self.llm.with_structured_output(SelectedProject)
                    result = structured_llm.invoke([similar_project_msg])  # type: ignore
                    similar_project = result.project  # type: ignore
                    print(f"  Selected template: {similar_project}")

                    # ── Step 2: Fetch full template tree (1 Cypher query) ──
                    print("Fetching full template tree...")
                    template = _fetch_full_template_tree(similar_project)
                    print(f"  Template has {len(template['phases'])} phases")

                    # ── Step 3: Adapt with LLM (1 LLM call) ──
                    print("Adapting WBS for user's project...")
                    adapted_wbs = _adapt_wbs_with_llm(template, user_intent)
                    wbs_data = adapted_wbs.model_dump()
                    print(f"  Adapted WBS has {len(wbs_data['phases'])} phases")

                    # Store adapted WBS in cache and interrupt for user confirmation
                    return Command(
                        update={
                            "sender": "phase_agent",
                            "messages": [],
                            "cache": {"wbs_data": wbs_data},
                            "interrupt": True,
                        },
                        goto="phase_agent",
                    )

                else:
                    # Correction re-entry: user gave feedback, we re-adapt
                    # The last HumanMessage contains the correction request
                    # and the previous WBS is in cache
                    state_cache = state.get("cache") or {}
                    prev_wbs = state_cache.get("wbs_data")
                    user_intent = state["user_intent"]

                    if prev_wbs:
                        # Get the latest user message (correction)
                        last_user_msg = ""
                        for msg in reversed(state["messages"]):
                            if isinstance(msg, HumanMessage):
                                last_user_msg = msg.content
                                break

                        correction_msg = HumanMessage(
                            content=f"""You are a construction WBS expert.

                            Here is the current Work Breakdown Structure (WBS) that was proposed:
                            {json.dumps(prev_wbs, indent=2)}

                            The user's project details:
                            {json.dumps(user_intent, indent=2) if isinstance(user_intent, dict) else user_intent}

                            The user wants the following changes:
                            "{last_user_msg}"

                            Apply the requested changes and return the COMPLETE updated WBS.
                            Preserve the hierarchical structure: phases → packages → tasks."""
                        )

                        structured_llm = self.llm.with_structured_output(FullProjectWBS)
                        result = structured_llm.invoke([correction_msg])
                        wbs_data = result.model_dump()  # type: ignore

                        return Command(
                            update={
                                "sender": "phase_agent",
                                "messages": [],
                                "cache": {"wbs_data": wbs_data},
                                "interrupt": True,
                            },
                            goto="phase_agent",
                        )
                    else:
                        # No previous WBS — shouldn't happen, but fallback
                        return {"messages": [], "interrupt": False, "cache": {}}  # type: ignore

            else:
                # ── Interrupted: show adapted WBS and ask for confirmation ──
                state_cache = state["cache"]

                if state_cache and "wbs_data" in state_cache:
                    wbs_data = state_cache["wbs_data"]

                    display_text = _format_wbs_for_display(wbs_data)

                    user_response = interrupt(
                        f"{display_text}\n\n"
                        "Please confirm:\n"
                        "• Type 'yes' or 'confirm' to proceed\n"
                        "• Type corrections if something needs to be changed\n"
                        "• Type 'cancel' to start over"
                    )
                    print(f"  → Interrupt resumed with response: '{user_response}'")

                    user_response_lower = (
                        user_response.lower().strip() if user_response else ""
                    )

                    if user_response_lower in ["yes", "confirm"]:
                        # ✅ User confirmed — proceed to details agent
                        print("✓ WBS confirmed by user")

                        # Extract phase names for backward compatibility
                        phase_names = [p["name"] for p in wbs_data["phases"]]

                        return Command(
                            update={
                                "messages": [],
                                "sender": "phase_agent",
                                "current_stage": WorkflowStage.DETAILS.value,
                                "phases": phase_names,
                                "project_wbs": wbs_data,
                                "current_phase_index": 0,
                                "cache": {},
                                "interrupt": False,
                            },
                            goto="details_agent",
                        )

                    elif user_response_lower in ["cancel", "start over", "reset"]:
                        # ❌ User cancelled — restart phase gathering
                        print("User cancelled - restarting phase gathering")

                        restart_msg = AIMessage(
                            content="No problem! Let's redo the phases. What changes would you like to make to the project phases?"
                        )

                        return Command(
                            update={
                                "messages": [restart_msg],
                                "sender": "user",
                                "current_stage": WorkflowStage.PHASES.value,
                                "phases": [],
                                "project_wbs": None,
                                "current_phase_index": None,
                                "interrupt": False,
                                "cache": {},
                            },
                            goto=END,
                        )

                    else:
                        # ✏️ User wants corrections — re-adapt via LLM
                        print(f"User requested corrections: {user_response_lower}")

                        correction_msg = HumanMessage(
                            content=f"Please update the WBS based on this feedback: {user_response}"
                        )

                        return Command(
                            update={
                                "messages": [correction_msg],
                                "sender": "user",
                                "current_stage": WorkflowStage.PHASES.value,
                                "interrupt": False,
                                "cache": {
                                    "wbs_data": wbs_data
                                },  # Keep previous WBS for correction
                            },
                            goto="phase_agent",
                        )
                else:
                    return {"messages": [], "interrupt": False, "cache": {}}  # type: ignore

        # Create Details Agent
        DETAIL_AGENT_SYSTEM_PROMPT = """
            You are a construction scheduling assistant.
            - Identify the tasks and dependencies and required resources for this phase.
            - Ask the user if you have any vague tasks.
            """
        details_agent = create_agent(
            system_prompt=DETAIL_AGENT_SYSTEM_PROMPT,
            tools=details_tools,
            model=self.llm,
            response_format=TaskList,
        )

        def details_node(state: AgentState) -> AgentState | Command:
            print("\n===== DETAILS NODE =====\n")

            interrupted = state["interrupt"]
            current_phase_idx = state["current_phase_index"]
            phases = state["phases"]
            user_intent = state["user_intent"]
            generated_tasks = state.get("generated_tasks") or {}
            state_cache = state["cache"]

            if not interrupted:

                # Check if all phases are complete
                if current_phase_idx is None or current_phase_idx >= len(phases):
                    print("All phases complete, moving to scheduling")
                    return Command(
                        update={
                            "current_stage": WorkflowStage.SCHEDULING.value,
                            "sender": "details_agent",
                            "generated_tasks": generated_tasks,
                        }
                    )

                if state_cache and "sub_tasks" in state_cache:
                    current_phase = phases[current_phase_idx]
                    task = state_cache["sub_tasks"]

                    # Extract just the name strings for the Cypher query
                    task_names = [t["name"] if isinstance(t, dict) else t for t in task]

                    # Fetch task details and required variables from Neo4j
                    task_records = _fetch_required_task_and_task_details(task_names)

                    # Build task summary for the LLM to generate smart questions
                    task_summary_lines = []
                    for t in task_records:
                        task_duration = t.get("task_duration", "")
                        variables = re.findall(r"\{(\w+)\}", str(task_duration))
                        non_prod_vars = [v for v in variables if v != "productivity"]
                        if non_prod_vars:
                            task_summary_lines.append(
                                f"- {t['name']}: task_duration='{task_duration}', "
                                f"needs values for: {', '.join(non_prod_vars)}, "
                                f"productivity={t.get('productivity')}, "
                                f"unit={t.get('unit')}"
                            )

                    # LLM call 1: Generate context-aware questions
                    question_prompt = HumanMessage(
                        content=f"""
                                The following tasks in the "{current_phase}" phase need measurements from the user to calculate their durations:

                                {chr(10).join(task_summary_lines)}

                                IMPORTANT: The same variable name (e.g., 'volume') can mean DIFFERENT things for different tasks.
                                For example:
                                - 'volume' for Excavation = volume of earth to excavate
                                - 'volume' for RC Footing Concrete = volume of concrete to pour

                                But:
                                - 'number of foundation' in formwork, reinforcement and others are the same

                                Generate a clear, numbered list of questions asking the user for each required measurement.
                                - Be specific about WHAT is being measured for EACH task
                                - Include the unit expected (m³, m², etc.) based on the task's unit field
                                - Group truly identical values together (e.g., if the same floor area applies to multiple finishes)
                                - **TRY TO GROUP THE REQUIRED VALUES AS MUCH AS POSSIBLE**
                                - Keep it concise and construction-professional
                                - If you GROUP together then do not ask them as seperate question, make it to a single question

                                Return ONLY the numbered questions, nothing else."""  # noqa: E501
                    )

                    llm_questions = self.llm.invoke([question_prompt])
                    question_text = llm_questions.content
                    question_text = f"  LLM generated questions:\n{question_text}"

                    return Command(
                        update=AgentState(
                            {
                                **state,
                                "sender": "details_agent",
                                "messages": [],
                                "cache": {
                                    "awaiting_variables": True,
                                    "task_records": task_records,
                                    "current_phase": current_phase,
                                    "question_text": question_text,
                                    "task_summary_lines": task_summary_lines,
                                },
                                "interrupt": True,
                            }
                        ),
                        goto="details_agent",
                    )

                else:
                    current_phase = phases[current_phase_idx]

                    print(
                        f"Processing phase {current_phase_idx + 1}/{len(phases)}: {current_phase}"
                    )

                    # ── Read tasks from the adapted WBS (set by phase_node) ──
                    project_wbs = state.get("project_wbs") or {}
                    wbs_tasks_for_phase: list[dict] = []

                    for wbs_phase in project_wbs.get("phases", []):
                        if wbs_phase["name"] == current_phase:
                            for pkg in wbs_phase.get("packages", []):
                                for task in pkg.get("tasks", []):
                                    wbs_tasks_for_phase.append(task)
                            break

                    if wbs_tasks_for_phase:
                        task_names_str = ", ".join(
                            t["name"] for t in wbs_tasks_for_phase
                        )
                        print(f"  WBS tasks for {current_phase}: {task_names_str}")
                    else:
                        print(
                            f"  No WBS tasks found for {current_phase}, LLM will generate from scratch"
                        )

                    #         Use the reference tasks as a starting point. For each task, provide:
                    #         - name: task name
                    #         - duration_days: estimated duration in days
                    #         - dependencies: list of [previous_task, relationship_type, lag_days]
                    #         - resources: list of [resource_name, amount]
                    #         """

                    # Store in cache and interrupt for user review
                    return Command(
                        update=AgentState(
                            {
                                **state,
                                "sender": "details_agent",
                                "messages": [],
                                "cache": {
                                    "sub_tasks": wbs_tasks_for_phase,
                                    "current_phase": current_phase,
                                },
                                "interrupt": False,
                            }
                        ),
                        goto="details_agent",
                    )

            else:

                if state_cache and state_cache.get("awaiting_variables"):
                    # ── LLM-powered variable collection ──
                    task_records = state_cache["task_records"]
                    phase_name = state_cache["current_phase"]
                    question_text = state_cache["question_text"]
                    task_summary_lines = state_cache["task_summary_lines"]

                    # Interrupt user with the LLM-generated questions
                    user_response = interrupt(
                        f'📐 To calculate task durations for "{phase_name}":\n\n'
                        f"{question_text}"
                    )
                    print(f"  → User response: '{user_response}'")

                    # LLM call 2: Parse user's free-text response into per-task values
                    parse_prompt = HumanMessage(
                        content=f"""You are a construction data parser.

                                Here are the tasks and the variables each one needs:
                                {chr(10).join(task_summary_lines)}

                                Here are the questions that were asked:
                                {question_text}

                                Here is the user's response:
                                "{user_response}"

                                Extract the numeric values for EACH task's variables from the user's response.
                                Each task needs its OWN variable values — do NOT share values between tasks unless the user explicitly says they are the same.
                                Use the exact task names as listed above."""  # noqa: E501
                    )

                    structured_llm = self.llm.with_structured_output(TaskVariableValues)
                    parsed_values: TaskVariableValues = structured_llm.invoke([parse_prompt])  # type: ignore

                    # Convert to per_task_values dict
                    per_task_values: dict[str, dict[str, float]] = {}
                    for tv in parsed_values.task_values:
                        per_task_values[tv.task_name] = {
                            entry.variable_name: entry.value
                            for entry in tv.variable_entries
                        }

                    print(f"  Parsed per-task values: {per_task_values}")

                    # Calculate durations using per-task values
                    print("  Calculating durations with per-task values...")
                    computed_tasks = _calculate_task_durations(
                        task_records, per_task_values
                    )

                    # Move to pending_tasks confirmation
                    return Command(
                        update=AgentState(
                            {
                                **state,
                                "sender": "details_agent",
                                "messages": [],
                                "cache": {
                                    "pending_tasks": computed_tasks,
                                    "current_phase": phase_name,
                                },
                                "interrupt": True,
                            }
                        ),
                        goto="details_agent",
                    )

                elif state_cache and "pending_tasks" in state_cache:
                    pending_tasks = state_cache["pending_tasks"]
                    phase_name = state_cache["current_phase"]

                    # Build a readable task list for the user
                    if pending_tasks:
                        task_list_str = "\n".join(
                            [
                                f"  {i+1}. {t['name']} ({t['duration_days']} days)"
                                for i, t in enumerate(pending_tasks)
                            ]
                        )
                    else:
                        task_list_str = "  (no tasks generated)"

                    user_response = interrupt(
                        f'📋 Generated Tasks for "{phase_name}" '
                        f"(Phase {current_phase_idx + 1}/{len(phases)}):\n"  # type: ignore
                        f"{task_list_str}\n\n"
                        "Please review:\n"
                        "• Type 'yes' or 'confirm' to accept and move to the next phase\n"
                        "• Type your changes (e.g. 'add a soil testing task', 'remove task 3', 'change duration of task 1 to 5 days')\n"
                        "• Type 'regenerate' to discard and regenerate from scratch"
                    )
                    print(f"  → Interrupt resumed with response: '{user_response}'")

                    user_response_lower = (
                        user_response.lower().strip() if user_response else ""
                    )

                    if user_response_lower in ["yes", "confirm"]:
                        # ✅ User confirmed — save tasks and advance to next phase
                        print(f"✓ Tasks for {phase_name} confirmed by user")

                        generated_tasks[phase_name] = pending_tasks

                        response_msg = AIMessage(
                            content=f"✅ Confirmed {len(pending_tasks)} tasks for {phase_name}."
                        )

                        return Command(
                            update=AgentState(
                                {
                                    **state,
                                    "messages": [response_msg],
                                    "sender": "details_agent",
                                    "current_stage": WorkflowStage.DETAILS.value,
                                    "current_phase_index": current_phase_idx + 1,  # type: ignore
                                    "generated_tasks": generated_tasks,
                                    "interrupt": False,
                                    "cache": {},
                                }
                            ),
                            goto="details_agent",
                        )

                    elif user_response_lower in ["regenerate", "redo", "retry"]:
                        # 🔄 User wants to regenerate from scratch
                        print(f"User requested regeneration for {phase_name}")

                        return Command(
                            update=AgentState(
                                {
                                    **state,
                                    "messages": [],
                                    "sender": "details_agent",
                                    "interrupt": False,
                                    "cache": {},
                                }
                            ),
                            goto="details_agent",
                        )

                    else:
                        # ✏️ User wants edits — re-invoke LLM with feedback
                        print(f"User requested task edits: {user_response}")

                        edit_prompt = f"""
                        You previously generated these tasks for the "{phase_name}" phase:
                        {json.dumps(pending_tasks, indent=2)}

                        The user wants the following changes:
                        "{user_response}"

                        PROJECT DETAILS: {user_intent}

                        Apply the requested changes and return the COMPLETE updated task list with:
                        - name: task name
                        - duration_days: estimated duration
                        - dependencies: list of [previous_task, relationship_type, lag_days]
                        - resources: list of [resource_name, amount]
                        """

                        try:
                            result = self.llm.with_structured_output(TaskList).invoke(
                                edit_prompt
                            )
                            updated_tasks = [task.model_dump() for task in result.tasks]  # type: ignore
                            print(
                                f"Regenerated {len(updated_tasks)} tasks after user edits"
                            )
                        except Exception as e:
                            print(f"Edit regeneration failed: {e}, keeping original")
                            updated_tasks = pending_tasks

                        # Store updated tasks in cache and interrupt again for review
                        return Command(
                            update=AgentState(
                                {
                                    **state,
                                    "messages": [],
                                    "sender": "details_agent",
                                    "cache": {
                                        "pending_tasks": updated_tasks,
                                        "current_phase": phase_name,
                                    },
                                    "interrupt": True,
                                }
                            ),
                            goto="details_agent",
                        )
                else:
                    return AgentState(
                        {**state, "messages": [], "interrupt": False, "cache": {}}
                    )

        def scheduling_node(state: AgentState) -> AgentState | Command:
            print("\n===== SCHEDULING NODE =====\n")

            interrupted = state["interrupt"]
            generated_tasks = state.get("generated_tasks") or {}

            if not interrupted:
                # ── Normal flow: run OR-Tools solver ──
                print("Running OR-Tools CP-SAT solver...")

                try:
                    schedule = solve_schedule(generated_tasks)
                    print(f"Solver produced {len(schedule)} scheduled tasks")

                    if schedule:
                        makespan = max(t["end_day"] for t in schedule)
                        start_date = schedule[0]["start_date"]
                        end_date = max(t["end_date"] for t in schedule)
                    else:
                        makespan = 0
                        start_date = "N/A"
                        end_date = "N/A"

                except Exception as e:
                    print(f"Scheduling failed: {e}")
                    schedule = []
                    makespan = 0
                    start_date = "N/A"
                    end_date = "N/A"

                # Store schedule and interrupt for approval
                return Command(
                    update=AgentState(
                        {
                            **state,
                            "sender": "scheduling_agent",
                            "messages": [],
                            "schedule_result": schedule,
                            "cache": {
                                "makespan": makespan,
                                "start_date": start_date,
                                "end_date": end_date,
                            },
                            "interrupt": True,
                        }
                    ),
                    goto="scheduling_agent",
                )

            else:
                # ── Interrupted: present schedule summary for approval ──
                state_cache = state.get("cache") or {}
                schedule = state.get("schedule_result") or []
                makespan = state_cache.get("makespan", 0)
                sched_start = state_cache.get("start_date", "N/A")
                sched_end = state_cache.get("end_date", "N/A")

                # Build summary by phase
                phase_summaries = {}
                for t in schedule:
                    phase = t["phase"]
                    if phase not in phase_summaries:
                        phase_summaries[phase] = []
                    phase_summaries[phase].append(
                        f"    {t['name']}: {t['start_date']} → {t['end_date']} ({t['duration_days']}d)"
                    )

                summary_lines = []
                for phase, tasks in phase_summaries.items():
                    summary_lines.append(f"  📌 {phase}:")
                    summary_lines.extend(tasks)

                user_response = interrupt(
                    f"📅 Optimised Schedule ({makespan} days total)\n"
                    f"   Start: {sched_start}  →  End: {sched_end}\n\n"
                    + "\n".join(summary_lines)
                    + "\n\nType 'yes' to finalise or provide feedback to adjust."
                )

                user_response_lower = (
                    user_response.lower().strip() if user_response else ""
                )

                if user_response_lower in ["yes", "confirm", "approve"]:
                    print("✓ Schedule approved by user")

                    response_msg = AIMessage(
                        content=f"✅ Schedule finalised! {len(schedule)} tasks over {makespan} days "
                        f"({sched_start} → {sched_end}).\n\n"
                        "The Gantt chart is now available in the side panel."
                    )

                    return Command(
                        update=AgentState(
                            {
                                **state,
                                "messages": [response_msg],
                                "sender": "scheduling_agent",
                                "interrupt": False,
                                "cache": {},
                            }
                        ),
                        goto=END,
                    )
                else:
                    # User wants changes — not yet implemented, just re-approve
                    print(f"User feedback on schedule: {user_response}")

                    feedback_msg = AIMessage(
                        content=f'Noted your feedback: "{user_response}". '
                        "Manual schedule adjustments will be supported in a future update. "
                        "For now, please type 'yes' to approve the current schedule."
                    )

                    return Command(
                        update=AgentState(
                            {
                                **state,
                                "messages": [feedback_msg],
                                "sender": "scheduling_agent",
                                "interrupt": True,
                            }
                        ),
                        goto="scheduling_agent",
                    )

        # Build the workflow with proper state
        workflow = StateGraph(AgentState)

        workflow.add_node("intent_agent", intent_node)
        workflow.add_node("phase_agent", phase_node)
        workflow.add_node("details_agent", details_node)
        workflow.add_node("scheduling_agent", scheduling_node)

        workflow.add_conditional_edges(
            START,
            agent_router,
            {
                "intent_agent": "intent_agent",
                "phase_agent": "phase_agent",
                "details_agent": "details_agent",
                "END": END,
            },
        )

        # Add EXIT edges from each agent
        def intent_exit_router(state: AgentState):
            return (
                "phase_agent"
                if state.get("current_stage", WorkflowStage.INTENT.value)
                == WorkflowStage.PHASES.value
                else END
            )

        def phase_exit_router(state: AgentState):
            return (
                "details_agent"
                if state["current_stage"] == WorkflowStage.DETAILS.value
                else END
            )

        def details_exit_router(state: AgentState):
            return (
                "scheduling_agent"
                if state["current_stage"] == WorkflowStage.SCHEDULING.value
                else END
            )

        workflow.add_conditional_edges("intent_agent", intent_exit_router)
        workflow.add_conditional_edges("phase_agent", phase_exit_router)
        workflow.add_conditional_edges("details_agent", details_exit_router)
        workflow.add_edge("scheduling_agent", END)

        return workflow.compile(checkpointer=MemorySaver())

    def extract_interrupt_message(self, state):
        """Helper to extract interrupt message from state"""
        if state.tasks:
            for task in state.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    for item in task.interrupts:
                        if isinstance(item, dict) and "value" in item:
                            return item["value"]
        return None

    def _visualize_graph(self):
        """Visualize the graph and save as PNG"""
        try:
            graph = self.workflow.get_graph(xray=True)
            png_data = graph.draw_mermaid_png()
            with open("workflow_graph.png", "wb") as f:
                f.write(png_data)
            print("✅ Graph visualization saved as 'workflow_graph.png'")
        except Exception as e:
            print(f"⚠️  Could not generate graph visualization: {e}")


if __name__ == "__main__":
    model = AgenticSchedulerModel()
    # model.chat_with_model()
