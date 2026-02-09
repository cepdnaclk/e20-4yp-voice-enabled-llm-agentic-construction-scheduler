from pydantic.fields import Field
from typing import Optional
import os
import operator
import uuid
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
from langchain.agents import create_agent
from src.tools import setup_tools, phase_tools, details_tools, intent_tools
from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_neo4j.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_openai import OpenAIEmbeddings
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

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
    user_intent: Optional[str]
    current_phase_index: Optional[int]


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


class Resource(BaseModel):
    """Represents a required resource"""

    name: str = Field(..., description="Resource name")
    amount: float = Field(..., description="Required amount/quantity")

    @classmethod
    def from_list(cls, res_list: List) -> "Resource":
        """Create from [resource_name, amount] format"""
        return cls(name=res_list[0], amount=res_list[1])


class Task(BaseModel):
    """Represents a single task in the project schedule"""

    name: str = Field(..., description="Task name")
    duration_days: int = Field(..., gt=0, description="Duration in days")
    dependencies: List[List] = Field(
        default_factory=list,
        description="List of dependencies as [previous_task, relationship, lag]",
    )
    resources: List[List] = Field(
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


class AgenticSchedulerModel:

    def __init__(self):
        setup_tools(self)
        print(os.getenv("NEO4J_URI"))
        self.llm = ChatOpenAI(
            model=os.getenv("MODEL") or "",
            api_key=SecretStr(os.getenv("OPENAI_API_KEY") or ""),
        )

        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI") or "",
            username=os.getenv("NEO4J_USERNAME") or "",
            password=os.getenv("NEO4J_PASSWORD") or "",
            database=os.getenv("NEO4J_DATABASE") or "",
        )

        self.vector_index = Neo4jVector.from_existing_graph(
            OpenAIEmbeddings(),
            url=os.getenv("NEO4J_URI") or "",
            username=os.getenv("NEO4J_USERNAME") or "",
            password=os.getenv("NEO4J_PASSWORD") or "",
            index_name="task",
            node_label="Subtask",
            text_node_properties=["name", "description"],
            embedding_node_property="embedding",
        )

        # TODO find better alternative to allow_dangerous_requests=True
        self.cypher_chain = GraphCypherQAChain.from_llm(
            llm=self.llm, graph=self.graph, verbose=True, allow_dangerous_requests=True
        )

        # TODO vectorizing the graph
        retriever = self.vector_index.as_retriever()

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

        self.chain = create_retrieval_chain(retriever, document_chain)

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
        - Special requirements (materials, sustainability, design preferences)
        - Timeline preferences
        - Budget range

        **IMPORTANT INSTRUCTIONS:**
        1. Start with a friendly greeting if the user says "hi" or "hello"
        2. Ask questions naturally, one or two at a time - don't overwhelm the user
        3. Listen to what the user provides and adapt your questions
        4. Only call the submit_construction_intent tool when you have enough information
        5. If the user gives vague answers, ask for clarification before submitting

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

            current_stage = state["current_stage"]

            if current_stage == WorkflowStage.INTENT.value:
                return "intent_agent"
            elif current_stage == WorkflowStage.PHASES.value:
                return "phase_agent"
            elif current_stage == WorkflowStage.DETAILS.value:
                return "details_agent"
            else:
                return "phase_agent"

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

            result = intent_agent.invoke(state)  # type: ignore
            messages = result["messages"]
            last_message = messages[-1]

            if (
                last_message
                and hasattr(last_message, "content")
                and last_message.content
            ):
                print(f"\n🤖 Assistant: {last_message.content}")

            intent_phase_tool_calls = extract_toolcall(
                messages, "submit_construction_intent"
            )

            if intent_phase_tool_calls:
                tool_data = intent_phase_tool_calls[-1]  # Get the last tool call
                tool_call = tool_data["tool_call"]
                messages = tool_data["messages"]
                user_response_lower = None
                structured_intent = None

                if tool_call["name"] == "submit_construction_intent":

                    args = tool_call["args"]

                    structured_intent = ConstructionIntent(
                        project_type=args["project_type"],
                        building_category=args["building_category"],
                        size={"value": args["size_value"], "unit": args["size_unit"]},
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
                    )

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

                if user_response_lower in [
                    "yes",
                    "confirm",
                ]:
                    print(f"✓ Intent confirmed by user")

                    msg = None
                    user_intent = None
                    if structured_intent:
                        user_intent = structured_intent.model_dump()
                        msg = f"List the major tasks in construction of {user_intent}"

                    return Command(
                        update={
                            "messages": [],
                            "sender": "intent_agent",
                            "current_stage": WorkflowStage.PHASES.value,
                            "user_intent": user_intent,
                            "phases": [],
                            "current_phase_index": None,
                        },
                    )

                elif user_response_lower in ["cancel", "start over", "reset"]:
                    print("User cancelled - restarting intent gathering")

                    # Add a system message explaining the restart
                    restart_msg = AIMessage(
                        content="No problem! Let's start fresh. What type of construction project would you like to plan?"
                    )

                    # Reset the DATA but keep conversation history
                    return {
                        "messages": messages + [restart_msg],  # Add restart message
                        "sender": "intent_agent",
                        "current_stage": WorkflowStage.INTENT.value,
                        "user_intent": None,  # Clear structured data
                        "phases": [],
                        "current_phase_index": None,
                    }

                else:
                    # User wants corrections
                    print(f"User requested corrections: {user_response_lower}")
                    from langchain_core.messages import HumanMessage, ToolMessage

                    # Send tool result and user correction
                    tool_msg = ToolMessage(
                        content="User requested changes before confirmation",
                        tool_call_id=tool_call["id"],
                    )
                    correction_msg = HumanMessage(
                        content=f"Please update the project details based on this feedback: {user_response_lower}"
                    )

                    return {
                        "messages": messages + [tool_msg, correction_msg],
                        "sender": "intent_agent",
                        "current_stage": WorkflowStage.INTENT.value,
                        "phases": state.get("phases", None),
                        "user_intent": None,
                        "current_phase_index": None,
                    }

            else:
                return {
                    "messages": [last_message],
                    "sender": "intent_agent",
                    "current_stage": WorkflowStage.INTENT.value,
                    "phases": state.get("phases", None),
                    "user_intent": None,
                    "current_phase_index": None,
                }  # type: ignore

        # Create Phase Agent
        PHASE_AGENT_SYSTEM_PROMPT = """
            You are a construction scheduling assistant. 
            - List the major phases for the project based on the user's intent.
            - No pre-construction phases are required (except Site Prep).
            - Ask the user to confirm if these phases look correct.
            - Don't List any task or subtasks in the phases. Only the major phases
            - Use the 'confirm_phases' tool when the user agrees with the phases.
            """

        phase_agent = create_agent(
            system_prompt=PHASE_AGENT_SYSTEM_PROMPT,
            tools=phase_tools,
            model=self.llm,
        )

        def phase_node(state: AgentState) -> AgentState | Command:

            print("\n===== PHASE NODE =====\n")

            sender = state["sender"]

            if sender == "intent_agent":
                user_intent = state["user_intent"]
                initial_msg = HumanMessage(
                    content=f"List the major tasks in construction of {user_intent}"
                )
                result = phase_agent.invoke({"messages": initial_msg})  # type: ignore
                messages = result["messages"]
                last_message = messages[-1]

                if hasattr(last_message, "content") and last_message.content:
                    print(f"\n🤖 Assistant: {last_message.content}")

                return {
                    "messages": [last_message],
                    "sender": "phase_agent",
                    "current_stage": WorkflowStage.PHASES.value,
                    "phases": [],
                    "user_intent": state.get("user_intent", ""),
                    "current_phase_index": None,
                }

            result = phase_agent.invoke(state)  # type: ignore
            messages = result["messages"]
            last_message = messages[-1]

            if hasattr(last_message, "content") and last_message.content:
                print(f"\n🤖 Assistant: {last_message.content}")

            phases_tool_calls = extract_toolcall(messages, "confirm_phases")

            if phases_tool_calls:
                tool_data = phases_tool_calls[-1]  # Get the last tool call
                tool_call = tool_data["tool_call"]
                messages = tool_data["messages"]
                user_response_lower = None
                structured_intent = None

                if tool_call["name"] == "confirm_phases":

                    phases_list = tool_call["args"]["phases_list"]
                    phases = [p.strip() for p in phases_list.split(",")]
                    # Interrupt for Phase Confirmation
                    user_response = interrupt(f"Do you approve these phases?")

                    user_response_lower = (
                        user_response.lower().strip() if user_response else ""
                    )

                    if user_response_lower in ["yes", "confirm"]:
                        print(f"✓ Phases confirmed by user")

                        return Command(
                            update={
                                "messages": [last_message],
                                "sender": "phase_agent",
                                "current_stage": WorkflowStage.DETAILS.value,
                                "phases": phases,
                                "user_intent": state.get("user_intent", ""),
                                "current_phase_index": 0,  # Reset to first phase
                            }
                        )

            # No tool call, return state as-is

            return {
                "messages": [last_message],
                "sender": "phase_agent",
                "current_stage": WorkflowStage.PHASES.value,
                "phases": state.get("phases", None),
                "user_intent": state.get("user_intent", ""),
                "current_phase_index": None,
            }

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

        def details_node(state: AgentState) -> AgentState | dict:

            print("\n===== DETAILS NODE =====\n")

            current_phase = state["current_phase_index"]
            phases = state["phases"]
            lastmsg = state["messages"][-1]

            print(f"\ncurrent_phase: {current_phase}\n")
            print(phases)

            if lastmsg.content:
                result1 = self.cypher_chain.invoke({"query": lastmsg.content})
                print("cypher: ", result1)

                result = self.chain.invoke({"input": lastmsg.content})

                print("vector: ", result["answer"])

                result2 = self.llm.invoke((f"{lastmsg.content},be brief"))
                print("llm: ", result2.content)

            # while (not (current_phase == None)) and current_phase < len(phases):
            #     phases = state["phases"]
            #     msg = SystemMessage(
            #         content=f"You are currently detailing the major construction phases [{phases[current_phase]}]"
            #     )
            #     result = details_agent.invoke({"messages": msg})  # type: ignore
            #     messages = result["messages"]
            #     last_message = messages[-1]

            #     if hasattr(last_message, "content") and last_message.content:
            #         print(f"\n🤖 Assistant: {last_message.content}")

            #     return {
            #         "messages": [last_message],
            #         "sender": "details_agent",
            #         "current_stage": WorkflowStage.DETAILS.value,
            #         "phases": state.get("phases", []),
            #         "user_intent": state.get("user_intent", ""),
            #         "current_phase_index": current_phase + 1,
            #     }

            return state

        def scheduling_node(state: AgentState) -> AgentState:
            final_approval = interrupt("Requesting final schedule approval.")
            print(f"Final Schedule Approval: {final_approval}")
            return state

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
                "details_agent": "details_agent",  # Add if needed
                "END": END,  # Add option to end directly if needed
            },
        )

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

    def chat_with_model(self):
        print("🚀 AI Assistant Started!")
        print("-" * 60)

        # Create a thread ID
        thread_id = str(uuid.uuid4())
        config = RunnableConfig(
            configurable={"thread_id": thread_id}, recursion_limit=50
        )

        print(f"Thread ID: {thread_id}")

        # START with initial state ONLY ONCE
        initial_state: AgentState = {
            "messages": [HumanMessage(content="", config=config)],
            "sender": "user",
            "current_stage": WorkflowStage.DETAILS.value,  # TODO this should be INTENT value
            "phases": [],
            "user_intent": None,
            "current_phase_index": None,
        }

        # Initial invoke with empty state
        self.workflow.invoke(initial_state, config=config)

        while True:
            input_from_interrupt = False
            user_input = None
            try:
                # Check for interrupts FIRST
                state_snapshot = self.workflow.get_state(config)

                if state_snapshot.tasks:
                    for task in state_snapshot.tasks:
                        if hasattr(task, "interrupts") and task.interrupts:
                            for item in task.interrupts:
                                print(f"\n🤖 Assistant: {item.value}")

                                # Get user response to interrupt
                                input_from_interrupt = True
                                user_input = input("\n👤 You: ").strip()

                                # Resume with JUST the response
                                self.workflow.invoke(
                                    Command(resume=user_input), config=config
                                )
                                continue

                # No interrupt? Get normal user input
                if not input_from_interrupt:
                    user_input = input("\n👤 You: ").strip()
                    input_from_interrupt = False

                if user_input and user_input.lower() in ["exit", "quit", "bye"]:
                    print("👋 Goodbye!")
                    break

                # Send ONLY the new message, not full state
                self.workflow.invoke(
                    {"messages": [HumanMessage(content=user_input)]},  # type: ignore
                    config=config,
                )

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break

            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback

                traceback.print_exc()
                break

        print("👋 Goodbye!")

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
    model.chat_with_model()
