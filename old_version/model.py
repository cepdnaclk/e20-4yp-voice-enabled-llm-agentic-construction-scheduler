import os
import json
from dotenv import load_dotenv
from typing import Annotated, Sequence, TypedDict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage
from langchain_core.messages import SystemMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage
from fyp.src.tools import tools_list
from fyp.src.core.scheduler import ConstructionScheduler


load_dotenv()


class ChatState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


class AgenticSchedulerModel:

    # Initialize Scheduler which will persist across all tool calls
    scheduler = ConstructionScheduler()
    current_task = {}
    ask_from_user = True

    def __init__(self):
        self.agent_llm = ChatOpenAI(
            model=os.getenv("MODEL") or "",
            api_key=SecretStr(os.getenv("OPENAI_API_KEY") or ""),
            # base_url="https://openrouter.ai/api/v1",
        ).bind_tools(tools=tools_list)

        workflow = StateGraph(ChatState)
        workflow.add_node("user_input", self.get_user_message)
        workflow.add_node("agent", self.model_call)
        workflow.set_entry_point("user_input")
        workflow.add_edge("user_input", "agent")
        workflow.add_node("tools", ToolNode(tools_list))
        workflow.add_edge("agent", "tools")
        workflow.add_conditional_edges(
            "agent",
            self.should_i_ask_user,
            {"continue_to_tools": "tools", "continue_to_userinput": "user_input"},
        )

        workflow.add_node("display", self.display_gantt_chart_call)
        workflow.add_conditional_edges(
            "tools",
            self.should_continue,
            {"continue_to_model": "agent", "end": "display"},
        )
        self.chatbot = workflow.compile()

    def get_user_message(self, state: ChatState) -> ChatState:
        user_input = input("\n👤 You: ").strip()
        return {"messages": [HumanMessage(content=user_input)]}

    def model_call(self, state: ChatState) -> ChatState:
        """This node will understand the input and decide to use tools"""

        # Strong instruction to use the extraction tool
        system_message = SystemMessage(
            content="""
                    You are a construction scheduling specialist. Help with extracting task and scheduling them.
                    
                    - You can use the Extract Tasks tool and Schedule tool for this purpose
                    - Do the next task sequence only if requested
                    - If not sure what to do next ask the user
                    """
        )

        new_message = [system_message] + list(state["messages"])

        print("\n🔄 Processing...")
        response = self.agent_llm.invoke(new_message)

        if (
            isinstance(response, AIMessage)
            and "tool_calls" in response.additional_kwargs
        ):
            self.ask_from_user = False
        return {"messages": [response]}

    def display_gantt_chart_call(self, state: ChatState) -> ChatState:
        print("\nVisualizing gantt chart")
        print(self.current_task)
        self.scheduler.display_gantt_chart(self.current_task)
        return state

    def should_continue(self, state: ChatState):
        messages = state["messages"]

        self.ask_from_user = True # Reset the ask user bool
        if not messages:
            return "continue_to_model"

        for message in reversed(messages):
            if (
                isinstance(message, ToolMessage)
                and isinstance(message.content, str)
                and "successfully" in message.content.lower()
                and "scheduled" in message.content.lower()
            ):
                return "end"

        return "continue_to_model"

    def should_i_ask_user(self, state: ChatState):
        if not self.ask_from_user:
            return "continue_to_tools"
        return "continue_to_userinput"

    def chat_with_model(self):
        user_input = []
        input_msg = [HumanMessage(content=user_input)]
        chatstate = ChatState(messages=input_msg)

        last_printed_content = None
        # Use values mode to avoid threading issues
        for s in self.chatbot.stream(chatstate, stream_mode="values"):
            message = s["messages"][-1]
            if isinstance(message, AIMessage) and not message.tool_calls:
                current_content = message.content
                # Only print if this is different from what we last printed
                if current_content != last_printed_content:
                    print(f"\n🤖 Assistant: {current_content}\n")
                    last_printed_content = current_content
            elif isinstance(message, AIMessage) and message.tool_calls:
                tool_call = message.tool_calls[0]
                print(f"\n🛠️ Agent decided to use: {tool_call['name']}")
            elif isinstance(message, ToolMessage):
                print(f"\n✅ Tool result: {message.content}")

    def start_chat_session(self):
        """Start an interactive terminal chat session"""
        print("🚀 AI Assistant Started!")
        print("💡 Available functions: Construction scheduling")
        print("❌ Type 'exit', 'quit', or 'bye' to end the session")
        print("-" * 60)

        self.chat_with_model()
