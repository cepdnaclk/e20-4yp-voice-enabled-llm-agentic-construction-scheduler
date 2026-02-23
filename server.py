"""
FastAPI server for the Multi-Agent Construction Scheduler.
Exposes the AgenticSchedulerModel via HTTP endpoints with SSE streaming.
"""

import uuid
import json
import asyncio
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel as PydanticBaseModel

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig

from src.model import AgenticSchedulerModel, WorkflowStage, AgentState

app = FastAPI(title="Construction Scheduler API")

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global model instance ---
model_instance: Optional[AgenticSchedulerModel] = None


def get_model() -> AgenticSchedulerModel:
    global model_instance
    if model_instance is None:
        model_instance = AgenticSchedulerModel()
    return model_instance


# --- Request/Response Models ---


class StartRequest(PydanticBaseModel):
    pass


class MessageRequest(PydanticBaseModel):
    thread_id: str
    message: str


class ResumeRequest(PydanticBaseModel):
    thread_id: str
    response: str


# --- SSE Helper ---


def sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event"""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# --- Endpoints ---


@app.post("/api/chat/start")
async def start_chat():
    """Create a new chat session and get initial AI greeting."""
    model = get_model()
    thread_id = str(uuid.uuid4())
    config = RunnableConfig(configurable={"thread_id": thread_id}, recursion_limit=50)

    initial_state: AgentState = {
        "messages": [SystemMessage(content="")],
        "sender": "user",
        "current_stage": WorkflowStage.INTENT.value,
        "phases": [],
        "user_intent": None,
        "current_phase_index": None,
        "generated_tasks": {},
        "interrupt": False,
        "cache": {},
    }

    # Run initial invoke in a thread to not block the event loop
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, lambda: model.workflow.invoke(initial_state, config=config)
    )

    # Get state to extract AI message
    state_snapshot = model.workflow.get_state(config)
    state_values = state_snapshot.values

    # Extract the last AI message
    ai_message = ""
    interrupt_value = None
    messages = state_values.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            ai_message = msg.content
            break

    return {
        "thread_id": thread_id,
        "message": ai_message,
        "interrupt": interrupt_value,
        "stage": state_values.get("current_stage", "intent"),
        "phases": state_values.get("phases", []),
        "generated_tasks": state_values.get("generated_tasks", {}),
    }


@app.post("/api/chat/message")
async def send_message(request: MessageRequest):
    """Send a user message and stream the AI response via SSE."""
    model = get_model()
    config = {"configurable": {"thread_id": request.thread_id}, "recursion_limit": 50}

    async def event_stream():
        loop = asyncio.get_event_loop()

        try:
            # Send user message
            result = await loop.run_in_executor(
                None,
                lambda: model.workflow.invoke(
                    {"messages": [HumanMessage(content=request.message)]},  # type: ignore
                    config=config,  # type: ignore
                ),
            )

            # ✅ CHECK FOR INTERRUPTS FIRST (from invoke result)
            state_snapshot = model.workflow.get_state(config)  # type: ignore
            interrupt_value = None
            if state_snapshot.tasks:
                for task in state_snapshot.tasks:
                    if hasattr(task, "interrupts") and task.interrupts:
                        for item in task.interrupts:
                            interrupt_value = item.value

            if interrupt_value:
                # Stream the interrupt message as chat message chunks
                chunk_size = 3
                for i in range(0, len(str(interrupt_value)), chunk_size):
                    chunk = str(interrupt_value)[i : i + chunk_size]
                    yield sse_event("message", {"chunk": chunk})
                    await asyncio.sleep(0.02)

                yield sse_event("interrupt", {"value": interrupt_value})
                yield sse_event("done", {"status": "interrupted"})
                return

            # Get state values (reuse state_snapshot from interrupt check above)
            state_values = state_snapshot.values

            # Extract the last AI message
            ai_message = ""
            messages = state_values.get("messages", [])
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    ai_message = msg.content
                    break

            # Stream the message character by character for typewriter effect
            chunk_size = 3  # Send 3 characters at a time for smooth streaming
            for i in range(0, len(ai_message), chunk_size):
                chunk = ai_message[i : i + chunk_size]
                yield sse_event("message", {"chunk": chunk})
                await asyncio.sleep(0.02)  # 20ms delay between chunks

            # Check for stage changes
            current_stage = state_values.get("current_stage", "intent")
            yield sse_event("stage_change", {"stage": current_stage})

            # Check for generated tasks (detailing phase)
            generated_tasks = state_values.get("generated_tasks", {})
            if generated_tasks:
                yield sse_event("tasks", {"tasks": generated_tasks})

            # Check for phases
            phases = state_values.get("phases", [])
            if phases:
                yield sse_event("phases", {"phases": phases})

            yield sse_event("done", {"status": "complete"})

        except Exception as e:
            yield sse_event("error", {"message": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat/resume")
async def resume_from_interrupt(request: ResumeRequest):
    """Resume the workflow from an interrupt (e.g., confirming intent/phases)."""
    model = get_model()
    config = {"configurable": {"thread_id": request.thread_id}, "recursion_limit": 50}

    async def event_stream():
        loop = asyncio.get_event_loop()

        try:
            # Resume with user response
            await loop.run_in_executor(
                None,
                lambda: model.workflow.invoke(
                    Command(resume=request.response),
                    config=config,  # type: ignore
                ),
            )

            # Get updated state
            state_snapshot = model.workflow.get_state(config)  # type: ignore

            # Check for new interrupts
            interrupt_value = None
            if state_snapshot.tasks:
                for task in state_snapshot.tasks:
                    if hasattr(task, "interrupts") and task.interrupts:
                        for item in task.interrupts:
                            interrupt_value = item.value

            if interrupt_value:
                # Stream the interrupt message as chat message chunks
                chunk_size = 3
                for i in range(0, len(str(interrupt_value)), chunk_size):
                    chunk = str(interrupt_value)[i : i + chunk_size]
                    yield sse_event("message", {"chunk": chunk})
                    await asyncio.sleep(0.02)

                yield sse_event("interrupt", {"value": interrupt_value})
                yield sse_event("done", {"status": "complete"})
                return

            state_values = state_snapshot.values

            # Extract AI messages added after resume
            ai_message = ""
            messages = state_values.get("messages", [])
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    ai_message = msg.content
                    break

            # Stream the message
            if ai_message:
                chunk_size = 3
                for i in range(0, len(ai_message), chunk_size):
                    chunk = ai_message[i : i + chunk_size]
                    yield sse_event("message", {"chunk": chunk})
                    await asyncio.sleep(0.02)

            # Stage change
            current_stage = state_values.get("current_stage", "intent")
            yield sse_event("stage_change", {"stage": current_stage})

            # Generated tasks
            generated_tasks = state_values.get("generated_tasks", {})
            if generated_tasks:
                yield sse_event("tasks", {"tasks": generated_tasks})

            # Phases
            phases = state_values.get("phases", [])
            if phases:
                yield sse_event("phases", {"phases": phases})

            yield sse_event("done", {"status": "complete"})

        except Exception as e:
            yield sse_event("error", {"message": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/chat/state")
async def get_state(thread_id: str):
    """Get the current workflow state."""
    model = get_model()
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}

    try:
        state_snapshot = model.workflow.get_state(config)  # type: ignore
        state_values = state_snapshot.values

        if not state_values:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "stage": state_values.get("current_stage", "intent"),
            "phases": state_values.get("phases", []),
            "generated_tasks": state_values.get("generated_tasks", {}),
            "current_phase_index": state_values.get("current_phase_index"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
