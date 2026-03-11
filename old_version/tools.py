import os
import json
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from typing import Any, Dict, Optional, Tuple

# This will be set by main.py
agentic_model = None


def setup_tools(model_instance):
    """Set the global model instance for tools to access"""
    from fyp.src.model import AgenticSchedulerModel

    model: AgenticSchedulerModel = model_instance
    global agentic_model
    agentic_model = model


@tool
def extract_tasks_and_dependencies(user_input: str) -> str:
    """Extract construction tasks, durations, and dependencies from natural language descriptions. Use this for any construction scheduling input."""

    system_message = """
        You are a strict task extraction engine for a construction scheduler.

        Your job:
        - Extract construction tasks from text.
        - Return ONLY valid JSON in this format:
        - A task name in dependency SHOULD EXIST in the tasks list if not change name in dependancy to match it. COMPULSORY.

        {
            "tasks": [
                {
                    "name": "task name",
                    "duration_days": number,
                    "dependencies": [
                        ["previous_task_name", "relationship_link", "lag_or_lead_by"]
                    ]
                }
            ]
        }

        Examples:
        - "Electrical Rough-In takes 3 days and starts 2 days after Framing starts."
        - "HVAC Installation takes 4 days and starts 3 days after Framing starts."
        - "Exterior Finishing takes 5 days and starts after Roofing finishes."
        - "Insulation takes 2 days and starts after Plumbing, Electrical, and HVAC have all finished."
        → {"tasks": [
                {"name": "Electrical Rough-In", "duration_days": 3, "dependencies": [["Framing","SS",2]]},
                {"name": "HVAC Installation", "duration_days": 4, "dependencies": [["Framing","SS",3]]},
                {"name": "Exterior Finishing", "duration_days": 5, "dependencies": [["Roofing","FS",0]]},
                {"name": "Insulation", "duration_days": 2, "dependencies": [["Plumbing Rough-In","FS",0],["Electrical Rough-In","FS",0],["HVAC Installation","FS",0]]}
            ]}

        
        """

    prompt = f"""
        Extract construction tasks from this description:
        "{user_input}"
        """
    try:

        json_llm = ChatOpenAI(
            model=os.getenv("MODEL") or "",
            api_key=SecretStr(os.getenv("OPENAI_API_KEY") or ""),
            # base_url="https://openrouter.ai/api/v1",
        )

        response = json_llm.invoke(
            [SystemMessage(content=system_message), HumanMessage(content=prompt)],
            response_format={"type": "json_object"},
        )

        # Always decode from content as response_format does not provide parsed output
        content = response.content
        parsed_data = None
        if content is not None:
            # Ensure content is a string before parsing
            if isinstance(content, str):
                # Remove the outer single quotes if present

                if content.startswith("'") and content.endswith("'"):
                    content = content[1:-1]

                parsed_data = json.loads(content)
            else:
                raise ValueError("Unexpected response content format.")

        if not parsed_data:
            tool_message = "No content returned from LLM response."
            return tool_message

        for task in parsed_data["tasks"]:
            # Convert dependency lists to tuples
            task["dependencies"] = [tuple(dep) for dep in task["dependencies"]]
        tool_message = f"📋 Extracted {len(parsed_data['tasks'])} tasks"
        # print(parsed_data)

        if (
            agentic_model is None
            or not hasattr(agentic_model, "scheduler")
            or agentic_model.scheduler is None
        ):
            return "Error: Model not initialized"

        agentic_model.current_task = parsed_data
        return tool_message

    except json.JSONDecodeError as e:
        tool_message = f"JSON parsing error: {e}"
        return tool_message

    except Exception as e:
        tool_message = f"LLM parsing failed: {e}"
        return tool_message


@tool
def add_tasks_to_scheduler(tasks: str) -> str:
    """
    This is a task scheduler. If list of task is given then this will find the optimized schedule
    and visualize in a Gantt Chart
    """

    if (
        agentic_model is None
        or not hasattr(agentic_model, "scheduler")
        or agentic_model.scheduler is None
    ):
        return "Error: Model not initialized"

    scheduler_model = agentic_model.scheduler

    # print(tasks)
    current_task = agentic_model.current_task
    if not (current_task):
        raise ValueError("Tasks is empty")
    try:
        tasks_data = current_task
        for task_data in tasks_data["tasks"]:
            task = scheduler_model.add_task(
                task_data["name"], task_data["duration_days"], task_data["dependencies"]
            )
            print(f"✅ Added: {task_data['name']} ({task_data['duration_days']} days)")

        scheduler_model.create_variables()
        scheduled_task, makespan = scheduler_model.add_dependencies()
        agentic_model.current_task = {"tasks": scheduled_task}
        # print(scheduled_task)
        return f"Successfully scheduled {len(tasks_data['tasks'])} tasks. Makespan: {makespan} days"

    except Exception as e:
        return f"Error adding tasks: {str(e)}"


tools_list = [extract_tasks_and_dependencies, add_tasks_to_scheduler]
