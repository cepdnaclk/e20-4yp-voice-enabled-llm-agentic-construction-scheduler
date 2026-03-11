"""
Main entry point for Construction Scheduler FYP
"""

from fyp.src.core.scheduler import ConstructionScheduler
from fyp.src.llm.natural_language import ConstructionLLM
from fyp.src.model import AgenticSchedulerModel
from fyp.src.tools import setup_tools

def main():
    # """Launch the construction scheduler application"""
    print("Construction Scheduler FYP - Starting...")

    scheduler = ConstructionScheduler()
    llm = ConstructionLLM()
    agentic_model = AgenticSchedulerModel()
    setup_tools(agentic_model)

    # retrieve the LLM task messages
    # llm_result = llm.parse_with_gpt(
    #     """
        # Site Preparation takes 3 days and has no dependencies.

        # Foundation takes 5 days and starts after Site Preparation finishes.

        # Framing takes 7 days and starts after Foundation finishes.

        # Roofing takes 4 days and starts after Framing finishes.

        # Plumbing Rough-In takes 4 days and starts 1 day after Framing starts.

        # Electrical Rough-In takes 3 days and starts 2 days after Framing starts.

        # HVAC Installation takes 4 days and starts 3 days after Framing starts.

        # Exterior Finishing takes 5 days and starts after Roofing finishes.

        # Insulation takes 2 days and starts after Plumbing, Electrical, and HVAC have all finished.

        # Drywall takes 4 days and starts after Insulation finishes.

        # Interior Painting takes 3 days and finishes 1 day after Drywall finishes.

        # Flooring takes 3 days and starts after Interior Painting finishes.

        # Fixtures Installation takes 2 days and starts after Flooring finishes.

        # Landscaping takes 4 days and starts after Exterior Finishing finishes.

        # Final Inspection takes 1 day and starts after Fixtures Installation finishes and finishes 1 day after Landscaping finishes.
    #     """
    # )

    # if not llm_result:
    #     print("llm result is Null")
    #     return

    # print(f"📋 Extracted {len(llm_result['tasks'])} tasks")

    # # adding task to scheduler

    # for task_data in llm_result["tasks"]:
    #     print(task_data)
    #     task = scheduler.add_task(
    #         task_data["name"], task_data["duration_days"], task_data["dependencies"]
    #     )

    # scheduler.create_variables()
    # scheduled_task, makespan = scheduler.add_dependencies()
    # print(scheduled_task)
    # scheduler.display_gantt_chart(scheduled_task)

    agentic_model.start_chat_session()


if __name__ == "__main__":
    main()
