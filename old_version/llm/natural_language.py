"""
Natural language processing for construction scheduling
"""

from typing import Any, Dict, Optional
import ollama
import json
from openai import OpenAI
from dotenv import load_dotenv
import os


class ConstructionLLM:

    def __init__(self):
        load_dotenv()
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.model = "phi3"

    def parse_with_gpt(self, user_input: str) -> Optional[Dict[str, Any]]:
        """Parse task schedules"""

        system_prompt = """
        You are a construction scheduling expert. You help convert natural language 
        construction descriptions into structured task lists.
        
        Always return valid JSON format. Be concise and accurate.
        """

        prompt = f"""
        Extract construction tasks from this description:
        "{user_input}"

        Return ONLY valid JSON in this format:
        {{
            "tasks": [
                {{
                    "name": "task name",
                    "duration_days": number,
                    "dependencies": [["previous_task_name","relationship_link","lag_or_lead_by"]]
                }}
            ]
        }}

        Examples:
        - "Build foundation for 5 days then framing for 10 days"
        → {{"tasks": [{{"name": "Foundation", "duration_days": 5, "dependencies": []}}, {{"name": "Framing", "duration_days": 10, "dependencies": [["Foundation","FS",0]]}}]}}

        - "Roofing takes 4 days and starts after Framing finishes.
           Plumbing Rough-In takes 4 days and starts 1 day after Framing starts."
        → {{"tasks": [{{"name": "Roofing", "duration_days": 4, "dependencies": [["Framing","FS",0]]}}, {{"name": "Plumbing Rough-In", "duration_days": 4, "dependencies": [["Framing","SS",1]]}}]}}

        - "Site prep 3 days, excavation 4 days after site prep, foundation 7 days after excavation"
        → {{"tasks": [{{"name": "Site Prep", "duration_days": 3, "dependencies": []}}, {{"name": "Excavation", "duration_days": 4, "dependencies": [["Site Prep","FS",0]]}}, {{"name": "Foundation", "duration_days": 7, "dependencies": [["Excavation","FS",0]]}}]}}
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-5-nano",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                # temperature=0.1,
                response_format={"type": "json_object"},  # Ensures structured JSON
            )

            # Print the response content from the first choice
            print(response.choices[0].message.content)

            # Always decode from content as response_format does not provide parsed output
            content = response.choices[0].message.content
            if content is not None:
                parsed_data = json.loads(content)

                for task in parsed_data["tasks"]:
                    # Convert dependency lists to tuples
                    task["dependencies"] = [tuple(dep) for dep in task["dependencies"]]
                return parsed_data

            else:
                print("No content returned from GPT-5 response.")
                return None

        except Exception as e:
            print(f"GPT-5 parsing failed: {e}")
            return None

    def parse_construction_chatbot(self, user_input: str) -> Dict[str, Any]:

        system_prompt = "You are a classifier. Return ONLY valid JSON with no additional text. Choose between: extract_task_state, schedule_task_state"

        prompt = f"""
        Classify this user input into ONE of these states:
        - "extract_task_state" 
        - "schedule_task_state"

        User input: "{user_input}"

        Return ONLY: {{"state": "state_name"}}
        """

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                options={
                    "temperature": 0.1,  # Low temperature for consistent JSON
                    "top_p": 0.9,
                },
            )

            # Extract and parse JSON
            content = response["message"]["content"]

            # Sometimes the response might have text before/after JSON
            # Try to extract JSON if it exists
            if "{" in content and "}" in content:
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                json_str = content[json_start:json_end]

                parsed_data = json.loads(json_str)

                return parsed_data
            else:
                # Fallback: try to parse the entire content
                return json.loads(content)

        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return {"state": []}
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return {"state": []}
