"""
Task Duration Calculation Evaluation
====================================
Tests the accuracy of the LLM in extracting variables from natural language
user inputs and the deterministic calculation of task durations based on
formulas retrieved from the Neo4j knowledge graph.
"""

from src.model import AgenticSchedulerModel, TaskVariableValues
from langchain_core.messages import HumanMessage


def get_model():
    return AgenticSchedulerModel()


# ---------------------------------------------------------------------------
# Test Data
# ---------------------------------------------------------------------------

# Mock task records as they would be returned from Neo4j
MOCK_TASK_RECORDS = [
    {
        "name": "Excavation",
        "task_duration": "{volume} / {productivity}",
        "productivity": 15.0,  # 15 m3 per day
        "unit": "m3",
    },
    {
        "name": "RC Footing Concrete",
        "task_duration": "{volume} / {productivity}",
        "productivity": 20.0,  # 20 m3 per day
        "unit": "m3",
    },
    {
        "name": "Formwork",
        "task_duration": "({area} * 1.5) / {productivity}",  # Complex formula test
        "productivity": 50.0,  # 50 m2 per day
        "unit": "m2",
    },
    {
        "name": "Blockwork Wall",
        "task_duration": "({length} * {height}) / {productivity}", # Multi-variable formula
        "productivity": 12.0,  # 12 m2 per day
        "unit": "m2"
    }
]

# Mock inputs to test the LLM extraction
EXTRACTION_TEST_CASES = [
    {
        "description": "Clear and direct single value",
        "question_text": "1. What is the total volume of earth to be excavated in m3 for the Excavation task?",
        "task_summary_lines": ["- Excavation: needs values for: volume"],
        "user_input": "The excavation volume is 300.",
        "expected_values": {"Excavation": {"volume": 300.0}},
    },
    {
        "description": "Grouped variables (System combines similar questions)",
        "question_text": "1. What is the volume of concrete required in m3 for the RC Footing Concrete?\n2. What is the area of the formwork required in m2?",
        "task_summary_lines": [
            "- RC Footing Concrete: needs values for: volume",
            "- Formwork: needs values for: area"
        ],
        "user_input": "Footing concrete volume is 45m3, and formwork area is 120 square meters.",
        "expected_values": {
            "RC Footing Concrete": {"volume": 45.0},
            "Formwork": {"area": 120.0}
        },
    },
    {
        "description": "Multi-variable task and conversational response",
        "question_text": "1. For the Blockwork Wall, what is the total length in meters and the height in meters?",
        "task_summary_lines": [
            "- Blockwork Wall: needs values for: length, height"
        ],
        "user_input": "The wall will be 50 meters long and 3 meters high.",
        "expected_values": {
            "Blockwork Wall": {"length": 50.0, "height": 3.0}
        },
    },
    {
        "description": "Highly complex grouped response with shared values",
        "question_text": "1. What is the volume of earth to excavate (m3)?\n2. What is the volume of concrete for the RC Footing (m3)?\n3. What is the length and height of the Blockwork Wall (m)?",
        "task_summary_lines": [
            "- Excavation: needs values for: volume",
            "- RC Footing Concrete: needs values for: volume",
            "- Blockwork Wall: needs values for: length, height"
        ],
        "user_input": "Excavation and footing concrete both need 60 cubes. The wall is 100m long and stands 2.5m tall.",
        "expected_values": {
            "Excavation": {"volume": 60.0},
            "RC Footing Concrete": {"volume": 60.0},
            "Blockwork Wall": {"length": 100.0, "height": 2.5}
        },
    }
]

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_variable_extraction_accuracy():
    """
    Test how accurately the LLM extracts variables from free-text user response.
    This replaces the 'interrupt' step in the details_agent node.
    """
    print("\n\n🧪 Testing LLM Variable Extraction Accuracy")
    print("=" * 60)

    model = get_model()
    correct_extractions = 0
    total_cases = len(EXTRACTION_TEST_CASES)

    for i, case in enumerate(EXTRACTION_TEST_CASES):
        print(f"\n--- Case {i+1}: {case['description']} ---")
        print(f"User Input: '{case['user_input']}'")

        parse_prompt = HumanMessage(
            content=f"""You are a construction data parser.

            Here are the tasks and the variables each one needs:
            {chr(10).join(case['task_summary_lines'])}

            Here are the questions that were asked:
            {case['question_text']}

            Here is the user's response:
            "{case['user_input']}"

            Extract the numeric values for EACH task's variables from the user's response.
            Each task needs its OWN variable values — do NOT share values between tasks unless the user explicitly says they are the same.
            """
        )

        structured_llm = model.llm.with_structured_output(TaskVariableValues)
        parsed_values: TaskVariableValues = structured_llm.invoke([parse_prompt]) # type: ignore

        # Convert to a flat dictionary for easy comparison
        extracted = {}
        for tv in parsed_values.task_values:
            extracted[tv.task_name] = {
                entry.variable_name: entry.value for entry in tv.variable_entries
            }

        expected = case["expected_values"]

        print(f"Extracted: {extracted}")
        print(f"Expected:  {expected}")

        if extracted == expected:
            print("✅ PASSED: Exact Match")
            correct_extractions += 1
        else:
            print("❌ FAILED: Mismatch in extraction")

    accuracy = (correct_extractions / total_cases) * 100
    print(
        f"\n🏆 Extraction Accuracy: {accuracy:.1f}% ({correct_extractions}/{total_cases})"
    )
    assert accuracy > 80.0, "The LLM extraction accuracy should be high (at least 80%)."


def test_deterministic_duration_calculation():
    """
    Test the pure Python deterministic eval() engine from model.py.
    Checks rounding logic, formula evaluation, and edge cases.
    """
    print("\n\n🧮 Testing Deterministic Duration Calculation")
    print("=" * 60)

    # We will simulate the `per_task_values` dictionary created after LLM parsing
    mock_per_task_values = {
        "Excavation": {"volume": 100.0},  # 100 / 15 = 6.66 -> 7 days
        "RC Footing Concrete": {"volume": 40.0},  # 40 / 20 = 2 -> 2 days
        "Formwork": {"area": 200.0},  # (200 * 1.5) / 50 = 6 -> 6 days
    }

    # _calculate_task_durations is an internal method in phase_node closure
    # but we can replicate its exact logic here to test it safely.
    import math

    computed_tasks = []

    for t in MOCK_TASK_RECORDS:
        task_name = t.get("name", "")
        task_duration = t.get("task_duration", "")
        productivity = t.get("productivity", 1)

        task_vars = mock_per_task_values.get(task_name, {})
        eval_context = {**task_vars, "productivity": float(productivity)}

        try:
            expression = task_duration
            for var_name, var_value in eval_context.items():
                expression = expression.replace(f"{{{var_name}}}", str(var_value))

            duration = eval(expression)
            duration_days = math.ceil(duration)

            print(
                f"✅ {task_name}: {task_duration} -> {expression} = {duration:.2f} -> {duration_days} days"
            )
            computed_tasks.append({"name": task_name, "duration_days": duration_days})

        except Exception as e:
            raise RuntimeError(f"Failed to calculate duration for {task_name}: {e}")

    # Assert correct math and ceil() behavior
    expected_durations = {
        "Excavation": 7,  # 100/15 = 6.66... ceil is 7
        "RC Footing Concrete": 2,  # 40/20 = 2.0 ceil is 2
        "Formwork": 6,  # (200 * 1.5) / 50 = 6.0 ceil is 6
    }

    for task in computed_tasks:
        name = task["name"]
        assert (
            task["duration_days"] == expected_durations[name]
        ), f"Incorrect calculation for {name}"

    print("\n🏆 All deterministic calculations evaluated correctly.")


if __name__ == "__main__":
    """Run directly with: python -m tests.test_task_duration_calculation"""
    print("🚀 Task Duration Calculation Test (standalone mode)")
    print("=" * 60)

    test_variable_extraction_accuracy()
    test_deterministic_duration_calculation()

    print("\n✅ Done!")
