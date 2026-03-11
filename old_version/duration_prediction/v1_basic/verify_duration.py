import sys
import logging
from datetime import date
from ..shared.models import TaskInput, ResourceConfig
from .engine import DurationPredictor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_prediction():
    print("\n--- Testing V1 Duration Prediction (Rule-Based) ---\n")
    
    predictor = DurationPredictor()
    
    # Define Test Task
    task = TaskInput(
        id="t1",
        name="Standard Framing",
        type="Foundation",
        quantity=1000,
        unit="sqft",
        #complexity_description="Standard residential foundation work with normal soil conditions, good site access, and no special constraints.",
        complexity_description= "Foundation construction in constrained site conditions with poor soil stability, limited access, and weather-related delays requiring additional safety and curing time.",
        resources=ResourceConfig(crew_size=4),
        target_start_date=date(2025, 12, 25)
    )
    
    # Print Task Context
    print(f"Subject Task: {task.name} ({task.type})")
    print(f"Conditions: {task.quantity} {task.unit}, Crew: {task.resources.crew_size}, Date: {task.target_start_date}")
    print(f"Description: {task.complexity_description}\n")

    print("Running Prediction...")
    result = predictor.predict(task)

    # Print Results in formatted style
    print("\n--- Prediction Results ---")
    print(f"Predicted Duration: {result.predicted_duration_days:.2f} working days")
    print(f"Predicted Hours: {result.predicted_duration_hours:.2f} hours")
    print(f"Completion Date: {result.completion_date}")
    print(f"Confidence: {result.confidence_score}")
    print(f"Explanation: {result.explanation}")
    print("Factors:")
    for k, v in result.factors.items():
        print(f"  - {k}: {v}")

if __name__ == "__main__":
    test_prediction()
