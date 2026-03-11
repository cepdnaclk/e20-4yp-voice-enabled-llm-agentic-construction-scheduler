import sys
import logging
from datetime import date
from ..shared.models import TaskInput, ResourceConfig
from .orchestrator import PredictionOrchestrator

# Configure logging to show info
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_scaled_prediction():
    print("\n--- Testing Scaled Duration Prediction System ---\n")
    
    # 1. Initialize Orchestrator (loads Graph, VectorDB, ML)
    try:
        orchestrator = PredictionOrchestrator()
        print("Orchestrator Initialized")
    except Exception as e:
        print(f"Failed to initialize orchestrator: {e}")
        return

    # 2. Define a Test Task
    # Scenario: Foundation work in Winter (should trigger Knowledge Graph risk)
    task1 = TaskInput(
        id="task_001",
        name="Winter Foundation",
        type="Foundation",
        quantity=500.0,
        unit="cubic_yards",
        complexity_description="Pouring concrete foundation on a slope during winter conditions. Access is tricky.",
        resources=ResourceConfig(crew_size=6),
        target_start_date=date(2025, 1, 15) # Winter date
    )
    
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
    
    print(f"\nSubject Task: {task.name} ({task.type})")
    print(f"Conditions: {task.quantity} {task.unit}, Date: {task.target_start_date}")
    print(f"Description: {task.complexity_description}\n")

    # 3. specific test for history (optional, to ensure vector store has something)
    # The VectorHistoricalStore init should have added sample data.

    # 4. Run Prediction
    print("Running Prediction Pipeline...")
    try:
        result = orchestrator.predict(task)
        
        print("\n--- Prediction Results ---")
        print(f"Predicted Duration: {result.predicted_duration_days:.2f} working days")
        print(f"Predicted Hours: {result.predicted_duration_hours:.2f} hours")
        print(f"Completion Date: {result.completion_date}")
        print(f"Confidence: {result.confidence_score}")
        print(f"Explanation: {result.explanation}")
        print("Factors:")
        for k, v in result.factors.items():
            print(f"  - {k}: {v}")
            
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_scaled_prediction()
