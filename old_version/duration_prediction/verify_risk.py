import sys
import os
from datetime import date

# Add src to path
# Assuming we are running from d:\SEM7\Research\code\research\FYP
sys.path.append(os.path.join(os.getcwd(), 'fyp', 'src'))
sys.path.append(os.path.join(os.getcwd(), 'fyp', 'src', 'duration_prediction'))

from models import TaskInput, ResourceConfig
from engine import DurationPredictor

def verify_risk():
    print("Initializing Predictor...")
    predictor = DurationPredictor()
    
    # Test Case 1: March Date -> Spring -> Condition "rain"
    # KB "rain" -> 0.7
    # Hardcoded "Spring" -> 0.9 (If fallback used)
    
    # Task: Framing near Spring
    task_rain = TaskInput(
        id="test-rain",
        name="Rain Task",
        type="framing",
        quantity=1000,
        unit="sqft",
        complexity_description="Standard",
        resources=ResourceConfig(crew_size=2),
        target_start_date=date(2025, 3, 10) # March
    )
    
    print("\nPredicting for March (Rain)...")
    result_rain = predictor.predict(task_rain)
    weather_factor_rain = result_rain.factors["weather_factor"]
    print(f"Weather Factor: {weather_factor_rain}")
    
    if abs(weather_factor_rain - 0.7) < 0.01:
        print("SUCCESS: Used KB factor for rain (0.7)")
    elif abs(weather_factor_rain - 0.9) < 0.01:
        print("FAIL: Used fallback factor for rain (0.9)")
    else:
        print(f"FAIL: Unexpected factor {weather_factor_rain}")

    # Test Case 2: Jan Date -> Winter -> Condition "snow"
    # KB "snow" -> 0.4
    # Hardcoded "Winter" -> 0.7
    
    task_snow = TaskInput(
        id="test-snow",
        name="Snow Task",
        type="framing",
        quantity=1000,
        unit="sqft",
        complexity_description="Standard",
        resources=ResourceConfig(crew_size=2),
        target_start_date=date(2025, 1, 15) # Jan
    )
    


    print("\nPredicting for Jan (Snow)...")
    result_snow = predictor.predict(task_snow)
    weather_factor_snow = result_snow.factors["weather_factor"]
    print(f"Weather Factor: {weather_factor_snow}")
    
    if abs(weather_factor_snow - 0.4) < 0.01:
        print("SUCCESS: Used KB factor for snow (0.4)")
    elif abs(weather_factor_snow - 0.7) < 0.01:
        print("FAIL: Used fallback factor for snow (0.7)")
    else:
        print(f"FAIL: Unexpected factor {weather_factor_snow}")


if __name__ == "__main__":
    verify_risk()
