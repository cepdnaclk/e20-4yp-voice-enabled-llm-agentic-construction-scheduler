import argparse
import json
import sys
from datetime import date
from engine import DurationPredictor
from models import TaskInput

def main():
    parser = argparse.ArgumentParser(description="Construction Task Duration Predictor")
    parser.add_argument("--type", type=str, help="Task type (e.g. framing, foundation)")
    parser.add_argument("--qty", type=float, help="Quantity")
    parser.add_argument("--unit", type=str, help="Unit (sqft, etc)")
    parser.add_argument("--desc", type=str, help="Complexity description")
    parser.add_argument("--crew", type=int, default=1, help="Crew size")
    parser.add_argument("--start", type=str, help="Start Date (YYYY-MM-DD)")
    
    # JSON input mode
    #parser.add_argument("--json", type=str, help="Pass a full JSON string")
    parser.add_argument("--json-file", type=str, help="Path to JSON task file")

    args = parser.parse_args()

    try:
        if args.json_file:
            with open(args.json_file, "r") as f:
                data = json.load(f)
            task = TaskInput(**data)
        else:
            # Make sure required CLI args are provided manually
            missing = [arg for arg in ["type", "qty", "unit"] if getattr(args, arg) is None]
            if missing:
                parser.error(f"Missing required arguments when not using --json-file: {', '.join(missing)}")

            task = TaskInput(
                id="cmd-1",
                name="CLI Task",
                type=args.type,
                quantity=args.qty,
                unit=args.unit,
                complexity_description=args.desc or "Standard construction conditions",
                resources={"crew_size": args.crew},
                target_start_date=date.fromisoformat(args.start) if args.start else None
            )
    except Exception as e:
        print(f"Error creating task input: {e}")
        sys.exit(1)


    # Run Prediction
    predictor = DurationPredictor()
    result = predictor.predict(task)

    # Output Result
    print(json.dumps(result.model_dump(mode='json'), indent=2))

if __name__ == "__main__":
    main()
