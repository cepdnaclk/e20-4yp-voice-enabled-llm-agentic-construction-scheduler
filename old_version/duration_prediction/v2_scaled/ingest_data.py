import json
import os
import logging
from ..shared.models import TaskInput, ResourceConfig
from .vector_store import VectorHistoricalStore
from .ml_model import QuantitativePredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest")

def load_data(filepath):
    """Load history data from JSON."""
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return []
    with open(filepath, 'r') as f:
        return json.load(f)

def run_ingestion():
    # 1. Load Data
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "history.json")
    data_path = os.path.abspath(data_path)
    logger.info(f"Loading data from {data_path}")
    
    historical_data = load_data(data_path)
    if not historical_data:
        logger.warning("No data found. Exiting.")
        return

    # 2. Populate Vector Store
    logger.info("Initializing Vector Store...")
    vector_store = VectorHistoricalStore(persistence_path="./chroma_db")
    
    logger.info(f"Ingesting {len(historical_data)} records into Vector Store...")
    for record in historical_data:
        # Create TaskInput object for structure, though we just need description/meta here
        task = TaskInput(
            id=record['id'],
            name=f"History: {record['type']}",
            type=record['type'],
            quantity=record['quantity'],
            unit=record['unit'],
            complexity_description=record['description'],
            resources=ResourceConfig(crew_size=record['crew_size'])
        )
        
        vector_store.add_project_history(
            task=task,
            actual_duration=record['actual_duration_days'],
            description=record['description']
        )
    logger.info("Vector Store ingestion complete.")

    # 3. Train ML Model
    logger.info("Training ML Model...")
    ml_predictor = QuantitativePredictor(model_path="./duration_model.joblib")
    
    # ML model expects raw dicts which it processes
    ml_predictor.train(historical_data)
    logger.info("ML Model training complete and saved.")

if __name__ == "__main__":
    run_ingestion()
