import chromadb
from chromadb.config import Settings
import logging
from typing import List, Dict, Optional
import json
from ..shared.models import TaskInput

logger = logging.getLogger(__name__)

class VectorHistoricalStore:
    def __init__(self, persistence_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persistence_path)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="construction_tasks",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Vector Store initialized at {persistence_path}")
        self._ensure_sample_data()

    def _ensure_sample_data(self):
        """Populate with some sample data if empty."""
        if self.collection.count() == 0:
            logger.info("Populating Vector Store with sample data...")
            
            samples = [
                {
                    "id": "hist_1",
                    "text": "Framing for a 2000 sqft residential house in summer",
                    "metadata": {"type": "framing", "duration_days": 10, "quantity": 2000, "unit": "sqft"}
                },
                {
                    "id": "hist_2",
                    "text": "Concrete foundation pouring 500 cubic yards in winter",
                    "metadata": {"type": "foundation", "duration_days": 15, "quantity": 500, "unit": "cy"}
                },
                {
                    "id": "hist_3",
                    "text": "Roofing installation 3000 sqft asphalt shingles",
                    "metadata": {"type": "roofing", "duration_days": 5, "quantity": 3000, "unit": "sqft"}
                }
            ]
            
            self.collection.add(
                documents=[s["text"] for s in samples],
                metadatas=[s["metadata"] for s in samples],
                ids=[s["id"] for s in samples]
            )

    def add_project_history(self, task: TaskInput, actual_duration: float, description: str):
        """Add a completed task to history."""
        try:
            self.collection.add(
                documents=[description],
                metadatas=[{
                    "type": task.type,
                    "duration_days": float(actual_duration),
                    "quantity": float(task.quantity),
                    "unit": task.unit
                }],
                ids=[f"task_{task.id}_history"]
            )
            logger.info(f"Added task {task.id} to history.")
        except Exception as e:
            logger.error(f"Failed to add to vector store: {e}")

    def find_similar_tasks(self, description: str, n_results: int = 3) -> List[Dict]:
        """Find similar past tasks based on text description."""
        try:
            results = self.collection.query(
                query_texts=[description],
                n_results=n_results
            )
            
            # Format results
            # Chroma returns lists of lists (one list per query)
            formatted = []
            if results["ids"]:
                for i, id_ in enumerate(results["ids"][0]):
                    formatted.append({
                        "id": id_,
                        "description": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i] if results["distances"] else 0
                    })
            return formatted
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
