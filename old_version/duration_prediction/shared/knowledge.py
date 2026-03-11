import json
import os
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class KnowledgeBase:
    def __init__(self, filepath: Optional[str] = None):
        if filepath is None:
            # Resolve relative to this file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(base_dir, "knowledge_base.json")
        self.filepath = filepath
        self.data = self._load_data()

    def _load_data(self) -> Dict:
        if not os.path.exists(self.filepath):
            # Default "learned" knowledge
            return {
                "custom_rates": {
                    "framing": 45.0, # Learned: slightly slower than standard 50.0
                    "complex_foundation": 0.4
                },
                "risk_factors": {
                     "rain": 0.8,
                     "snow": 0.5
                }
            }
        try:
            with open(self.filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
            return {}

    def get_custom_rate(self, task_type: str) -> Optional[float]:
        """
        Retrieve a learned productivity rate if it exists.
        """
        return self.data.get("custom_rates", {}).get(task_type.lower())

    def save_data(self):
        """Persist updates to disk."""
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save knowledge base: {e}")

    def get_risk_factor(self, condition: str) -> Optional[float]:
        """
        Retrieve a learned risk factor if it exists.
        """
        return self.data.get("risk_factors", {}).get(condition.lower())
