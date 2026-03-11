from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal
from datetime import date

class Dependency(BaseModel):
    task_id: str
    type: Literal['FS', 'FF', 'SS', 'SF'] = 'FS'
    lag_days: int = 0

class ResourceConfig(BaseModel):
    crew_size: int = 1
    # Equipment list could be strings or enums
    equipment: List[str] = [] 

class TaskInput(BaseModel):
    id: str
    name: str
    type: str  # e.g., "framing", "foundation"
    
    # Quantitative Size
    quantity: float = Field(..., description="Primary dimension quantity (e.g. 1000)")
    unit: str = Field(..., description="Unit of measurement (e.g. sqft, cubic_yards)")
    
    # Qualitative Context for LLM
    complexity_description: Optional[str] = Field(
        None, 
        description="Natural language description of specific site conditions or complexity (e.g. 'steep slope', 'limited access')"
    )
    
    dependencies: List[Dependency] = []
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    
    # Optional scheduling constraint
    target_start_date: Optional[date] = None

class PredictionOutput(BaseModel):
    task_id: str
    predicted_duration_days: float
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    explanation: str
    completion_date: Optional[date] = None
    
    # Detailed breakdown of factors affecting the duration
    factors: Dict[str, float] = Field(
        default_factory=dict, 
        description="Factors that influenced the result (e.g. {'base_days': 5.0, 'weather_penalty': 0.2, 'complexity_multiplier': 1.1})"
    )
    
    predicted_duration_hours: float
