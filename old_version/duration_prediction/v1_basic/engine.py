import openai
import json
import logging
from ..shared.models import TaskInput, PredictionOutput
from ..shared.config import Config
from ..shared.context import EnvironmentContext

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from ..shared.knowledge import KnowledgeBase

class DurationPredictor:
    def __init__(self):
        self.kb = KnowledgeBase()
        self.context = EnvironmentContext(knowledge_base=self.kb)
        # Initialize OpenAI client if key is present
        self.client = None
        if Config.OPENAI_API_KEY:
            try:
                self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
                logger.info("OpenAI Client initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")

    def calculate_base_duration(self, task: TaskInput) -> float:
        """
        Level 1: Rule-based calculation.
        Formula: Quantity / (Rate * CrewSize * 8hrs)
        """
        # 1. Check Knowledge Base for "Learned Rate"
        rate = self.kb.get_custom_rate(task.type)
        
        # 2. Fallback to Config
        if not rate:
            rate = Config.BASE_PRODUCTIVITY_RATES.get(task.type.lower(), 10.0)
        
        if rate <= 0: 
            logger.warning(f"Invalid rate {rate} for task {task.type}, defaulting to 1.0")
            rate = 1.0
        
        # Calculate Total Man-Hours required
        total_man_hours = task.quantity / rate
        
        # Adjust for Crew Size
        crew_size = max(1, task.resources.crew_size)
        daily_hours = 8 # Standard work day
        
        # Raw days (effort)
        days = total_man_hours / (crew_size * daily_hours)
        return days


    def get_llm_adjustment(self, task: TaskInput) -> dict:
        """
        Level 2: LLM analysis for complexity.
        """
        # Return neutral factor if no LLM or no description
        if not self.client or not task.complexity_description:
            logger.info(f"OpenAI key present: {bool(Config.OPENAI_API_KEY)}")
            logger.info(f"OpenAI client: {self.client}")
            return {"factor": 1.0, "reason": "No LLM available or no complexity description provided."}
        
        prompt = f"""
        You are an expert construction scheduler. Analyze the following task and determine a "Complexity Factor" multiplier for the duration.
        - Standard condition = 1.0
        - Difficult/Complex = > 1.0 (e.g., 1.2 for 20% more time)
        - Simple/Easy = < 1.0

        Task Details:
        - Type: {task.type}
        - Quantity: {task.quantity} {task.unit}
        - Description/Context: "{task.complexity_description}"
        - Crew Size: {task.resources.crew_size}

        Return ONLY a JSON object with keys:
        - "complexity_factor": float
        - "reasoning": string (concise explanation)
        """
        
        try:
             response = self.client.chat.completions.create(
                model="gpt-5-nano", # Cost-effective model for this logic
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=1
            )
             content = response.choices[0].message.content
             data = json.loads(content)
             return {
                 "factor": float(data.get("complexity_factor", 1.0)),
                 "reason": data.get("reasoning", "LLM analysis complete")
             }
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return {"factor": 1.0, "reason": f"LLM Error: {e}"}

    def predict(self, task: TaskInput) -> PredictionOutput:
        """
        Main prediction pipeline.
        """
        # 1. Base Duration
        base_days = self.calculate_base_duration(task)
        
        # 2. LLM Adjustment
        llm_result = self.get_llm_adjustment(task)
        complexity_factor = llm_result["factor"]
        
        # 3. Calculate Final Duration
        # Weather Impact
        weather_factor = 1.0
        if task.target_start_date:
            weather_factor = self.context.get_weather_factor(task.target_start_date)
        
        # Avoid division by zero
        if weather_factor < 0.1: weather_factor = 0.1
        
        # Formula: Duration = (Base * Complexity) / Weather
        final_days = (base_days * complexity_factor) / weather_factor
        final_hours = final_days * 8.0 # Assuming 8-hour work days
        
        # 4. Schedule Adjustment (Context)
        completion_date = None
        if task.target_start_date:
            # Round up to nearest whole working day for calendar calculation
            needed_work_days = int(final_days + 0.99)
            # Use Calendar Context to find end date
            completion_date = self.context.add_working_days(task.target_start_date, needed_work_days)

        # Build Explanation
        confidence = 0.9 if self.client else 0.5
        explanation = (
            f"Estimated {final_days:.1f} working days ({final_hours:.1f} hours). "
            f"Base: {base_days:.1f}d. Complexity: {complexity_factor}x ({llm_result['reason']}). "
            f"Weather: {weather_factor}x factor."
        )
        if completion_date:
            explanation += f" Finish by: {completion_date} (considering working days)."

        return PredictionOutput(
            task_id=task.id,
            predicted_duration_days=final_days,
            predicted_duration_hours=final_hours,
            completion_date=completion_date,
            confidence_score=confidence,
            explanation=explanation,
            factors={
                "base_days": base_days,
                "complexity_factor": complexity_factor,
                "weather_factor": weather_factor,
                "rate_used": self.kb.get_custom_rate(task.type) or Config.BASE_PRODUCTIVITY_RATES.get(task.type.lower(), 10.0)
            }
        )
