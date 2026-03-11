import logging
import json
from typing import Optional
import openai

from ..shared.models import TaskInput, PredictionOutput
from ..shared.config import Config
from ..shared.context import EnvironmentContext
from ..shared.knowledge import KnowledgeBase

# New Components
from .graph import KnowledgeGraph
from .vector_store import VectorHistoricalStore
from .ml_model import QuantitativePredictor

logger = logging.getLogger(__name__)

class PredictionOrchestrator:
    def __init__(self):
        # 1. Core utilities
        self.kb = KnowledgeBase()
        self.context = EnvironmentContext(knowledge_base=self.kb)
        
        # 2. Key Subsystems
        self.graph = KnowledgeGraph()
        self.vector_store = VectorHistoricalStore()
        self.ml_predictor = QuantitativePredictor()
        
        # 3. LLM Client
        self.client = None
        if Config.OPENAI_API_KEY:
            try:
                self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
                logger.info("Orchestrator: LLM Connected")
            except Exception as e:
                logger.error(f"Orchestrator: LLM Connection Failed: {e}")

    def predict(self, task: TaskInput) -> PredictionOutput:
        """
        Synthesize inputs from all systems to generate a prediction.
        """
        logger.info(f"Orchestrating prediction for task: {task.type}")
        
        # A. Knowledge Graph Scan
        # ------------------------
        # Determine risks based on task type + season (inferred from date)
        condition_context = []
        if task.target_start_date:
            month = task.target_start_date.month
            if month in [12, 1, 2]: condition_context.append("Winter")
            if month in [3, 4, 5]: condition_context.append("Rain")
        
        graph_risks = self.graph.get_task_risks(task.type, condition_context)
        graph_context = self.graph.get_semantic_context(task.type)
        
        # B. Vector History Search
        # ------------------------
        # Construct a search query from the task description
        search_query = f"{task.type} {task.quantity} {task.unit}. {task.complexity_description or ''}"
        similar_cases = self.vector_store.find_similar_tasks(search_query)
        
        # C. ML Quantitative Prediction
        # -----------------------------
        ml_days = self.ml_predictor.predict(task)
        
        # D. LLM Synthesis
        # ----------------
        final_prediction = self._synthesize_with_llm(
            task, 
            ml_days, 
            graph_risks, 
            similar_cases, 
            graph_context
        )
        
        # E. Final Schedule Calculation (Calendar)
        # ----------------------------------------
        completion_date = None
        final_days = final_prediction.get("days", ml_days)
        final_hours = final_days * 8.0
        
        if task.target_start_date:
            needed_work_days = int(final_days + 0.99)
            completion_date = self.context.add_working_days(task.target_start_date, needed_work_days)

        return PredictionOutput(
            task_id=task.id,
            predicted_duration_days=final_days,
            predicted_duration_hours=final_hours,
            completion_date=completion_date,
            confidence_score=final_prediction.get("confidence", 0.5),
            explanation=final_prediction.get("explanation", "Based on quantitative model."),
            factors={
                "ml_estimate": ml_days,
                "complexity_adjustment": final_prediction.get("complexity_factor", 1.0),
                "risk_count": len(graph_risks),
                "similar_cases_found": len(similar_cases)
            }
        )

    def _synthesize_with_llm(self, task, ml_val, risks, history, graph_context) -> dict:
        """
        Prompt the LLM to act as the final judge.
        """
        if not self.client:
            return {"days": ml_val, "confidence": 0.5, "explanation": "ML Model Estimation (No LLM)"}

        # Format inputs for prompt
        risk_str = "\n".join([f"- {r['description']} (Impact: {r.get('impact_factor','?')})" for r in risks])
        history_str = "\n".join([f"- {h['description']} (Took: {h['metadata']['duration_days']}d)" for h in history])
        
        prompt = f"""
        You are an expert construction project manager. Determine the final duration for a task based on multiple inputs.
        
        Task: {task.type} - {task.quantity} {task.unit}
        Context: {task.complexity_description}
        
        Inputs:
        1. Quantitative Model (Baseline): {ml_val} days
        2. Knowledge Graph Rules:
           {graph_context}
           Identified Risks:
           {risk_str}
        3. Historical Similar Cases:
           {history_str}
           
        Instruction:
        - Analyze the context and risks.
        - If risks are high or context is complex, increase the baseline.
        - If history shows longer times for similar cases, adjust accordingly.
        - If history suggests shorter times, you may reduce it.
        
        Output JSON:
        {{
            "days": float (the final predicted working days),
            "confidence": float (0.0-1.0),
            "complexity_factor": float (multiplier applied to baseline),
            "explanation": "Concise reasoning citing specific risks or history."
        }}
        """

        try:
             response = self.client.chat.completions.create(
                model="gpt-5-nano",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
             content = response.choices[0].message.content
             return json.loads(content)
        except Exception as e:
            logger.error(f"LLM Synthesis failed: {e}")
            return {"days": ml_val, "confidence": 0.5, "explanation": "Fallback to ML due to LLM error."}
