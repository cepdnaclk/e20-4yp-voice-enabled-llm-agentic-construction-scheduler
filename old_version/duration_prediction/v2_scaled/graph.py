import networkx as nx
from typing import List, Dict, Optional
import logging

from ..shared.knowledge import KnowledgeBase

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    def __init__(self, kb: Optional[KnowledgeBase] = None):
        self.kb = kb or KnowledgeBase()
        self.graph = nx.DiGraph()
        self._initialize_graph()


    def _initialize_graph(self):
        """
        Initialize the graph with domain knowledge:
        - Task Types
        - Conditions (Weather, Site)
        - Constraints
        - Risks
        """
        # Task Types
        self.graph.add_node("Framing", type="TaskType")
        self.graph.add_node("Foundation", type="TaskType")
        self.graph.add_node("Roofing", type="TaskType")
        self.graph.add_node("Excavation", type="TaskType")

        # Conditions
        self.graph.add_node("Rain", type="Condition")
        self.graph.add_node("Snow", type="Condition")
        self.graph.add_node("HighWind", type="Condition")
        self.graph.add_node("Winter", type="Condition")

        # Risks
        self.graph.add_node("SlipperyCheck", type="Risk", severity="High")
        self.graph.add_node("ConcreteCureDelay", type="Risk", severity="High")
        self.graph.add_node("CraneDanger", type="Risk", severity="Critical")

        # Edges (Relationships)
        # Roofing is affected by High Wind and Rain
        self.graph.add_edge("Roofing", "HighWind", relationship="vulnerable_to", impact=1.5)
        self.graph.add_edge("Roofing", "Rain", relationship="vulnerable_to", impact=1.3)
        
        # Foundation affects Concrete Cure
        self.graph.add_edge("Foundation", "Winter", relationship="sensitive_to", impact=1.4)
        self.graph.add_edge("Winter", "ConcreteCureDelay", relationship="causes")

        # Excavation affected by frequent rain
        self.graph.add_edge("Excavation", "Rain", relationship="vulnerable_to", impact=1.6)

        # Dynamic Ingestion from Knowledge Base
        # If KB has risk factors, we can infer some general graph edges or just use them as attributes
        if self.kb:
            risk_data = self.kb.data.get("risk_factors", {})
            for condition, factor in risk_data.items():
                cond_node = condition.capitalize()
                if cond_node not in self.graph:
                    self.graph.add_node(cond_node, type="Condition")
                
                # Assume all outdoor tasks are somewhat affected if not explicitly defined?
                # For now, let's just ensure the nodes exist.
                pass
            
            # We could also add specific rules if the JSON structure supported "task_risks"
            # But currently it only has global 'risk_factors' and 'custom_rates'.
            # We might want to expand the KnowledgeBase class later to support graph-like rules.


    def get_task_risks(self, task_type: str, conditions: List[str]) -> List[Dict]:
        """
        Find risks associated with a task type given current conditions.
        """
        risks = []
        task_node = None
        
        # Find the node matching the task type (case-insensitive)
        for node in self.graph.nodes:
            if str(node).lower() == task_type.lower():
                task_node = node
                break
        
        if not task_node:
            return []

        # Check direct vulnerabilities
        for condition in conditions:
            # Normalize condition
            cond_node = None
            for n in self.graph.nodes:
                if str(n).lower() == condition.lower():
                    cond_node = n
                    break
            
            if cond_node:
                # Check for direct edge
                if self.graph.has_edge(task_node, cond_node):
                    edge_data = self.graph.get_edge_data(task_node, cond_node)
                    risks.append({
                        "condition": cond_node,
                        "impact_factor": edge_data.get("impact", 1.0),
                        "description": f"{task_node} is {edge_data.get('relationship')} {cond_node}"
                    })
                    
                    # Check 2nd order risks (Condition -> Risk)
                    for neighbor in self.graph.successors(cond_node):
                        if self.graph.nodes[neighbor].get("type") == "Risk":
                            risks.append({
                                "risk": neighbor,
                                "source": cond_node,
                                "description": f"{cond_node} causes {neighbor}"
                            })

        return risks

    def get_semantic_context(self, task_type: str) -> str:
        """
        Get a text description of the task's dependencies in the graph.
        """
        task_node = None
        for node in self.graph.nodes:
            if str(node).lower() == task_type.lower():
                task_node = node
                break
        
        if not task_node:
            return ""

        context = []
        for neighbor in self.graph.successors(task_node):
            edge = self.graph.get_edge_data(task_node, neighbor)
            context.append(f"Is {edge.get('relationship')} {neighbor}")
            
        return f"{task_node} context: " + ", ".join(context)
