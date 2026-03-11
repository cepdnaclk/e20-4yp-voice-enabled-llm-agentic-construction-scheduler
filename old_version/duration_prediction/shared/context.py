from datetime import date, timedelta
from typing import List, Optional
from .knowledge import KnowledgeBase

class EnvironmentContext:
    """
    Handles external factors like weather and calendar constraints.
    """
    def __init__(self, holidays: Optional[List[date]] = None, work_days: List[int] = [0, 1, 2, 3, 4], knowledge_base: Optional[KnowledgeBase] = None):
        # Default: Mon-Fri work week
        self.holidays = set(holidays) if holidays else set()
        self.work_days = set(work_days) # 0=Mon, 6=Sun
        self.kb = knowledge_base

    def is_working_day(self, check_date: date) -> bool:
        if check_date in self.holidays:
            return False
        if check_date.weekday() not in self.work_days:
            return False
        return True

    def items_working_days_between(self, start: date, end: date) -> int:
        """Count working days between start and end (inclusive)."""
        count = 0
        curr = start
        while curr <= end:
            if self.is_working_day(curr):
                count += 1
            curr += timedelta(days=1)
        return count
    
    def add_working_days(self, start: date, num_days: int) -> date:
        """Add working days to a date to find the finish date."""
        curr = start
        days_added = 0
        while days_added < num_days:
            curr += timedelta(days=1)
            if self.is_working_day(curr):
                days_added += 1
        return curr

    def get_weather_factor(self, target_date: date, location: str = "default") -> float:
        """
        Get weather impact factor for a date.
        Uses KnowledgeBase if available, otherwise falls back to hardcoded seasonal patterns.
        """
        month = target_date.month
        condition = "normal"
        default_factor = 1.0
        
        if month in [12, 1, 2]:
            condition = "snow"
            default_factor = 0.7 # Hardcoded fallback
        elif month in [3, 4, 5]:
            condition = "rain"
            default_factor = 0.9 # Hardcoded fallback
            
        if hasattr(self, 'kb') and self.kb:
            factor = self.kb.get_risk_factor(condition)
            if factor is not None:
                return factor
                
        return default_factor

