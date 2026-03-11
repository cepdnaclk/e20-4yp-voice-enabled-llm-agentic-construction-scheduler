import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # Default productivity rates (could be loaded from DB later)
    BASE_PRODUCTIVITY_RATES = {
        "framing": 50.0, # sqft per man-hour
        "foundation": 0.5, # cubic yards per man-hour
        "drywall": 40.0, # sqft per man-hour
        "painting": 100.0, # sqft per man-hour
    }

    @staticmethod
    def check_api_key():
        if not Config.OPENAI_API_KEY:
            print("Warning: OPENAI_API_KEY not found in environment variables. LLM features will be disabled or mocked.")
