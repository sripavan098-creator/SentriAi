import os
import sys

# Export top-level FastAPI app for Vercel Serverless deployment
from src.engine.api import app

# Execute Streamlit dashboard if invoked directly
if __name__ == "__main__" or "streamlit" in sys.modules:
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard", "app.py")
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            code = f.read()
        exec(compile(code, dashboard_path, 'exec'))
