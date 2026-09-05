import os
import sys

# Guarantee project root is in sys.path for Vercel & Serverless Runtimes
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.engine.api import app

__all__ = ["app"]

# Local Streamlit runner fallback when executed via CLI
if __name__ == "__main__" or "streamlit" in sys.modules:
    dashboard_path = os.path.join(BASE_DIR, "dashboard", "app.py")
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            code = f.read()
        exec(compile(code, dashboard_path, 'exec'))

