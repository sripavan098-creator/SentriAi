import os
import sys

# Guarantee project root is in sys.path for Vercel & Serverless Runtimes
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Top-level FastAPI app export for Vercel / Uvicorn serverless deployments
from src.engine.api import app

__all__ = ["app"]

# Check if running under Streamlit CLI environment (e.g., streamlit run app.py)
is_streamlit_runner = False
try:
    import streamlit as st
    # Streamlit sets _is_running_with_streamlit when launched via `streamlit run`
    if getattr(st, "_is_running_with_streamlit", False):
        is_streamlit_runner = True
except Exception:
    pass

if is_streamlit_runner:
    dashboard_path = os.path.join(BASE_DIR, "dashboard", "app.py")
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            code = f.read()
        exec(compile(code, dashboard_path, 'exec'))


