import os
import sys

# Redirect to dashboard/app.py
dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard", "app.py")
with open(dashboard_path, "r", encoding="utf-8") as f:
    code = f.read()

exec(compile(code, dashboard_path, 'exec'))
