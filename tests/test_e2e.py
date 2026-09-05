import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine.api import app

client = TestClient(app)

def test_api_health():
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ONLINE"

def test_full_e2e_incident_flow():
    # 1. Ingest 5 raw alerts
    incident_id = None
    for i in range(5):
        resp = client.post("/api/alerts", json={
            "service": "payment-service",
            "environment": "prod",
            "alert_name": "HighMemoryUsage",
            "severity": "CRITICAL",
            "host": f"node-prod-0{i+1}",
            "error_code": "OOMKilled",
            "log_snippet": "Memory leak detected in worker process",
            "tags": {"env": "prod", "service": "payment"}
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "INGESTED"
        incident_id = data["incident_id"]

    assert incident_id is not None

    # 2. Process the incident
    proc_resp = client.post(f"/api/process_incident/{incident_id}")
    assert proc_resp.status_code == 200
    pdata = proc_resp.json()
    
    assert pdata["incident_id"] == incident_id
    assert "routing" in pdata
    assert "llm_output" in pdata
    assert "execution" in pdata
    assert pdata["execution"]["success"] is True

    # 3. Check Audit Logs & Metrics endpoints
    audit_resp = client.get("/api/audit_logs")
    assert audit_resp.status_code == 200
    logs = audit_resp.json()
    assert len(logs) >= 1

    metrics_resp = client.get("/api/metrics")
    assert metrics_resp.status_code == 200
    mdata = metrics_resp.json()
    assert mdata["total_incidents"] >= 1
