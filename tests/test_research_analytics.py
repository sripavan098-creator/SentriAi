import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine.api import app
from src.engine.research_analytics import (
    DowntimeFinancialRiskPredictor,
    RogueExecutionSafetyGuard,
    AlertFatigueAnalyzer,
    TokenEconomicsCalculator
)

client = TestClient(app)

def test_downtime_financial_risk_predictor():
    pred = DowntimeFinancialRiskPredictor.predict_risk("INC-TEST-10M", duration_minutes=10.0)
    assert pred.gartner_cost_usd == 56000.0
    assert pred.ponemon_cost_usd == 88510.0
    assert pred.mttr_reduction_percent == 60.0
    assert pred.sentri_saved_cost_usd == 88510.0 * 0.60

def test_rogue_execution_safety_guard():
    # Low risk auto-approve
    cert_low = RogueExecutionSafetyGuard.audit_remediation_action("ACTION_RESTART_CONTAINER", "payment-api")
    assert cert_low.safety_passed is True
    assert cert_low.risk_level == "LOW_RISK"
    assert cert_low.requires_human_approval is False

    # High blast radius requires approval
    cert_high = RogueExecutionSafetyGuard.audit_remediation_action("ACTION_SCALE_SERVICE", "payment-api", human_approved=False)
    assert cert_high.safety_passed is False
    assert cert_high.risk_level == "HIGH_BLAST_RADIUS"
    assert cert_high.requires_human_approval is True

    # High blast radius with approval
    cert_high_ok = RogueExecutionSafetyGuard.audit_remediation_action("ACTION_SCALE_SERVICE", "payment-api", human_approved=True)
    assert cert_high_ok.safety_passed is True

    # Command injection attempt rejection
    cert_inj = RogueExecutionSafetyGuard.audit_remediation_action("rm -rf /", "payment-api")
    assert cert_inj.safety_passed is False
    assert cert_inj.command_injection_blocked is True

def test_alert_fatigue_analyzer():
    metrics = AlertFatigueAnalyzer.analyze_noise(raw_alert_count=20, correlated_incident_count=2)
    assert metrics.noise_reduction_percent == 90.0
    assert metrics.burnout_risk_level in ["LOW", "MODERATE", "CRITICAL"]

def test_token_economics_calculator():
    summary_cached = TokenEconomicsCalculator.calculate_savings(input_tokens=10000, output_tokens=500, is_cache_hit=True)
    assert summary_cached.sentri_tiered_cached_cost_usd == 0.0
    assert summary_cached.cost_savings_percent == 100.0

    summary_uncached = TokenEconomicsCalculator.calculate_savings(input_tokens=10000, output_tokens=500, is_cache_hit=False)
    assert summary_uncached.sentri_tiered_cached_cost_usd < summary_uncached.direct_frontier_cost_usd

def test_research_brief_api_endpoints():
    res_brief = client.get("/api/sentri/research_brief")
    assert res_brief.status_code == 200
    bdata = res_brief.json()
    assert "downtime_benchmarks" in bdata
    assert "crowdstrike_case_study" in bdata

    res_risk = client.post("/api/sentri/predict_downtime_risk", json={"incident_id": "INC-TEST-15M", "duration_minutes": 15.0})
    assert res_risk.status_code == 200
    rdata = res_risk.json()
    assert rdata["gartner_cost_usd"] == 84000.0

    res_safety = client.post("/api/sentri/verify_remediation_safety", json={"action_key": "ACTION_RESTART_CONTAINER", "target_name": "payment-api"})
    assert res_safety.status_code == 200
    sdata = res_safety.json()
    assert sdata["safety_passed"] is True
