import os
import sys
import time
import pytest

# Ensure src is in pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine.correlator import AlertCorrelator, RawAlert
from src.engine.router import IncidentRouter
from src.engine.llm_tiering import TieredLLMSolver
from src.execution.guardrails import ActionGuardrailController, GuardrailViolationError, AllowedRemediationAction
from src.execution.executor import SafeRemediationExecutor
from src.execution.audit_logger import AuditLogger

def test_correlator_sliding_window():
    correlator = AlertCorrelator(window_seconds=60)
    
    # Ingest 3 alerts for payment service
    for i in range(3):
        incident, alert = correlator.ingest_alert({
            "service": "payment-service",
            "environment": "prod",
            "alert_name": "HighMemoryUsage",
            "severity": "CRITICAL",
            "host": f"host-0{i+1}",
            "error_code": "OOMKilled",
            "log_snippet": "Memory leak detected in worker container"
        })
    
    assert incident is not None
    assert incident.raw_alerts_count == 3
    assert incident.service == "payment-service"
    assert incident.compression_ratio > 0.0
    assert len(incident.distinct_hosts) == 3

def test_vector_heuristic_router():
    router = IncidentRouter()
    
    # 1. Simple memory leak incident -> Tier 1
    correlator = AlertCorrelator()
    incident_simple, _ = correlator.ingest_alert({
        "service": "payment-service",
        "environment": "prod",
        "alert_name": "MemoryLeak",
        "severity": "CRITICAL",
        "error_code": "OOMKilled",
        "log_snippet": "Container payment-api memory limit 2048MB reached (98% utilization)"
    })
    
    res_simple = router.evaluate_incident(incident_simple)
    assert res_simple.complexity_score < 0.75
    assert res_simple.selected_tier == "TIER1_FAST"

    # 2. Complex cascading deadlock incident -> Tier 2
    incident_complex, _ = correlator.ingest_alert({
        "service": "core-db",
        "environment": "prod",
        "alert_name": "CascadingDeadlock",
        "severity": "CRITICAL",
        "error_code": "DISTRIBUTED_DEADLOCK_BGP_FLAP",
        "log_snippet": "CRITICAL: BGP route flap, corrupt TLS handshake, cross-region distributed database deadlock"
    })
    
    res_complex = router.evaluate_incident(incident_complex)
    assert res_complex.complexity_score >= 0.75
    assert res_complex.selected_tier == "TIER2_FRONTIER"

def test_guardrails_validation():
    # Allowed action
    is_valid, msg = ActionGuardrailController.validate_action("ACTION_RESTART_CONTAINER", "mock-nginx-prod")
    assert is_valid is True
    
    # Unauthorized action key -> MUST raise GuardrailViolationError
    with pytest.raises(GuardrailViolationError):
        ActionGuardrailController.validate_action("EXECUTE_ARBITRARY_BASH_SCRIPT", "mock-nginx-prod")

    # Command injection attempt -> MUST raise GuardrailViolationError
    with pytest.raises(GuardrailViolationError):
        ActionGuardrailController.validate_action("ACTION_RESTART_CONTAINER", "mock-nginx; rm -rf /")

def test_executor_safe_execution():
    executor = SafeRemediationExecutor()
    res = executor.execute_action("ACTION_RESTART_CONTAINER", "mock-nginx-prod")
    assert res.success is True
    assert "mock-nginx-prod" in res.target

def test_audit_logger(tmp_path):
    db_file = os.path.join(tmp_path, "test_audit.db")
    logger = AuditLogger(db_file)
    
    logger.log_incident_resolution(
        incident_id="INC-TEST01",
        service="payment-service",
        environment="prod",
        raw_alert_count=15,
        compression_ratio=82.5,
        complexity_score=0.31,
        routed_tier="TIER1_FAST",
        cache_hit=False,
        root_cause_summary="Memory pressure in payment service",
        remediation_key="ACTION_RESTART_CONTAINER",
        target_name="mock-nginx-prod",
        execution_status="SUCCESS",
        execution_message="Container restarted",
        baseline_cost_usd=0.045,
        actual_cost_usd=0.0015,
        mttr_seconds=10.5
    )
    
    metrics = logger.get_summary_metrics()
    assert metrics["total_incidents"] == 1
    assert metrics["total_savings_usd"] == pytest.approx(0.0435, abs=1e-3)
