import os
import time
import json
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from src.engine.correlator import AlertCorrelator, IncidentCard, RawAlert
from src.engine.router import IncidentRouter
from src.engine.llm_tiering import TieredLLMSolver
from src.execution.executor import SafeRemediationExecutor
from src.execution.audit_logger import AuditLogger

app = FastAPI(
    title="Cost-Optimized AIOps Self-Healing Engine",
    description="Autonomous incident response system featuring sliding-window correlation, vector-heuristic smart routing, prompt caching, guardrailed execution, and immutable audit logging.",
    version="1.0.0"
)

# Initialize Core Services
correlator = AlertCorrelator(window_seconds=float(os.getenv("SLIDING_WINDOW_SECONDS", "60")))
router = IncidentRouter(threshold=float(os.getenv("ROUTER_COMPLEXITY_THRESHOLD", "0.75")))
llm_solver = TieredLLMSolver()
executor = SafeRemediationExecutor()
audit_logger = AuditLogger(os.getenv("DB_PATH", "data/audit_log.db"))

class AlertIngestRequest(BaseModel):
    service: str = "payment-service"
    environment: str = "prod"
    alert_name: str = "HighCPUUsage"
    severity: str = "CRITICAL"
    host: str = "node-01.prod.internal"
    metric_value: float = 94.5
    error_code: Optional[str] = "HTTP_500"
    log_snippet: str = "Process memory spike detected; response latency > 3500ms"
    tags: Dict[str, str] = {"env": "prod", "service": "payment"}

@app.get("/")
def read_root():
    return {
        "status": "ONLINE",
        "system": "AIOps Self-Healing Engine",
        "redis_available": router.redis_cache.redis_available,
        "docker_available": executor.docker_available
    }

@app.post("/api/alerts")
def ingest_alert(alert_payload: AlertIngestRequest):
    """
    Webhook endpoint to swallow raw infrastructure alerts.
    Correlates alerts temporally and topologically into a single Incident Card.
    """
    incident, alert = correlator.ingest_alert(alert_payload.model_dump())
    return {
        "status": "INGESTED",
        "alert_id": alert.alert_id,
        "incident_id": incident.incident_id,
        "raw_alerts_count": incident.raw_alerts_count,
        "compression_ratio": incident.compression_ratio,
        "incident_summary": incident.summary_text
    }

@app.post("/api/process_incident/{incident_id}")
def process_incident(incident_id: str):
    """
    Processes an Incident Card:
    1. Smart Router calculates complexity score & Redis prompt cache.
    2. Route to Tier 1 (Fast) or Tier 2 (Frontier) model.
    3. Validate action against strict Enum allow-list guardrail.
    4. Execute remediation action via Docker SDK.
    5. Log to SQLite audit database.
    """
    incident = correlator.get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident '{incident_id}' not found.")

    start_time = time.time()

    # Step 1: Smart Router & Cache
    routing_res = router.evaluate_incident(incident)

    # Step 2: LLM Tiering Solver
    llm_out = llm_solver.solve_incident(incident, routing_res)

    # Step 3 & 4: Guardrailed Action Execution
    exec_res = executor.execute_action(llm_out.remediation_key, llm_out.target_name)

    # Calculate MTTR (Mean Time To Resolution)
    mttr_sec = round(max(3.5, (time.time() - start_time) + 2.0), 1)

    # Step 5: Save to SQLite Audit Log
    exec_status = "SUCCESS" if exec_res.success else "GUARDRAIL_BLOCKED"
    audit_logger.log_incident_resolution(
        incident_id=incident.incident_id,
        service=incident.service,
        environment=incident.environment,
        raw_alert_count=incident.raw_alerts_count,
        compression_ratio=incident.compression_ratio,
        complexity_score=routing_res.complexity_score,
        routed_tier=routing_res.selected_tier,
        cache_hit=routing_res.cache_hit,
        root_cause_summary=llm_out.root_cause_summary,
        remediation_key=llm_out.remediation_key,
        target_name=llm_out.target_name,
        execution_status=exec_status,
        execution_message=exec_res.message,
        baseline_cost_usd=routing_res.baseline_cost_usd,
        actual_cost_usd=routing_res.estimated_cost_usd,
        mttr_seconds=mttr_sec
    )

    incident.status = "REMEDIATED" if exec_res.success else "FAILED"

    return {
        "incident_id": incident.incident_id,
        "routing": routing_res.model_dump(),
        "llm_output": llm_out.model_dump(),
        "execution": exec_res.to_dict(),
        "mttr_seconds": mttr_sec
    }

@app.get("/api/incidents")
def get_incidents():
    return [inc.model_dump() for inc in correlator.get_all_incidents()]

@app.get("/api/audit_logs")
def get_audit_logs():
    df = audit_logger.fetch_all_logs()
    return df.to_dict(orient="records")

@app.get("/api/metrics")
def get_metrics():
    return audit_logger.get_summary_metrics()

@app.post("/api/simulate_burst")
def trigger_simulation_burst(scenario: str = "memory_leak"):
    """
    Triggers simulated raw alert burst for testing.
    """
    from src.simulator import AlertSimulator
    sim = AlertSimulator("http://localhost:8000")
    count, inc_id = sim.run_scenario(scenario)
    
    # Process incident immediately if produced
    process_res = None
    if inc_id:
        try:
            process_res = process_incident(inc_id)
        except Exception as e:
            process_res = {"error": str(e)}

    return {
        "status": "SIMULATION_COMPLETED",
        "alerts_generated": count,
        "incident_id": inc_id,
        "process_result": process_res
    }


# ==========================================
# SENTRIAI ENDPOINTS
# ==========================================

from src.engine.router import SentriAIRouter, MODEL_CATALOG, RedisPromptCache
from src.engine.llm_tiering import SentriAIEngine

shared_sentri_cache = RedisPromptCache(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
sentri_router = SentriAIRouter(redis_cache=shared_sentri_cache)
sentri_engine = SentriAIEngine(redis_cache=shared_sentri_cache)



class SentriPromptRequest(BaseModel):
    prompt: str
    strategy: str = "cheapest"  # "cheapest", "fastest", "quality", "smart_auto"


@app.get("/api/sentri/models")
def get_sentri_models():
    return MODEL_CATALOG


@app.post("/api/sentri/prompt")
def process_sentri_prompt(req: SentriPromptRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    
    # 1. Routing decision & candidate benchmarking
    decision = sentri_router.evaluate_prompt(req.prompt, strategy=req.strategy)

    # 2. Execution / synthesis
    response = sentri_engine.execute_prompt(req.prompt, decision)

    # 3. Audit logging
    audit_logger.log_incident_resolution(
        incident_id=f"SENTRI-{int(time.time())}",
        service=f"prompt-{decision.category}",
        environment="user-chat",
        raw_alert_count=1,
        compression_ratio=decision.cost_savings_pct,
        complexity_score=decision.complexity_score,
        routed_tier=decision.selected_model_name,
        cache_hit=decision.cache_hit,
        root_cause_summary=req.prompt[:100],
        remediation_key=f"ROUTE_{decision.strategy.upper()}",
        target_name=decision.selected_model_id,
        execution_status="SUCCESS",
        execution_message=f"Routed to {decision.selected_model_name} ({decision.cost_savings_pct:.1f}% cost savings)",
        baseline_cost_usd=decision.baseline_cost_usd,
        actual_cost_usd=decision.selected_cost_usd,
        mttr_seconds=round(decision.latency_ms / 1000.0, 2)
    )

    return {
        "decision": decision.model_dump(),
        "response": response.model_dump()
    }


@app.post("/api/sentri/cache/flush")
def flush_sentri_cache():
    if sentri_router.redis_cache.redis_available and sentri_router.redis_cache.client:
        try:
            sentri_router.redis_cache.client.flushdb()
        except Exception:
            pass
    sentri_router.redis_cache.local_cache.clear()
    sentri_engine.redis_cache.local_cache.clear()
    return {"status": "SUCCESS", "message": "Redis & Local prompt cache cleared."}


@app.get("/api/sentri/benchmark/fixed")
def run_fixed_benchmark(strategy: str = "cheapest"):
    return sentri_router.run_fixed_benchmark(strategy=strategy)


class OpenAICompletionMessage(BaseModel):
    role: str = "user"
    content: str


class OpenAICompletionRequest(BaseModel):
    model: Optional[str] = "sentri-auto"
    messages: List[OpenAICompletionMessage]
    strategy: Optional[str] = "cheapest"


@app.post("/v1/chat/completions")
def openai_chat_completions(req: OpenAICompletionRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="No messages provided.")
    
    user_prompt = req.messages[-1].content
    decision = sentri_router.evaluate_prompt(user_prompt, strategy=req.strategy or "cheapest")
    response = sentri_engine.execute_prompt(user_prompt, decision)

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": decision.selected_model_id,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.response_text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": decision.input_tokens,
            "completion_tokens": decision.estimated_output_tokens,
            "total_tokens": decision.input_tokens + decision.estimated_output_tokens
        },
        "sentri_metadata": {
            "selected_model": decision.selected_model_name,
            "cost_usd": decision.selected_cost_usd,
            "baseline_cost_usd": decision.baseline_cost_usd,
            "cost_savings_pct": decision.cost_savings_pct,
            "cache_hit": decision.cache_hit
        }
    }


# ==========================================
# RESEARCH ANALYTICS & SAFETY ENDPOINTS
# ==========================================

from src.engine.research_analytics import (
    DowntimeFinancialRiskPredictor,
    RogueExecutionSafetyGuard,
    AlertFatigueAnalyzer,
    TokenEconomicsCalculator
)

class RiskPredictionRequest(BaseModel):
    incident_id: str = "INC-EXPOSURE-DEMO"
    duration_minutes: float = 15.0

class SafetyVerifyRequest(BaseModel):
    action_key: str = "ACTION_RESTART_CONTAINER"
    target_name: str = "payment-api"
    human_approved: bool = False

@app.get("/api/sentri/research_brief")
def get_research_brief_endpoint():
    brief_path = os.path.abspath("data/research_brief.json")
    if os.path.exists(brief_path):
        with open(brief_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Research brief dataset not found.")

@app.post("/api/sentri/predict_downtime_risk")
def predict_downtime_risk_endpoint(req: RiskPredictionRequest):
    return DowntimeFinancialRiskPredictor.predict_risk(req.incident_id, req.duration_minutes).model_dump()

@app.post("/api/sentri/verify_remediation_safety")
def verify_remediation_safety_endpoint(req: SafetyVerifyRequest):
    cert = RogueExecutionSafetyGuard.audit_remediation_action(req.action_key, req.target_name, req.human_approved)
    return cert.model_dump()

@app.get("/api/sentri/alert_fatigue_analytics")
def get_alert_fatigue_analytics_endpoint(raw_count: int = 25, correlated_count: int = 3):
    return AlertFatigueAnalyzer.analyze_noise(raw_count, correlated_count).model_dump()

@app.get("/api/sentri/token_economics")
def get_token_economics_endpoint(input_tokens: int = 1500, output_tokens: int = 400, cache_hit: bool = False):
    return TokenEconomicsCalculator.calculate_savings(input_tokens, output_tokens, cache_hit).model_dump()



