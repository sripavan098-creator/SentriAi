import os
import time
import random
import requests
from typing import Tuple, Optional, List, Dict, Any

class AlertSimulator:
    def __init__(self, base_url: str = None):
        self.base_url = (base_url or os.getenv("API_ENDPOINT", "http://localhost:8001")).rstrip("/")

    def _post_alert(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            res = requests.post(f"{self.base_url}/api/alerts", json=payload, timeout=3.0)
            if res.status_code == 200:
                return res.json()
        except Exception:
            pass
        return None

    def run_scenario(self, scenario_name: str = "memory_leak") -> Tuple[int, Optional[str]]:
        alerts = []
        
        if scenario_name == "memory_leak":
            services = ["payment-service"]
            hosts = ["node-prod-01", "node-prod-02", "node-prod-03"]
            errors = ["OOMKilled", "HeapSpaceExhausted", "HTTP_500", "WorkerTimeout"]
            snippets = [
                "cgroup out of memory: Process (java) total-vm:4120300kB, anon-rss:3890200kB killed",
                "java.lang.OutOfMemoryError: Java heap space at com.payment.Gateway.process()",
                "HTTP 500 Internal Server Error - Connection pool timeout during heap GC pause",
                "Container payment-api-v2 memory limit 2048MB reached (98.6% utilization)"
            ]
            count = random.randint(15, 30)
            for i in range(count):
                alerts.append({
                    "service": "payment-service",
                    "environment": "prod",
                    "alert_name": "MemoryThresholdExceeded",
                    "severity": "CRITICAL" if i % 2 == 0 else "WARNING",
                    "host": random.choice(hosts),
                    "metric_value": round(90.0 + random.uniform(1.0, 9.5), 1),
                    "error_code": random.choice(errors),
                    "log_snippet": random.choice(snippets),
                    "tags": {"env": "prod", "service": "payment"}
                })

        elif scenario_name == "redis_saturation":
            hosts = ["redis-node-01", "redis-node-02"]
            count = random.randint(10, 20)
            for i in range(count):
                alerts.append({
                    "service": "redis-cache",
                    "environment": "prod",
                    "alert_name": "RedisMaxMemoryReached",
                    "severity": "CRITICAL",
                    "host": random.choice(hosts),
                    "metric_value": 99.1,
                    "error_code": "OOM_COMMAND_NOT_ALLOWED",
                    "log_snippet": "OOM command not allowed when used memory > 'maxmemory'. Eviction policy noeviction failed.",
                    "tags": {"env": "prod", "service": "cache"}
                })

        elif scenario_name == "cascading_deadlock":
            count = random.randint(25, 45)
            for i in range(count):
                alerts.append({
                    "service": "payment-service",
                    "environment": "prod",
                    "alert_name": "CascadingSystemDeadlock",
                    "severity": "CRITICAL",
                    "host": f"core-db-node-0{i%4+1}",
                    "metric_value": 100.0,
                    "error_code": "DISTRIBUTED_DEADLOCK_BGP_FLAP",
                    "log_snippet": "CRITICAL: BGP route flap, corrupt TLS handshake timeout, cross-region distributed database deadlock across microservices.",
                    "tags": {"env": "prod", "service": "payment"}
                })

        else:  # traffic_spike
            count = random.randint(8, 15)
            for i in range(count):
                alerts.append({
                    "service": "payment-service",
                    "environment": "prod",
                    "alert_name": "WorkerQueueBacklog",
                    "severity": "WARNING",
                    "host": "node-prod-01",
                    "metric_value": 85.0,
                    "error_code": "HTTP_503",
                    "log_snippet": "Worker pool saturated under 4500 RPS traffic surge; queue length > 500 requests.",
                    "tags": {"env": "prod", "service": "payment"}
                })

        last_incident_id = None
        for alert_payload in alerts:
            res = self._post_alert(alert_payload)
            if res and "incident_id" in res:
                last_incident_id = res["incident_id"]
            time.sleep(0.02)  # fast burst stream

        return len(alerts), last_incident_id

if __name__ == "__main__":
    import sys
    sim = AlertSimulator()
    scenario = sys.argv[1] if len(sys.argv) > 1 else "memory_leak"
    cnt, inc_id = sim.run_scenario(scenario)
    print(f"[SIMULATOR] Fired {cnt} raw alerts for scenario '{scenario}'. Generated Incident ID: {inc_id}")
