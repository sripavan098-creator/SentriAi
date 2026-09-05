import time
import uuid
from collections import deque
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

class RawAlert(BaseModel):
    alert_id: str = Field(default_factory=lambda: f"ALT-{uuid.uuid4().hex[:8].upper()}")
    timestamp: float = Field(default_factory=time.time)
    service: str = "payment-service"
    environment: str = "prod"
    alert_name: str = "HighCPUUsage"
    severity: str = "CRITICAL"
    host: str = "node-01.prod.internal"
    metric_value: float = 94.5
    error_code: Optional[str] = "HTTP_500"
    log_snippet: str = "Process memory spike detected; response latency > 3500ms"
    tags: Dict[str, str] = Field(default_factory=lambda: {"env": "prod", "service": "payment"})

class IncidentCard(BaseModel):
    incident_id: str = Field(default_factory=lambda: f"INC-{uuid.uuid4().hex[:8].upper()}")
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    service: str
    environment: str
    raw_alert_ids: List[str] = Field(default_factory=list)
    raw_alerts_count: int = 0
    distinct_error_codes: List[str] = Field(default_factory=list)
    distinct_hosts: List[str] = Field(default_factory=list)
    severity_counts: Dict[str, int] = Field(default_factory=dict)
    summary_text: str = ""
    raw_payload_bytes: int = 0
    compressed_payload_bytes: int = 0
    compression_ratio: float = 0.0
    status: str = "CORRELATING"  # CORRELATING, ROUTED, REMEDIATED, FAILED

class AlertCorrelator:
    def __init__(self, window_seconds: float = 60.0):
        self.window_seconds = window_seconds
        self.raw_alerts_queue: deque = deque()
        self.active_incidents: Dict[str, IncidentCard] = {}
        self.alert_to_incident: Dict[str, str] = {}

    def ingest_alert(self, alert_data: Dict[str, Any]) -> Tuple[IncidentCard, RawAlert]:
        """
        Ingest a raw alert JSON dict, maintain the 60s sliding window,
        and group into an IncidentCard based on environment and service tags.
        """
        alert = RawAlert(**alert_data) if not isinstance(alert_data, RawAlert) else alert_data
        now = time.time()
        self.raw_alerts_queue.append((now, alert))
        
        # Evict old alerts outside sliding window
        self._purge_expired(now)

        # Correlate based on service + environment key
        group_key = f"{alert.environment}:{alert.service}"
        
        incident = None
        # Check if active correlating incident exists for this group_key
        for inc in self.active_incidents.values():
            inc_key = f"{inc.environment}:{inc.service}"
            if inc_key == group_key and inc.status == "CORRELATING":
                incident = inc
                break

        if not incident:
            incident = IncidentCard(
                service=alert.service,
                environment=alert.environment,
                status="CORRELATING"
            )
            self.active_incidents[incident.incident_id] = incident

        # Append alert details to incident
        incident.raw_alert_ids.append(alert.alert_id)
        incident.raw_alerts_count += 1
        self.alert_to_incident[alert.alert_id] = incident.incident_id

        if alert.error_code and alert.error_code not in incident.distinct_error_codes:
            incident.distinct_error_codes.append(alert.error_code)

        if alert.host and alert.host not in incident.distinct_hosts:
            incident.distinct_hosts.append(alert.host)

        sev = alert.severity.upper()
        incident.severity_counts[sev] = incident.severity_counts.get(sev, 0) + 1
        incident.updated_at = now

        # Update metrics & compressed summary text
        self._update_incident_compression(incident)

        return incident, alert

    def _purge_expired(self, current_time: float):
        cutoff = current_time - self.window_seconds
        while self.raw_alerts_queue and self.raw_alerts_queue[0][0] < cutoff:
            self.raw_alerts_queue.popleft()

    def _update_incident_compression(self, incident: IncidentCard):
        # Calculate raw payload size by collecting alerts for this incident
        relevant_alerts = [alt for t, alt in self.raw_alerts_queue if alt.alert_id in incident.raw_alert_ids]
        raw_str = "".join([str(alt.model_dump()) for alt in relevant_alerts])
        raw_bytes = len(raw_str.encode('utf-8'))
        incident.raw_payload_bytes = max(raw_bytes, incident.raw_alerts_count * 250)

        # Build compressed Incident Card summary text
        err_str = ", ".join(incident.distinct_error_codes) if incident.distinct_error_codes else "N/A"
        hosts_str = ", ".join(incident.distinct_hosts[:3])
        sample_logs = "; ".join(list(set([alt.log_snippet for alt in relevant_alerts]))[:2])

        incident.summary_text = (
            f"[INCIDENT CARD {incident.incident_id}] Service: {incident.service} ({incident.environment}) | "
            f"Alerts: {incident.raw_alerts_count} in {self.window_seconds}s window | "
            f"Errors: [{err_str}] | Hosts: [{hosts_str}] | Sample Logs: \"{sample_logs}\""
        )
        compressed_bytes = len(incident.summary_text.encode('utf-8'))
        incident.compressed_payload_bytes = compressed_bytes

        if incident.raw_payload_bytes > 0:
            ratio = (1.0 - (compressed_bytes / incident.raw_payload_bytes)) * 100.0
            incident.compression_ratio = max(0.0, min(98.0, round(ratio, 1)))
        else:
            incident.compression_ratio = 0.0

    def get_incident(self, incident_id: str) -> Optional[IncidentCard]:
        return self.active_incidents.get(incident_id)

    def get_all_incidents(self) -> List[IncidentCard]:
        return list(self.active_incidents.values())
