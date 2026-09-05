import os
import sqlite3
import time
from typing import List, Dict, Any, Optional
import pandas as pd

class AuditLogger:
    def __init__(self, db_path: str = "data/audit_log.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    iso_time TEXT,
                    incident_id TEXT UNIQUE,
                    service TEXT,
                    environment TEXT,
                    raw_alert_count INTEGER,
                    compression_ratio REAL,
                    complexity_score REAL,
                    routed_tier TEXT,
                    cache_hit INTEGER,
                    root_cause_summary TEXT,
                    remediation_key TEXT,
                    target_name TEXT,
                    execution_status TEXT,
                    execution_message TEXT,
                    baseline_cost_usd REAL,
                    actual_cost_usd REAL,
                    savings_cost_usd REAL,
                    mttr_seconds REAL
                );
            """)
            conn.commit()

    def log_incident_resolution(
        self,
        incident_id: str,
        service: str,
        environment: str,
        raw_alert_count: int,
        compression_ratio: float,
        complexity_score: float,
        routed_tier: str,
        cache_hit: bool,
        root_cause_summary: str,
        remediation_key: str,
        target_name: str,
        execution_status: str,
        execution_message: str,
        baseline_cost_usd: float,
        actual_cost_usd: float,
        mttr_seconds: float = 12.5
    ) -> int:
        now = time.time()
        iso_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
        savings_cost_usd = max(0.0, baseline_cost_usd - actual_cost_usd)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO audit_log (
                    timestamp, iso_time, incident_id, service, environment,
                    raw_alert_count, compression_ratio, complexity_score,
                    routed_tier, cache_hit, root_cause_summary, remediation_key,
                    target_name, execution_status, execution_message,
                    baseline_cost_usd, actual_cost_usd, savings_cost_usd, mttr_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now, iso_time, incident_id, service, environment,
                raw_alert_count, compression_ratio, complexity_score,
                routed_tier, 1 if cache_hit else 0, root_cause_summary, remediation_key,
                target_name, execution_status, execution_message,
                baseline_cost_usd, actual_cost_usd, savings_cost_usd, mttr_seconds
            ))
            conn.commit()
            return cursor.lastrowid

    def fetch_all_logs(self) -> pd.DataFrame:
        with self._get_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM audit_log ORDER BY id DESC", conn)
            return df

    def get_summary_metrics(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_incidents,
                    SUM(savings_cost_usd) as total_savings,
                    SUM(baseline_cost_usd) as total_baseline,
                    SUM(actual_cost_usd) as total_actual,
                    AVG(cache_hit) as cache_hit_rate,
                    AVG(mttr_seconds) as avg_mttr,
                    SUM(CASE WHEN routed_tier LIKE '%TIER1%' THEN 1 ELSE 0 END) as tier1_count,
                    SUM(CASE WHEN routed_tier LIKE '%TIER2%' THEN 1 ELSE 0 END) as tier2_count
                FROM audit_log
            """)
            row = dict(cursor.fetchone() or {})
            
            total = row.get("total_incidents") or 0
            if total > 0:
                cache_hit_pct = round((row.get("cache_hit_rate") or 0.0) * 100.0, 1)
                tier1_pct = round(((row.get("tier1_count") or 0) / total) * 100.0, 1)
                tier2_pct = round(((row.get("tier2_count") or 0) / total) * 100.0, 1)
            else:
                cache_hit_pct = 0.0
                tier1_pct = 0.0
                tier2_pct = 0.0

            return {
                "total_incidents": total,
                "total_savings_usd": round(row.get("total_savings") or 0.0, 4),
                "total_baseline_cost_usd": round(row.get("total_baseline") or 0.0, 4),
                "total_actual_cost_usd": round(row.get("total_actual") or 0.0, 4),
                "cache_hit_rate_pct": cache_hit_pct,
                "avg_mttr_seconds": round(row.get("avg_mttr") or 0.0, 1),
                "tier1_routing_pct": tier1_pct,
                "tier2_routing_pct": tier2_pct
            }
