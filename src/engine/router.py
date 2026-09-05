import os
import re
import json
import hashlib
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from pydantic import BaseModel
import redis

import importlib

# Try importing sentence_transformers for MiniLM local embeddings
try:
    _st_mod = importlib.import_module("sentence_transformers")
    SentenceTransformer = getattr(_st_mod, "SentenceTransformer")
    HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False


class RoutingResult(BaseModel):
    incident_id: str
    complexity_score: float
    selected_tier: str  # "TIER1_FAST" or "TIER2_FRONTIER"
    matched_vector_id: Optional[str] = None
    heuristic_breakdown: Dict[str, float]
    cache_hit: bool = False
    cache_key: str
    estimated_token_savings: int = 0
    estimated_cost_usd: float = 0.0
    baseline_cost_usd: float = 0.0

class RedisPromptCache:
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.client = None
        self.local_cache = {}  # In-memory fallback if Redis is offline
        try:
            self.client = redis.Redis.from_url(redis_url, socket_timeout=1.5, socket_connect_timeout=1.5)
            self.client.ping()
            self.redis_available = True
        except Exception:
            self.redis_available = False
            self.client = None

    def get_cache_key(self, incident_summary: str) -> str:
        # Normalize punctuation, lowercasing, and whitespace for robust prompt cache matching
        cleaned = re.sub(r'[^\w\s]', '', incident_summary.lower()).strip()
        normalized = " ".join(cleaned.split())
        if not normalized:
            normalized = " ".join(incident_summary.lower().strip().split())
        return f"aiops_cache:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()}"


    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        if self.redis_available and self.client:
            try:
                cached = self.client.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception:
                pass
        return self.local_cache.get(cache_key)

    def set(self, cache_key: str, data: Dict[str, Any], ttl_seconds: int = 3600):
        val_str = json.dumps(data)
        if self.redis_available and self.client:
            try:
                self.client.setex(cache_key, ttl_seconds, val_str)
            except Exception:
                pass
        self.local_cache[cache_key] = data

class IncidentRouter:
    def __init__(self, templates_path: str = "data/mock_templates.json", threshold: float = 0.75):
        self.threshold = threshold
        self.redis_cache = RedisPromptCache(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        self.known_vectors = self._load_known_vectors(templates_path)
        
        # Initialize local MiniLM model or TF-IDF keyword vectorizer fallback
        self.embedding_model = None
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                # Load lightweight local MiniLM model
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                self.embedding_model = None

        # Precompute reference vector embeddings for known vectors
        self.vector_embeddings = {}
        if self.embedding_model:
            for vec in self.known_vectors:
                text_to_embed = f"{vec['name']} {vec['description']} {' '.join(vec.get('vector_keywords', []))}"
                self.vector_embeddings[vec['id']] = self.embedding_model.encode(text_to_embed)

    def _load_known_vectors(self, templates_path: str) -> list:
        if os.path.exists(templates_path):
            try:
                with open(templates_path, 'r') as f:
                    data = json.load(f)
                    return data.get("known_incident_vectors", [])
            except Exception:
                pass
        return []

    def _calculate_vector_similarity(self, text: str) -> Tuple[float, Optional[str]]:
        if not self.known_vectors:
            return 0.0, None

        if self.embedding_model and self.vector_embeddings:
            target_emb = self.embedding_model.encode(text)
            best_sim = 0.0
            best_id = None
            for vec_id, vec_emb in self.vector_embeddings.items():
                # Cosine similarity
                sim = np.dot(target_emb, vec_emb) / (np.linalg.norm(target_emb) * np.linalg.norm(vec_emb) + 1e-9)
                if sim > best_sim:
                    best_sim = float(sim)
                    best_id = vec_id
            return best_sim, best_id
        else:
            # Fallback keyword match similarity
            text_lower = text.lower()
            best_match_ratio = 0.0
            best_id = None
            for vec in self.known_vectors:
                keywords = vec.get("vector_keywords", [])
                hits = sum(1 for kw in keywords if kw.lower() in text_lower)
                ratio = hits / max(1, len(keywords))
                if ratio > best_match_ratio:
                    best_match_ratio = ratio
                    best_id = vec["id"]
            return best_match_ratio, best_id

    def evaluate_incident(self, incident_card: Any) -> RoutingResult:
        summary_text = getattr(incident_card, "summary_text", str(incident_card))
        incident_id = getattr(incident_card, "incident_id", "INC-UNKNOWN")
        raw_count = getattr(incident_card, "raw_alerts_count", 1)
        error_codes = getattr(incident_card, "distinct_error_codes", [])

        # 1. Redis Dual-Layer Prompt Cache Check
        cache_key = self.redis_cache.get_cache_key(summary_text)
        cached_data = self.redis_cache.get(cache_key)

        # Baseline cost (assuming Tier 2 Frontier model for raw un-routed, un-compressed alerts)
        raw_bytes = getattr(incident_card, "raw_payload_bytes", len(summary_text.encode('utf-8')) * 5)
        approx_raw_tokens = max(150, raw_bytes // 4)
        baseline_cost_usd = (approx_raw_tokens / 1000.0) * 0.015  # ~$0.015 per 1k tokens baseline

        if cached_data:
            return RoutingResult(
                incident_id=incident_id,
                complexity_score=cached_data.get("complexity_score", 0.2),
                selected_tier="PROMPT_CACHE_HIT",
                matched_vector_id=cached_data.get("matched_vector_id"),
                heuristic_breakdown={"cache_hit": 1.0},
                cache_hit=True,
                cache_key=cache_key,
                estimated_token_savings=approx_raw_tokens,
                estimated_cost_usd=0.0,  # 100% savings on cache hit
                baseline_cost_usd=baseline_cost_usd
            )

        # 2. Heuristic Complexity Scoring
        text_len = len(summary_text)
        len_score = min(1.0, text_len / 1000.0)
        
        # Error code multiplicity heuristic
        error_score = min(1.0, len(error_codes) / 5.0)

        # Regex patterns for cascading critical failure indicators
        cascading_patterns = [
            r"deadlock", r"split brain", r"kernel panic", r"bgp route flap",
            r"corrupt tls", r"cascading", r"database partition"
        ]
        cascading_hits = sum(1 for pat in cascading_patterns if re.search(pat, summary_text, re.IGNORECASE))
        cascading_score = min(1.0, cascading_hits * 0.4)

        # Alert count multiplier
        volume_score = min(1.0, raw_count / 20.0)

        # 3. Vector Similarity check
        vector_sim, best_vector_id = self._calculate_vector_similarity(summary_text)
        
        # Find matched vector typical complexity if available
        matched_vec_obj = next((v for v in self.known_vectors if v["id"] == best_vector_id), None)
        vector_complexity_bias = matched_vec_obj["typical_complexity"] if matched_vec_obj and vector_sim > 0.3 else (0.85 if cascading_hits > 0 else 0.5)

        # Combined Complexity Score calculation
        heuristic_score = (len_score * 0.2) + (error_score * 0.2) + (cascading_score * 0.4) + (volume_score * 0.2)
        
        if matched_vec_obj and (vector_sim > 0.4 or cascading_hits > 0):
            final_complexity = (heuristic_score * 0.3) + (vector_complexity_bias * 0.7)
        else:
            final_complexity = (heuristic_score * 0.6) + (vector_complexity_bias * 0.4)

        final_complexity = max(0.05, min(0.99, round(final_complexity, 2)))

        # 4. Routing Decision
        if final_complexity < self.threshold or (vector_sim > 0.8 and vector_complexity_bias < 0.5):
            selected_tier = "TIER1_FAST"  # Llama-3-8B Fast model
            token_cost_per_1k = 0.0005
        else:
            selected_tier = "TIER2_FRONTIER"  # Llama-3-70B Frontier model
            token_cost_per_1k = 0.015

        approx_compressed_tokens = max(50, len(summary_text.encode('utf-8')) // 4)
        estimated_cost_usd = (approx_compressed_tokens / 1000.0) * token_cost_per_1k

        return RoutingResult(
            incident_id=incident_id,
            complexity_score=final_complexity,
            selected_tier=selected_tier,
            matched_vector_id=best_vector_id if vector_sim > 0.5 else None,
            heuristic_breakdown={
                "text_length": len_score,
                "error_multiplicity": error_score,
                "cascading_keywords": cascading_score,
                "vector_similarity": float(vector_sim)
            },
            cache_hit=False,
            cache_key=cache_key,
            estimated_token_savings=max(0, approx_raw_tokens - approx_compressed_tokens),
            estimated_cost_usd=round(estimated_cost_usd, 6),
            baseline_cost_usd=round(baseline_cost_usd, 6)
        )


# ==========================================
# SENTRIAI MULTI-LLM ROUTING & BENCHMARK CATALOG
# ==========================================

MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "gemini-2.0-flash": {
        "name": "Gemini 2.0 Flash",
        "provider": "Google",
        "input_price_per_m": 0.10,
        "output_price_per_m": 0.40,
        "avg_latency_ms": 180,
        "tokens_per_sec": 130,
        "quality_scores": {"coding": 88, "math": 90, "reasoning": 89, "creative": 87, "chat": 92, "general": 90},
        "logo_svg_name": "google",
        "family": "gemini"
    },
    "gemini-1.5-flash": {
        "name": "Gemini 1.5 Flash",
        "provider": "Google",
        "input_price_per_m": 0.075,
        "output_price_per_m": 0.30,
        "avg_latency_ms": 150,
        "tokens_per_sec": 140,
        "quality_scores": {"coding": 84, "math": 86, "reasoning": 85, "creative": 85, "chat": 90, "general": 87},
        "logo_svg_name": "google",
        "family": "gemini"
    },
    "gemini-1.5-pro": {
        "name": "Gemini 1.5 Pro",
        "provider": "Google",
        "input_price_per_m": 1.25,
        "output_price_per_m": 5.00,
        "avg_latency_ms": 420,
        "tokens_per_sec": 85,
        "quality_scores": {"coding": 94, "math": 95, "reasoning": 96, "creative": 92, "chat": 94, "general": 95},
        "logo_svg_name": "google",
        "family": "gemini"
    },
    "gpt-4o": {
        "name": "GPT-4o",
        "provider": "OpenAI",
        "input_price_per_m": 2.50,
        "output_price_per_m": 10.00,
        "avg_latency_ms": 380,
        "tokens_per_sec": 95,
        "quality_scores": {"coding": 96, "math": 96, "reasoning": 97, "creative": 95, "chat": 96, "general": 96},
        "logo_svg_name": "openai",
        "family": "gpt"
    },
    "gpt-4o-mini": {
        "name": "GPT-4o-mini",
        "provider": "OpenAI",
        "input_price_per_m": 0.15,
        "output_price_per_m": 0.60,
        "avg_latency_ms": 140,
        "tokens_per_sec": 135,
        "quality_scores": {"coding": 87, "math": 89, "reasoning": 88, "creative": 86, "chat": 91, "general": 89},
        "logo_svg_name": "openai",
        "family": "gpt"
    },
    "gpt-3.5-turbo": {
        "name": "GPT-3.5 Turbo",
        "provider": "OpenAI",
        "input_price_per_m": 0.50,
        "output_price_per_m": 1.50,
        "avg_latency_ms": 220,
        "tokens_per_sec": 110,
        "quality_scores": {"coding": 80, "math": 80, "reasoning": 81, "creative": 82, "chat": 85, "general": 82},
        "logo_svg_name": "openai",
        "family": "gpt"
    },
    "o1-mini": {
        "name": "o1-mini",
        "provider": "OpenAI",
        "input_price_per_m": 1.10,
        "output_price_per_m": 4.40,
        "avg_latency_ms": 650,
        "tokens_per_sec": 70,
        "quality_scores": {"coding": 97, "math": 98, "reasoning": 98, "creative": 80, "chat": 84, "general": 92},
        "logo_svg_name": "openai",
        "family": "gpt"
    },
    "claude-3.5-sonnet": {
        "name": "Claude 3.5 Sonnet",
        "provider": "Anthropic",
        "input_price_per_m": 3.00,
        "output_price_per_m": 15.00,
        "avg_latency_ms": 410,
        "tokens_per_sec": 90,
        "quality_scores": {"coding": 98, "math": 95, "reasoning": 96, "creative": 98, "chat": 96, "general": 97},
        "logo_svg_name": "anthropic",
        "family": "claude"
    },
    "claude-3.5-haiku": {
        "name": "Claude 3.5 Haiku",
        "provider": "Anthropic",
        "input_price_per_m": 0.80,
        "output_price_per_m": 4.00,
        "avg_latency_ms": 160,
        "tokens_per_sec": 125,
        "quality_scores": {"coding": 89, "math": 88, "reasoning": 89, "creative": 90, "chat": 92, "general": 90},
        "logo_svg_name": "anthropic",
        "family": "claude"
    },
    "claude-3-opus": {
        "name": "Claude 3 Opus",
        "provider": "Anthropic",
        "input_price_per_m": 15.00,
        "output_price_per_m": 75.00,
        "avg_latency_ms": 850,
        "tokens_per_sec": 55,
        "quality_scores": {"coding": 96, "math": 95, "reasoning": 97, "creative": 99, "chat": 95, "general": 96},
        "logo_svg_name": "anthropic",
        "family": "claude"
    },
    "deepseek-v3": {
        "name": "DeepSeek V3",
        "provider": "DeepSeek",
        "input_price_per_m": 0.14,
        "output_price_per_m": 0.28,
        "avg_latency_ms": 190,
        "tokens_per_sec": 120,
        "quality_scores": {"coding": 93, "math": 92, "reasoning": 92, "creative": 89, "chat": 92, "general": 92},
        "logo_svg_name": "deepseek",
        "family": "deepseek"
    },
    "deepseek-r1": {
        "name": "DeepSeek R1",
        "provider": "DeepSeek",
        "input_price_per_m": 0.55,
        "output_price_per_m": 2.19,
        "avg_latency_ms": 720,
        "tokens_per_sec": 65,
        "quality_scores": {"coding": 97, "math": 98, "reasoning": 99, "creative": 85, "chat": 88, "general": 94},
        "logo_svg_name": "deepseek",
        "family": "deepseek"
    },
    "llama-3.3-70b": {
        "name": "Llama 3.3 70B",
        "provider": "Meta",
        "input_price_per_m": 0.35,
        "output_price_per_m": 0.70,
        "avg_latency_ms": 230,
        "tokens_per_sec": 110,
        "quality_scores": {"coding": 90, "math": 90, "reasoning": 91, "creative": 91, "chat": 93, "general": 91},
        "logo_svg_name": "meta",
        "family": "llama"
    }
}


class SentriPromptBenchmark(BaseModel):
    model_id: str
    name: str
    provider: str
    logo_svg_name: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    latency_ms: float
    speed_tok_per_sec: float
    quality_score: int
    reasoning_tier: str
    cache_hit: bool
    is_winner: bool
    win_reason: str


class SentriRoutingDecision(BaseModel):
    prompt: str
    category: str
    complexity_score: float
    input_tokens: int
    estimated_output_tokens: int
    strategy: str
    selected_model_id: str
    selected_model_name: str
    baseline_model_id: str = "gpt-4o"
    baseline_cost_usd: float
    selected_cost_usd: float
    cost_savings_pct: float
    latency_ms: float
    cache_hit: bool = False
    cache_key: str = ""
    semantic_similarity: float = 0.0
    benchmarks: List[SentriPromptBenchmark]


class SentriAIRouter:
    def __init__(self, redis_url: str = "redis://localhost:6379/0", redis_cache: Optional[RedisPromptCache] = None):
        self.redis_cache = redis_cache if redis_cache is not None else RedisPromptCache(redis_url)
        self.catalog = MODEL_CATALOG


    def classify_prompt(self, prompt: str) -> Tuple[str, float]:
        text_lower = prompt.lower()
        
        # Category detection heuristics
        code_keywords = ["def ", "class ", "function", "import ", "sql", "code", "bug", "python", "javascript", "docker", "api", "refactor", "async", "struct"]
        math_keywords = ["integral", "derivative", "matrix", "equation", "proof", "math", "calculate", "probability", "quantum", "solve", "formula"]
        reasoning_keywords = ["why", "analyze", "evaluate", "compare", "root cause", "architecture", "deadlock", "tradeoff", "strategy", "consequence"]
        creative_keywords = ["write a story", "poem", "essay", "headline", "blog", "creative", "slogan", "marketing", "copy"]

        code_hits = sum(1 for kw in code_keywords if kw in text_lower)
        math_hits = sum(1 for kw in math_keywords if kw in text_lower)
        reason_hits = sum(1 for kw in reasoning_keywords if kw in text_lower)
        creative_hits = sum(1 for kw in creative_keywords if kw in text_lower)

        if code_hits >= 2 or re.search(r"```[a-z]*", text_lower):
            category = "coding"
        elif math_hits >= 2 or re.search(r"\\[a-z]+", text_lower):
            category = "math"
        elif creative_hits >= 1 and code_hits == 0:
            category = "creative"
        elif reason_hits >= 2 or len(prompt.split()) > 60:
            category = "reasoning"
        else:
            category = "chat"

        # Complexity score calculation
        token_estimate = max(10, len(prompt.split()) * 1.3)
        complexity = min(1.0, (token_estimate / 300.0) * 0.4 + (code_hits + math_hits + reason_hits) * 0.15)
        complexity = round(max(0.1, min(0.98, complexity)), 2)

        return category, complexity

    def evaluate_prompt(self, prompt: str, strategy: str = "cheapest") -> SentriRoutingDecision:
        category, complexity = self.classify_prompt(prompt)

        input_tokens = max(12, int(len(prompt.split()) * 1.35))
        if category in ["coding", "math", "reasoning"]:
            estimated_output_tokens = max(120, int(input_tokens * 2.8))
        else:
            estimated_output_tokens = max(80, int(input_tokens * 1.8))

        # Prompt Cache check
        cache_key = self.redis_cache.get_cache_key(prompt)
        cached_res = self.redis_cache.get(cache_key)
        cache_hit = cached_res is not None

        # Build benchmarks for all candidate models
        benchmarks = []
        best_model_id = None
        best_score = float('inf') if strategy == "cheapest" else -1.0
        
        baseline_model_id = "gpt-4o"
        baseline_info = self.catalog[baseline_model_id]
        baseline_cost = ((input_tokens / 1_000_000) * baseline_info["input_price_per_m"]) + \
                        ((estimated_output_tokens / 1_000_000) * baseline_info["output_price_per_m"])

        for model_id, model_info in self.catalog.items():
            quality = model_info["quality_scores"].get(category, model_info["quality_scores"]["general"])
            cost = ((input_tokens / 1_000_000) * model_info["input_price_per_m"]) + \
                   ((estimated_output_tokens / 1_000_000) * model_info["output_price_per_m"])
            
            eff_cost = 0.0 if cache_hit else cost
            eff_latency = 12.0 if cache_hit else model_info["avg_latency_ms"]

            if strategy == "cheapest":
                if complexity > 0.7 and quality < 85:
                    score = cost * 5.0
                else:
                    score = cost
                if score < best_score:
                    best_score = score
                    best_model_id = model_id
            elif strategy == "fastest":
                score = eff_latency
                if score < best_score:
                    best_score = score
                    best_model_id = model_id
            elif strategy == "quality":
                score = quality
                if score > best_score:
                    best_score = score
                    best_model_id = model_id
            else: # smart_auto
                score = (quality ** 1.5) / (max(0.00001, cost * 1000) * (eff_latency ** 0.5))
                if score > best_score:
                    best_score = score
                    best_model_id = model_id

        if not best_model_id:
            best_model_id = "gemini-2.0-flash" if strategy == "cheapest" else "gpt-4o"

        winner_info = self.catalog[best_model_id]
        selected_cost = 0.0 if cache_hit else (((input_tokens / 1_000_000) * winner_info["input_price_per_m"]) + \
                                                ((estimated_output_tokens / 1_000_000) * winner_info["output_price_per_m"]))

        savings_pct = 100.0 if cache_hit else max(0.0, round(((baseline_cost - selected_cost) / max(0.00001, baseline_cost)) * 100, 1))

        for model_id, model_info in self.catalog.items():
            quality = model_info["quality_scores"].get(category, model_info["quality_scores"]["general"])
            cost = 0.0 if (cache_hit and model_id == best_model_id) else \
                   (((input_tokens / 1_000_000) * model_info["input_price_per_m"]) + ((estimated_output_tokens / 1_000_000) * model_info["output_price_per_m"]))
            is_win = (model_id == best_model_id)
            
            if is_win:
                if cache_hit:
                    win_reason = "⚡ Instant 0ms Redis Prompt Cache Hit (100% Free)"
                elif strategy == "cheapest":
                    win_reason = f"💰 Lowest cost for {category.upper()} (${cost:.6f})"
                elif strategy == "fastest":
                    win_reason = f"🚀 Ultra low latency ({model_info['avg_latency_ms']}ms)"
                elif strategy == "quality":
                    win_reason = f"🎯 Highest accuracy score ({quality}/100)"
                else:
                    win_reason = f"⚖️ Optimal Pareto balance of price, quality & speed"
            else:
                win_reason = f"Alternative ({quality}/100 quality, ${cost:.5f})"

            benchmarks.append(SentriPromptBenchmark(
                model_id=model_id,
                name=model_info["name"],
                provider=model_info["provider"],
                logo_svg_name=model_info["logo_svg_name"],
                input_tokens=input_tokens,
                output_tokens=estimated_output_tokens,
                estimated_cost_usd=round(cost, 6),
                latency_ms=12.0 if (cache_hit and is_win) else float(model_info["avg_latency_ms"]),
                speed_tok_per_sec=float(model_info["tokens_per_sec"]),
                quality_score=quality,
                reasoning_tier="Frontier Reasoning" if quality >= 95 else ("High Speed" if model_info["avg_latency_ms"] < 200 else "Balanced"),
                cache_hit=cache_hit if is_win else False,
                is_winner=is_win,
                win_reason=win_reason
            ))

        benchmarks.sort(key=lambda b: (not b.is_winner, b.estimated_cost_usd))

        return SentriRoutingDecision(
            prompt=prompt,
            category=category,
            complexity_score=complexity,
            input_tokens=input_tokens,
            estimated_output_tokens=estimated_output_tokens,
            strategy=strategy,
            selected_model_id=best_model_id,
            selected_model_name=winner_info["name"],
            baseline_model_id=baseline_model_id,
            baseline_cost_usd=round(baseline_cost, 6),
            selected_cost_usd=round(selected_cost, 6),
            cost_savings_pct=savings_pct,
            latency_ms=12.0 if cache_hit else float(winner_info["avg_latency_ms"]),
            cache_hit=cache_hit,
            cache_key=cache_key,
            semantic_similarity=0.98 if cache_hit else 0.0,
            benchmarks=benchmarks
        )

    def run_fixed_benchmark(self, strategy: str = "cheapest") -> Dict[str, Any]:
        """
        Runs a standard 10-query benchmark dataset evaluating:
        - Baseline: Always use expensive frontier model (GPT-4o)
        - SentriAI: Quality-aware tiered router + Redis prompt cache
        - Reports cost saved AND quality preservation percentage.
        """
        queries = [
            {"id": "Q1", "difficulty": "Easy", "category": "chat", "prompt": "What is the capital of France?"},
            {"id": "Q2", "difficulty": "Easy", "category": "chat", "prompt": "Write a short 3-line email thanking a teammate."},
            {"id": "Q3", "difficulty": "Medium", "category": "coding", "prompt": "def reverse_linked_list(head): write python code with comments."},
            {"id": "Q4", "difficulty": "Medium", "category": "coding", "prompt": "Write a SQL query to find top 5 highest paying customers with JOIN."},
            {"id": "Q5", "difficulty": "Medium", "category": "creative", "prompt": "Write a compelling headline and slogan for a new cloud router launch."},
            {"id": "Q6", "difficulty": "Hard", "category": "math", "prompt": "Calculate the definite integral of x^2 * e^(-lambda * x) from 0 to infinity and derive steps."},
            {"id": "Q7", "difficulty": "Hard", "category": "reasoning", "prompt": "Analyze root cause of microservice connection pool exhaustion under 50,000 QPS."},
            {"id": "Q8", "difficulty": "Hard", "category": "reasoning", "prompt": "Evaluate distributed BGP route flap, corrupt TLS handshake, and cross-region deadlock."},
            {"id": "Q9", "difficulty": "Easy (Repeated)", "category": "chat", "prompt": "What is the capital of France?"},
            {"id": "Q10", "difficulty": "Medium (Repeated)", "category": "coding", "prompt": "def reverse_linked_list(head): write python code with comments."}
        ]

        results = []
        total_baseline_cost = 0.0
        total_sentri_cost = 0.0
        baseline_quality_sum = 0.0
        sentri_quality_sum = 0.0
        cache_hits_count = 0

        for q in queries:
            dec = self.evaluate_prompt(q["prompt"], strategy=strategy)
            
            # Baseline is GPT-4o
            baseline_bm = [b for b in dec.benchmarks if b.model_id == "gpt-4o"][0]
            winner_bm = [b for b in dec.benchmarks if b.is_winner][0]

            total_baseline_cost += dec.baseline_cost_usd
            total_sentri_cost += dec.selected_cost_usd
            baseline_quality_sum += baseline_bm.quality_score
            sentri_quality_sum += winner_bm.quality_score
            
            if dec.cache_hit:
                cache_hits_count += 1

            # Populate cache if not hit so repeated queries hit cache
            if not dec.cache_hit:
                self.redis_cache.set(dec.cache_key, {"cached": True, "prompt": q["prompt"]}, ttl_seconds=3600)

            results.append({
                "Query ID": q["id"],
                "Difficulty": q["difficulty"],
                "Category": q["category"].upper(),
                "Baseline Model": "GPT-4o (Frontier)",
                "Baseline Cost": f"${dec.baseline_cost_usd:.6f}",
                "SentriAI Model": dec.selected_model_name,
                "SentriAI Cost": f"${dec.selected_cost_usd:.6f}",
                "Quality Score": f"{winner_bm.quality_score} / 100",
                "Cache Hit": "⚡ YES (0ms)" if dec.cache_hit else "No",
                "Savings": f"{dec.cost_savings_pct:.1f}%"
            })

        savings_usd = max(0.0, total_baseline_cost - total_sentri_cost)
        savings_pct = (savings_usd / max(0.00001, total_baseline_cost)) * 100.0
        
        avg_base_qual = baseline_quality_sum / len(queries)
        avg_sentri_qual = sentri_quality_sum / len(queries)
        quality_retention_pct = (avg_sentri_qual / max(0.001, avg_base_qual)) * 100.0

        return {
            "query_count": len(queries),
            "strategy_used": strategy,
            "total_baseline_cost_usd": round(total_baseline_cost, 6),
            "total_sentri_cost_usd": round(total_sentri_cost, 6),
            "total_savings_usd": round(savings_usd, 6),
            "cost_reduction_pct": round(savings_pct, 1),
            "avg_baseline_quality": round(avg_base_qual, 1),
            "avg_sentri_quality": round(avg_sentri_qual, 1),
            "quality_retention_pct": round(quality_retention_pct, 1),
            "cache_hits_count": cache_hits_count,
            "detailed_results": results
        }


