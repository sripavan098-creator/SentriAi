import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine.router import SentriAIRouter, MODEL_CATALOG
from src.engine.llm_tiering import SentriAIEngine

def test_sentri_model_catalog():
    assert "gemini-2.0-flash" in MODEL_CATALOG
    assert "gpt-4o" in MODEL_CATALOG
    assert "claude-3.5-sonnet" in MODEL_CATALOG
    assert "deepseek-v3" in MODEL_CATALOG
    assert "llama-3.3-70b" in MODEL_CATALOG

def test_sentri_prompt_classification():
    router = SentriAIRouter()
    
    cat_code, _ = router.classify_prompt("def solve_async_queue(): import asyncio")
    assert cat_code == "coding"

    cat_math, _ = router.classify_prompt("Calculate the integral of x^2 * e^(-lambda*x) dx from 0 to infinity")
    assert cat_math == "math"

    cat_creative, _ = router.classify_prompt("Write a creative slogan for an AI product launch")
    assert cat_creative == "creative"

def test_sentri_prompt_evaluation():
    router = SentriAIRouter()
    
    # Evaluate prompt under cheapest strategy
    decision = router.evaluate_prompt("Write Python function to sort array", strategy="cheapest")
    assert decision.selected_model_id is not None
    assert decision.cost_savings_pct >= 0.0
    assert len(decision.benchmarks) >= 10
    
    # Winner should be benchmark winner
    winner_bm = [b for b in decision.benchmarks if b.is_winner][0]
    assert winner_bm.model_id == decision.selected_model_id

def test_sentri_engine_execution():
    router = SentriAIRouter()
    engine = SentriAIEngine()
    
    prompt = "def ping_redis(): import redis"
    decision = router.evaluate_prompt(prompt, strategy="cheapest")
    response = engine.execute_prompt(prompt, decision)
    
    assert response.response_text is not None
    assert "def " in response.response_text or "Response" in response.response_text
    assert response.tokens_used > 0

def test_multi_model_outputs_generation():
    router = SentriAIRouter()
    engine = SentriAIEngine()
    
    prompt = "def solve_math(): return 42"
    decision = router.evaluate_prompt(prompt, strategy="cheapest")
    response = engine.execute_prompt(prompt, decision)
    
    assert response.all_model_responses is not None
    assert len(response.all_model_responses) >= 10
    assert "gemini-2.0-flash" in response.all_model_responses
    assert "gpt-4o" in response.all_model_responses
    assert "claude-3.5-sonnet" in response.all_model_responses

def test_fixed_benchmark_suite():
    router = SentriAIRouter()
    res = router.run_fixed_benchmark(strategy="cheapest")
    
    assert res["query_count"] == 10
    assert res["cost_reduction_pct"] > 50.0
    assert res["quality_retention_pct"] > 90.0
    assert len(res["detailed_results"]) == 10

def test_repeated_prompt_cache_hit():
    from src.engine.router import RedisPromptCache
    shared_cache = RedisPromptCache()
    router = SentriAIRouter(redis_cache=shared_cache)
    engine = SentriAIEngine(redis_cache=shared_cache)
    
    prompt = "Write a python script to test redis stream queue"
    
    # First Execution -> Cache Miss
    dec1 = router.evaluate_prompt(prompt, strategy="cheapest")
    res1 = engine.execute_prompt(prompt, dec1)
    
    # Second Execution -> MUST BE Cache Hit
    dec2 = router.evaluate_prompt(prompt, strategy="cheapest")
    res2 = engine.execute_prompt(prompt, dec2)
    
    assert dec2.cache_hit is True
    assert res2.cache_hit is True
    assert res2.cost_usd == 0.0
    assert res2.response_text == res1.response_text




