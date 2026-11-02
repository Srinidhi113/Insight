"""Opt-in real model integration test. Requires weights and requirements-ai.txt."""
import os
import pytest

@pytest.mark.skipif(os.getenv('RUN_AI_TESTS') != '1', reason='Set RUN_AI_TESTS=1 after installing/caching neural models')
def test_real_model_pipeline():
    from pathlib import Path
    from insightflow.schemas import AnalyzeRequest, Options
    from insightflow.pipeline import analyze
    from insightflow.models import get_models
    text = (Path(__file__).resolve().parents[1] / 'examples/discussion.txt').read_text()
    result = analyze(AnalyzeRequest(text=text, options=Options(mode='ai', num_topics=1, chunk_tokens=96, summary_tokens=40)))
    model = get_models('ai')
    assert len(result['topics']) == 1
    topic = result['topics'][0]
    assert topic['summary'] and topic['compression_rounds'] > 1
    assert len(model.tokens(topic['summary'])) <= 40
    assert all(len(model.tokens(c['input'])) <= 96 for c in topic['chunks'])
    assert topic['sentence_count'] == len(result['preprocessing']['sentences'])
