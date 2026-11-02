import numpy as np
import pytest
from insightflow.schemas import AnalyzeRequest, Options
from insightflow.preprocessing import preprocess
from insightflow.models import DemoModels
from insightflow.pipeline import analyze, detect_topics, make_chunks, summarize_texts


def test_clean_reddit_tree():
    request = AnalyzeRequest(discussion={'data': {'children': [{'data': {'body': 'u/alex **Protected** bicycle lanes improve road safety. https://example.com', 'replies': {'data': {'children': [{'data': {'body': 'Reliable bus services reduce waiting times.'}}]}}}}]}})
    result = preprocess(request)
    texts = [s['text'] for s in result['sentences']]
    assert len(texts) == 2
    assert all('http' not in s and 'u/alex' not in s for s in texts)
    assert 'Protected bicycle' in texts[0]


def test_duplicate_short_empty_and_long_input():
    result = preprocess(AnalyzeRequest(text='A useful sentence about remote work.\nA useful sentence about remote work.\n[deleted]\nOK'))
    assert len(result['sentences']) == 1
    assert result['exact_duplicates_removed'] == 1
    with pytest.raises(ValueError):
        preprocess(AnalyzeRequest(text='OK [deleted]'))
    words = ' '.join(['transport']*301)
    assert len(preprocess(AnalyzeRequest(text=words))['sentences']) == 2  # identical 100-word fragments deduplicated


def test_topic_count_small_and_keyword_fallback():
    sentences = [{'id': 0, 'text': 'the and this that'}]
    topics = detect_topics(sentences, np.array([[1.,0.]]), 7)
    assert len(topics) == 1 and topics[0]['label'] == 'Topic 1'


def test_chunk_budget_preserves_words():
    model = DemoModels()
    texts = [' '.join(f'token{i}' for i in range(310))]
    chunks = make_chunks(texts, model, 64)
    assert all(len(model.tokens(c)) <= 64 for c in chunks)
    assert ' '.join(chunks) == texts[0]


def test_two_pass_long_input_and_no_compression():
    model = DemoModels()
    texts = [f'The transport project number {i} will improve protected bicycle routes in local neighborhoods.' for i in range(90)]
    result = summarize_texts(texts, model, Options(mode='demo', chunk_tokens=64, summary_tokens=40))
    assert len(result['summary'].split()) <= 40
    assert result['compression_rounds'] >= 2
    assert len(result['chunks']) > 1
    result = summarize_texts(texts[:4], model, Options(mode='demo', second_pass=False))
    assert result['compression_rounds'] == 0


def test_end_to_end_example():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / 'examples/discussion.txt').read_text()
    stages = []
    result = analyze(AnalyzeRequest(text=text, options=Options(mode='demo')), stages.append)
    assert len(result['topics']) == 3
    assert result['mode'] == 'demo'
    assert 'extractive' in result['models']
    ids = [s['id'] for t in result['topics'] for s in t['sources']]
    assert sorted(ids) == list(range(len(result['preprocessing']['sentences'])))
    assert all(t['summary'] and t['keywords'] for t in result['topics'])
    assert 'summarization' in result['timings_seconds']
    assert stages[-1] == 'Summarizing topic 3 of 3'


def test_ai_orchestration_with_injected_model(monkeypatch):
    # Verifies AI routing and orchestration, not actual neural inference.
    import insightflow.pipeline as pipeline
    called = []
    def provider(mode):
        called.append(mode)
        return DemoModels()
    monkeypatch.setattr(pipeline, 'get_models', provider)
    result = analyze(AnalyzeRequest(text='Reliable buses improve city transport. Bicycle lanes protect vulnerable road users.', options=Options(mode='ai')))
    assert called == ['ai'] and result['topics']


def test_refinement_keeps_evidence_but_selects_representatives():
    from insightflow.pipeline import select_sources
    vectors = np.array([[1.,0.],[.999,.001],[0.,1.]])
    selected = select_sources([0,1,2], vectors, .92, True)
    assert len(selected) == 2 and 2 in selected
    assert select_sources([0,1,2], vectors, .92, False) == [0,1,2]
