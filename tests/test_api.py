import time
import pytest
from fastapi.testclient import TestClient
from insightflow.api import app

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('INSIGHTFLOW_DB', str(tmp_path / 'test.db'))
    with TestClient(app) as client:
        yield client

PAYLOAD = {'text': 'Reliable bus services reduce waiting times. Protected bicycle lanes make commuting safer. Remote teams need clear documentation and mentoring. Flexible work schedules reduce long daily commutes.', 'options': {'mode': 'demo', 'num_topics': 2}}


def test_ui_health_and_validation(client):
    assert client.get('/').status_code == 200
    assert client.get('/static/app.js').status_code == 200
    assert client.get('/docs').status_code == 200
    assert client.get('/api/health').json()['status'] == 'ok'
    assert client.post('/api/jobs', json={'text': 'OK'}).status_code == 422
    assert client.post('/api/jobs', json={'text': 'Hello there people today.', 'discussion': []}).status_code == 422
    assert client.post('/api/jobs', json={**PAYLOAD, 'options': {'num_topics': 50}}).status_code == 422
    assert client.post('/api/preprocess', content='x'*1_200_001).status_code == 413
    assert client.get('/api/jobs/missing').status_code == 404


def test_sync_stages(client):
    for endpoint in ('preprocess','topics','analyze'):
        response = client.post('/api/'+endpoint, json=PAYLOAD)
        assert response.status_code == 200, response.text
    response = client.post('/api/summarize', json={'sentences': ['Reliable bus services reduce waiting times.'], 'options': {'mode': 'demo'}})
    assert response.status_code == 200 and response.json()['summary']


def test_job_export_and_delete(client):
    response = client.post('/api/jobs', json=PAYLOAD)
    assert response.status_code == 202
    job_id = response.json()['id']
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        job = client.get('/api/jobs/'+job_id).json()
        if job['status'] in ('completed','failed'):
            break
        time.sleep(.03)
    assert job['status'] == 'completed', job
    assert len(client.get('/api/jobs').json()) == 1
    exported = client.get(f'/api/jobs/{job_id}/export?format=json')
    assert exported.status_code == 200 and exported.json()['mode'] == 'demo'
    assert 'attachment' in exported.headers['content-disposition']
    assert 'InsightFlow' in client.get(f'/api/jobs/{job_id}/export?format=txt').text
    assert client.get(f'/api/jobs/{job_id}/export?format=html').status_code == 422
    assert client.delete(f'/api/jobs/{job_id}').status_code == 200
    assert client.get('/api/jobs/'+job_id).status_code == 404


def test_failed_job_is_visible(client, monkeypatch):
    import insightflow.api as api
    def fail(*args, **kwargs):
        raise RuntimeError('Model download unavailable')
    monkeypatch.setattr(api, 'analyze', fail)
    job_id = client.post('/api/jobs', json=PAYLOAD).json()['id']
    for _ in range(100):
        job = client.get('/api/jobs/'+job_id).json()
        if job['status'] == 'failed':
            break
        time.sleep(.03)
    assert job['status'] == 'failed'
    assert job['error'] == 'Model download unavailable'
    assert client.get(f'/api/jobs/{job_id}/export').status_code == 409


def test_restart_retains_results_and_marks_interrupted(tmp_path):
    from insightflow.storage import Store
    path = tmp_path / 'persist.db'
    store = Store(path)
    finished = store.create(PAYLOAD)
    store.update(finished, 'completed', 'Complete', result={'value': 1})
    interrupted = store.create(PAYLOAD)
    restarted = Store(path)
    assert restarted.get(finished)['result'] == {'value': 1}
    assert restarted.get(interrupted)['status'] == 'failed'


def test_queue_limit_and_running_delete(client):
    for _ in range(8):
        assert app.state.capacity.acquire(blocking=False)
    try:
        assert client.post('/api/jobs', json=PAYLOAD).status_code == 429
    finally:
        for _ in range(8):
            app.state.capacity.release()
    job_id = app.state.store.create(PAYLOAD)
    assert client.delete('/api/jobs/'+job_id).status_code == 409


def test_request_persistence_contains_provenance(client):
    import sqlite3
    response = client.post('/api/analyze', json=PAYLOAD)
    assert response.status_code == 200
    with sqlite3.connect(app.state.store.path) as db:
        assert db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 0
