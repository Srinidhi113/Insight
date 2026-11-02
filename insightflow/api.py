"""Single-process local REST gateway with persisted jobs and a bounded worker."""
import json
import logging
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from threading import Lock, BoundedSemaphore
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from .schemas import AnalyzeRequest, SummarizeRequest
from .preprocessing import preprocess
from .pipeline import analyze, detect_topics, summarize_texts, plain_text
from .models import get_models
from .storage import Store

ROOT = Path(__file__).resolve().parent
log = logging.getLogger('insightflow')

@asynccontextmanager
async def lifespan(app):
    app.state.store = Store(os.getenv('INSIGHTFLOW_DB', str(ROOT.parent / 'data' / 'insightflow.sqlite3')))
    app.state.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='insightflow')
    app.state.pipeline_lock = Lock()
    app.state.capacity = BoundedSemaphore(8)
    yield
    app.state.executor.shutdown(wait=True)

app = FastAPI(title='InsightFlow', version='1.0.0', lifespan=lifespan)

@app.middleware('http')
async def bounded_body(request: Request, call_next):
    if request.method == 'POST':
        # Read a bounded stream; do not trust only Content-Length.
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > 1_200_000:
                return JSONResponse({'detail': 'Request body exceeds 1.2 MB.'}, status_code=413)
        request._body = bytes(data)
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    if request.url.path not in ('/docs', '/redoc', '/docs/oauth2-redirect'):
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
    return response

@app.get('/api/health')
def health():
    return {'status': 'ok', 'ai_packages_installed': __import__('importlib.util', fromlist=['find_spec']).find_spec('sentence_transformers') is not None,
            'embedding_model': __import__('insightflow.models', fromlist=['EMBEDDING_MODEL']).EMBEDDING_MODEL,
            'summary_model': __import__('insightflow.models', fromlist=['SUMMARY_MODEL']).SUMMARY_MODEL}


def execute_job(job_id, payload):
    store = app.state.store
    try:
        with app.state.pipeline_lock:
            store.update(job_id, 'running', 'Starting')
            result = analyze(payload, lambda stage: store.update(job_id, 'running', stage))
            store.update(job_id, 'completed', 'Complete', result=result)
    except Exception as exc:
        log.exception('Analysis failed: %s', job_id)
        store.update(job_id, 'failed', 'Failed', error=str(exc)[:1000])
    finally:
        app.state.capacity.release()

@app.post('/api/jobs', status_code=202)
def submit(payload: AnalyzeRequest):
    try:
        preprocess(payload)  # Reject invalid input before consuming queue capacity.
    except (ValueError, RecursionError) as exc:
        raise HTTPException(422, str(exc)) from exc
    if not app.state.capacity.acquire(blocking=False):
        raise HTTPException(429, 'Worker queue is full. Wait for an analysis to finish.')
    try:
        job_id = app.state.store.create(payload.model_dump())
        app.state.executor.submit(execute_job, job_id, payload)
    except Exception:
        app.state.capacity.release()
        raise
    return {'id': job_id, 'status': 'queued'}

@app.get('/api/jobs')
def history():
    return app.state.store.list()

@app.get('/api/jobs/{job_id}')
def get_job(job_id: str):
    job = app.state.store.get(job_id)
    if job is None:
        raise HTTPException(404, 'Analysis not found.')
    return job

@app.delete('/api/jobs/{job_id}')
def delete_job(job_id: str):
    get_job(job_id)
    if not app.state.store.delete(job_id):
        raise HTTPException(409, 'Wait until this analysis finishes before deleting it.')
    return {'deleted': True}

@app.get('/api/jobs/{job_id}/export')
def export(job_id: str, format: str = 'json'):
    job = get_job(job_id)
    if job['status'] != 'completed':
        raise HTTPException(409, 'Analysis is not complete.')
    if format not in ('json', 'txt'):
        raise HTTPException(422, 'Choose json or txt.')
    headers = {'Content-Disposition': f'attachment; filename="insightflow-{job_id}.{format}"'}
    if format == 'txt':
        return PlainTextResponse(plain_text(job['result']), headers=headers)
    return JSONResponse(job['result'], headers=headers)

async def locked_call(function):
    if not app.state.capacity.acquire(blocking=False):
        raise HTTPException(429, 'Worker queue is full.')
    def run():
        with app.state.pipeline_lock:
            return function()
    try:
        return await run_in_threadpool(run)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    finally:
        app.state.capacity.release()

@app.post('/api/analyze')
async def synchronous_analysis(payload: AnalyzeRequest):
    return await locked_call(lambda: analyze(payload))

@app.post('/api/preprocess')
def preprocessing(payload: AnalyzeRequest):
    try:
        return preprocess(payload)
    except (ValueError, RecursionError) as exc:
        raise HTTPException(422, str(exc)) from exc

@app.post('/api/topics')
async def topics_only(payload: AnalyzeRequest):
    def run():
        prepared = preprocess(payload)
        vectors = get_models(payload.options.mode).embed([s['text'] for s in prepared['sentences']])
        topics = detect_topics(prepared['sentences'], vectors, payload.options.num_topics)
        for topic in topics:
            topic['sources'] = [prepared['sentences'][i] for i in topic.pop('indices')]
        return {'mode': payload.options.mode, 'topics': topics}
    return await locked_call(run)

@app.post('/api/summarize')
async def summaries_only(payload: SummarizeRequest):
    if sum(len(t) for t in payload.sentences) > 250_000 or not any(t.strip() for t in payload.sentences):
        raise HTTPException(422, 'Supply nonempty sentences totaling at most 250,000 characters.')
    return await locked_call(lambda: {'mode': payload.options.mode, **summarize_texts(payload.sentences, get_models(payload.options.mode), payload.options)})

@app.get('/')
def home():
    return FileResponse(ROOT / 'static' / 'index.html')

@app.get('/api/example')
def example():
    return PlainTextResponse((ROOT.parent / 'examples' / 'discussion.txt').read_text(encoding='utf-8'))

app.mount('/static', StaticFiles(directory=ROOT / 'static'), name='static')
