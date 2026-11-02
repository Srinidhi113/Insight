import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
import uuid

class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, created TEXT, status TEXT, stage TEXT, request TEXT, result TEXT, error TEXT)')
            db.execute("UPDATE jobs SET status='failed', stage='Interrupted', error='Server stopped before completion. Submit again.' WHERE status IN ('queued', 'running')")

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def create(self, request):
        job_id = str(uuid.uuid4())
        with self.connect() as db:
            db.execute('INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?)',
                (job_id, datetime.now(timezone.utc).isoformat(), 'queued', 'Waiting for worker', json.dumps(request), None, None))
        return job_id

    def update(self, job_id, status, stage, result=None, error=None):
        with self.connect() as db:
            db.execute('UPDATE jobs SET status=?, stage=?, result=COALESCE(?,result), error=? WHERE id=?',
                (status, stage, json.dumps(result) if result is not None else None, error, job_id))

    def get(self, job_id):
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            row = db.execute('SELECT id,created,status,stage,result,error FROM jobs WHERE id=?', (job_id,)).fetchone()
        if row is None:
            return None
        value = dict(row)
        if value['result']:
            value['result'] = json.loads(value['result'])
        return value

    def list(self):
        with self.connect() as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute('SELECT id,created,status,stage,error FROM jobs ORDER BY created DESC LIMIT 50')]

    def delete(self, job_id):
        with self.connect() as db:
            return db.execute("DELETE FROM jobs WHERE id=? AND status NOT IN ('running','queued')", (job_id,)).rowcount > 0
