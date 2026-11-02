# InsightFlow

An end-to-end local implementation of **InsightFlow: An AI-Driven Service for Semantic Topic Detection and Context-Aware Summarization of Online Discussions**.

This implementation follows the report's system requirements and methodology. Its conclusions, reported quality scores and runtime claims are not used as targets or presented as reproduced results. No training dataset or pre-existing project code was included in the supplied archive; this is a new implementation using pretrained models.

## Start in VS Code

1. Install **Python 3.11** (3.10–3.12 supported), VS Code, and the Microsoft Python extension. On Windows, enable “Add Python to PATH” during Python installation.
2. Extract this ZIP. Open the **inner `insightflow` folder** containing `run.py` and `README.md` with **File → Open Folder**. Do not run inside the ZIP.
3. Open **Terminal → New Terminal**, then run:

   ```sh
   python setup_project.py --ai
   ```

   On macOS/Linux, use `python3` if `python` is unavailable. On Windows, `py -3.11 setup_project.py --ai` also works. Setup requires internet access and several GB of free disk space. PyTorch packages can be large; installation time varies.

4. Press **Ctrl+Shift+P → Python: Select Interpreter** and choose the interpreter inside this project's `.venv` folder. If it is not listed, choose **Enter interpreter path**: `.venv/Scripts/python.exe` on Windows or `.venv/bin/python` on macOS/Linux.
5. Press **F5**, choose **InsightFlow: local server**, and open **http://127.0.0.1:8000**. The Python Debugger extension may need to install when prompted.
6. Click **Load example**, keep **MPNet + DistilBART**, and click **Analyze discussion**. The initial run downloads the models. The progress area shows the current stage. Keep the terminal running.
7. Expand source evidence and pipeline traces, or download JSON/Text results. **Ctrl+C** in the terminal stops the server.

### Quick demo without neural model downloads

Run `python setup_project.py` (omit `--ai`), select the `.venv` interpreter, and press F5. Choose **Offline demo** in the interface before analyzing. Demo uses TF-IDF clustering and extractive summarization; it is explicitly labeled and is **not** the report's neural method. AI mode never silently falls back to demo.

### Run without the debugger

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe run.py
```

macOS/Linux:

```sh
./.venv/bin/python run.py
```

No Node.js, npm, paid API, API key, database server or Reddit account is required. The browser interface is served by the Python application.

## Neural model setup

Defaults:

- Sentence encoder: `sentence-transformers/all-mpnet-base-v2` (768-dimensional embeddings).
- Summarizer: `sshleifer/distilbart-cnn-12-6` through `AutoModelForSeq2SeqLM.generate`.
- Device: CPU. CUDA is optional with a compatible PyTorch installation.

After selecting the `.venv` interpreter, this optional command downloads models and runs a real neural smoke check before opening the UI:

```sh
python scripts/download_models.py
```

Models are cached by Hugging Face, usually under `~/.cache/huggingface/hub`. Allow several GB of free disk space; **8 GB RAM or more is recommended**, with additional memory for large inputs. These are practical setup estimates, not measured requirements. First use requires access to Hugging Face and its model storage hosts. Once the weights are cached, inference runs locally. To force cached-only operation set `HF_HUB_OFFLINE=1`. The ZIP does not contain model weights.

Environment variables may be set before starting the server:

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8000` | Local HTTP port |
| `INSIGHTFLOW_DB` | `data/insightflow.sqlite3` | SQLite storage location |
| `INSIGHTFLOW_CPU_THREADS` | Up to `4` | CPU inference thread count |
| `INSIGHTFLOW_DEVICE` | `cpu` | `cpu` or `cuda` for neural models |
| `INSIGHTFLOW_EMBEDDING_MODEL` | MPNet model above | Model ID or local compatible sentence encoder |
| `INSIGHTFLOW_SUMMARY_MODEL` | DistilBART model above | Model ID or local BART-compatible seq2seq model |
| `HF_HOME` | Hugging Face default | Cache location |

There is no `.env` loader; use terminal environment variables or add an `env` object in `.vscode/launch.json`. Restart after changing model configuration. Custom models with different context limits may require code changes.

## What the application does

1. Accepts plain text, Markdown or JSON via paste, file import or REST. JSON supports `title`, `selftext`, `body`, `text`, nested `comments`/`replies` and Reddit Listing `data.children`.
2. Cleans HTML, links, Markdown, Reddit usernames, votes, timestamps and removed comments. Segments text and filters fragments shorter than four words. Long sentences are split into at most 100-word source units. Exact duplicate fragments are removed.
3. Encodes sentences with MPNet and clusters with deterministic K-means (`random_state=42`, `n_init=10`). Requested topics are reduced when fewer distinct vectors exist.
4. Extracts five TF-IDF keywords per cluster; uses three keywords as a transparent topic label.
5. Optionally removes semantically redundant sentences for summary generation using cosine similarity, while preserving all cluster source evidence. More central sentences take precedence; selected sentences retain source order.
6. Uses adjacent TF-IDF cosine similarity and the actual summarizer tokenizer to form context-aware chunks within a strict token budget.
7. Generates a first-pass BART summary for each chunk, recursively integrates intermediate summaries when needed, and produces the final second-pass summary. Intermediate input and output are retained. Disabling second pass returns concatenated first-pass summaries.
8. Saves request, pipeline trace, summaries, provenance and actually measured timing values in SQLite. Offers persisted history and JSON/text downloads.

The report calls the topic stage “BERTopic” in one overview but specifies MPNet vectors and a K-means objective in its detailed methodology. This project follows that detailed fixed-K method; it does not install the BERTopic package. See `docs/REPORT_MAPPING.md` for the requirement mapping and deliberate scope choices.

## Controls and limits

- Topic count: 1–12; default 3.
- Chunk budget: 64–900 tokenizer tokens; default 700.
- Summary budget: 40–256 generated tokens; default 200. It is a maximum/target, not a guaranteed word or sentence count. Short source text produces shorter summaries. The report's simultaneous 2–3-sentence and 120–150-word description is treated as a flexible presentation goal.
- Semantic duplicate threshold: 0.75–1; default 0.92. Higher values retain more near-duplicates.
- English text; at most 250,000 characters and 1,500 cleaned sentence units per analysis.
- One inference job runs at a time; up to eight running/waiting requests. HTTP 429 indicates a full queue. This is a local single-process service: do not run multiple workers against the same database.
- Existing completed jobs survive a restart. Interrupted jobs are marked failed and can be resubmitted. There is no cancellation control or automatic retry.
- Source evidence preserves cleaned text and source order IDs, not full Reddit parent-child semantics. JSON trees are flattened in traversal order.
- This package handles individual discussions and batches submitted separately. It does not download Reddit data or ingest a million-comment corpus in one request.

## Command-line file analysis

```sh
python -m insightflow.cli examples/discussion.txt --mode ai --topics 3 --output result.json
python -m insightflow.cli examples/discussion.json --mode demo --output result.txt --format txt
```

CLI results are written to your chosen file and are not added to the browser history.

## REST API

Interactive API docs: **http://127.0.0.1:8000/docs** (Swagger UI assets require internet). Raw OpenAPI schema: `/openapi.json`.

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Service and AI package availability; not proof that weights are downloaded |
| `POST /api/jobs` | Submit background analysis; returns HTTP 202 and job ID |
| `GET /api/jobs` | Latest 50 jobs |
| `GET /api/jobs/{id}` | Status, stage, error and completed result |
| `GET /api/jobs/{id}/export?format=json` | Download structured result; `format=txt` for text |
| `DELETE /api/jobs/{id}` | Delete a completed/failed job |
| `POST /api/analyze` | Synchronous pipeline result (not added to history) |
| `POST /api/preprocess` | Cleaned sentence units |
| `POST /api/topics` | Topic discovery only |
| `POST /api/summarize` | Summarize a `sentences` array with `options` |

Example JSON body for `/api/jobs`, `/api/analyze`, `/api/topics` or `/api/preprocess`:

```json
{
  "text": "Remote teams need clear documentation and mentoring. Laptop memory is important when running development containers. Reliable bus services reduce commuting costs.",
  "options": {
    "mode": "ai",
    "num_topics": 3,
    "chunk_tokens": 700,
    "summary_tokens": 200,
    "semantic_refinement": true,
    "dedup_threshold": 0.92,
    "second_pass": true
  }
}
```

For structured JSON, replace `text` with `discussion` containing an object or array. Supply exactly one input field. `examples/discussion.json` is a raw discussion to import in the UI, not a complete API request. `examples/request-demo.json` is a complete API request.

## Tests

With the `.venv` interpreter selected:

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests cover cleaning, JSON trees, invalid input, topic count bounds, lossless chunk splitting, hierarchical token budgets, pipeline provenance, background jobs, failed jobs, exports, deletion and persistence. Neural inference can be checked separately with `python scripts/download_models.py`. To run the opt-in integration test, set `RUN_AI_TESTS=1` and run `python -m pytest tests/test_ai_integration.py -q`. See `docs/VALIDATION.md` for the checks actually performed while building this package.

## Troubleshooting

- **ModuleNotFoundError:** VS Code is using a different Python environment. Select the `.venv` interpreter and install the requirements there.
- **AI packages missing:** run `python setup_project.py --ai`, reselect `.venv`, restart the server. Alternatively run `.venv`'s Python with `-m pip install -r requirements-ai.txt`.
- **Model loading fails:** run `python scripts/download_models.py` using `.venv` and inspect the terminal error; check network access, disk space, RAM and your Hugging Face cache. Offline demo works without the models.
- **Address already in use:** stop the other server or set `PORT=8001`; then browse to the new port.
- **No usable text:** include English sentences with four or more words. Avoid submitting only links or removed comments.
- **First analysis is slow:** downloads and model initialization happen once per process. Subsequent requests reuse the loaded models. No fixed runtime is promised.
- **Outputs seem incorrect:** inspect source evidence, vary topic count, or disable semantic refinement to retain every cleaned sentence. BART was trained for news summarization and may hallucinate or lose conversational nuance.

The server binds to `127.0.0.1`. It has no authentication and is intended for your local computer. Results and original request text are stored in `data/insightflow.sqlite3`; delete a job using its history row, or stop the server and remove the database to reset all history.

## Source references

- Supplied InsightFlow report, system design and methodology sections (III–VII).
- [Sentence Transformers encode documentation](https://www.sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html)
- [DistilBART model card](https://huggingface.co/sshleifer/distilbart-cnn-12-6)
- [FastAPI static file documentation](https://fastapi.tiangolo.com/tutorial/static-files/)

Model checkpoints and third-party packages retain their respective licenses and terms. No dataset or model weights are redistributed.
