# Validation

Checked on 2026-11-02 with Python 3.12 on Linux. No report conclusions, quality scores or performance numbers were used as acceptance criteria.

## Completed checks

- 15 automated unit/API tests passed. These cover preprocessing, nested Reddit JSON, empty/invalid/oversized inputs, duplicate removal, topic limits, token-safe chunking, multi-stage compression, source provenance, synchronous stage APIs, background jobs, queue saturation, failed jobs, exports, deletion, persisted results and interrupted-job recovery.
- Real MPNet and DistilBART checkpoints downloaded and loaded successfully. Encoder smoke check returned a 768-dimensional vector, and BART produced a summary.
- Complete neural CLI run on `examples/discussion.txt` completed with three topic summaries and source evidence. All cleaned sentences were assigned to topics.
- Opt-in real-model integration test passed. A 96-token chunk budget exercised multiple integration rounds. Chunk inputs and final generated output stayed within their configured token budgets.
- JavaScript syntax check passed.
- DOM interaction smoke check passed for loading the example, constructing the request, polling, rendering three topic cards, exposing source evidence and creating download links.
- Python source compilation passed.

A full graphical browser rendering check could not be completed in the build environment because a local browser process could not launch. The DOM interaction check does not establish pixel-level appearance. Windows and macOS were not directly tested; the setup and run commands use platform-specific virtual-environment paths.

## Versions used for validation

| Component | Version |
|---|---|
| Python | 3.12 |
| FastAPI | 0.162.0 |
| Uvicorn | 0.54.0 |
| NumPy | 2.3.5 |
| scikit-learn | 1.8.0 |
| SciPy | 1.17.0 |
| PyTorch | 2.15.0+cpu |
| Transformers | 4.57.6 |
| Sentence Transformers | 3.4.1 |
| pytest | 8.4.2 |
| HTTPX | 0.28.1 |

Neural tests used CPU inference and cached Hugging Face checkpoints. Checkpoint weights are not bundled. No ROUGE, BERTScore, C_v, human quality ratings or report benchmark results are claimed.

## Repeat the checks

Install `requirements-dev.txt`, then run `python -m pytest -q` from the project root. The neural integration test is skipped unless `RUN_AI_TESTS=1` is set. Install `requirements-ai.txt` and cache the models before enabling it. Running the full test suite without this variable should show 15 passes and one skip.

Use `python scripts/download_models.py` for a real model loading, embedding and generation check. Use `python -m insightflow.cli examples/discussion.txt --mode ai --topics 3 --output result.json` for a full neural analysis.
