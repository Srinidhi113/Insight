# InsightFlow

InsightFlow identifies topics and summarizes online discussions using MPNet embeddings, K-means clustering, and two-pass DistilBART summarization.

It supports text, Markdown, and JSON input, with a web interface for viewing summaries, exploring source sentences, and exporting results.

## Getting Started

Use Python 3.10–3.12. Open the project folder in VS Code and run:

```bash
python setup_project.py --ai
```

Select the `.venv` Python interpreter, press **F5**, and open http://127.0.0.1:8000. Models download automatically on first use.

## Technologies

Python, FastAPI, Sentence Transformers, PyTorch, scikit-learn, SQLite, HTML, CSS, and JavaScript.
