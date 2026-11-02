# Report-to-implementation mapping

Scope is drawn from sections III–VII of the attached report. The evaluation, future work and conclusions do not establish required results. No quality score or runtime claim from those sections has been embedded in this project.

| Report requirement or design | Implementation |
|---|---|
| Lightweight web interface + REST gateway | FastAPI gateway and locally served HTML/CSS/JavaScript; no separate frontend server |
| Text, Markdown and structured JSON | `preprocessing.py`, UI file importer and API schemas |
| Filter platform artifacts, short text, URLs and HTML | Deterministic preprocessing with cleaned source IDs |
| Transformer sentence encoder, 768 dimensions, MPNet | `AIModels.encoder`, `all-mpnet-base-v2` |
| Unsupervised user-configurable K, minimize within-cluster squared distance | K-means on normalized sentence vectors; clamp K to available distinct vectors |
| BERTopic in overview | Detailed section VII K-means objective takes precedence; no claim of using BERTopic package |
| Top five TF-IDF topic terms | TF-IDF across cluster documents; transparent keyword labels |
| Optional LLM label refinement | Not enabled or included; labels remain keyword-based, requiring no extra LLM credentials |
| Semantic sentence selection/deduplication | Centroid-ranked cosine deduplication, followed by source-order restoration; excluded sources remain visible |
| Context-aware TF-IDF chunking | Adjacent lexical similarity boundaries, preserving local sequence, bounded by tokenizer token count |
| First-pass BART summaries | DistilBART seq2seq generation per chunk |
| Second-pass compression | Token-safe recursive integration and final generation; can be disabled |
| 2–3 sentences, roughly 120–150 words | Configurable output token budget, not a forced word/sentence count; short sources remain short |
| Source sentences and intermediate outputs | Stored per topic; expandable in UI and included in JSON export |
| Structured and readable output | JSON and plain-text downloads |
| Storage for traceability | SQLite request, job state, source units, first-pass inputs/summaries and final output |
| Progress indicator | Persisted per-stage status and browser polling |
| CPU-accessible zero-shot models | CPU default, pretrained checkpoints, no training required |
| Configurable parameters | Topics, chunk size, summary length, deduplication threshold, refinement and second pass |
| Modular services | Separate preprocessing, model adapters, pipeline, schemas, storage and REST modules in one local process |

## Deliberate implementation boundaries

- No Reddit scraping, credentials, paid LLM or external summarization API is required.
- Model loading is lazy and process-cached rather than unconditional at startup. This allows the interface and demo to run before neural dependencies are installed.
- SQLite replaces an unspecified structured storage layer. One bounded worker replaces a distributed service deployment, appropriate for local VS Code use.
- Source IDs refer to cleaned sentence fragments. The implementation does not claim dialogue-level coreference resolution, sarcasm understanding or complete preservation of reply-tree semantics.
- The included sample discussion is authored sample data; it is not the report's experimental Reddit corpus.
- Demo is a separate lexical/extractive engine for setup verification. It must not be used to claim neural summary performance.
- LDA comparisons, C_v coherence studies, GPT scoring, ROUGE, BERTScore and the report's final metrics are omitted.
- Multilingual models, automatic K selection, streaming analysis, hierarchical topic trees and fine-tuning are future-work ideas in the report and are outside this implementation.
