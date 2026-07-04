# BOOFS — Bootstrapped Ontology and Object Frame Semantics

**Corpus-Driven, Schema-Free Ontology Learning with an Interactive Web UI**

BOOFS is an unsupervised ontology-learning system that extracts concepts and relations from raw text, and — unlike most extraction systems — *induces the relation types themselves* from corpus statistics instead of relying on a hand-written schema. It combines classical NLP (dependency parsing, coreference resolution), DIRT-style distributional relation induction, active learning, and knowledge-graph embeddings, wrapped in a FastAPI web application for interactive use.

No LLMs or transformer-based relation extraction are used anywhere in the extraction pipeline. spaCy provides the parse; PyKEEN (RotatE) is used only downstream, for knowledge-graph embedding and link prediction.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Pipeline Overview](#pipeline-overview)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Usage — Step by Step](#usage--step-by-step)
- [API Endpoints](#api-endpoints)
- [Persistent Data Stores](#persistent-data-stores)
- [Configuration](#configuration)
- [Evaluation Metrics](#evaluation-metrics)
- [Known Limitations](#known-limitations)
- [Team](#team)

---

## Features

- **Schema-free relation induction** — relation types emerge from clustering dependency paths (DIRT algorithm), not from a predefined list.
- **Self-growing corpus memory** — path statistics persist across runs, so induction quality improves as more documents are processed.
- **Coreference resolution** with automatic backend fallback (`fastcoref` → `spacy-experimental` → `neuralcoref` → rule-based).
- **OpenIE proposition extraction** over dependency parses — no trigger-word lists.
- **Unsupervised entity clustering** for distributional `SIMILAR_TO` hypotheses.
- **Genuine active learning** — margin-sampling uncertainty selection with an incrementally retrained classifier.
- **Knowledge-graph embeddings** (RotatE via PyKEEN) for link prediction and entity similarity.
- **Interactive web UI** — run the pipeline, browse concepts/relations/propositions/schema/evaluation in tabs, view a live knowledge graph, and label uncertain relations directly in the browser.
- **Two run modes** — ephemeral (in-memory, nothing written to disk) or persistent (corpus memory grows in `data/`).

---

## Project Structure

```
boofs-studio/
├── boofs.py                 # Core NLP/ML pipeline engine (unmodified by the web layer)
├── boofs_eval.py             # Evaluation harness (CaRB-style P/R/F1, B-cubed, ECE/Brier, drift)
├── server.py                 # FastAPI backend — wraps and orchestrates the pipeline
├── requirements.txt          # Python dependencies
├── static/
│   └── index.html             # Single-page frontend (vanilla HTML/CSS/JS, no framework)
└── data/                      # Auto-created on first persistent run
    ├── boofs_path_stats.jsonl            # Persistent dependency-path corpus statistics
    ├── boofs_path_stats.jsonl.induction  # Cluster/label carryover state
    └── boofs_al_labels.jsonl             # Active-learning label store
```

> `server.py` serves `index.html` straight from `static/` — there is no `templates/` folder and no server-side templating; the frontend is a single static file plus its own inline/linked JS and CSS.

---

## Pipeline Overview

Each call to `BOOFSOntologyLearner.process()` runs the following stages, in order:

| Stage | Description |
|---|---|
| 0. Coreference resolution | Resolves pronouns to their referent entities |
| 1. Concept extraction | Extracts named entities and noun chunks |
| 2. Candidate entity pairs | Distant-supervision pairing of co-occurring entities |
| 3. OpenIE proposition extraction | Extracts dependency paths connecting entity pairs (walks to the lowest common ancestor) |
| 4. DIRT relation induction | Clusters paths across the corpus into induced relation types; derives a relation schema (domain/range/subsumption) |
| 5. Unsupervised entity clustering | Discovers `SIMILAR_TO` relations via distributional similarity |
| 6. Active learning | Seeds and (optionally) queries a classifier for uncertain relation labels |
| 7. Consolidation | Merges everything into a deduplicated relation list |
| 8. Knowledge-graph embeddings | Trains RotatE embeddings for link prediction and similarity |

---

## Tech Stack

| Layer | Technology |
|---|---|
| NLP parsing | spaCy (`en_core_web_lg`, fallback `en_core_web_sm`) |
| Coreference | fastcoref → spacy-experimental → neuralcoref → rule-based fallback |
| Clustering | scikit-learn (DBSCAN, Agglomerative Clustering) |
| Active-learning classifier | scikit-learn `SGDClassifier` + `HashingVectorizer` |
| Knowledge-graph embeddings | PyKEEN (RotatE) |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML / CSS / JavaScript (no framework) |
| Persistence | Append-only JSONL files |

---

## Prerequisites

- Python **3.9 – 3.12**
- pip
- ~2 GB free disk space (for spaCy language models and PyKEEN/torch)
- Internet access for the initial spaCy model download

---

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd boofs-studio
   ```

2. **Create and activate a virtual environment** (recommended)
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   `requirements.txt` covers the core pipeline (`numpy`, `spacy`, `scikit-learn`) and the web layer (`fastapi`, `uvicorn`, `pydantic`). Two optional packages are commented out there and installed separately:
   ```bash
   pip install pykeen        # stage 8 KG embeddings + Hits@10 / similarity
   pip install fastcoref     # best coreference backend (rule-based fallback otherwise)
   ```
   > `pykeen` pulls in `torch`; on machines without a GPU it installs a CPU-only build automatically. If `fastcoref` fails to install on your platform, that's fine — the pipeline falls back to `spacy-experimental`, then `neuralcoref`, then a rule-based resolver.

4. **Download a spaCy language model**
   ```bash
   python -m spacy download en_core_web_sm
   # or, for better NER quality:
   python -m spacy download en_core_web_lg
   ```

---

## Running the Application

Start the FastAPI server directly with Python (Uvicorn is launched from inside `server.py`):

```bash
python server.py
```

The app starts on:

```
http://127.0.0.1:8000
```

Open that URL in your browser.

---

## Usage — Step by Step

This is the exact sequence that produces output, end to end:

1. **Load input text**
   In the web UI, either paste your own text into the input box, or click **"Load Default Sample"**. This calls `GET /api/sample`, which returns the built-in Bill Hewlett / Dave Packard biographical passage taken verbatim from `boofs.py`'s `__main__` block.

2. **Choose run options**
   - **Coreference resolution** — on by default; toggles `resolve_coreference`.
   - **Persist corpus memory** — off by default. Off = `BOOFSOntologyLearner.for_evaluation()`, an in-memory learner that doesn't touch disk. On = a learner backed by `data/boofs_path_stats.jsonl` and `data/boofs_al_labels.jsonl`, so the corpus grows across runs.
   - **Coref before/after comparison** — runs an extra throwaway no-coref pass so you can see the pronoun-free proxy precision.

3. **Click "Run Pipeline"**
   This sends `POST /api/run` with `{ "text": ..., "resolve_coreference": ..., "persist": ..., "compare_no_coref": ... }`. Server-side, `learner.process(...)` runs all 9 stages (0–8) described above and returns concepts, relations, propositions, the induced schema, and evaluation metrics as JSON.

4. **Explore the results tabs**
   - **Concepts** — searchable extracted entities
   - **Relations** — final consolidated relations with confidence, source, and evidence
   - **Propositions** — raw OpenIE triples before consolidation
   - **Induced Schema** — the auto-derived relation hierarchy (`induce_ontology`)
   - **Evaluation** — coreference improvement, relation-precision proxy, Hits@10, entity-similarity quality
   - **Knowledge Graph** — rendered from the consolidated relations

5. **Active learning loop**
   - Click **"Get suggestions"** → `GET /api/al/queries`, which calls the backend's own `ActiveLearningModule.select_queries()` (margin-sampling uncertainty selection) and returns the most uncertain entity pairs.
   - **Accept / Reject / enter a custom label** for a suggestion → `POST /api/al/label`. This runs the same chain the backend's own oracle loop uses: `LabelStore.add(source='oracle')` → `retrain()` → `_update_calibration()` → `_consolidate_relations()`.
   - Relations and the knowledge graph update immediately after each label using the returned payload — no need to re-run the whole pipeline.

6. **Re-fetch the last run (optional)**
   `GET /api/last` returns the most recent run's full payload — useful for refreshing the UI without recomputation.

7. **Persistent learning (if enabled)**
   If "Persist corpus memory" was on, the new document's path statistics and any labels are already saved to `data/` by the time step 3–5 complete — the next run in the same session (or a future session pointed at the same `data/` folder) starts from that improved state.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the web UI (`static/index.html`) |
| `GET` | `/api/status` | Session/model status: spaCy model loaded, coref backend in use, whether the eval module is available |
| `GET` | `/api/sample` | Returns the built-in default sample text |
| `POST` | `/api/run` | Runs the full pipeline on submitted text. Body: `{ "text": "...", "resolve_coreference": true, "persist": false, "compare_no_coref": true }` |
| `GET` | `/api/al/queries?k=6` | Returns the `k` most uncertain entity-pair labels via `select_queries()` |
| `POST` | `/api/al/label` | Applies a human label. Body: `{ "key": "...", "label": "..." }` (use `"NO_RELATION"` to reject) |
| `GET` | `/api/last` | Returns the payload from the most recent run in this session |

---

## Persistent Data Stores

These files are created automatically under `data/` on the first **persistent** run and grow as more text is processed — this is what makes the system "self-growing" across sessions:

| File | Contents |
|---|---|
| `boofs_path_stats.jsonl` | Dependency-path corpus statistics (support counts, argument fillers, types) |
| `boofs_path_stats.jsonl.induction` | Cluster state and path→label mapping, so induced relation labels stay stable as the corpus grows |
| `boofs_al_labels.jsonl` | Every relation label ever applied (seed-induced or human), used to train the active-learning classifier |

Delete the `data/` folder to reset the system to a blank corpus state. Ephemeral runs (persist = off) never touch these files.

---

## Configuration

Pipeline behavior (spaCy model choice, clustering thresholds, confidence weights, active-learning parameters, etc.) is controlled centrally in the `BOOFSConfig` dataclass at the top of `boofs.py`. Notable options:

- `spacy_model_preferred` / `spacy_model_fallback` — which spaCy model to load
- `max_path_len` — maximum dependency-path length considered for a relation
- `adaptive_threshold_min` / `adaptive_threshold_max` — bounds for per-block clustering thresholds
- `dbscan_eps_unsup` — DBSCAN epsilon for unsupervised entity clustering
- `path_stats_path` — default location of the persistent path-statistics store (overridden by `server.py` to point into `data/`)

---

## Evaluation Metrics

Metrics wired into the live web UI (via `server.py`):

| Metric | Purpose |
|---|---|
| Coreference improvement | Compares raw vs. coreference-resolved text to quantify how much pronoun resolution helped downstream extraction |
| Relation precision (proxy) | Pronoun-free ratio of extracted relations — an acknowledged proxy, not gold-standard precision |
| Hits@10 | Knowledge-graph link-prediction accuracy |
| Entity similarity quality | Sanity-checks that entities the KG embedding considers similar are semantically sensible |
| Incremental drift | From `boofs_eval.py` — measures how much induced relation clusters shift as new documents are added |

`boofs_eval.py` additionally implements a CaRB-style P/R/F1 harness, B-cubed/pairwise induction scoring, and ECE/Brier calibration reporting, for offline evaluation against gold-labeled data — these are not currently surfaced in the web UI's Evaluation tab.

---

## Known Limitations

- Relation precision in the live demo is measured via a proxy metric (pronoun-free ratio), not gold-standard precision.
- Coreference quality depends on which backend (`fastcoref` / `spacy-experimental` / `neuralcoref` / rule-based) successfully installs on the host machine.
- KG embedding training is skipped gracefully if there are too few triples to train on.
- JSONL-based persistence; no database backend.
- The tool is single-session by design (one local learner instance behind a lock), matching how the pipeline itself is used.

---

## Team

HPE Mentorship Project — The National Institute of Engineering, Mysuru

---

