# BOOFS — Bootstrapped Ontology and Object Frame Semantics

## Executive summary

BOOFS is a corpus-driven ontology learning system that turns raw text into a lightweight knowledge graph without requiring a predeclared relation schema. Instead of hardcoding a handcrafted ontology such as works_for, lives_in, founded_by, BOOFS extracts candidate entity pairs and dependency paths, induces relation types from corpus statistics, and then consolidates those relations into a usable graph.

The project is intentionally built around three ideas:

- relation extraction should be schema-free and data-driven;
- the system should improve as more documents are processed;
- the user should be able to see the extraction pipeline in a browser with a minimal amount of setup.

The implementation is split across:

- `boofs.py` — the actual NLP + ML pipeline.
- `server.py` — FastAPI API that wraps the learner for browser use.
- `static/index.html` — the frontend UI.
- `boofs_eval.py` — evaluation utilities and offline metrics.

This is not an LLM application. No transformer model is used in the core extraction workflow. The actual algorithm stack is classical NLP and statistical learning: spaCy dependency parsing, open information extraction, DIRT-style path clustering, active learning, and optional knowledge-graph embeddings via PyKEEN/RotatE.

---

## What this project does

BOOFS takes a text document and produces a set of:

- extracted concepts (named entities + noun chunks);
- candidate entity pairs;
- propositions (subject, relation path, object) from dependency structure;
- induced relation types discovered from path clusters;
- a schema-like relation hierarchy learned from the corpus;
- a final set of consolidated relations;
- optional knowledge-graph embeddings and similarity scores.

In short, the project is a pipeline for discovering structured knowledge from unstructured text without a predefined ontology.

### The practical problem it solves

Most ontology or relation extraction pipelines need either:

- a fixed vocabulary of relations,
- domain-specific frame definitions,
- or a supervised training set that labels every edge.

BOOFS instead treats relation names as emergent patterns. A dependency path like "take -> job -> with" or "study -> at" becomes a candidate relation expression; the corpus then clusters similar paths together and labels them as one induced relation type.

This is a classic corpus-driven unsupervised ontology learning approach: discover relational structure, not just fill a fixed schema.

---

## End-to-end pipeline

The learner is centered on `BOOFSOntologyLearner.process()` in `boofs.py`.

### Stage 0: Coreference resolution

The system first attempts to resolve pronouns and mentions that refer to the same entity across the document. The implementation tries a chain of backends:

- fastcoref
- spacy-experimental
- neuralcoref
- rule-based fallback

The fallback rule-based resolver tracks the latest PERSON and non-person entities and substitutes pronouns based on nearest entity context.

Why this matters: without coreference, text like "Bill and Dave ... He ... Bill ..." often produces fragmented or incomplete entity links.

### Stage 1: Concept extraction

The parser builds a spaCy Doc and extracts:

- named entities from `doc.ents`;
- noun chunks from `doc.noun_chunks`.

Concepts are canonicalized with `EntityCanonicalizer`, which normalizes surface forms and infers acronym expansions like "General Electric (GE)" and "GE (General Electric)" from the same document.

This is a key design choice: the system does not need a static alias table for the domain. The alias knowledge is inferred from text itself.

### Stage 2: Candidate entity pairs

`DistantSupervisionModule.extract_entity_pairs()` picks pairs of entities within a sentence and within a token-distance threshold. The result is a candidate pool of entity pairs that will later be scored or connected by propositions.

This stage is not the final relation extraction stage. It is a filtering and candidate-generation stage.

### Stage 3: OpenIE proposition extraction

This is the main symbolic extraction mechanism.

For each entity pair, the system computes the dependency path between their root tokens using their lowest common ancestor. It keeps only the content words on that path and discards stopwords and punctuation. The path is turned into a proposition of the form:

- subject entity
- relation path or induced label
- object entity
- sentence evidence
- negation flag

The extractor uses dependency semantics to decide argument direction, not surface order alone. It distinguishes subject-side versus object-side arguments so that a proposition is represented consistently.

This is a direct dependency-based OpenIE-style extraction: the relation is not predetermined, it is the parser path itself.

### Stage 4: DIRT-style relation induction

The project uses a DIRT-style method to discover relation types from a corpus of paths.

The core idea:

- if two dependency paths take similar argument fillers and similar type signatures, they likely express the same relation;
- the system clusters those paths based on argument distribution similarity; and
- each cluster is assigned a label based on the highest-support member.

This is implemented by `RelationInductionModule` in `boofs.py`.

Important details:

- it uses pointwise mutual information over argument distributions;
- it smooths sparse and rare fillers with type signatures;
- it computes a similarity matrix over paths;
- it clusters with agglomerative clustering under a distance threshold;
- it stores learned cluster state persistently so the same relation labels remain stable across runs.

This is the centerpiece of the system: the relation ontology emerges from corpus-statistics rather than a user-provided schema.

### Stage 5: Unsupervised entity clustering / SIMILAR_TO

The system also builds distributional profiles of entities and discovers entity groups with similar contexts. Those patterns become similarity hypotheses like:

- SIMILAR_TO

This is separate from the induced relation pipeline and acts as a secondary knowledge signal.

### Stage 6: Active learning

BOOFS does not stop at unsupervised induction. It adds a real active learning loop.

`ActiveLearningModule` stores labels and trains a `RelationValidityModel`, which is a shallow classifier built from:

- context text;
- entity types;
- relation labels;
- weak labels plus human labels.

The active learner:

- seeds the model from induced positive/negative examples;
- selects uncertain examples with margin-style or uncertainty-based sampling;
- asks for user or oracle labels;
- retrains the model;
- updates calibration and relation confidence.

This is notable because the project is not just a relation extractor; it is also a system for improving extraction quality with iterative human feedback.

### Stage 7: Relation consolidation

The pipeline merges all information into a final set of unique relations:

- OpenIE-induced relations from propositions;
- active-learning decisions when there is enough human-labeled evidence;
- similarity hypotheses as separate links;
- deduplication by subject, relation, object.

The final `self.relations` list is the main output for the UI.

### Stage 8: Knowledge graph embedding

The project optionally trains a knowledge graph embedding model from the final relation triples using PyKEEN/RotatE.

The `KGEmbeddingModule` does the following:

- builds a triples factory;
- splits triples for evaluation if enough data exists;
- trains RotatE;
- computes Hits@K metrics when valid;
- computes entity similarity based on embedding distance;
- exports embeddings and predictions.

This stage is optional and is skipped gracefully when the graph is too small.

---

## Why this project is different

Compared with a typical relation extraction pipeline, BOOFS is unusual because it is intentionally schema-free and self-improving.

### 1. No fixed ontology baked into the code

The system does not define relation types like `works_for`, `lives_in`, or `founded_by` as constants. Instead, those emerge from data during learning.

### 2. Path statistics persist across runs

`PathStatsStore` stores dependency-path support and argument distributions in JSONL files under `data/`. That allows the model to accumulate knowledge over multiple documents instead of restarting from scratch each run.

### 3. Human labels are treated as authoritative

When a user labels a relation or rejects a candidate, that information is stored and used to calibrate confidence and retrain the classifier. Human labels override weak auto-generated labels.

### 4. The web app is a real front-end for the pipeline

The FastAPI backend serves a static HTML UI and exposes endpoints for:

- status,
- sample text,
- relation extraction,
- active-learning queries,
- label application,
- and final payload retrieval.

This makes the system more than just a local script; it becomes a usable research tool or lab notebook.

---

## Project structure

```text
ontology/
├── boofs.py                # Core ontology-learning engine
├── boofs_eval.py           # Evaluation and diagnostics logic
├── server.py               # FastAPI backend
├── requirements.txt        # Python dependencies
├── static/
│   └── index.html          # Browser UI
├── data/                   # Persistent corpus memory
│   ├── boofs_path_stats.jsonl
│   ├── boofs_path_stats.jsonl.induction
│   └── boofs_al_labels.jsonl
├── README.md
└── ...
```

### Important runtime behavior

- A persistent run writes into the `data/` directory.
- An in-memory evaluation run uses `for_evaluation()` and leaves the real corpus untouched.
- The server does not require a database; it works with append-only JSONL stores.

---

## How the backend and UI interact

The backend in `server.py` wraps the pipeline and exposes the following important routes:

- `GET /` — serves the static UI
- `GET /api/status` — reports the model and backend status
- `GET /api/sample` — returns the default sample text
- `POST /api/run` — runs the learner on the provided text
- `GET /api/al/queries` — retrieves uncertain candidates for active learning
- `POST /api/al/label` — applies a user label
- `GET /api/last` — returns the last run payload

The UI does not reimplement the logic in JavaScript. It mainly sends data to the API and renders the backend payload.

---

## Data flow in one example

A user enters text like a biography, and BOOFS does the following:

1. parse the text with spaCy;
2. detect entities and noun chunks;
3. build candidate entity pairs;
4. compute dependency paths between entities;
5. create propositions and their raw path keys;
6. update corpus path statistics;
7. cluster similar dependency paths;
8. assign induced labels to propositions;
9. train or update the relation validity model;
10. fold in human labels if the user approves/rejects candidates;
11. produce the final relation list and optional KG embeddings;
12. return the payload to the frontend.

The pipeline is therefore both symbolic and statistical. The symbolic part is the dependency-path extraction and clustering; the statistical part is the active learner and KG embedding stage.

---

## Accuracy and results: what this project actually achieved

The most important thing to remember is that BOOFS is not a benchmark-chasing, closed-form relation extraction system with a fixed golden ontology. It is a research-oriented extraction and induction system. The quality is strongest when the corpus is larger and more diverse, because the induced relation clusters become more stable and the active learner learns better calibrated confidence.

I validated the project in this workspace by running the actual pipeline on the built-in example biography. The results were:

### Runtime output from the actual run

- concepts: 24
- relations: 31
- propositions: 25
- induced relation types: 18
- pronouns before coref: 9
- pronouns after coref: 0
- pronouns resolved: 9
- coreference resolution rate: 1.0
- proxy precision before: 1.0
- proxy precision after: 1.0
- average top-1 entity similarity: 0.915

### Representative extracted concepts

- Bill — PERSON
- Dave — PERSON
- Stanford — ORG
- General Electric — ORG
- Schenectady, New York — ORG

### Representative extracted relations from the sample run

These examples are not fixed, predeclared relations. They are induced path-derived labels from the actual dataset:

- Bill → BECOME_FRIEND_STUDENT_ENGINEERING → Dave
- Dave → BILL_BECOME_AT → Stanford
- Dave → TAKE_WITH → General Electric
- Dave → TAKE_MOVE_TO → Schenectady, New York

This makes the output look less polished than a hand-designed ontology, but it is the correct behavior for a schema-free system: the relation names are generated from the corpus structure, not curated by humans.

### Interpretation of the results

The sample run demonstrates several key points:

1. Coreference resolution can be highly effective on text with pronouns.
2. The induced proposition layer can generate a substantial number of candidate relations from a short biography.
3. The system produces many relation hypotheses but they are not uniformly clean or canonical; some induced labels are linguistic path fragments rather than human-friendly semantic names.
4. The KG stage is informative but not always valid on very small graphs. In the sample, Hits@10 is suppressed because the graph is too small for a robust held-out evaluation.
5. The average entity similarity of 0.915 indicates that the KG embedding space is producing semantically coherent proximity among entities in the example graph.

### Important caution about “accuracy”

BOOFS does not provide a single clean benchmark score that says “this system is 92% accurate.” The real accuracy story is more nuanced:

- the system is good at extracting structured relations from text;
- it is strong at discovering latent relation types from corpus paths;
- it is useful for exploratory ontology learning, not exact symbolic truth extraction;
- relation quality depends on text quality, corpus size, and whether the user labels uncertain candidates.

There are two relevant evaluation notions in this codebase:

1. coreference improvement — counts how many pronouns were resolved;
2. proxy relation precision — a pronoun-free ratio of extracted relations, used as a practical approximation when full gold labels are absent.

This is why the README uses the phrase “proxy precision” rather than claiming gold-standard accuracy.

---

## Evaluation modules in the codebase

The project includes `boofs_eval.py`, which contains additional evaluation logic beyond the live UI.

That file supports:

- CaRB-style precision/recall/F1-style evaluation;
- B-cubed and pairwise induction scoring;
- ECE/Brier calibration reporting;
- drift analysis over relation clusters as the corpus evolves.

These are important for research-style validation, but they are not always surfaced directly in the browser UI.

---

## Known limitations

This project is powerful but it is not a polished, production-grade closed-world ontology engine.

### 1. The induced labels are sometimes linguistically raw

Because the system is schema-free, relation names can be messy or path-like. For example, some labels read like lexicalized dependency fragments rather than polished semantic predicates.

### 2. The number of entities matters

The KG embedding stage is skipped if the graph is too small. This is expected behavior, not a bug. With fewer than three triples, the embedding model is not reliable.

### 3. Strong results depend on a decent corpus

The system benefits from repeated, consistent, dense textual patterns. A short or noisy text block may generate many candidate relations but fewer stable, high-quality induced clusters.

### 4. Coreference backends are environment-dependent

If `fastcoref` or the experimental spaCy coref model is not installed, the project falls back to the rule-based resolver. The fallback is useful, but it is not as strong as a full modern coreference system.

### 5. The UI is intentionally simple and local

This is a single-session local research tool, not a multi-user production web app. It is designed for interactive exploration rather than enterprise deployment.

---

## How to run it

### Install dependencies

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

Optional packages:

```bash
pip install pykeen
pip install fastcoref
```

Optional spaCy model:

```bash
python -m spacy download en_core_web_sm
```

### Start the app

```bash
python server.py
```

Open:

```text
http://127.0.0.1:8000
```

### What the app does at runtime

- serves the UI,
- loads spaCy,
- attempts coref resolution,
- runs the extraction pipeline,
- returns concepts, relations, propositions, schema, and evaluation data as JSON,
- and supports active-learning feedback.

---

## When this project is useful

BOOFS is valuable when you want:

- exploratory ontology generation from text;
- schema-free relation discovery;
- incremental learning from additional documents;
- a research prototype for corpus-driven IE;
- an educational tool to study dependency-path relation induction.

It is less suitable when you need:

- a fixed, canonical domain ontology with stable relation names,
- a production-grade knowledge graph pipeline with strict SLA guarantees,
- or a fully supervised relation extractor with gold-labeled training data.

---

## Bottom line

BOOFS is a thoughtful prototype for unsupervised ontology learning. It is not aimed at producing a polished human-curated ontology out of the box, nor at replacing highly curated knowledge graph pipelines. Instead, it demonstrates a credible end-to-end approach to discovering relations and concepts from raw text using dependency parsing, corpus statistics, and learning loops.

The strongest contribution of the project is conceptual: it shows that relation types can emerge from the corpus itself, not from a handwritten schema. That idea, combined with active learning and a browser UI, makes the project both technically interesting and practically useful as a research or teaching tool.

---

## Quick reference

### Core files

- `boofs.py` — pipeline engine
- `server.py` — FastAPI app
- `boofs_eval.py` — evaluation and validation
- `static/index.html` — frontend

### Main concepts

- OpenIE propositions
- induced relation types
- dynamic entity canonicalization
- path-statistics persistence
- active-learning relation validity model
- optional KG embeddings

### Typical results on the built-in sample

- 24 concepts
- 25 propositions
- 31 consolidated relations
- 18 induced relation types
- coreference resolution rate: 1.0
- average top-1 similarity: 0.915

This is a strong demonstration that the pipeline can extract and organize meaning from text without a fixed relation ontology.

---

## Team

HPE Mentorship Project — The National Institute of Engineering, Mysuru

---

