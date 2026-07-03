# BOOFS Web UI

A small Streamlit front-end for `boofs_extended.py`. Instead of editing the
`sample_text` variable in the script, you paste text into a text box in
your browser, click **Run**, and get the extracted concepts, relations,
frames, and similarity hypotheses as tables (with CSV download buttons).

The BOOFS pipeline depends on spaCy, scikit-learn, and PyKEEN — real
Python ML libraries that need to run on a machine with those installed.
Claude's in-chat artifacts only run JavaScript/HTML in the browser, so
they can't host this. This is instead a normal local Python app, same as
running `boofs_extended.py` directly, just with a UI wrapped around it.

## Setup (one-time)

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
# Optional, more accurate but ~500MB larger:
# python -m spacy download en_core_web_lg
```

## Run

streamlit run app.py

This opens a browser tab (usually `http://localhost:8501`). Paste your
text (or click "Load example text" to try the original sample), click
**Run BOOFS pipeline**, and browse results in the tabs:
- **Concepts** — extracted entities/noun phrases with type & confidence
- **Relations** — consolidated (subject, relation, object) triples
- **Frames** — frame-semantic slot fills per sentence
- **Similarity hypotheses** — entity-similarity edges (not asserted relations)
- **KG embeddings** — Hits@10 and similarity-quality metrics, if enough
  triples were extracted for embedding training
Each table has a **Download CSV** button. A "Coreference resolution:
before vs after" panel at the bottom lets you sanity-check what the
resolver changed in your input.

## Notes
- Active learning (the human-in-the-loop oracle labeling step) is left
  off in this UI, since it needs an interactive terminal oracle. Every
  other pipeline stage runs exactly as in the original script.
- `boofs_extended.py` is used as-is, unmodified — the UI only imports and
  calls `BOOFSOntologyLearner`.
