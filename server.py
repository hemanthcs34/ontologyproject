"""
BOOFS Studio — API wrapper around the (unmodified) BOOFS pipeline.

This file contains NO extraction logic. It only:
  * imports `boofs` (and, if present, `boofs_eval`) exactly as they are,
  * calls the public methods the backend already exposes,
  * serializes their outputs to JSON for the web UI,
  * captures the logs/summaries the backend already prints.

Nothing in boofs.py / boofs_eval.py is monkey-patched, subclassed, wrapped
with altered behavior, or bypassed. Accept/Reject in the Active Learning
panel maps 1:1 onto the backend's own oracle-labeling mechanism
(LabelStore.add(..., source="oracle") -> retrain -> calibration ->
_consolidate_relations), i.e. exactly what CLIOracle / run_round do.
"""

import io
import os
import sys
import time
import json
import logging
import threading
import contextlib
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── the user's backend, imported as-is ──────────────────────────────────────
import boofs
from boofs import (
    BOOFSOntologyLearner, LabelStore, PathStatsStore, Oracle,
    evaluate_coreference_improvement, evaluate_relation_precision,
    evaluate_hits_at_k, evaluate_entity_similarity_quality,
)

try:
    import boofs_eval
    HAS_EVAL = True
except Exception:
    boofs_eval = None
    HAS_EVAL = False

# The exact sample text from boofs.py's __main__ block (it lives inside
# `if __name__ == "__main__"` so it cannot be imported; reproduced verbatim).
DEFAULT_SAMPLE_TEXT = """
    Bill and Dave became friends when they were both engineering students at Stanford.
    After graduation, Dave took a job with General Electric and moved to Schenectady,
    New York, where he married his college sweetheart Lucile Salter in 1938.
    But he and Bill stayed in touch. The two were encouraged by their former professor Fred Terman
    to start a technology company of their own.
    Taking a leave of absence from his job at GE, Dave and his new bride drove to California with
    a used drill press in the rumble seat.
    Bill scouted for places where the newlyweds could live. He found the ideal rental at 367 Addison
    Avenue in Palo Alto for $45 per month.
    """

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(title="BOOFS Studio", version="1.0")


@app.exception_handler(Exception)
async def _unhandled_exception(request, exc):
    """Surface unexpected errors as JSON so the UI can display them,
    instead of Starlette's plain-text 'Internal Server Error' page."""
    import traceback
    logging.getLogger("boofs_studio").error("Unhandled error:\n%s", traceback.format_exc())
    return JSONResponse(status_code=500,
                        content={"detail": f"{type(exc).__name__}: {exc}"})


def _clean(o):
    """Recursively convert backend outputs to plain JSON types.
    Handles numpy scalars/arrays (sklearn confidences, eval metrics),
    tuples, sets, and anything else non-serializable."""
    if o is None or isinstance(o, (str, bool, int)):
        return o
    if isinstance(o, float):
        return o if o == o and o not in (float("inf"), float("-inf")) else None
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_clean(x) for x in o]
    item = getattr(o, "item", None)          # numpy scalar → python scalar
    if callable(item):
        try:
            return _clean(item())
        except Exception:
            pass
    tolist = getattr(o, "tolist", None)      # numpy array → list
    if callable(tolist):
        try:
            return _clean(tolist())
        except Exception:
            pass
    return str(o)

# ── single-session state (local research tool) ──────────────────────────────
_lock = threading.Lock()
SESSION: Dict[str, Any] = {
    "learner": None,        # the live BOOFSOntologyLearner
    "pairs": None,          # candidate pairs from the last run
    "patterns": None,       # distributional patterns from the last run
    "results": None,        # raw results dict returned by process()
    "query_cache": {},      # LabelStore.key -> example dict (for AL labeling)
    "last_run": None,       # serialized payload of the last run
    "mode": None,           # "persistent" | "ephemeral"
}


# ── log capture (reads what the backend already emits; changes nothing) ─────
class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: List[str] = []
        self.setFormatter(logging.Formatter("%(levelname)s  %(name)s  %(message)s"))

    def emit(self, record):
        try:
            self.lines.append(self.format(record))
        except Exception:
            pass


@contextlib.contextmanager
def capture_backend_output():
    """Capture stdout (the pipeline's verbose stage prints) and log records
    while the backend runs, without altering its behavior."""
    handler = _ListHandler()
    root = logging.getLogger()
    root.addHandler(handler)
    buf = io.StringIO()
    out = {"stdout": "", "log_lines": handler.lines}
    try:
        with contextlib.redirect_stdout(buf):
            yield out
    finally:
        out["stdout"] = buf.getvalue()
        root.removeHandler(handler)


# ── serialization helpers (read-only over backend objects) ──────────────────
def _serialize_relations(relations) -> List[Dict]:
    return [r.to_dict() for r in relations]


def _serialize_run(learner: BOOFSOntologyLearner, results: Dict, pairs: List[Dict],
                   logs: Dict, elapsed: float, evaluation: Dict) -> Dict:
    concepts = [c.to_dict() for c in results["concepts"]]
    model = learner.relation_model
    payload = {
        "elapsed_seconds": round(elapsed, 3),
        "mode": SESSION["mode"],
        "raw_text": learner.raw_text,
        "resolved_text": learner.resolved_text,
        "coref_backend": learner.coref_resolver.backend or "rule-based fallback",
        "concepts": concepts,
        "relations": _serialize_relations(results["relations"]),
        "propositions": [p.to_dict() for p in results["propositions"]],
        "similarity_hypotheses": _serialize_relations(results["similarity_hypotheses"]),
        "patterns": results["patterns"],
        "induced_relation_types": results["induced_relation_types"],
        "relation_schema": results["relation_schema"],
        "candidate_pairs": pairs,
        "path_stats": learner.path_stats.stats(),
        "label_store": learner.label_store.stats(),
        "model": {
            "fitted": bool(model.is_fitted),
            "n_train": int(model.n_train),
            "classes": list(model.classes_ or []),
            "has_human_labels": bool(learner.active_learner.has_human_labels()),
        },
        "kg": _kg_summary(learner),
        "evaluation": evaluation,
        "logs": {
            "stdout": logs.get("stdout", ""),
            "records": logs.get("log_lines", []),
        },
    }
    return payload


def _kg_summary(learner: BOOFSOntologyLearner) -> Dict:
    kg = learner.kg_embedder
    out = {"trained": bool(kg is not None and getattr(kg, "is_trained", False)),
           "evaluation_valid": bool(kg is not None and getattr(kg, "evaluation_valid", False)),
           "entity_similarity": [], "link_predictions": []}
    if not out["trained"]:
        return out
    try:
        for entity in list(kg._entity_to_id.keys())[:12]:
            sims = kg.get_entity_similarity(entity, top_k=3)
            out["entity_similarity"].append(
                {"entity": entity,
                 "similar": [{"entity": e, "score": round(s, 4)} for e, s in sims]})
    except Exception as e:
        out["entity_similarity_error"] = str(e)
    try:
        df = kg.predict_missing_links(top_k=10)
        out["link_predictions"] = json.loads(df.to_json(orient="records"))
    except Exception as e:
        out["link_predictions_error"] = str(e)
    return out


def _run_evaluation(learner: BOOFSOntologyLearner, compare_no_coref: bool,
                    resolve_coreference: bool) -> Dict:
    """Calls only the evaluation functions the backend / eval module already
    provide. Any failure is reported, never fatal."""
    ev: Dict[str, Any] = {}
    try:
        if resolve_coreference:
            ev["coreference"] = evaluate_coreference_improvement(
                learner.raw_text, learner.resolved_text)
    except Exception as e:
        ev["coreference_error"] = str(e)
    try:
        ev["hits_at_10"] = evaluate_hits_at_k(learner.kg_embedder, k=10)
    except Exception as e:
        ev["hits_at_10_error"] = str(e)
    try:
        ev["entity_similarity_quality"] = evaluate_entity_similarity_quality(learner.kg_embedder)
    except Exception as e:
        ev["entity_similarity_quality_error"] = str(e)
    if HAS_EVAL:
        try:
            ev["incremental_drift"] = boofs_eval.incremental_drift(
                learner.relation_inducer, learner.path_stats)
        except Exception as e:
            ev["incremental_drift_error"] = str(e)
        try:
            baseline = boofs_eval.baseline_cooccurrence_tuples(learner)
            ev["baseline_cooccurrence_pairs"] = len(baseline)
        except Exception as e:
            ev["baseline_error"] = str(e)
    if compare_no_coref:
        # Same before/after comparison the backend's __main__ performs:
        # a throwaway in-memory learner run WITHOUT coreference.
        try:
            _no_coref = BOOFSOntologyLearner.for_evaluation()
            _no_coref.process(learner.raw_text, use_active_learning=False,
                              resolve_coreference=False, verbose=False,
                              persist_path_stats=False, enable_relation_model=False)
            ev["coref_off_vs_on"] = evaluate_relation_precision(
                _no_coref.relations, learner.relations)
        except Exception as e:
            ev["coref_off_vs_on_error"] = str(e)
    return ev


# ── request models ───────────────────────────────────────────────────────────
class RunRequest(BaseModel):
    text: str
    resolve_coreference: bool = True
    persist: bool = False          # False = in-memory session (for_evaluation)
    compare_no_coref: bool = True  # run the no-coref comparison from __main__


class LabelRequest(BaseModel):
    key: str
    label: str                     # relation label, or "NO_RELATION" to reject


# ── endpoints ─────────────────────────────────────────────────────────────────
@app.get("/api/status")
def status():
    return {
        "ready": SESSION["learner"] is not None,
        "mode": SESSION["mode"],
        "spacy_model": getattr(boofs.nlp, "meta", {}).get("name") if boofs.nlp else None,
        "coref_backend": boofs._COREF_BACKEND or "rule-based fallback",
        "eval_module": HAS_EVAL,
        "no_relation_token": Oracle.NO_RELATION,
    }


@app.get("/api/sample")
def sample():
    return {"text": DEFAULT_SAMPLE_TEXT}


@app.post("/api/run")
def run_pipeline(req: RunRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty. Paste some text or load the default sample.")
    if boofs.nlp is None:
        raise HTTPException(status_code=503, detail="No spaCy model installed. Run: python -m spacy download en_core_web_sm")

    with _lock:
        t0 = time.perf_counter()
        if req.persist:
            learner = BOOFSOntologyLearner(
                label_store_path=os.path.join(DATA_DIR, "boofs_al_labels.jsonl"),
                path_stats_path=os.path.join(DATA_DIR, "boofs_path_stats.jsonl"))
        else:
            learner = BOOFSOntologyLearner.for_evaluation()

        with capture_backend_output() as logs:
            try:
                results = learner.process(
                    req.text,
                    use_active_learning=False,   # AL is driven interactively via the panel
                    verbose=True,
                    resolve_coreference=req.resolve_coreference,
                    enable_relation_model=True,
                    persist_path_stats=req.persist,
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

        pairs = learner.distant_supervisor.entity_pairs
        evaluation = _run_evaluation(learner, req.compare_no_coref, req.resolve_coreference)
        elapsed = time.perf_counter() - t0

        SESSION.update({
            "learner": learner,
            "pairs": pairs,
            "patterns": results["patterns"],
            "results": results,
            "query_cache": {},
            "mode": "persistent" if req.persist else "ephemeral",
        })
        payload = _serialize_run(learner, results, pairs, logs, elapsed, evaluation)
        SESSION["last_run"] = payload
        return JSONResponse(_clean(payload))


def _require_session():
    if SESSION["learner"] is None:
        raise HTTPException(status_code=409, detail="No pipeline run in this session. Run the pipeline first.")
    return SESSION["learner"]


@app.get("/api/al/queries")
def al_queries(k: int = 6):
    """Uncertainty-sampled queries via the backend's own select_queries()."""
    learner = _require_session()
    with _lock:
        pairs = SESSION["pairs"] or []
        learner.active_learner.retrain()
        queries = learner.active_learner.select_queries(pairs, k=k)
        out = []
        SESSION["query_cache"] = {}
        for ex in queries:
            key = LabelStore.key(ex)
            SESSION["query_cache"][key] = ex
            suggested = ex.get("frame_agreement", "none")
            pred_label, pred_prob = learner.active_learner.predict(ex)
            out.append({
                "key": key,
                "entity1": ex.get("entity1", ""), "type1": ex.get("type1", ""),
                "entity2": ex.get("entity2", ""), "type2": ex.get("type2", ""),
                "context": ex.get("context", ""),
                "sentence": ex.get("sentence", ""),
                "confidence": ex.get("confidence", 0.0),
                "suggested": None if suggested in (None, "none") else suggested,
                "model_prediction": pred_label,
                "model_probability": round(pred_prob, 3) if pred_label else None,
                "already_labeled": learner.label_store.has_human(ex),
            })
        return JSONResponse(_clean({
            "queries": out,
            "label_store": learner.label_store.stats(),
            "induced_types": sorted(set(learner.relation_inducer.path_to_label.values())),
            "model": {"fitted": bool(learner.relation_model.is_fitted),
                       "classes": list(learner.relation_model.classes_ or [])},
            "no_relation_token": Oracle.NO_RELATION,
        }))


@app.post("/api/al/label")
def al_label(req: LabelRequest):
    """Apply a human label exactly the way the backend's oracle loop does:
    LabelStore.add(source='oracle') -> retrain -> calibration -> consolidation."""
    learner = _require_session()
    with _lock:
        ex = SESSION["query_cache"].get(req.key)
        if ex is None:
            for p in SESSION["pairs"] or []:
                if LabelStore.key(p) == req.key:
                    ex = p
                    break
        if ex is None:
            raise HTTPException(status_code=404, detail="Query not found; refresh suggestions and try again.")
        label = (req.label or "").strip() or Oracle.NO_RELATION

        learner.label_store.add(ex, label, source="oracle")     # backend method
        learner.active_learner.retrain()                        # backend method
        learner._update_calibration()                           # backend method
        # Re-consolidate through the backend's own consolidation path, which
        # switches to the classifier once human labels exist (its designed rule).
        learner.relations = learner._consolidate_relations(
            learner.propositions, SESSION["patterns"] or [], pairs=SESSION["pairs"])

        return JSONResponse(_clean({
            "labeled": {"key": req.key, "label": label,
                        "entity1": ex.get("entity1", ""), "entity2": ex.get("entity2", "")},
            "relations": _serialize_relations(learner.relations),
            "label_store": learner.label_store.stats(),
            "model": {
                "fitted": bool(learner.relation_model.is_fitted),
                "n_train": int(learner.relation_model.n_train),
                "classes": list(learner.relation_model.classes_ or []),
                "has_human_labels": bool(learner.active_learner.has_human_labels()),
            },
        }))


@app.get("/api/last")
def last_run():
    if SESSION["last_run"] is None:
        raise HTTPException(status_code=404, detail="No run yet.")
    return SESSION["last_run"]


# ── static frontend ───────────────────────────────────────────────────────────
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)