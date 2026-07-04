"""
BOOFS evaluation framework
==========================

A MODULAR, standalone evaluation layer for the BOOFS pipeline. It imports from
`boofs` but adds nothing to the extraction architecture — every function here is
read-only over BOOFS outputs (propositions, induced partitions, hierarchy,
confidence) plus optional external gold data. Nothing in `boofs.py` depends on
this file, so evaluation can be added, changed, or removed freely.

What it covers (the metrics an OpenIE / relation-induction paper reports):

  1. Proposition-layer OpenIE scoring: Precision / Recall / F1 / Coverage against
     gold (arg1, relation, arg2) tuples, with CaRB/Re-OIE2016-style tuple
     matching (argument + relation-phrase token overlap).
  2. Relation-INDUCTION scoring that needs no gold label names: B-cubed and
     pairwise Precision/Recall/F1 between the induced partition of entity pairs
     and a gold partition (the standard label-agnostic way to score unsupervised
     relation induction — e.g. gold = Wikidata/DocRED relation of each pair).
  3. Hierarchy scoring: parent-accuracy / edge P/R against a gold taxonomy.
  4. Confidence calibration quality: reliability curve + Expected Calibration
     Error (ECE) + Brier score.
  5. Incremental-vs-full divergence (approximation error) over time.
  6. Runtime + peak memory of any callable.
  7. A trivial symbolic baseline (co-occurring entity pairs) for comparison.
  8. Ablation runner (toggles config flags), before/after coref, report + a
     text confusion matrix.
  9. Loaders for CaRB / Re-OIE2016 / DocRED / Wikidata-slice gold files.

All metrics are plain frequency/statistics computations — no learning, no neural
components, no hardcoded domain knowledge.
"""

from __future__ import annotations

import io
import json
import time
import math
import tracemalloc
from collections import defaultdict, Counter
from dataclasses import replace
from typing import List, Dict, Tuple, Set, Optional, Callable, Iterable, Any


# ════════════════════════════════════════════════════════════════════════════
# small helpers
# ════════════════════════════════════════════════════════════════════════════

def _prf(tp: float, fp: float, fn: float) -> Dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {'precision': round(p, 4), 'recall': round(r, 4), 'f1': round(f, 4)}


def _norm(s: str) -> str:
    return ' '.join((s or '').strip().lower().split())


def _toks(s: str) -> Set[str]:
    return set(_norm(s).split())


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ════════════════════════════════════════════════════════════════════════════
# 1. Proposition-layer OpenIE scoring (CaRB / Re-OIE2016 style)
# ════════════════════════════════════════════════════════════════════════════

def _tuple_match(pred: Tuple[str, str, str], gold: Tuple[str, str, str],
                 arg_thresh: float, rel_thresh: float) -> bool:
    """A predicted (arg1, rel, arg2) matches a gold triple if both arguments and
    the relation phrase pass a token-Jaccard threshold. This mirrors CaRB's
    lexical matcher and is robust to the fact that induced relation labels are
    surface path strings, not gold predicate strings."""
    pa1, pr, pa2 = pred
    ga1, gr, ga2 = gold
    if _jaccard(_toks(pa1), _toks(ga1)) < arg_thresh:
        return False
    if _jaccard(_toks(pa2), _toks(ga2)) < arg_thresh:
        return False
    # relation phrase: compare the readable path tokens to gold relation tokens
    if rel_thresh <= 0.0:
        return True
    return _jaccard(_toks(pr.replace('_', ' ')), _toks(gr.replace('_', ' '))) >= rel_thresh


def evaluate_openie(pred_tuples: List[Tuple[str, str, str]],
                    gold_tuples: List[Tuple[str, str, str]],
                    arg_thresh: float = 0.5, rel_thresh: float = 0.0) -> Dict[str, Any]:
    """Precision / Recall / F1 / Coverage for the proposition layer.

    rel_thresh=0.0 scores argument-only extraction quality (does the system find
    the right entity pairs at all); rel_thresh>0 additionally requires the
    relation phrase to overlap the gold predicate. Greedy 1-1 matching.
    """
    gold_remaining = list(gold_tuples)
    tp = 0
    matched_gold = set()
    for pred in pred_tuples:
        for gi, gold in enumerate(gold_remaining):
            if gi in matched_gold:
                continue
            if _tuple_match(pred, gold, arg_thresh, rel_thresh):
                tp += 1
                matched_gold.add(gi)
                break
    fp = len(pred_tuples) - tp
    fn = len(gold_tuples) - tp
    out = _prf(tp, fp, fn)
    gold_pairs = {(_norm(a), _norm(b)) for a, _, b in gold_tuples}
    pred_pairs = {(_norm(a), _norm(b)) for a, _, b in pred_tuples}
    out['coverage'] = round(len(gold_pairs & pred_pairs) / len(gold_pairs), 4) if gold_pairs else 0.0
    out['tp'], out['fp'], out['fn'] = tp, fp, fn
    out['n_pred'], out['n_gold'] = len(pred_tuples), len(gold_tuples)
    return out


def propositions_to_tuples(propositions) -> List[Tuple[str, str, str]]:
    """Convert BOOFS Proposition objects to (arg1, relation_phrase, arg2)."""
    out = []
    for p in propositions:
        rel = (p.induced_label or p.readable_path())
        out.append((p.arg1, rel, p.arg2))
    return out


# ════════════════════════════════════════════════════════════════════════════
# 2. Relation-INDUCTION scoring (label-agnostic: B-cubed + pairwise)
# ════════════════════════════════════════════════════════════════════════════

def bcubed(pred: Dict[Any, str], gold: Dict[Any, str]) -> Dict[str, float]:
    """B-cubed Precision/Recall/F1 between two clusterings of the SAME items
    (items = entity pairs; cluster id = induced label vs gold relation). This is
    the standard metric for unsupervised relation induction because it needs no
    alignment between induced label names and gold relation names."""
    items = [k for k in pred if k in gold]
    if not items:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'n': 0}
    pred_c = defaultdict(set)
    gold_c = defaultdict(set)
    for it in items:
        pred_c[pred[it]].add(it)
        gold_c[gold[it]].add(it)
    P = R = 0.0
    for it in items:
        pc, gc = pred_c[pred[it]], gold_c[gold[it]]
        inter = len(pc & gc)
        P += inter / len(pc)
        R += inter / len(gc)
    P /= len(items)
    R /= len(items)
    f = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    return {'precision': round(P, 4), 'recall': round(R, 4), 'f1': round(f, 4), 'n': len(items)}


def pairwise_prf(pred: Dict[Any, str], gold: Dict[Any, str]) -> Dict[str, float]:
    """Pairwise Precision/Recall/F1: over all item pairs, is 'same cluster'
    agreement between the induced and gold partitions correct?"""
    items = [k for k in pred if k in gold]
    tp = fp = fn = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            same_pred = pred[a] == pred[b]
            same_gold = gold[a] == gold[b]
            if same_pred and same_gold:
                tp += 1
            elif same_pred and not same_gold:
                fp += 1
            elif not same_pred and same_gold:
                fn += 1
    return _prf(tp, fp, fn)


def induced_partition_from_learner(learner) -> Dict[Tuple[str, str], str]:
    """Map each extracted entity pair -> its induced relation label."""
    part = {}
    for p in learner.propositions:
        part[(p.arg1, p.arg2)] = p.induced_label or p.readable_path()
    return part


# ════════════════════════════════════════════════════════════════════════════
# 3. Hierarchy scoring
# ════════════════════════════════════════════════════════════════════════════

def evaluate_hierarchy(schema: Dict[str, Dict], gold_parent: Dict[str, Optional[str]]) -> Dict[str, Any]:
    """Score induced parent links against a gold taxonomy (parent map). Reports
    edge Precision/Recall/F1 and domain/range presence. Only labels present in
    both are scored."""
    labels = [L for L in schema if L in gold_parent]
    pred_edges = {(L, schema[L].get('parent')) for L in labels if schema[L].get('parent')}
    gold_edges = {(L, gold_parent[L]) for L in labels if gold_parent[L]}
    tp = len(pred_edges & gold_edges)
    fp = len(pred_edges - gold_edges)
    fn = len(gold_edges - pred_edges)
    out = _prf(tp, fp, fn)
    out['n_pred_edges'], out['n_gold_edges'] = len(pred_edges), len(gold_edges)
    return out


# ════════════════════════════════════════════════════════════════════════════
# 4. Confidence calibration quality
# ════════════════════════════════════════════════════════════════════════════

def calibration_report(conf_correct: List[Tuple[float, bool]], bins: int = 10) -> Dict[str, Any]:
    """Reliability curve + Expected Calibration Error + Brier score from
    (confidence, correct?) samples. ECE measures how far predicted confidence is
    from observed accuracy; lower is better."""
    if not conf_correct:
        return {'ece': None, 'brier': None, 'curve': [], 'n': 0}
    buckets = defaultdict(list)
    for c, ok in conf_correct:
        b = min(bins - 1, max(0, int(c * bins)))
        buckets[b].append((c, 1 if ok else 0))
    n = len(conf_correct)
    ece = 0.0
    curve = []
    for b in range(bins):
        vals = buckets.get(b, [])
        if not vals:
            continue
        conf = sum(c for c, _ in vals) / len(vals)
        acc = sum(o for _, o in vals) / len(vals)
        ece += (len(vals) / n) * abs(conf - acc)
        curve.append({'bucket': b, 'avg_conf': round(conf, 3),
                      'accuracy': round(acc, 3), 'count': len(vals)})
    brier = sum((c - o) ** 2 for c, o in [(c, 1 if ok else 0) for c, ok in conf_correct]) / n
    return {'ece': round(ece, 4), 'brier': round(brier, 4), 'curve': curve, 'n': n}


# ════════════════════════════════════════════════════════════════════════════
# 5. Incremental vs full divergence (approximation error over time)
# ════════════════════════════════════════════════════════════════════════════

def partition_agreement(a: Dict[Any, str], b: Dict[Any, str]) -> Dict[str, float]:
    """Agreement between two partitions of the same keys (here: paths).

    Reports pairwise P/R/F1 AND a B-cubed F1. Approximation error is derived from
    B-cubed, not pairwise: pairwise F1 is undefined (0) when both partitions are
    all-singletons even though they agree perfectly, whereas B-cubed correctly
    scores identical partitions as 1.0 regardless of singleton structure."""
    keys = [k for k in a if k in b]
    tp = fp = fn = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            x, y = keys[i], keys[j]
            sa, sb = a[x] == a[y], b[x] == b[y]
            if sa and sb:
                tp += 1
            elif sa and not sb:
                fp += 1
            elif not sa and sb:
                fn += 1
    prf = _prf(tp, fp, fn)
    prf['pairwise_f1'] = prf.pop('f1')
    bf1 = bcubed(a, b)['f1']              # treats `b` (full pass) as reference
    prf['bcubed_f1'] = bf1
    prf['approx_error'] = round(1.0 - bf1, 4)
    prf['n_paths'] = len(keys)
    return prf


def incremental_drift(inducer, stats) -> Dict[str, float]:
    """Compare the inducer's CURRENT (possibly incremental) partition against a
    fresh FULL pass on the same statistics — the approximation error introduced
    by incremental medoid assignment. Non-mutating (uses full_pass_partition)."""
    full = inducer.full_pass_partition(stats)
    return partition_agreement(dict(inducer.path_to_label), full)


# ════════════════════════════════════════════════════════════════════════════
# 6. Runtime + memory
# ════════════════════════════════════════════════════════════════════════════

def measure_cost(fn: Callable, *args, **kwargs) -> Dict[str, Any]:
    """Wall-clock seconds + peak memory (KB) of a callable. Returns the metrics
    and the callable's result."""
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {'seconds': round(elapsed, 4), 'peak_kb': round(peak / 1024, 1), 'result': result}


# ════════════════════════════════════════════════════════════════════════════
# 7. Trivial symbolic baseline
# ════════════════════════════════════════════════════════════════════════════

def baseline_cooccurrence_tuples(learner) -> List[Tuple[str, str, str]]:
    """A minimal non-learning baseline: every co-occurring entity pair the
    candidate generator found becomes a relation whose phrase is the raw
    between-entity context. Used to show the induction layer adds value over
    'link every nearby pair'."""
    out = []
    for pr in learner.distant_supervisor.entity_pairs:
        out.append((pr['entity1'], pr.get('context', '') or 'RELATED', pr['entity2']))
    return out


# ════════════════════════════════════════════════════════════════════════════
# 8. Ablation runner + before/after + report + confusion matrix
# ════════════════════════════════════════════════════════════════════════════

def run_ablations(learner_factory: Callable[[Any], Any], text: str,
                  gold_tuples: List[Tuple[str, str, str]],
                  base_config, ablations: Dict[str, Dict[str, Any]],
                  arg_thresh: float = 0.5) -> Dict[str, Dict]:
    """Run the pipeline under several config overrides and score each on the
    proposition layer. `learner_factory(config)` must return a fresh learner
    bound to that config; each is processed on `text` in isolation (in-memory).

    ablations: name -> {config_field: value, ...}. The base run is always included.
    Does not change the architecture — only flips existing config flags.
    """
    results = {}
    runs = {'base': {}}
    runs.update(ablations)
    for name, overrides in runs.items():
        cfg = replace(base_config, **overrides) if overrides else base_config
        learner = learner_factory(cfg)
        cost = measure_cost(
            learner.process, text,
            use_active_learning=False, resolve_coreference=True,
            verbose=False, persist_path_stats=False, enable_relation_model=False)
        preds = propositions_to_tuples(learner.propositions)
        scores = evaluate_openie(preds, gold_tuples, arg_thresh=arg_thresh)
        scores['seconds'] = cost['seconds']
        scores['peak_kb'] = cost['peak_kb']
        results[name] = scores
    return results


def before_after_coref(learner_with, learner_without,
                       gold_tuples: List[Tuple[str, str, str]] = None,
                       arg_thresh: float = 0.5) -> Dict[str, Any]:
    """Compare relations extracted WITH vs WITHOUT coreference. If gold is given,
    scores both; otherwise reports the pronoun-free proxy and set sizes."""
    from boofs import evaluate_relation_precision
    out = {'proxy': evaluate_relation_precision(learner_without.relations, learner_with.relations)}
    if gold_tuples:
        pw = propositions_to_tuples(learner_with.propositions)
        pn = propositions_to_tuples(learner_without.propositions)
        out['with_coref'] = evaluate_openie(pw, gold_tuples, arg_thresh=arg_thresh)
        out['without_coref'] = evaluate_openie(pn, gold_tuples, arg_thresh=arg_thresh)
    return out


def confusion_matrix(pred: Dict[Any, str], gold: Dict[Any, str], top: int = 12) -> str:
    """A text confusion matrix aligning induced labels to gold relations by their
    dominant co-occurrence (label-agnostic display only)."""
    items = [k for k in pred if k in gold]
    if not items:
        return "(no overlapping items)"
    cell = Counter((gold[it], pred[it]) for it in items)
    gold_labels = [g for g, _ in Counter(gold[it] for it in items).most_common(top)]
    pred_labels = [p for p, _ in Counter(pred[it] for it in items).most_common(top)]
    buf = io.StringIO()
    header = "gold\\pred".ljust(16) + "".join(pl[:12].ljust(13) for pl in pred_labels)
    buf.write(header + "\n")
    for g in gold_labels:
        row = g[:15].ljust(16)
        for p in pred_labels:
            row += str(cell.get((g, p), 0)).ljust(13)
        buf.write(row + "\n")
    return buf.getvalue()


def format_report(sections: Dict[str, Any]) -> str:
    """Render a nested metrics dict as a readable text report."""
    buf = io.StringIO()
    buf.write("=" * 70 + "\nBOOFS EVALUATION REPORT\n" + "=" * 70 + "\n")

    def _emit(d, indent=0):
        pad = "  " * indent
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, (dict, list)):
                    buf.write(f"{pad}{k}:\n")
                    _emit(v, indent + 1)
                else:
                    buf.write(f"{pad}{k}: {v}\n")
        elif isinstance(d, list):
            for item in d:
                _emit(item, indent)
                buf.write(f"{pad}---\n")
        else:
            buf.write(f"{pad}{d}\n")

    _emit(sections)
    return buf.getvalue()


# ════════════════════════════════════════════════════════════════════════════
# 9. Dataset loaders (CaRB / Re-OIE2016 / DocRED / Wikidata slice)
# ════════════════════════════════════════════════════════════════════════════

def load_carb(path: str) -> Dict[str, List[Tuple[str, str, str]]]:
    """CaRB / Re-OIE2016 gold: tab-separated `sentence \\t arg1 \\t rel \\t arg2`
    (extra columns ignored). Returns sentence -> list of gold triples."""
    gold = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            sent, a1, rel, a2 = parts[0], parts[1], parts[2], parts[3]
            gold[sent.strip()].append((a1.strip(), rel.strip(), a2.strip()))
    return dict(gold)



load_reoie2016 = load_carb


def load_docred_slice(path: str) -> Dict[str, Any]:
    """DocRED JSON slice -> (a) gold triples per document (title-keyed) and
    (b) a pair->relation gold partition for B-cubed. Uses only fields present in
    the standard DocRED format; missing pieces are skipped gracefully."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    triples: Dict[str, List[Tuple[str, str, str]]] = {}
    partition: Dict[Tuple[str, str], str] = {}
    for doc in data:
        title = doc.get('title', str(len(triples)))
        ents = doc.get('vertexSet', [])

        def _name(idx):
            try:
                return _norm(ents[idx][0]['name'])
            except (IndexError, KeyError, TypeError):
                return None
        tlist = []
        for lab in doc.get('labels', []):
            h, t, r = _name(lab.get('h')), _name(lab.get('t')), lab.get('r')
            if h and t and r:
                tlist.append((h, r, t))
                partition[(h, t)] = r
        triples[title] = tlist
    return {'triples': triples, 'partition': partition}


def load_wikidata_partition(path: str) -> Dict[Tuple[str, str], str]:
    """Wikidata slice as JSON list of {"arg1","arg2","relation"} -> pair->relation
    gold partition for B-cubed induction scoring."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {(_norm(r['arg1']), _norm(r['arg2'])): r['relation'] for r in data
            if r.get('arg1') and r.get('arg2') and r.get('relation')}


# ════════════════════════════════════════════════════════════════════════════
# self-test (metrics only; no spaCy needed)
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # --- OpenIE tuple scoring ---
    gold = [("dave", "work for", "general electric"),
            ("dave", "study at", "stanford"),
            ("bill", "study at", "stanford")]
    pred = [("dave", "TAKE_JOB_WITH", "general electric"),   # arg match, rel differs
            ("dave", "STUDY_AT", "stanford"),
            ("dave", "MARRY", "lucile")]                     # false positive
    print("OpenIE (arg-only):", evaluate_openie(pred, gold, arg_thresh=0.5, rel_thresh=0.0))
    print("OpenIE (arg+rel): ", evaluate_openie(pred, gold, arg_thresh=0.5, rel_thresh=0.3))

    # --- B-cubed induction scoring ---
    pred_part = {("dave", "stanford"): "A", ("bill", "stanford"): "A",
                 ("dave", "ge"): "B", ("alice", "mit"): "A"}
    gold_part = {("dave", "stanford"): "EDU", ("bill", "stanford"): "EDU",
                 ("dave", "ge"): "EMP", ("alice", "mit"): "EDU"}
    print("B-cubed:", bcubed(pred_part, gold_part))
    print("pairwise:", pairwise_prf(pred_part, gold_part))

    # --- hierarchy ---
    schema = {"STUDY": {"parent": "AFFILIATE", "domain": "PERSON", "range": "ORG"},
              "AFFILIATE": {"parent": "RELATED"}, "RELATED": {"parent": None}}
    goldp = {"STUDY": "AFFILIATE", "AFFILIATE": "RELATED", "RELATED": None}
    print("hierarchy:", evaluate_hierarchy(schema, goldp))

    # --- calibration ---
    cc = [(0.9, i < 6) for i in range(10)] + [(0.5, i < 5) for i in range(10)]
    print("calibration:", {k: v for k, v in calibration_report(cc).items() if k != 'curve'})

    # --- partition divergence ---
    print("agreement:", partition_agreement({"p": "A", "q": "A", "r": "B"},
                                             {"p": "A", "q": "B", "r": "B"}))

    # --- cost ---
    print("cost:", {k: v for k, v in measure_cost(lambda: sum(range(100000))).items() if k != 'result'})

    # --- confusion matrix ---
    print(confusion_matrix(pred_part, gold_part))
    print("SELF-TEST OK")
