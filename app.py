"""
BOOFS Pipeline — Flask Web Frontend
====================================

Wraps the existing finalaaryacode.py pipeline with a web UI.
No modifications are made to finalaaryacode.py — this file imports
and orchestrates its classes as-is.
"""

import sys
import os
import io
import logging
import traceback

# ── Fix Windows console encoding ─────────────────────────────
# finalaaryacode.py prints Unicode characters (✓, 📦, 🔗, etc.) via verbose
# print() calls. On Windows, the default console encoding (CP1252) cannot
# encode these, causing UnicodeEncodeError. We reconfigure stdout/stderr to
# use UTF-8 with error replacement so these prints succeed silently.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

from flask import Flask, render_template, request, jsonify

# ── Ensure the project directory is on sys.path ──────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# ── Import BOOFS classes from finalaaryacode ─────────────────
from finalaaryacode import (
    BOOFSOntologyLearner,
    DictOracle,
    Oracle,
    ActiveLearningModule,
    evaluate_coreference_improvement,
    evaluate_relation_precision,
    evaluate_hits_at_k,
    evaluate_entity_similarity_quality,
)

# ── Flask app ────────────────────────────────────────────────
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['JSON_SORT_KEYS'] = False

logger = logging.getLogger(__name__)

# ── Default sample text (same as in finalaaryacode.py __main__) ──
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

DEFAULT_GOLD = {
    ('dave', 'general electric'): 'WORKS_FOR',
    ('dave', 'ge'): 'WORKS_FOR',
    ('dave', 'stanford'): 'STUDIED_AT',
    ('bill', 'stanford'): 'STUDIED_AT',
}

# ── In-memory learner state ─────────────────────────────────
# Held as a module-level singleton so active learning state persists
# across HTTP requests within a server session.
_learner = None
_last_text_hash = None


def _get_learner(force_new=False):
    """Return the current learner, creating a fresh one if needed."""
    global _learner
    if _learner is None or force_new:
        _learner = BOOFSOntologyLearner()
    return _learner


def _serialize_results(learner, results):
    """Convert pipeline results into a JSON-safe dictionary."""

    concepts = [c.to_dict() for c in learner.concepts]
    relations = [r.to_dict() for r in learner.relations]

    propositions = []
    for p in learner.propositions:
        d = p.to_dict()
        propositions.append({
            'arg1': d['arg1'],
            'type1': d['type1'],
            'relation': d['relation'],
            'path_key': d['path_key'],
            'arg2': d['arg2'],
            'type2': d['type2'],
            'negated': d['negated'],
            'sentence': d['sentence'],
        })

    # induced relation types
    induced_types = results.get('induced_relation_types', [])

    # relation schema
    schema = {}
    raw_schema = results.get('relation_schema', {})
    for name, info in raw_schema.items():
        entry = {}
        if isinstance(info, dict):
            entry = {
                'domain': info.get('domain'),
                'range': info.get('range'),
                'parent': info.get('parent'),
                'confidence': info.get('confidence'),
            }
        schema[name] = entry

    # evaluation metrics
    evaluation = {}
    try:
        evaluation['coreference'] = evaluate_coreference_improvement(
            learner.raw_text, learner.resolved_text)
    except Exception as e:
        logger.warning(f"Coreference evaluation failed: {e}")

    try:
        evaluation['relation_precision'] = evaluate_relation_precision(
            learner.relations, learner.relations)
    except Exception as e:
        logger.warning(f"Relation precision evaluation failed: {e}")

    try:
        evaluation['hits_at_10'] = evaluate_hits_at_k(learner.kg_embedder, k=10)
    except Exception as e:
        logger.warning(f"Hits@10 evaluation failed: {e}")

    try:
        evaluation['entity_similarity'] = evaluate_entity_similarity_quality(
            learner.kg_embedder)
    except Exception as e:
        logger.warning(f"Entity similarity evaluation failed: {e}")

    # active learning candidates (most uncertain pairs)
    al_candidates = []
    try:
        pairs = learner.distant_supervisor.entity_pairs
        if pairs:
            uncertain = ActiveLearningModule.select_informative_examples(pairs, k=8)
            al_candidates = [{
                'entity1': p.get('entity1', ''),
                'entity2': p.get('entity2', ''),
                'type1': p.get('type1', ''),
                'type2': p.get('type2', ''),
                'context': p.get('context', ''),
                'confidence': p.get('confidence', 0.5),
            } for p in uncertain]
    except Exception as e:
        logger.warning(f"AL candidate selection failed: {e}")

    # pipeline stats
    pipeline_stats = {
        'path_store_paths': 0,
        'path_store_support': 0,
        'label_store_total': 0,
        'model_fitted': False,
    }
    try:
        ps = learner.path_stats.stats()
        pipeline_stats['path_store_paths'] = ps.get('paths', 0)
        pipeline_stats['path_store_support'] = ps.get('total_support', 0)
    except Exception:
        pass
    try:
        pipeline_stats['label_store_total'] = len(learner.label_store.training_data())
    except Exception:
        pass
    try:
        pipeline_stats['model_fitted'] = learner.relation_model.is_fitted
    except Exception:
        pass

    return {
        'concepts': concepts,
        'relations': relations,
        'propositions': propositions,
        'induced_relation_types': induced_types,
        'relation_schema': schema,
        'evaluation': evaluation,
        'al_candidates': al_candidates,
        'pipeline_stats': pipeline_stats,
    }


# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/default-text', methods=['GET'])
def get_default_text():
    return jsonify({'text': DEFAULT_SAMPLE_TEXT.strip()})


@app.route('/api/run', methods=['POST'])
def run_pipeline():
    """Run the full BOOFS pipeline on the submitted text."""
    global _learner, _last_text_hash

    data = request.get_json(force=True)
    text = data.get('text', '').strip()
    use_gold = data.get('use_gold', False)

    if not text:
        return jsonify({'error': 'No text provided.'}), 400

    try:
        # create a fresh learner for each new text
        _learner = _get_learner(force_new=True)
        _last_text_hash = hash(text)

        # When use_gold is True, the DictOracle provides automatic labels for
        # known entity pairs. When False, we disable interactive active learning
        # entirely (otherwise CLIOracle.label() calls input(), blocking the server).
        # In both cases, relation seeding from induction still runs.
        oracle = DictOracle(DEFAULT_GOLD) if use_gold else None

        results = _learner.process(
            text,
            use_active_learning=use_gold,   # only if we have an oracle
            oracle=oracle,
            verbose=False,                   # avoid print() encoding issues
            enable_relation_model=True,
            al_query_batch=5,
        )

        return jsonify(_serialize_results(_learner, results))

    except Exception as e:
        logger.error(f"Pipeline error: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/active-learn', methods=['POST'])
def active_learn():
    """Accept human labels and retrain the active learner."""
    global _learner

    if _learner is None:
        return jsonify({'error': 'Run the pipeline first.'}), 400

    data = request.get_json(force=True)
    labels = data.get('labels', {})

    if not labels:
        return jsonify({'error': 'No labels provided.'}), 400

    try:
        pairs = _learner.distant_supervisor.entity_pairs

        # apply each human label
        for key, label_val in labels.items():
            parts = key.split('|', 1)
            if len(parts) != 2:
                continue
            e1, e2 = parts

            # find the matching candidate pair
            match = None
            for p in pairs:
                if p.get('entity1', '') == e1 and p.get('entity2', '') == e2:
                    match = p
                    break
                if p.get('entity1', '') == e2 and p.get('entity2', '') == e1:
                    match = p
                    break

            if match is None:
                # build a minimal pair dict
                match = {
                    'entity1': e1,
                    'entity2': e2,
                    'type1': '',
                    'type2': '',
                    'context': '',
                }

            _learner.label_store.add(match, label_val.strip(), source='oracle')

        # retrain
        _learner.active_learner.retrain()

        # reconsolidate relations with updated model
        patterns = _learner.relation_discoverer.discovered_patterns
        _learner.relations = _learner._consolidate_relations(
            _learner.propositions, patterns, pairs=pairs)

        # build response
        relations = [r.to_dict() for r in _learner.relations]

        # new AL candidates
        al_candidates = []
        try:
            uncertain = ActiveLearningModule.select_informative_examples(pairs, k=8)
            al_candidates = [{
                'entity1': p.get('entity1', ''),
                'entity2': p.get('entity2', ''),
                'type1': p.get('type1', ''),
                'type2': p.get('type2', ''),
                'context': p.get('context', ''),
                'confidence': p.get('confidence', 0.5),
            } for p in uncertain]
        except Exception:
            pass

        pipeline_stats = {
            'path_store_paths': 0,
            'label_store_total': len(_learner.label_store.training_data()),
            'model_fitted': _learner.relation_model.is_fitted,
        }
        try:
            ps = _learner.path_stats.stats()
            pipeline_stats['path_store_paths'] = ps.get('paths', 0)
        except Exception:
            pass

        return jsonify({
            'relations': relations,
            'al_candidates': al_candidates,
            'pipeline_stats': pipeline_stats,
        })

    except Exception as e:
        logger.error(f"Active learning error: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  BOOFS Pipeline — Web UI")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60 + "\n")
    app.run(debug=False, host='127.0.0.1', port=5000, threaded=True)
