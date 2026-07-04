"""
BOOFS: Bootstrapped Ontology and Object Frame Semantics
========================================================

Domain-independent (English) symbolic relation induction over parsed text.

This system performs symbolic, unsupervised Open Information Extraction (OpenIE)
followed by DIRT-style distributional relation induction. Nothing about the
relation inventory is declared in source: relations are the dependency paths that
connect entities in the text, and relation *types* emerge by clustering those
paths across the corpus. Path statistics accumulate on disk across runs, so each
new document sharpens the induced inventory without any code change.

Scope and honest framing:
  * This is a novel *integration* of established symbolic techniques, not a novel
    core algorithm. The building blocks (dependency-path OpenIE, DIRT relation
    induction [Lin & Pantel 2001], distant-supervision candidate generation
    [Mintz 2009], pool-based active learning [Settles 2009]) are pre-existing.
  * The incremental, corpus-driven refinements added here — type-signature
    blocking, an EMA-smoothed largest-gap merge threshold with a corpus-learned
    reference distance for sparse blocks, medoid-anchored incremental induction
    with a drift trigger, containment-based relation subsumption, and empirical
    confidence calibration — are useful engineering improvements to that pipeline;
    none is claimed as a standalone published algorithm.
  * "Symbolic" describes the induction/consolidation logic. Parsing, NER, and
    (optionally) coreference use spaCy's neural models as a PREPROCESSING layer;
    the optional PyKEEN/RotatE KG stage is neural and orthogonal to extraction.
  * Domain-independent means no relation schema, trigger lists, frames, or alias
    tables — it works across domains without code changes. It is English-tied
    (dependency labels, acronym patterns), so it is not language-independent.

Pipeline: coreference -> concept extraction -> distant-supervision candidate
  pairs -> OpenIE proposition extraction -> DIRT relation induction ->
  distributional entity clustering (SIMILAR_TO) -> active learning ->
  consolidation -> ontology (domain/range + subsumption) -> KG embeddings.
"""

import csv
import re
import os
import math
import json
import hashlib
import logging
from dataclasses import dataclass, replace
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Optional, FrozenSet
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# CENTRALIZED CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
#
# Every tunable lives here. Note what is NOT here anymore: there are no relation
# names, trigger words, slot definitions, evidence prepositions, or alias tables.
# Those were the domain-specific knobs; they have been deleted, not moved.

@dataclass
class BOOFSConfig:
    # --- spaCy model selection ---
    spacy_model_preferred: str = "en_core_web_lg"
    spacy_model_fallback: str = "en_core_web_sm"

    # --- entity pairing (candidate generation) ---
    max_entity_token_distance: int = 10

    # --- OpenIE proposition extraction ---
    # Maximum number of tokens on a dependency path (excluding the two argument
    # tokens). Longer paths are almost always spurious multi-clause artifacts.
    max_path_len: int = 5
    # Content POS tags that may appear on a relation path. This is a *linguistic*
    # (language-level) filter, not a domain one: it keeps predicative material
    # (verbs, relational nouns, adpositions, particles, auxiliaries) and drops
    # determiners/pronouns/etc. It contains no lexical items.
    path_content_pos: Tuple[str, ...] = ('VERB', 'NOUN', 'PROPN', 'ADP', 'PART', 'AUX')

    # --- DIRT relation induction ---
    # A path must occur at least this many times (across the accumulated corpus)
    # to participate in clustering; rarer paths become their own singleton
    # relation labeled by their surface form.
    min_path_support: int = 1
    # --- induction quality / stability (all corpus-derived, no relation schema) ---
    # Cluster only within blocks that share a dominant argument NER-type signature
    # (prevents nonsensical cross-type merges and bounds the O(P^2) cost to per
    # block). The signatures come from the parser, not a hand-written table.
    type_signature_blocking: bool = True
    # The per-block merge threshold is derived from that block's own distance
    # distribution (cut at the widest natural gap between "same" and "different"
    # pairs), clamped to these bounds so a block can neither merge everything nor
    # nothing. Bounds only — the cut point itself comes from the data.
    adaptive_threshold_min: float = 0.30
    adaptive_threshold_max: float = 0.75
    # Full re-clustering is expensive (O(P^2) per block). Between full passes, new
    # or changed paths are assigned incrementally to the nearest cluster medoid;
    # a full pass runs at least every this many fits to reconsolidate. This is a
    # runtime optimization only — a full pass reproduces the exact same result.
    recluster_every: int = 25
    # PMI cold-start smoothing (OPT-IN): rare (hapax) fillers back off to their
    # argument NER type, giving same-typed rare paths a non-zero similarity signal
    # instead of zero. Off by default (0.0) because on a single example it cannot
    # distinguish synonyms from non-synonyms without risking over-merge; the robust
    # cold-start mechanism is cross-run accumulation + drift reconsolidation below.
    # Set >0 to trade precision for recall on sparse corpora. Corpus-derived.
    type_smoothing_max_alpha: float = 0.0
    # Per-block clustering cap (bounds the O(block^2) matrix for a pathologically
    # large type block); paths beyond the cap in a block are still assigned to that
    # block's clusters incrementally, so NO path is exiled to a forced singleton.
    max_block_size: int = 800
    # Adaptive per-block thresholds are smoothed across full passes with this EMA
    # weight on the previous value, so cluster boundaries don't jitter as the
    # corpus grows. Both the old and new values are corpus-derived.
    threshold_ema_beta: float = 0.5
    # In a sparse block (too few paths for a meaningful within-block gap, e.g. two
    # paths), the gap threshold is self-referential and merges nothing. Instead we
    # fall back to a corpus-wide reference distance learned from POPULATED blocks
    # that actually merged (the typical "same-relation" distance). A pair in a
    # sparse block merges if its distance is below that learned level. This bound
    # is the minimum block size for which the within-block gap is trusted.
    min_block_for_gap: int = 3
    # A new cluster inherits a previous run's label when their path-sets overlap
    # by at least this Jaccard ratio -> labels stay stable as the corpus grows.
    label_carryover_jaccard: float = 0.5
    # Confidence = mean(cluster cohesion, support/median_support); this is the
    # relative weight of cohesion (the rest goes to support). Corpus-derived.
    conf_cohesion_weight: float = 0.5
    # Empirical confidence calibration: once at least this many human-labelled
    # induced edges exist in a confidence bucket, the raw score is mapped to the
    # bucket's observed reliability (how often that confidence was actually
    # correct). Calibration is corpus/label-driven and sharpens as labels grow.
    calib_bins_count: int = 10
    calib_min_evidence: int = 20
    # Relation A is subsumed by a more general B when A's argument fillers are
    # (on both slots) contained in B's by at least this ratio. Strength cutoff
    # only; the containment values themselves are corpus-derived.
    subsumption_min_containment: float = 0.6
    # Optional sidecar for cross-run label/cluster state (set by the learner).
    induction_state_path: Optional[str] = None
    # Persistent path-statistics store (the self-growing corpus memory).
    path_stats_path: str = "boofs_path_stats.jsonl"
    # Per-document persistence appends only that document's deltas; a full
    # (atomic) compaction runs once every this many appends to bound file growth.
    path_stats_compact_every: int = 200

    # --- clustering (legacy distributional discovery) ---
    dbscan_eps_unsup: float = 0.4

    # --- negation handling ---
    skip_negated: bool = True
    negation_confidence_penalty: float = 0.4

    # --- KG embedding evaluation ---
    min_triples_for_eval: int = 20
    kg_test_ratio: float = 0.2

    # --- confidence constants ---
    conf_ner: float = 0.9
    conf_noun_chunk: float = 0.6
    conf_distant_base: float = 0.7
    conf_similarity: float = 0.25

    # --- active learning ---
    al_confidence_shrinkage: float = 10.0
    # Relation sources excluded from KG-embedding training (non-factual edges).
    kg_excluded_sources: tuple = ('distributional_similarity',)
    al_refit_epochs: int = 12
    al_incremental_epochs: int = 3
    al_human_weight: float = 1.0
    al_seed_weight: float = 0.3


CONFIG = BOOFSConfig()


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURABLE spaCy MODEL LOADER
# ════════════════════════════════════════════════════════════════════════════

def load_spacy_model(preferred: str = None, fallback: str = None):
    """Load the preferred spaCy model, falling back to a smaller one if needed."""
    preferred = preferred or CONFIG.spacy_model_preferred
    fallback = fallback or CONFIG.spacy_model_fallback
    for name in (preferred, fallback):
        try:
            model = spacy.load(name)
            if name != preferred:
                logger.warning(f"Preferred model '{preferred}' unavailable; using '{name}'.")
            else:
                logger.info(f"Loaded spaCy model '{name}'.")
            return model
        except OSError:
            continue
    logger.error(
        f"No spaCy model found. Install one with: "
        f"python -m spacy download {preferred}  (or {fallback})"
    )
    raise OSError(f"Neither '{preferred}' nor '{fallback}' is installed.")


# Import-time load is made non-fatal: the module imports even without a model so
# that the model-independent components (induction, stores, exports) are usable
# and testable; anything that actually parses text raises a clear error on first
# use if `nlp` is None.
try:
    nlp = load_spacy_model()
except OSError as _e:
    logger.error(f"{_e} — text-parsing stages will be unavailable until a model is installed.")
    nlp = None


def _require_nlp():
    if nlp is None:
        raise RuntimeError(
            "No spaCy model is loaded. Install one, e.g. "
            "`python -m spacy download en_core_web_sm`, then re-run.")
    return nlp


# ════════════════════════════════════════════════════════════════════════════
# COREFERENCE RESOLUTION (PREPROCESSING STAGE) — unchanged
# ════════════════════════════════════════════════════════════════════════════

try:
    from fastcoref import FCoref
    _COREF_BACKEND = "fastcoref"
except ImportError:
    try:
        import spacy_experimental  # noqa: F401  # type: ignore
        _COREF_BACKEND = "spacy_experimental"
    except ImportError:
        try:
            import neuralcoref  # noqa: F401  # type: ignore
            _COREF_BACKEND = "neuralcoref"
        except ImportError:
            _COREF_BACKEND = None


class CoreferenceResolver:
    """
    Resolves pronouns to canonical entity mentions before BOOFS extraction.
    Tries fastcoref, then spacy-experimental, then neuralcoref, then a
    lightweight rule-based resolver, so it always works.
    """

    def __init__(self):
        self.backend = _COREF_BACKEND
        self._coref_nlp = None
        self._fcoref_model = None

        if self.backend == "fastcoref":
            try:
                self._fcoref_model = FCoref()
            except Exception:
                logger.warning("fastcoref failed to initialize; falling back to next backend.")
                self.backend = None

        if self.backend is None:
            try:
                import spacy_experimental  # noqa: F401  # type: ignore
                self.backend = "spacy_experimental"
            except ImportError:
                try:
                    import neuralcoref  # noqa: F401  # type: ignore
                    self.backend = "neuralcoref"
                except ImportError:
                    self.backend = None

        if self.backend == "spacy_experimental":
            try:
                self._coref_nlp = spacy.load("en_coreference_web_trf")
            except Exception:
                logger.warning("spacy-experimental coref model not found; using rule-based resolver.")
                self.backend = None

        elif self.backend == "neuralcoref":
            try:
                neuralcoref.add_to_pipe(nlp)
                self._coref_nlp = nlp
            except Exception:
                logger.warning("neuralcoref failed to attach; using rule-based resolver.")
                self.backend = None

    def resolve(self, text: str) -> str:
        if self.backend == "fastcoref" and self._fcoref_model is not None:
            return self._resolve_fastcoref(text)
        if self.backend == "spacy_experimental" and self._coref_nlp is not None:
            return self._resolve_spacy_experimental(text)
        if self.backend == "neuralcoref" and self._coref_nlp is not None:
            return self._resolve_neuralcoref(text)
        return self._resolve_rule_based(text)

    def _resolve_fastcoref(self, text: str) -> str:
        try:
            preds = self._fcoref_model.predict(texts=[text])
            clusters = preds[0].get_clusters(as_strings=False)
        except Exception as e:
            logger.warning(f"fastcoref prediction failed ({e}); returning original text.")
            return text
        span_clusters = []
        for cluster in clusters:
            spans = [type("Span", (), {"start_char": s, "end_char": e, "text": text[s:e]})() for s, e in cluster]
            span_clusters.append(spans)
        return self._apply_clusters(text, span_clusters)

    def _resolve_spacy_experimental(self, text: str) -> str:
        doc = self._coref_nlp(text)
        clusters = [v for k, v in doc.spans.items() if k.startswith("coref_clusters")]
        return self._apply_clusters(text, clusters)

    def _resolve_neuralcoref(self, text: str) -> str:
        doc = self._coref_nlp(text)
        if doc._.has_coref:
            return doc._.coref_resolved
        return text

    def _apply_clusters(self, text: str, clusters) -> str:
        replacements = []
        for cluster in clusters:
            if not cluster:
                continue

            def _looks_like_proper_name(span_text: str) -> bool:
                words = span_text.split()
                return bool(words) and words[0][0:1].isupper() and len(words) <= 4

            proper_candidates = [s for s in cluster if _looks_like_proper_name(s.text)]
            if proper_candidates:
                main = min(proper_candidates, key=lambda s: len(s.text))
            else:
                main = max(cluster, key=lambda s: len(s.text))

            for mention in cluster:
                if mention.text.lower() != main.text.lower():
                    replacements.append((mention.start_char, mention.end_char, main.text))

        replacements.sort(key=lambda r: r[0], reverse=True)
        accepted = []
        for start, end, repl in replacements:
            if any(not (end <= a_start or start >= a_end) for a_start, a_end, _ in accepted):
                continue
            accepted.append((start, end, repl))

        resolved = text
        for start, end, repl in accepted:
            resolved = resolved[:start] + repl + resolved[end:]
        return resolved

    PRONOUNS = {'he', 'him', 'his', 'she', 'her', 'hers', 'they', 'them', 'their', 'it', 'its'}
    PERSONAL_PRONOUNS = {'he', 'him', 'his', 'she', 'her', 'hers'}
    IMPERSONAL_PRONOUNS = {'it', 'its'}

    def _resolve_rule_based(self, text: str) -> str:
        doc = _require_nlp()(text)
        last_person = None
        last_nonperson = None
        replacements = []

        token_to_entity = {}
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "ORG", "GPE", "PRODUCT"):
                for tok in ent:
                    token_to_entity[tok.i] = ent

        for token in doc:
            word = token.text.lower()
            if word in self.PRONOUNS:
                if word in self.PERSONAL_PRONOUNS and last_person:
                    replacements.append((token.idx, token.idx + len(token.text), last_person))
                elif word in self.IMPERSONAL_PRONOUNS and last_nonperson:
                    replacements.append((token.idx, token.idx + len(token.text), last_nonperson))
                elif word not in self.PERSONAL_PRONOUNS and word not in self.IMPERSONAL_PRONOUNS:
                    fallback = last_person or last_nonperson
                    if fallback:
                        replacements.append((token.idx, token.idx + len(token.text), fallback))

            ent = token_to_entity.get(token.i)
            if ent is not None:
                if ent.label_ == "PERSON":
                    last_person = ent.text
                else:
                    last_nonperson = ent.text

        replacements.sort(key=lambda r: r[0], reverse=True)
        resolved = text
        for start, end, repl in replacements:
            resolved = resolved[:start] + repl + resolved[end:]
        return resolved


# Universal (language-level, not domain-level) dependency-label -> role map,
# used ONLY to pick a canonical argument direction (subject-ish vs object-ish)
# for an OpenIE proposition. It contains no relation names and no lexical items.
SUBJECT_DEPS = {'nsubj', 'nsubjpass', 'agent', 'expl', 'csubj'}
OBJECT_DEPS = {'dobj', 'obj', 'pobj', 'dative', 'attr', 'oprd', 'iobj'}
# Passive voice inverts surface roles: in "Y was acquired by X", Y is the
# grammatical subject (nsubjpass) but the SEMANTIC object, and X (agent) is the
# semantic subject. These are handled explicitly so direction stays correct.
PASSIVE_SUBJ_DEPS = {'nsubjpass', 'nsubj:pass', 'auxpass', 'csubjpass'}
AGENT_DEPS = {'agent'}


# ════════════════════════════════════════════════════════════════════════════
# DYNAMIC ENTITY CANONICALIZATION (no hardcoded aliases)
# ════════════════════════════════════════════════════════════════════════════
#
# All static alias/suffix tables are gone. Canonicalization now derives entirely
# from the document itself:
#   (a) surface normalization (lowercasing, punctuation/whitespace folding);
#   (b) parenthetical acronym glosses in the text, e.g. "General Electric (GE)"
#       or "GE (General Electric)" — a fully domain-independent textual signal;
#   (c) initial-letter matching among org/geo entities that CO-OCCUR in the same
#       document, so "GE" maps to "General Electric" only when both appear.
# An optional runtime glossary can still be injected via register(), but nothing
# domain-specific is baked into the source.

class EntityCanonicalizer:
    # Parenthetical gloss patterns (both orders). Purely structural, no lexicon.
    _PAREN_FULL_THEN_ABBR = re.compile(
        r'([A-Z][A-Za-z.&\-]*(?:\s+[A-Za-z.&\-]+){0,4})\s*\(\s*([A-Z][A-Za-z.&]{1,9})\s*\)')
    _PAREN_ABBR_THEN_FULL = re.compile(
        r'\b([A-Z][A-Z.&]{1,9})\s*\(\s*([A-Z][A-Za-z.&\-]*(?:\s+[A-Za-z.&\-]+){0,4})\s*\)')

    def __init__(self, aliases: Dict[str, str] = None):
        # Starts EMPTY. Any entries come from register() at runtime, never source.
        self.aliases: Dict[str, str] = {}
        if aliases:
            self.aliases.update({self._normalize(k): v.strip().lower() for k, v in aliases.items()})
        self._doc_map: Dict[str, str] = {}  # per-document acronym -> expansion

    def register(self, alias: str, canonical: str):
        """Runtime-injectable glossary (e.g. a user-supplied domain lexicon).
        Kept out of source so the code stays domain-agnostic."""
        self.aliases[self._normalize(alias)] = canonical.strip().lower()

    @staticmethod
    def _normalize(surface: str) -> str:
        s = (surface or '').strip().lower().replace('\u2019', "'").replace('.', '')
        return ' '.join(t for t in s.split() if t)

    def build_from_doc(self, doc):
        """Derive per-document acronym<->expansion mappings from (a) parenthetical
        glosses in the raw text and (b) initial-letter matches among co-occurring
        ORG/GPE/PRODUCT entities. Nothing is assumed about the domain."""
        self._doc_map = {}
        text = doc.text

        # (a) parenthetical glosses, both orders
        for full, abbr in self._PAREN_FULL_THEN_ABBR.findall(text):
            f, a = self._normalize(full), self._normalize(abbr)
            if a and f and a != f and len(a.replace(' ', '')) <= 9:
                self._doc_map[a] = f
        for abbr, full in self._PAREN_ABBR_THEN_FULL.findall(text):
            f, a = self._normalize(full), self._normalize(abbr)
            if a and f and a != f and len(a.replace(' ', '')) <= 9:
                self._doc_map[a] = f

        # (b) initial-letter matching among co-occurring entities
        ents = [(self._normalize(e.text)) for e in doc.ents
                if e.label_ in ('ORG', 'GPE', 'PRODUCT')]
        for raw in ents:
            compact = raw.replace(' ', '')
            # a candidate acronym: short, all-caps-ish single token
            if raw and ' ' not in raw and 1 < len(compact) <= 6 and raw == raw.lower():
                # (surfaces are lowercased already; require original all-caps below)
                pass
        # Re-scan using original casing to detect acronyms robustly.
        acronyms = {}
        expansions = {}
        for e in doc.ents:
            if e.label_ not in ('ORG', 'GPE', 'PRODUCT'):
                continue
            norm = self._normalize(e.text)
            compact = norm.replace(' ', '')
            if e.text.replace('.', '').isupper() and 1 < len(compact) <= 6:
                acronyms[compact] = norm
            words = norm.split()
            if len(words) >= 2:
                initials = ''.join(w[0] for w in words if w)
                expansions[initials] = norm
        for compact, acr_norm in acronyms.items():
            if compact in expansions and expansions[compact] != acr_norm:
                self._doc_map.setdefault(acr_norm, expansions[compact])

    def canon(self, surface: str) -> str:
        n = self._normalize(surface)
        if n in self.aliases:
            return self.aliases[n]
        if n in self._doc_map:
            return self._doc_map[n]
        return n


# ════════════════════════════════════════════════════════════════════════════
# CORE DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════

class ConceptExtract:
    """Extracted concept with metadata."""
    def __init__(self, text: str, entity_type: str, surface: str, confidence: float = 0.5):
        self.text = text.lower()
        self.type = entity_type
        self.surface = surface
        self.confidence = confidence
        self.sources = []

    def to_dict(self):
        return {
            'concept': self.text,
            'type': self.type,
            'surface': self.surface,
            'confidence': round(self.confidence, 3),
            'sources': ','.join(set(self.sources))
        }


class RelationExtract:
    """Extracted relation with metadata."""
    def __init__(self, subject: str, relation: str, object_: str, confidence: float = 0.5):
        self.subject = subject.lower()
        self.relation = relation
        self.object = object_.lower()
        self.confidence = confidence
        self.source = None
        self.evidence = None

    def to_dict(self):
        return {
            'subject': self.subject,
            'relation': self.relation,
            'object': self.object,
            'confidence': round(self.confidence, 3),
            'source': self.source,
            'evidence': self.evidence
        }

    def __hash__(self):
        return hash((self.subject, self.relation, self.object))

    def __eq__(self, other):
        return (self.subject == other.subject and
                self.relation == other.relation and
                self.object == other.object)


@dataclass
class Proposition:
    """
    An OpenIE proposition: two entity arguments connected by a lexicalized
    dependency path. The path IS the (pre-induction) relation — no frame, no
    slot, no trigger. `induced_label` is filled in later by relation induction.
    """
    arg1: str
    type1: str
    arg2: str
    type2: str
    path_key: str            # canonical path string, e.g. "take→job→with"
    path_lemmas: Tuple[str, ...]
    negated: bool
    sentence: str
    induced_label: Optional[str] = None

    def readable_path(self) -> str:
        label = "_".join(self.path_lemmas).upper()
        label = re.sub(r'[^A-Z0-9_]', '', label)[:40]
        return label or "RELATED_TO"

    def to_dict(self):
        return {
            'arg1': self.arg1, 'type1': self.type1,
            'arg2': self.arg2, 'type2': self.type2,
            'path_key': self.path_key,
            'relation': self.induced_label or self.readable_path(),
            'negated': self.negated,
            'sentence': self.sentence,
        }


# ════════════════════════════════════════════════════════════════════════════
# ALGORITHM 1: DISTANT SUPERVISION  (candidate-pair generator — unchanged core)
# ════════════════════════════════════════════════════════════════════════════

class DistantSupervisionModule:
    """
    Generates candidate entity pairs (the pool the OpenIE extractor and the
    active learner draw from). The clustering method is retained but is no longer
    the source of asserted relations — those now come from OpenIE + induction.
    """

    def __init__(self, max_entity_distance: int = None):
        self.max_entity_distance = (max_entity_distance
                                     if max_entity_distance is not None
                                     else CONFIG.max_entity_token_distance)
        self.entity_pairs = []

    def extract_entity_pairs(self, doc) -> List[Dict]:
        pairs = []
        for sent in doc.sents:
            entities = list(sent.ents)
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    ent1, ent2 = entities[i], entities[j]
                    token_distance = ent2.start - ent1.end
                    if 0 <= token_distance <= self.max_entity_distance:
                        context = self._extract_context(sent, ent1, ent2)
                        pairs.append({
                            'entity1': ent1.text.lower(),
                            'type1': ent1.label_,
                            'entity2': ent2.text.lower(),
                            'type2': ent2.label_,
                            'context': context,
                            'sentence': sent.text,
                            'confidence': CONFIG.conf_distant_base
                        })
        self.entity_pairs = pairs
        logger.info(f"[Distant Supervision] Found {len(pairs)} entity pairs")
        return pairs

    def _extract_context(self, sent, ent1, ent2):
        try:
            left, right = (ent1, ent2) if ent1.start <= ent2.start else (ent2, ent1)
            span = sent.doc[left.end:right.start]
            lemmas = [t.lemma_ for t in span if not t.is_stop and not t.is_punct]
            return " ".join(lemmas)
        except (AttributeError, ValueError, IndexError) as e:
            logger.debug(f"[Distant Supervision] context extraction failed: {e}")
            return ""


# ════════════════════════════════════════════════════════════════════════════
# ALGORITHM 2a: OPEN INFORMATION EXTRACTION  (replaces frame slot filling)
# ════════════════════════════════════════════════════════════════════════════

class PropositionExtractor:
    """
    Symbolic, schema-free relation extractor. For each ordered co-occurring
    entity pair in a sentence, it computes the shortest dependency path between
    the two entity heads (via their lowest common ancestor) and emits a
    Proposition whose relation IS that lexicalized path. There are no trigger
    lists, no frames, no slots, and no domain assumptions: whatever predicate
    structure the parser finds between two entities becomes a candidate relation.

    Argument direction (which entity is arg1 vs arg2) is chosen by dependency
    role — the subject-side argument becomes arg1 — falling back to surface order.
    This is a language-level rule (subjecthood), not a domain rule.
    """

    def __init__(self, canonicalizer: EntityCanonicalizer = None,
                 max_path_len: int = None, max_entity_distance: int = None):
        self.canonicalizer = canonicalizer
        self.max_path_len = max_path_len if max_path_len is not None else CONFIG.max_path_len
        self.max_entity_distance = (max_entity_distance if max_entity_distance is not None
                                    else CONFIG.max_entity_token_distance)
        self.propositions: List[Proposition] = []
        # pairs joined by a SHORT dependency path that carries NO content predicate
        # -> structurally justified NO_RELATION negatives (distinct from pairs the
        # parser simply fails to connect or connects only via an over-long path).
        self.structural_negatives: Set[FrozenSet[str]] = set()

    def _canon(self, surface: str) -> str:
        if self.canonicalizer is not None:
            return self.canonicalizer.canon(surface)
        return (surface or '').strip().lower()

    def extract(self, doc) -> List[Proposition]:
        props: List[Proposition] = []
        self.structural_negatives = set()
        for sent in doc.sents:
            ents = list(sent.ents)
            for i in range(len(ents)):
                for j in range(i + 1, len(ents)):
                    e1, e2 = ents[i], ents[j]
                    # Locality window: only consider pairs within the same token
                    # distance the candidate generator uses. Far-apart in-sentence
                    # pairs almost never yield a content path and cost O(n^2) LCA
                    # walks on entity-dense text. Same linguistic assumption as
                    # DistantSupervisionModule — no new heuristic, no hardcoding.
                    if e2.start - e1.end > self.max_entity_distance:
                        continue
                    parsed = self._dependency_path(e1.root, e2.root)
                    if parsed is None:
                        # no usable path (unconnected or over-long) -> UNKNOWN,
                        # deliberately NOT treated as a negative example.
                        continue
                    if parsed == 'NO_CONTENT':
                        # short path but no predicate on it -> structural negative
                        n1, n2 = self._canon(e1.text), self._canon(e2.text)
                        if n1 and n2 and n1 != n2:
                            self.structural_negatives.add(frozenset((n1, n2)))
                        continue
                    lemmas, negated, e1_is_subj = parsed
                    if not lemmas:
                        continue
                    # direction: subject-side argument is arg1
                    if e1_is_subj is False:
                        se, oe = e2, e1
                    else:
                        se, oe = e1, e2
                    a1, a2 = self._canon(se.text), self._canon(oe.text)
                    if not a1 or not a2 or a1 == a2:
                        continue
                    props.append(Proposition(
                        arg1=a1, type1=se.label_, arg2=a2, type2=oe.label_,
                        path_key="\u2192".join(lemmas),
                        path_lemmas=tuple(lemmas), negated=negated,
                        sentence=sent.text))
        self.propositions = props
        logger.info(f"[OpenIE] Extracted {len(props)} propositions")
        return props

    def _dependency_path(self, a, b):
        """Return (content_lemmas, negated, a_is_subject) for the path a<->b;
        the string 'NO_CONTENT' if a short path exists but has no predicate on it
        (a structural negative); or None if the heads are not connected within the
        length budget (an unknown, never treated as a negative)."""
        a_chain = [a] + list(a.ancestors)
        a_pos = {t.i: k for k, t in enumerate(a_chain)}
        b_chain = [b] + list(b.ancestors)
        lca = None
        b_k = None
        for k, t in enumerate(b_chain):
            if t.i in a_pos:
                lca, b_k = t, k
                break
        if lca is None:
            return None
        a_k = a_pos[lca.i]
        up = a_chain[:a_k + 1]                 # a ... lca
        down = b_chain[:b_k + 1]               # b ... lca
        path = up + list(reversed(down[:-1]))  # a ... lca ... b
        if len(path) > self.max_path_len + 2:
            return None

        arg_ids = {a.i, b.i}
        content = [t for t in path
                   if t.i not in arg_ids and t.pos_ in CONFIG.path_content_pos]
        if not content:
            # A short path exists but carries no content predicate: this is
            # positive evidence of NO relation (structural negative), unlike the
            # None cases above which are genuine "unknowns".
            return 'NO_CONTENT'
        lemmas = [t.lemma_.lower() for t in content]

        negated = any(c.dep_ == 'neg' for t in path for c in t.children) \
            or any(t.dep_ == 'neg' for t in path)

        # subjecthood: does a's own attachment (or its side of the path) look like
        # a subject? Check the dependency labels along a's climb to the LCA.
        a_is_subj = None
        a_deps = {t.dep_ for t in up[:-1]}     # deps on a's side (exclude lca)
        b_deps = {t.dep_ for t in down[:-1]}

        # Passive voice first: the `agent` (by-phrase) is the semantic subject and
        # `nsubjpass` is the semantic object, so surface roles are inverted.
        a_agent, b_agent = a_deps & AGENT_DEPS, b_deps & AGENT_DEPS
        a_pass, b_pass = a_deps & PASSIVE_SUBJ_DEPS, b_deps & PASSIVE_SUBJ_DEPS
        if a_agent or b_pass:
            a_is_subj = True
        elif b_agent or a_pass:
            a_is_subj = False
        # Active voice: subject-side argument is arg1.
        elif (a_deps & SUBJECT_DEPS) and not (b_deps & SUBJECT_DEPS):
            a_is_subj = True
        elif (b_deps & SUBJECT_DEPS) and not (a_deps & SUBJECT_DEPS):
            a_is_subj = False
        elif (a_deps & OBJECT_DEPS) and not (b_deps & OBJECT_DEPS):
            a_is_subj = False
        elif (b_deps & OBJECT_DEPS) and not (a_deps & OBJECT_DEPS):
            a_is_subj = True
        # else: leave None -> caller keeps surface order (a=arg1)
        return lemmas, negated, a_is_subj


# ════════════════════════════════════════════════════════════════════════════
# SELF-GROWING PATH STATISTICS STORE  (cross-run corpus memory)
# ════════════════════════════════════════════════════════════════════════════

class PathStatsStore:
    """
    Persistent, append-then-compact store of per-path argument-filler
    distributions. This is what makes the system self-growing: every processed
    document folds its propositions' fillers into the store, so relation
    induction sharpens over the whole corpus history rather than a single doc.

    For each path key we keep:
      slotX[path] : Counter of arg1 fillers
      slotY[path] : Counter of arg2 fillers
      support[path]: number of propositions with that path
    Persisted as JSONL (one path per line) and compacted on save.
    """

    # Pass path=":memory:" for a non-persistent store (used during evaluation so
    # the real learned knowledge base is never mutated).
    IN_MEMORY = ":memory:"

    def __init__(self, path: str = None):
        if path == self.IN_MEMORY:
            self.path = None
        else:
            self.path = path if path is not None else CONFIG.path_stats_path
        self.slotX: Dict[str, Counter] = defaultdict(Counter)
        self.slotY: Dict[str, Counter] = defaultdict(Counter)
        # Per-path argument NER-type distributions (corpus-derived, from the
        # parser's own labels — not a schema). Used only as a distributional
        # feature for type-signature blocking during induction.
        self.typeX: Dict[str, Counter] = defaultdict(Counter)
        self.typeY: Dict[str, Counter] = defaultdict(Counter)
        self.support: Counter = Counter()
        # content hashes of documents already folded in -> prevents double-counting
        self.seen_docs: Set[str] = set()
        # appends since last full compaction (bounds delta-log growth)
        self._appends_since_compact = 0
        self.compact_every = CONFIG.path_stats_compact_every
        self._load()

    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Robustness: a single corrupt/incomplete record must never abort
                # loading and lose all previously learned knowledge — skip it.
                try:
                    r = json.loads(line)
                    if 'doc_id' in r:                       # ingested-document marker
                        self.seen_docs.add(str(r['doc_id']))
                        continue
                    key = r.get('path')
                    if not key:
                        continue
                    x = {k: int(v) for k, v in r.get('x', {}).items()}
                    y = {k: int(v) for k, v in r.get('y', {}).items()}
                    tx = {k: int(v) for k, v in r.get('tx', {}).items()}
                    ty = {k: int(v) for k, v in r.get('ty', {}).items()}
                    support = int(r.get('support', sum(x.values())))
                except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
                    logger.warning("[PathStatsStore] skipping corrupt record during load.")
                    continue
                self.slotX[key].update(x)
                self.slotY[key].update(y)
                self.typeX[key].update(tx)
                self.typeY[key].update(ty)
                self.support[key] += support

    def add_propositions(self, props: List[Proposition], doc_id: str = None) -> bool:
        """Fold a document's propositions into the store. If doc_id was already
        ingested, this is a no-op (idempotent) so re-processing the same document
        never inflates path support. Returns True if the document was counted."""
        if doc_id is not None and doc_id in self.seen_docs:
            return False
        for p in props:
            self.slotX[p.path_key][p.arg1] += 1
            self.slotY[p.path_key][p.arg2] += 1
            self.typeX[p.path_key][p.type1 or '?'] += 1
            self.typeY[p.path_key][p.type2 or '?'] += 1
            self.support[p.path_key] += 1
        if doc_id is not None:
            self.seen_docs.add(doc_id)
        return True

    def persist_delta(self, props: List[Proposition], doc_id: str = None):
        """Append ONLY this document's contribution to the on-disk log, instead
        of rewriting the entire file. The loader already sums per-line counts, so
        appended deltas accumulate to the correct totals. A full (atomic)
        compaction runs periodically to keep the log from growing unboundedly.
        Call after add_propositions() has returned True for the same document."""
        if not self.path:
            return
        dx: Dict[str, Counter] = defaultdict(Counter)
        dy: Dict[str, Counter] = defaultdict(Counter)
        dtx: Dict[str, Counter] = defaultdict(Counter)
        dty: Dict[str, Counter] = defaultdict(Counter)
        ds: Counter = Counter()
        for p in props:
            dx[p.path_key][p.arg1] += 1
            dy[p.path_key][p.arg2] += 1
            dtx[p.path_key][p.type1 or '?'] += 1
            dty[p.path_key][p.type2 or '?'] += 1
            ds[p.path_key] += 1
        with open(self.path, "a", encoding="utf-8") as f:
            for key in ds:
                f.write(json.dumps({
                    'path': key, 'support': int(ds[key]),
                    'x': dict(dx[key]), 'y': dict(dy[key]),
                    'tx': dict(dtx[key]), 'ty': dict(dty[key]),
                }) + "\n")
            if doc_id is not None:
                f.write(json.dumps({'doc_id': doc_id}) + "\n")
        self._appends_since_compact += 1
        if self._appends_since_compact >= self.compact_every:
            self.save()

    def save(self):
        """Full, atomic, compact rewrite of the whole store (dedup-safe). Written
        to a temp file then os.replace()d so a crash mid-write cannot corrupt the
        existing learned knowledge base."""
        if not self.path:
            return
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for key in self.support:
                f.write(json.dumps({
                    'path': key,
                    'support': int(self.support[key]),
                    'x': dict(self.slotX[key]),
                    'y': dict(self.slotY[key]),
                    'tx': dict(self.typeX[key]),
                    'ty': dict(self.typeY[key]),
                }) + "\n")
            for doc_id in self.seen_docs:
                f.write(json.dumps({'doc_id': doc_id}) + "\n")
        os.replace(tmp, self.path)  # atomic on POSIX and Windows
        self._appends_since_compact = 0

    def stats(self) -> Dict[str, int]:
        return {'paths': len(self.support),
                'total_support': int(sum(self.support.values())),
                'documents': len(self.seen_docs)}


# ════════════════════════════════════════════════════════════════════════════
# ALGORITHM 2b: DIRT-STYLE RELATION INDUCTION  (replaces UNIVERSAL_FRAMES)
# ════════════════════════════════════════════════════════════════════════════

class RelationInductionModule:
    """
    Discovers relation TYPES by clustering dependency paths on the distributional
    hypothesis for paths (Lin & Pantel, 2001, "DIRT"): two paths express the same
    relation if they take the same kinds of arguments. Similarity between two
    paths is the geometric mean of their per-slot filler-overlap similarities,
    where overlap is weighted by pointwise mutual information. Paths are then
    clustered by complete-linkage agglomeration over the distance matrix (a
    diameter-bounded, deterministic strategy that avoids density chaining), and
    each cluster is
    labeled by the surface form of its highest-support member.

    No relation names exist a priori; the inventory is whatever the corpus yields.
    On a biography corpus this induces employment/education/founding-like classes;
    on a chemistry or finance corpus it induces entirely different ones — with no
    code change. That is the operational definition of domain independence.
    """

    def __init__(self, config: BOOFSConfig = CONFIG):
        self.cfg = config
        self.path_to_label: Dict[str, str] = {}
        # per-label cluster cohesion (mean intra-cluster DIRT similarity) and the
        # corpus median path support — both feed corpus-derived confidence.
        self.label_cohesion: Dict[str, float] = {}
        self.corpus_median_support: float = 1.0
        # previous run's clusters as (frozenset(paths), label) for label carry-over
        self._prev_clusters: List[Tuple[FrozenSet[str], str]] = []
        # signature of the (path, support) set last clustered; lets fit() skip the
        # O(P^2) re-clustering when accumulated statistics are unchanged.
        self._last_sig: Optional[int] = None
        # --- incremental-induction state (in-memory within a process) ----------
        # medoid path per label, the adaptive threshold learned per type-block,
        # a snapshot of supports at the last full pass, and a fit counter. These
        # let a fit assign only NEW/CHANGED paths to existing clusters instead of
        # re-clustering everything, while a periodic full pass reconsolidates.
        self._medoids: Dict[str, str] = {}
        self._label_sig: Dict[str, Tuple[str, str]] = {}     # label -> block signature
        self._block_thr: Dict[Tuple[str, str], float] = {}
        # EMA of each block's adaptive threshold across full passes (stability);
        # count of singletons created incrementally since the last full pass
        # (a corpus-relative drift signal that triggers an early full pass).
        self._block_thr_ema: Dict[Tuple[str, str], float] = {}
        self._new_singletons_since_full: int = 0
        # Corpus-wide "same-relation" reference distance, learned from populated
        # blocks that actually merged pairs; used to decide merges in sparse blocks
        # where the within-block gap is undefined. Accumulates across full passes.
        self._merge_ref_dist: Optional[float] = None
        # Reliability calibration: for confidence buckets we accumulate how often a
        # (later human-confirmed) induced edge was correct, so confidence can be
        # mapped to an empirical reliability as the corpus/labels grow.
        self._calib_bins: Dict[int, List[int]] = {}   # bucket -> [correct, total]
        self._support_snapshot: Dict[str, int] = {}
        self._fits_since_full: int = 0
        self._state_path = getattr(config, 'induction_state_path', None)
        self._load_state()

    # ---- cross-run state (label stability) -----------------------------------
    def _load_state(self):
        if not self._state_path or not os.path.exists(self._state_path):
            return
        try:
            with open(self._state_path, encoding="utf-8") as f:
                data = json.load(f)
            self._prev_clusters = [(frozenset(m), lbl) for m, lbl in data.get('clusters', [])]
            self.path_to_label = dict(data.get('path_to_label', {}))
            self._merge_ref_dist = data.get('merge_ref_dist', None)
            self._block_thr_ema = {tuple(k.split('\t')): v
                                   for k, v in data.get('block_thr_ema', {}).items()}
            self._calib_bins = {int(k): list(v) for k, v in data.get('calib_bins', {}).items()}
        except (json.JSONDecodeError, ValueError, TypeError, OSError):
            logger.warning("[Relation Induction] could not load induction state; starting fresh.")

    def _save_state(self):
        if not self._state_path:
            return
        tmp = f"{self._state_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({
                'clusters': [[sorted(m), lbl] for m, lbl in self._prev_clusters],
                'path_to_label': self.path_to_label,
                'merge_ref_dist': self._merge_ref_dist,
                'block_thr_ema': {'\t'.join(k): v for k, v in self._block_thr_ema.items()},
                'calib_bins': {str(k): v for k, v in self._calib_bins.items()},
            }, f)
        os.replace(tmp, self._state_path)

    @staticmethod
    def _make_clusterer(distance_threshold):
        """Complete-linkage agglomerative clusterer, robust to the sklearn
        rename of the precomputed-metric argument (metric= in >=1.2,
        affinity= in older releases)."""
        try:
            return AgglomerativeClustering(
                n_clusters=None, distance_threshold=distance_threshold,
                metric='precomputed', linkage='complete')
        except TypeError:
            return AgglomerativeClustering(
                n_clusters=None, distance_threshold=distance_threshold,
                affinity='precomputed', linkage='complete')

    # ---- DIRT mutual-information machinery -----------------------------------
    @staticmethod
    def _slot_mi(counts_by_path: Dict[str, Counter]):
        """Return (mi[path][word], mass[path]) with mi = max(PMI, 0)."""
        word_tot = Counter()
        path_tot: Dict[str, int] = {}
        N = 0
        for p, c in counts_by_path.items():
            s = sum(c.values())
            path_tot[p] = s
            N += s
            for w, n in c.items():
                word_tot[w] += n
        mi: Dict[str, Dict[str, float]] = {}
        mass: Dict[str, float] = {}
        for p, c in counts_by_path.items():
            d: Dict[str, float] = {}
            total = 0.0
            for w, n in c.items():
                denom = path_tot[p] * word_tot[w]
                if denom <= 0 or n <= 0 or N <= 0:
                    continue
                val = math.log((n * N) / denom)
                if val > 0:
                    d[w] = val
                    total += val
            mi[p] = d
            mass[p] = total
        return mi, mass

    @staticmethod
    def _slot_sim(p, q, mi, mass) -> float:
        dp, dq = mi.get(p, {}), mi.get(q, {})
        if not dp or not dq:
            return 0.0
        shared = dp.keys() & dq.keys()
        if not shared:
            return 0.0
        num = sum(dp[w] + dq[w] for w in shared)
        den = mass.get(p, 0.0) + mass.get(q, 0.0)
        return num / den if den > 0 else 0.0

    def _smooth_slot(self, paths, slot_counts, type_counts):
        """Corpus-derived cold-start smoothing. A path's RARE (globally hapax)
        fillers carry little PMI signal, so a fraction of each path's mass — equal
        to its hapax fraction, capped — is mixed in as its argument-TYPE
        distribution (namespaced pseudo-tokens). Paths with rich, frequent fillers
        get alpha~0 and are untouched; paths whose arguments are all rare fall back
        to their type, letting disjoint-but-same-typed synonyms share signal on
        small corpora. Types come from the parser, not a schema."""
        global_freq = Counter()
        for p in paths:
            global_freq.update(slot_counts.get(p, {}))
        out: Dict[str, Counter] = {}
        for p in paths:
            c = Counter(slot_counts.get(p, {}))
            total = sum(c.values())
            if total <= 0:
                out[p] = c
                continue
            hapax = sum(n for f, n in c.items() if global_freq[f] <= 1)
            alpha = min(self.cfg.type_smoothing_max_alpha, hapax / total)
            if alpha > 0:
                tc = type_counts.get(p, Counter())
                ttotal = sum(tc.values())
                if ttotal > 0:
                    type_mass = alpha * total
                    for t, tn in tc.items():
                        c[f"\u00a7T:{t}"] += type_mass * (tn / ttotal)
            out[p] = c
        return out

    def _global_mi(self, paths, stats: 'PathStatsStore'):
        """Compute DIRT mutual information over the FULL candidate population.
        MI must be global: a filler's informativeness depends on how rare it is
        across the whole corpus, not within one tiny type-block (where every
        shared filler would look ubiquitous and collapse to zero PMI). Clustering
        is still restricted to within-block, but the *signal* is corpus-wide.
        Rare fillers are backed off to their argument type first (see _smooth_slot)."""
        sx = self._smooth_slot(paths, {p: stats.slotX[p] for p in paths},
                               {p: stats.typeX.get(p, Counter()) for p in paths})
        sy = self._smooth_slot(paths, {p: stats.slotY[p] for p in paths},
                               {p: stats.typeY.get(p, Counter()) for p in paths})
        mi_x, mass_x = self._slot_mi(sx)
        mi_y, mass_y = self._slot_mi(sy)
        return mi_x, mass_x, mi_y, mass_y

    def _block_distance(self, block, mis) -> np.ndarray:
        mi_x, mass_x, mi_y, mass_y = mis
        n = len(block)
        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                sx = self._slot_sim(block[i], block[j], mi_x, mass_x)
                sy = self._slot_sim(block[i], block[j], mi_y, mass_y)
                sim = math.sqrt(sx * sy) if (sx > 0 and sy > 0) else 0.0
                D[i, j] = D[j, i] = 1.0 - sim
        return D

    # ---- type signatures / adaptive threshold / cohesion ---------------------
    @staticmethod
    def _dominant(counter: Counter) -> str:
        if not counter:
            return '?'
        # highest count, ties broken lexicographically (deterministic)
        return min(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0]

    def _dominant_sig(self, stats: 'PathStatsStore', path: str) -> Tuple[str, str]:
        return (self._dominant(stats.typeX.get(path, Counter())),
                self._dominant(stats.typeY.get(path, Counter())))

    def _block_threshold(self, D: np.ndarray) -> float:
        """Merge threshold for a block.

        For a POPULATED block (>= min_block_for_gap paths) the cut is read off the
        block's own distance distribution at the WIDEST gap between consecutive
        sorted non-zero distances — the natural separation between 'same' and
        'different' pairs, more principled than a median.

        For a SPARSE block (too few paths for a meaningful gap — the classic
        two-path case where the gap is self-referential and merges nothing) we
        fall back to a corpus-wide reference distance learned from populated
        blocks (_merge_ref_dist): the typical distance between paths that DID
        merge elsewhere. This removes the two-path degeneracy without a hand-set
        constant — the reference is entirely corpus-derived."""
        if D.shape[0] < 2:
            return self.cfg.adaptive_threshold_min
        nz = np.sort(D[np.triu_indices(D.shape[0], 1)])
        nz = nz[nz > 0]
        if nz.size == 0:
            return self.cfg.adaptive_threshold_max  # all identical -> merge freely
        if D.shape[0] < self.cfg.min_block_for_gap:
            # sparse: prefer the corpus-learned "same-relation" level if we have
            # one; otherwise stay conservative (min clamp).
            ref = self._merge_ref_dist if self._merge_ref_dist is not None \
                else self.cfg.adaptive_threshold_min
            # merge a sparse pair only if it is at least as close as the corpus
            # typically is for same-relation pairs (+ tiny epsilon so equality
            # merges, fixing the boundary non-merge).
            thr = ref + 1e-9
            return min(max(thr, self.cfg.adaptive_threshold_min), self.cfg.adaptive_threshold_max)
        cut = int(np.argmax(np.diff(nz)))          # index before the widest gap
        thr = float((nz[cut] + nz[cut + 1]) / 2.0)  # midpoint of that gap
        return min(max(thr, self.cfg.adaptive_threshold_min), self.cfg.adaptive_threshold_max)

    def _update_merge_reference(self, members: List[str], block: List[str], D: np.ndarray):
        """Accumulate the corpus-wide same-relation reference distance from a
        merged (>=2 member) cluster in a POPULATED block: the mean intra-cluster
        distance. Kept as a running mean across full passes (corpus-derived)."""
        if len(members) < 2 or len(block) < self.cfg.min_block_for_gap:
            return
        idx = {p: i for i, p in enumerate(block)}
        tot, k = 0.0, 0
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                tot += D[idx[members[a]], idx[members[b]]]; k += 1
        if k == 0:
            return
        d = tot / k
        self._merge_ref_dist = d if self._merge_ref_dist is None \
            else 0.5 * self._merge_ref_dist + 0.5 * d

    @staticmethod
    def _cohesion(members: List[str], block: List[str], D: np.ndarray) -> Optional[float]:
        if len(members) < 2:
            return None
        idx = {p: i for i, p in enumerate(block)}
        sims, k = 0.0, 0
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                sims += 1.0 - D[idx[members[a]], idx[members[b]]]
                k += 1
        return sims / k if k else None

    # ---- labeling ------------------------------------------------------------
    @staticmethod
    def _readable(path_key: str) -> str:
        lbl = re.sub(r'[^A-Z0-9_]', '', path_key.replace("\u2192", "_").upper())[:40]
        return lbl or "RELATED_TO"

    def _unique(self, label: str, used: Set[str]) -> str:
        if label not in used:
            used.add(label)
            return label
        k = 2
        while f"{label}_{k}" in used:
            k += 1
        out = f"{label}_{k}"
        used.add(out)
        return out

    def _carryover_label(self, members: List[str], used: Set[str], rep: str) -> str:
        """Reuse a previous run's label when this cluster substantially overlaps a
        prior cluster (Jaccard >= threshold). Keeps relation labels stable as the
        corpus grows; falls back to the readable form of the highest-support
        member. Deterministic: best overlap, ties broken by label string."""
        mset = frozenset(members)
        best_lbl, best_j = None, 0.0
        for pset, lbl in self._prev_clusters:
            inter = len(mset & pset)
            if not inter:
                continue
            j = inter / len(mset | pset)
            if (j > best_j) or (j == best_j and best_lbl is not None and lbl < best_lbl):
                best_j, best_lbl = j, lbl
        if best_lbl is not None and best_j >= self.cfg.label_carryover_jaccard and best_lbl not in used:
            used.add(best_lbl)
            return best_lbl
        return self._unique(self._readable(rep), used)

    def fit(self, stats: 'PathStatsStore') -> Dict[str, str]:
        """(Re)induce the relation inventory from accumulated path statistics.

        Deterministic and corpus-driven. A FULL pass clusters within dominant
        argument-type blocks at a corpus-derived per-block threshold, with label
        carry-over for stability. Between full passes, only NEW/CHANGED paths are
        assigned to the nearest existing cluster medoid (O(changed x clusters)),
        so unchanged relations keep their assignment and the O(P^2) matrix is not
        rebuilt every document. A full pass runs periodically to reconcile.
        """
        candidates = sorted(
            (p for p, s in stats.support.items() if s >= self.cfg.min_path_support),
            key=lambda p: (-stats.support[p], p))

        sig = hash(tuple((p, stats.support[p]) for p in candidates))
        if sig == self._last_sig and self.path_to_label:
            return self.path_to_label
        self._last_sig = sig

        # corpus-derived reference for confidence (refreshed every fit)
        sup_sorted = sorted(stats.support[p] for p in candidates) or [1]
        self.corpus_median_support = float(sup_sorted[len(sup_sorted) // 2])

        changed = [p for p in candidates
                   if stats.support.get(p) != self._support_snapshot.get(p)]
        new_block = self.cfg.type_signature_blocking and any(
            self._dominant_sig(stats, p) not in self._block_thr for p in changed)
        # Drift signal: too many incremental singletons accumulated relative to the
        # number of clusters means the incremental approximation has diverged from
        # what a full pass would produce -> reconsolidate early. Corpus-relative.
        drift = self._new_singletons_since_full > max(2, len(self._medoids) // 10)
        need_full = (not self._medoids
                     or self._fits_since_full >= self.cfg.recluster_every
                     or len(changed) > max(1, len(candidates) // 2)
                     or new_block or drift)

        if need_full:
            self._full_cluster(candidates, stats)
            self._fits_since_full = 0
            self._new_singletons_since_full = 0
        else:
            self._incremental_update(changed, candidates, stats)
            self._fits_since_full += 1

        self._support_snapshot = {p: stats.support[p] for p in candidates}
        self._save_state()

        n_types = len(set(self.path_to_label.values()))
        logger.info(f"[Relation Induction] {len(self.path_to_label)} paths -> "
                    f"{n_types} induced types "
                    f"({'full' if need_full else 'incremental'} pass)")
        return self.path_to_label

    def _ema_threshold(self, sigkey, new_thr: float) -> float:
        """Smooth a block's adaptive threshold across full passes so cluster
        boundaries don't jitter as the corpus grows (both values corpus-derived)."""
        prev = self._block_thr_ema.get(sigkey)
        beta = self.cfg.threshold_ema_beta
        thr = new_thr if prev is None else (beta * prev + (1.0 - beta) * new_thr)
        self._block_thr_ema[sigkey] = thr
        return thr

    def _full_cluster(self, candidates: List[str], stats: 'PathStatsStore'):
        self.path_to_label = {}
        self.label_cohesion = {}
        self._medoids = {}
        self._label_sig = {}
        self._block_thr = {}
        used: Set[str] = set()
        new_clusters: List[Tuple[FrozenSet[str], str]] = []

        def _rep_key(p):
            return (-stats.support[p], p)

        # Partition ALL candidates by dominant type signature — no global cap, so
        # rare relations are never exiled to forced singletons.
        if self.cfg.type_signature_blocking:
            blocks: Dict[Tuple[str, str], List[str]] = defaultdict(list)
            for p in candidates:
                blocks[self._dominant_sig(stats, p)].append(p)
        else:
            blocks = {('*', '*'): list(candidates)}

        mis = self._global_mi(candidates, stats) if candidates else None

        # Process POPULATED blocks before sparse ones: populated blocks teach the
        # corpus-wide same-relation reference distance that sparse blocks then use
        # (deterministic key: populated-first, then signature).
        def _block_order(sig):
            return (len(blocks[sig]) < self.cfg.min_block_for_gap, sig)

        for sigkey in sorted(blocks, key=_block_order):
            full_block = blocks[sigkey]           # already support-sorted
            # Per-block cap bounds the O(block^2) matrix; the tail is assigned to
            # the block's clusters afterwards (still no forced singletons).
            head = full_block[:self.cfg.max_block_size]
            tail = full_block[self.cfg.max_block_size:]
            if len(head) >= 2:
                D = self._block_distance(head, mis)
                thr = self._ema_threshold(sigkey, self._block_threshold(D))
                self._block_thr[sigkey] = thr
                labels = self._make_clusterer(thr).fit_predict(D)
                groups: Dict[int, List[str]] = defaultdict(list)
                for p, l in zip(head, labels):
                    groups[int(l)].append(p)
                for l in sorted(groups, key=lambda cid: _rep_key(min(groups[cid], key=_rep_key))):
                    members = groups[l]
                    rep = min(members, key=_rep_key)
                    label = self._carryover_label(members, used, rep)
                    for p in members:
                        self.path_to_label[p] = label
                    coh = self._cohesion(members, head, D)
                    if coh is not None:
                        self.label_cohesion[label] = coh
                    self._medoids[label] = self._medoid(members, head, D)
                    self._label_sig[label] = sigkey
                    self._update_merge_reference(members, head, D)
                    new_clusters.append((frozenset(members), label))
            else:
                p = head[0]
                self._block_thr.setdefault(sigkey, self.cfg.adaptive_threshold_min)
                label = self._carryover_label([p], used, p)
                self.path_to_label[p] = label
                self._medoids[label] = p
                self._label_sig[label] = sigkey
                new_clusters.append((frozenset([p]), label))

            # assign the block's tail to the nearest in-block cluster medoid
            thr = self._block_thr.get(sigkey, self.cfg.adaptive_threshold_min)
            for p in tail:
                best_lbl, best_d = None, None
                for lbl, m in sorted(self._medoids.items()):
                    if self._label_sig.get(lbl) != sigkey or m == p:
                        continue
                    d = self._pair_distance(p, m, mis)
                    if best_d is None or d < best_d or (d == best_d and lbl < best_lbl):
                        best_d, best_lbl = d, lbl
                if best_lbl is not None and best_d is not None and best_d <= thr:
                    self.path_to_label[p] = best_lbl
                    new_clusters.append((frozenset([p]), best_lbl))
                else:
                    lbl = self._carryover_label([p], used, p)
                    self.path_to_label[p] = lbl
                    self._medoids[lbl] = p
                    self._label_sig[lbl] = sigkey
                    new_clusters.append((frozenset([p]), lbl))

        self._prev_clusters = new_clusters

    def _incremental_update(self, changed: List[str], candidates: List[str], stats: 'PathStatsStore'):
        """Assign only NEW/CHANGED paths to the nearest existing cluster medoid in
        the same type block; unchanged paths keep their label (stable membership).
        Distances use current global MI so informativeness stays corpus-wide."""
        mis = self._global_mi(candidates, stats)
        used: Set[str] = set(self.path_to_label.values())
        for p in sorted(changed, key=lambda q: (-stats.support[q], q)):
            sigk = self._dominant_sig(stats, p) if self.cfg.type_signature_blocking else ('*', '*')
            thr = self._block_thr.get(sigk, self.cfg.adaptive_threshold_min)
            best_lbl, best_d = None, None
            for lbl, m in sorted(self._medoids.items()):
                if self._label_sig.get(lbl) != sigk or m == p:
                    continue
                d = self._pair_distance(p, m, mis)
                if best_d is None or d < best_d or (d == best_d and lbl < best_lbl):
                    best_d, best_lbl = d, lbl
            if best_lbl is not None and best_d is not None and best_d <= thr:
                self.path_to_label[p] = best_lbl
            elif p not in self.path_to_label:
                newlbl = self._unique(self._readable(p), used)
                self.path_to_label[p] = newlbl
                self._medoids[newlbl] = p
                self._label_sig[newlbl] = sigk
                self._new_singletons_since_full += 1

    def _pair_distance(self, p: str, q: str, mis) -> float:
        mi_x, mass_x, mi_y, mass_y = mis
        sx = self._slot_sim(p, q, mi_x, mass_x)
        sy = self._slot_sim(p, q, mi_y, mass_y)
        sim = math.sqrt(sx * sy) if (sx > 0 and sy > 0) else 0.0
        return 1.0 - sim

    @staticmethod
    def _medoid(members: List[str], block: List[str], D: np.ndarray) -> str:
        if len(members) == 1:
            return members[0]
        idx = {p: i for i, p in enumerate(block)}
        best, best_sum = members[0], None
        for m in sorted(members):
            s = sum(D[idx[m], idx[n]] for n in members if n != m)
            if best_sum is None or s < best_sum:
                best_sum, best = s, m
        return best

    def label_for(self, path_key: str) -> str:
        """Induced relation label for a path (falls back to its surface form)."""
        return self.path_to_label.get(path_key) or self._readable(path_key)

    def cohesion_for(self, path_key: str) -> Optional[float]:
        return self.label_cohesion.get(self.path_to_label.get(path_key))

    def set_calibration(self, samples):
        """Rebuild the reliability calibration from (confidence, correct) samples
        of human-confirmed induced edges. Idempotent (recomputed from scratch)."""
        bins: Dict[int, List[int]] = {}
        n = self.cfg.calib_bins_count
        for conf, correct in samples:
            b = min(n - 1, max(0, int(conf * n)))
            c, t = bins.get(b, [0, 0])
            bins[b] = [c + (1 if correct else 0), t + 1]
        self._calib_bins = bins

    def calibrated_confidence(self, raw: float) -> float:
        """Map a raw corpus-derived confidence to the empirical reliability its
        bucket has shown against human labels. Below an evidence threshold the raw
        score is returned unchanged; otherwise a Laplace-smoothed empirical
        accuracy is blended in, weighted by how much evidence the bucket has — so
        confidence becomes more meaningful (and better calibrated) as labels grow."""
        n = self.cfg.calib_bins_count
        b = min(n - 1, max(0, int(raw * n)))
        c, t = self._calib_bins.get(b, [0, 0])
        if t < self.cfg.calib_min_evidence:
            return round(raw, 3)
        emp = (c + 1) / (t + 2)                       # Laplace-smoothed reliability
        w = t / (t + self.cfg.calib_min_evidence)     # trust grows with evidence
        return round(w * emp + (1.0 - w) * raw, 3)

    def flush_state(self):
        """Persist induction state on demand (used after calibration updates)."""
        self._save_state()

    def annotate(self, props: List[Proposition]):
        for p in props:
            p.induced_label = self.label_for(p.path_key)

    def full_pass_partition(self, stats: 'PathStatsStore') -> Dict[str, str]:
        """Return the path->label partition a FULL clustering pass would produce
        on the current statistics, WITHOUT disturbing this module's live state.
        Used by the evaluation layer to measure how far the cheap incremental
        assignment has drifted from an exact full pass (approximation error)."""
        ref = RelationInductionModule(config=replace(self.cfg, induction_state_path=None))
        return dict(ref.fit(stats))

    def induce_ontology(self, stats: 'PathStatsStore') -> Dict[str, Dict]:
        """Derive a light T-Box from accumulated statistics: each induced relation
        gets a corpus-derived DOMAIN and RANGE (dominant argument NER types) and,
        where the evidence supports it, a PARENT relation via argument-set
        containment — if (nearly) everything relation A connects is also connected
        by a more general B with the same domain/range, A is a specialization of B.

        This is genuinely symbolic and fully corpus-driven: domain/range come from
        the parser's type counts, subsumption from set inclusion over observed
        argument fillers. No predefined hierarchy, relation names, or schema.
        Deterministic: labels and ties are resolved lexicographically.
        """
        agg: Dict[str, Dict] = {}
        for path, label in self.path_to_label.items():
            a = agg.setdefault(label, {'X': set(), 'Y': set(),
                                       'dx': Counter(), 'dy': Counter(), 'support': 0})
            a['X'].update(stats.slotX.get(path, {}))
            a['Y'].update(stats.slotY.get(path, {}))
            a['dx'].update(stats.typeX.get(path, Counter()))
            a['dy'].update(stats.typeY.get(path, Counter()))
            a['support'] += int(stats.support.get(path, 0))

        labels = sorted(agg)
        schema: Dict[str, Dict] = {
            L: {'domain': self._dominant(agg[L]['dx']),
                'range': self._dominant(agg[L]['dy']),
                'support': agg[L]['support'],
                'parent': None, 'containment': 0.0}
            for L in labels
        }
        # Subsumption from ASYMMETRIC argument-set containment. Generality is read
        # from filler-set BREADTH (which clustering preserves as the per-label
        # union), not support counts: B is a candidate parent of A when A's
        # arguments are largely contained in B's, B is broader, and the inclusion
        # is asymmetric (B not equally contained in A). Among qualifying parents
        # the NEAREST (smallest broader) is chosen, so multi-level taxonomies form.
        for L in labels:
            a = agg[L]
            if not a['X'] or not a['Y']:
                continue
            a_size = len(a['X']) + len(a['Y'])
            candidates_parent = []
            for B in labels:
                if B == L:
                    continue
                b = agg[B]
                if not b['X'] or not b['Y']:
                    continue
                if (schema[B]['domain'] != schema[L]['domain']
                        or schema[B]['range'] != schema[L]['range']):
                    continue
                if len(b['X']) + len(b['Y']) <= a_size:
                    continue  # a parent must be strictly broader in argument spread
                cont_a_in_b = min(len(a['X'] & b['X']) / len(a['X']),
                                  len(a['Y'] & b['Y']) / len(a['Y']))
                cont_b_in_a = min(len(a['X'] & b['X']) / len(b['X']),
                                  len(a['Y'] & b['Y']) / len(b['Y']))
                if cont_a_in_b >= self.cfg.subsumption_min_containment and cont_b_in_a < cont_a_in_b:
                    candidates_parent.append((len(b['X']) + len(b['Y']), -cont_a_in_b, B))
            if candidates_parent:
                _, negc, parent = min(candidates_parent)
                schema[L]['parent'] = parent
                schema[L]['containment'] = round(-negc, 3)

        # Derive hierarchy depth (root = 0) for downstream ontology consumers.
        def _depth(L, seen=()):
            p = schema[L]['parent']
            if p is None or p in seen:
                return 0
            return 1 + _depth(p, seen + (L,))
        for L in labels:
            schema[L]['depth'] = _depth(L)
        return schema


# ════════════════════════════════════════════════════════════════════════════
# ALGORITHM 3: UNSUPERVISED ENTITY CLUSTERING  (SIMILAR_TO — unchanged)
# ════════════════════════════════════════════════════════════════════════════

class UnsupervisedRelationDiscoveryModule:
    """Cluster entities by distributional context similarity -> SIMILAR_TO."""

    def __init__(self):
        self.entity_contexts = defaultdict(list)
        self.discovered_patterns = []

    def build_context_profiles(self, doc):
        # Per-document temporary state: reset so repeated process() calls do not
        # accumulate stale cross-document contexts (the persistent corpus memory
        # lives in PathStatsStore, not here).
        self.entity_contexts = defaultdict(list)
        self.discovered_patterns = []
        for sent in doc.sents:
            entities = [(ent.text.lower(), ent.label_, ent.root) for ent in sent.ents]
            for entity_text, ent_type, root_token in entities:
                context = {
                    'text': entity_text,
                    'type': ent_type,
                    'left': [t.lemma_ for t in root_token.lefts if not t.is_stop],
                    'right': [t.lemma_ for t in root_token.rights if not t.is_stop],
                    'sentence': sent.text
                }
                self.entity_contexts[entity_text].append(context)

    def discover_relations(self, min_cluster_size: int = 2) -> List[Dict]:
        if len(self.entity_contexts) < 3:
            logger.warning("Need at least 3 entities for discovery")
            return []
        entity_names = list(self.entity_contexts.keys())
        feature_vectors = []
        for entity in entity_names:
            contexts = self.entity_contexts[entity]
            all_words = []
            for ctx in contexts:
                all_words.extend(ctx['left'])
                all_words.extend(ctx['right'])
            feature_vectors.append(" ".join(all_words) if all_words else "EMPTY")
        try:
            vectorizer = TfidfVectorizer(max_features=50)
            X = vectorizer.fit_transform(feature_vectors)
            clustering = DBSCAN(eps=CONFIG.dbscan_eps_unsup, min_samples=min_cluster_size, metric='cosine')
            labels = clustering.fit_predict(X.toarray())
            patterns = []
            for cluster_id in set(labels):
                if cluster_id == -1:
                    continue
                cluster_entities = [entity_names[i] for i, l in enumerate(labels) if l == cluster_id]
                all_contexts = []
                for entity in cluster_entities:
                    for ctx in self.entity_contexts[entity]:
                        all_contexts.extend(ctx['left'])
                        all_contexts.extend(ctx['right'])
                signature = [w for w, c in Counter(all_contexts).most_common(3)]
                patterns.append({
                    'cluster_id': cluster_id,
                    'entities': cluster_entities,
                    'signature': signature,
                    'size': len(cluster_entities)
                })
            self.discovered_patterns = patterns
            logger.info(f"[Unsupervised Discovery] Found {len(patterns)} patterns")
            return patterns
        except (ValueError, AttributeError) as e:
            logger.debug(f"Discovery failed: {e}")
            return []


# ════════════════════════════════════════════════════════════════════════════
# ALGORITHM 4: ACTIVE LEARNING  (pool-based loop — preserved, retargeted)
# ════════════════════════════════════════════════════════════════════════════

class Oracle:
    NO_RELATION = "NO_RELATION"

    def label(self, example: Dict) -> str:
        raise NotImplementedError


class CLIOracle(Oracle):
    def label(self, example: Dict) -> str:
        print(f"\n  ({example.get('entity1','')}) <-> ({example.get('entity2','')})"
              f"  | types: {example.get('type1','')}/{example.get('type2','')}"
              f"  | ctx: {(example.get('context','') or '')[:80]}")
        ans = input("  relation label (blank = NO_RELATION): ").strip()
        return ans or Oracle.NO_RELATION


class DictOracle(Oracle):
    def __init__(self, gold: Dict[Tuple[str, str], str]):
        self.gold = {(a.lower(), b.lower()): rel for (a, b), rel in gold.items()}

    def label(self, example: Dict) -> str:
        key = (example.get('entity1', '').lower(), example.get('entity2', '').lower())
        return self.gold.get(key, Oracle.NO_RELATION)


class LabelStore:
    """Persistent append-only label store (JSONL). Human labels authoritative;
    weak seeds never overwrite an existing entry. Pass path=":memory:" for a
    non-persistent store (used during evaluation so the real KB is untouched)."""
    IN_MEMORY = ":memory:"

    def __init__(self, path: str = "boofs_al_labels.jsonl"):
        self.path = None if path == self.IN_MEMORY else path
        self.records: Dict[str, Dict] = {}
        self._load()

    @staticmethod
    def key(ex: Dict) -> str:
        return f"{ex.get('entity1','')}|{ex.get('entity2','')}|{(ex.get('context','') or '')[:60]}"

    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    key = r['key']
                except (json.JSONDecodeError, KeyError, TypeError):
                    logger.warning("[LabelStore] skipping corrupt label record during load.")
                    continue
                self.records[key] = r

    def _record(self, ex: Dict, label: str, source: str) -> Dict:
        return {
            'key': self.key(ex), 'label': label, 'source': source,
            'entity1': ex.get('entity1', ''), 'entity2': ex.get('entity2', ''),
            'type1': ex.get('type1', ''), 'type2': ex.get('type2', ''),
            'context': ex.get('context', ''),
            'frame_agreement': ex.get('frame_agreement', 'none'),
            'confidence': float(ex.get('confidence', 0.0)),
        }

    def _persist(self, r: Dict):
        if not self.path:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\n")

    def add(self, ex: Dict, label: str, source: str = "oracle") -> Dict:
        r = self._record(ex, label, source)
        self.records[r['key']] = r
        self._persist(r)
        return r

    def seed(self, ex: Dict, label: str, source: str = "frame_seed") -> Optional[Dict]:
        existing = self.records.get(self.key(ex))
        if existing is not None:
            # Human ('oracle') labels are authoritative and immutable. A weak
            # (auto-generated) label is REFRESHED when the induced relation for
            # this pair has since changed, so improving induction propagates into
            # the classifier's supervision instead of being frozen at first sight.
            if existing.get('source') == 'oracle' or existing.get('label') == label:
                return None
            return self.add(ex, label, source)   # overwrite stale weak label
        return self.add(ex, label, source)

    def has_human(self, ex: Dict) -> bool:
        r = self.records.get(self.key(ex))
        return bool(r) and r.get('source') == 'oracle'

    def training_data(self) -> List[Dict]:
        return list(self.records.values())

    def stats(self) -> Dict[str, int]:
        return dict(Counter(r['source'] for r in self.records.values()))

    def compact(self):
        if not self.path:
            return
        with open(self.path, "w", encoding="utf-8") as f:
            for r in self.records.values():
                f.write(json.dumps(r) + "\n")


class RelationValidityModel:
    """
    Shallow probabilistic classifier (logistic regression via SGD) over candidate
    pairs; predicts an induced relation label or NO_RELATION and yields calibrated
    posteriors for margin-based uncertainty sampling. Its class space is NOT fixed
    to any schema — it starts with only NO_RELATION and grows to whatever induced
    or human labels appear, so it inherits the pipeline's domain independence.
    Features are the between-entity context + the two NER types (the induced label
    is deliberately excluded to prevent trivial leakage).
    """
    def __init__(self, n_features: int = 2 ** 14, initial_classes: List[str] = None):
        self.vec = HashingVectorizer(n_features=n_features, alternate_sign=False)
        self._make_clf()
        self.is_fitted = False
        self.classes_: Optional[List[str]] = None
        self.initial_classes = list(initial_classes) if initial_classes else [Oracle.NO_RELATION]
        self._trained: Dict[str, Tuple[str, str]] = {}
        self.n_train = 0

    def _make_clf(self):
        self.clf = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=42)

    @staticmethod
    def _weight(source: str) -> float:
        return CONFIG.al_human_weight if source == 'oracle' else CONFIG.al_seed_weight

    @staticmethod
    def _features(ex: Dict) -> str:
        ctx = ex.get('context', '') or ''
        return f"{ctx} TYPE1={ex.get('type1','')} TYPE2={ex.get('type2','')}"

    def _is_dirty(self, r: Dict, key_fn) -> bool:
        prev = self._trained.get(key_fn(r))
        if prev is None:
            return True
        prev_label, prev_source = prev
        if prev_label != r['label']:
            return True
        if prev_source != 'oracle' and r.get('source') == 'oracle':
            return True
        return False

    def update(self, records: List[Dict], key_fn) -> bool:
        labels_all = [r['label'] for r in records]
        if len(records) < 2 or len(set(labels_all)) < 2:
            self.is_fitted = False
            return False
        classes = sorted(set(labels_all) | set(self.initial_classes))
        need_refit = (not self.is_fitted) or set(classes) != set(self.classes_ or [])
        if need_refit:
            self._make_clf()
            self.classes_ = classes
            X = self.vec.transform([self._features(r) for r in records])
            sw = np.array([self._weight(r.get('source', '')) for r in records])
            rng = np.random.RandomState(42)
            idx = np.arange(len(records))
            for epoch in range(max(1, CONFIG.al_refit_epochs)):
                rng.shuffle(idx)
                if epoch == 0:
                    self.clf.partial_fit(X[idx], [labels_all[i] for i in idx],
                                         classes=classes, sample_weight=sw[idx])
                else:
                    self.clf.partial_fit(X[idx], [labels_all[i] for i in idx],
                                         sample_weight=sw[idx])
            self._trained = {key_fn(r): (r['label'], r.get('source', '')) for r in records}
            self.n_train = len(records)
        else:
            dirty = [r for r in records if self._is_dirty(r, key_fn)]
            if dirty:
                Xd = self.vec.transform([self._features(r) for r in dirty])
                yd = [r['label'] for r in dirty]
                swd = np.array([self._weight(r.get('source', '')) for r in dirty])
                for _ in range(max(1, CONFIG.al_incremental_epochs)):
                    self.clf.partial_fit(Xd, yd, sample_weight=swd)
                for r in dirty:
                    self._trained[key_fn(r)] = (r['label'], r.get('source', ''))
                self.n_train = len(self._trained)
        self.is_fitted = True
        return True

    def fit_from_records(self, records: List[Dict]) -> bool:
        self._trained = {}
        self.is_fitted = False
        self.classes_ = None
        return self.update(records, lambda r: r.get('key') or LabelStore.key(r))

    def scaled_confidence(self, prob: float) -> float:
        k = CONFIG.al_confidence_shrinkage
        n = self.n_train
        return round(float(prob) * n / (n + k), 3) if n else 0.0

    def predict_proba(self, examples: List[Dict]) -> np.ndarray:
        X = self.vec.transform([self._features(e) for e in examples])
        return self.clf.predict_proba(X)

    def predict(self, ex: Dict) -> Tuple[Optional[str], float]:
        if not self.is_fitted:
            return (None, 0.0)
        p = self.predict_proba([ex])[0]
        i = int(np.argmax(p))
        return (self.classes_[i], float(p[i]))

    def uncertainty(self, examples: List[Dict]) -> np.ndarray:
        if not self.is_fitted:
            return np.ones(len(examples))
        P = np.sort(self.predict_proba(examples), axis=1)
        if P.shape[1] == 1:
            return 1.0 - P[:, -1]
        return 1.0 - (P[:, -1] - P[:, -2])


class ActiveLearningModule:
    """Pool-based active-learning controller: query -> label -> retrain loop.
    Weak self-supervision now comes from INDUCED relations (data-derived), not
    from any hand-written frame schema."""

    def __init__(self, store: LabelStore = None, oracle: Oracle = None,
                 model: RelationValidityModel = None, query_batch: int = 5):
        self.store = store or LabelStore()
        self.oracle = oracle
        self.model = model or RelationValidityModel()
        self.query_batch = query_batch

    def has_human_labels(self) -> bool:
        """True once at least one genuine human ('oracle') label exists. Weak
        self-supervised seeds do NOT count — they must never override OpenIE."""
        return any(r.get('source') == 'oracle' for r in self.store.training_data())

    @staticmethod
    def compute_uncertainty(example: Dict) -> float:
        score = 0.0
        context = example.get('context', '')
        if len(context.split()) < 2:
            score += 0.3
        type1 = example.get('type1', '')
        type2 = example.get('type2', '')
        if type1 == type2:
            score += 0.3
        conf = example.get('confidence', 0.5)
        score += (1.0 - conf) * 0.4
        return min(score, 1.0)

    @staticmethod
    def select_informative_examples(candidates: List[Dict], k: int = 5) -> List[Dict]:
        if len(candidates) <= k:
            return candidates
        scored = [(ActiveLearningModule.compute_uncertainty(c), c) for c in candidates]
        scored.sort(reverse=True, key=lambda x: x[0])
        return [item[1] for item in scored[:k]]

    def seed_from_induction(self, pairs: List[Dict], induced_examples: List[Dict],
                            negative_pairs: Set[FrozenSet[str]], max_neg: int = None):
        """
        Cold-start the learner with WEAK labels distilled from the pipeline's own
        induced relations (self-supervision). Positives: candidate pairs an
        induced relation supports, labeled with that induced relation and given
        the natural between-entity context. Negatives: pairs joined by a short
        dependency path that carries NO predicate (structural NO_RELATION) — a
        data-driven signal with no schema or type table. Pairs the parser simply
        fails to connect are treated as UNKNOWN, not negatives, and left for the
        human oracle.
        """
        # Look up candidate pairs in either argument order, since a proposition's
        # subject-first direction may be reversed relative to the pair's surface
        # order — this keeps a positive aligned to its natural context/types.
        pair_lookup = {}
        for p in pairs:
            pair_lookup[(p['entity1'], p['entity2'])] = p
            pair_lookup.setdefault((p['entity2'], p['entity1']), p)
        positives = 0
        for fe in induced_examples:
            ex = dict(fe)
            match = pair_lookup.get((fe['entity1'], fe['entity2']))
            if match:
                ex['type1'] = match.get('type1', ex.get('type1', ''))
                ex['type2'] = match.get('type2', ex.get('type2', ''))
                if match.get('context'):
                    ex['context'] = match['context']
            if self.store.seed(ex, fe['relation'], source="induced_seed_pos") is not None:
                positives += 1

        negatives_pool = [
            p for p in pairs
            if frozenset((p['entity1'], p['entity2'])) in negative_pairs
        ]
        cap = max_neg if max_neg is not None else max(positives, 1)
        for p in negatives_pool[:cap]:
            self.store.seed(p, Oracle.NO_RELATION, source="induced_seed_neg")

    # backward-compatible alias for any external caller of the old name
    def seed_from_frames(self, pairs, frame_examples, max_neg=None):
        return self.seed_from_induction(pairs, frame_examples, set(), max_neg=max_neg)

    def retrain(self) -> bool:
        return self.model.update(self.store.training_data(), LabelStore.key)

    def select_queries(self, candidates: List[Dict], k: int = None) -> List[Dict]:
        k = k or self.query_batch
        pool = [c for c in candidates if not self.store.has_human(c)]
        if not pool:
            return []
        if self.model.is_fitted:
            scores = self.model.uncertainty(pool)
        else:
            scores = np.array([self.compute_uncertainty(c) for c in pool])
        order = np.argsort(-np.asarray(scores))
        return [pool[i] for i in order[:k]]

    def run_round(self, candidates: List[Dict], k: int = None) -> List[Dict]:
        self.retrain()
        queries = self.select_queries(candidates, k)
        if self.oracle is not None and queries:
            for ex in queries:
                self.store.add(ex, self.oracle.label(ex), source="oracle")
            self.retrain()
        return queries

    def predict(self, ex: Dict) -> Tuple[Optional[str], float]:
        return self.model.predict(ex)


# ════════════════════════════════════════════════════════════════════════════
# MAIN ONTOLOGY LEARNING SYSTEM
# ════════════════════════════════════════════════════════════════════════════

class BOOFSOntologyLearner:
    """Complete ontology learning system with corpus-driven relation induction."""

    def __init__(self, label_store_path: str = "boofs_al_labels.jsonl",
                 path_stats_path: str = None):
        self.distant_supervisor = DistantSupervisionModule()
        self.relation_discoverer = UnsupervisedRelationDiscoveryModule()

        # dynamic, document-derived entity linking (no static alias tables)
        self.canonicalizer = EntityCanonicalizer()
        # OpenIE + induction replace the hand-crafted frame subsystem
        self.proposition_extractor = PropositionExtractor(canonicalizer=self.canonicalizer)
        self.path_stats = PathStatsStore(path_stats_path)
        # Cross-run label-stability state lives alongside the stats file (None for
        # in-memory/evaluation stores, so evaluation never persists label state).
        _state_path = f"{self.path_stats.path}.induction" if self.path_stats.path else None
        self.relation_inducer = RelationInductionModule(
            config=replace(CONFIG, induction_state_path=_state_path))

        self.label_store = LabelStore(label_store_path)
        # class space starts open (only NO_RELATION); it grows to induced/human
        # labels — so no relation schema is baked into the learner either.
        self.relation_model = RelationValidityModel(initial_classes=[Oracle.NO_RELATION])
        self.active_learner = ActiveLearningModule(store=self.label_store,
                                                   model=self.relation_model)
        self.coref_resolver = CoreferenceResolver()
        self.kg_embedder = None

        self.concepts = []
        self.relations = []
        self.propositions = []
        self.similarity_hypotheses = []
        self.relation_schema = {}
        self.raw_text = None
        self.resolved_text = None

    @classmethod
    def for_evaluation(cls) -> "BOOFSOntologyLearner":
        """A learner backed entirely by in-memory stores, so running the
        evaluation pipeline never mutates the persistent learned knowledge base.
        Combine with process(..., persist_path_stats=False)."""
        return cls(label_store_path=LabelStore.IN_MEMORY,
                   path_stats_path=PathStatsStore.IN_MEMORY)

    def process(self, text: str, use_active_learning: bool = False, verbose: bool = True,
                resolve_coreference: bool = True, oracle: 'Oracle' = None,
                enable_relation_model: bool = True, al_query_batch: int = 5,
                persist_path_stats: bool = True):
        if verbose:
            print("\n" + "=" * 70)
            print("BOOFS: UNIVERSAL ONTOLOGY LEARNING SYSTEM (induced relations)")
            print("=" * 70)

        # Stage 0: coreference resolution
        self.raw_text = text
        if resolve_coreference:
            if verbose: print("\n[0] Resolving coreferences...")
            text = self.coref_resolver.resolve(text)
            if verbose: print(f"    ✓ Coreference backend: {self.coref_resolver.backend or 'rule-based fallback'}")
        self.resolved_text = text

        doc = _require_nlp()(text)
        self.canonicalizer.build_from_doc(doc)

        if verbose: print("\n[1] Extracting concepts...")
        self.concepts = self._extract_concepts(doc)
        if verbose: print(f"    ✓ Found {len(self.concepts)} concepts")

        if verbose: print("\n[2] Candidate entity pairs (distant supervision)...")
        pairs = self.distant_supervisor.extract_entity_pairs(doc)
        for p in pairs:
            p['entity1'] = self.canonicalizer.canon(p['entity1'])
            p['entity2'] = self.canonicalizer.canon(p['entity2'])
        if verbose: print(f"    ✓ {len(pairs)} candidate pairs")

        if verbose: print("\n[3] OpenIE proposition extraction...")
        self.propositions = self.proposition_extractor.extract(doc)
        if verbose: print(f"    ✓ {len(self.propositions)} propositions")

        if verbose: print("\n[4] DIRT relation induction...")
        # Idempotent ingestion: the same document (by content hash) is folded in
        # at most once, so re-processing never double-counts path statistics.
        doc_id = hashlib.sha1(self.resolved_text.encode("utf-8")).hexdigest()
        counted = self.path_stats.add_propositions(self.propositions, doc_id=doc_id)
        if verbose and not counted:
            print("    (document already ingested; path statistics left unchanged)")
        self.relation_inducer.fit(self.path_stats)
        self.relation_inducer.annotate(self.propositions)
        # Persist only this document's delta (append), not the whole file. Skip
        # when the document was a duplicate (nothing new to record).
        if persist_path_stats and counted:
            self.path_stats.persist_delta(self.propositions, doc_id=doc_id)
        n_types = len(set(p.induced_label for p in self.propositions if p.induced_label))
        if verbose: print(f"    ✓ Induced {n_types} relation type(s) over this document")

        # Ontology refinement: derive relation domain/range + subsumption hierarchy
        # from the accumulated corpus statistics (symbolic, no predefined schema).
        self.relation_schema = self.relation_inducer.induce_ontology(self.path_stats)
        if verbose:
            n_parents = sum(1 for v in self.relation_schema.values() if v.get('parent'))
            print(f"    ✓ Induced relation schema: {len(self.relation_schema)} types, "
                  f"{n_parents} subsumption link(s)")

        if verbose: print("\n[5] Unsupervised entity clustering (SIMILAR_TO)...")
        self.relation_discoverer.build_context_profiles(doc)
        patterns = self.relation_discoverer.discover_relations()
        if verbose: print(f"    ✓ Found {len(patterns)} distributional patterns")

        # ── Stage [6] ACTIVE LEARNING ──────────────────────────────────────
        if verbose: print("\n[6] Active learning...")
        if oracle is not None:
            self.active_learner.oracle = oracle
        elif use_active_learning and self.active_learner.oracle is None:
            self.active_learner.oracle = CLIOracle()

        induced_examples = self._induced_examples_from_props(self.propositions)
        self._annotate_induced_relation(pairs, self.propositions)

        if enable_relation_model:
            self.active_learner.seed_from_induction(
                pairs, induced_examples, self.proposition_extractor.structural_negatives)
            if use_active_learning and self.active_learner.oracle is not None and pairs:
                queried = self.active_learner.run_round(pairs, k=al_query_batch)
                self._update_calibration()   # learn reliability from new human labels
                if verbose:
                    print(f"    ✓ Queried {len(queried)} example(s); "
                          f"label store = {self.label_store.stats()}")
            else:
                self.active_learner.retrain()
            if verbose:
                print(f"    ✓ Relation model fitted={self.relation_model.is_fitted}, "
                      f"total labels={len(self.label_store.training_data())}")
        elif verbose:
            print("    (relation model disabled; using induced OpenIE edges)")

        if verbose: print("\n[7] Consolidating relations...")
        self.relations = self._consolidate_relations(self.propositions, patterns, pairs=pairs)
        if verbose: print(f"    ✓ Consolidated {len(self.relations)} unique relations")

        if verbose: print("\n[8] Training knowledge graph embeddings...")
        triples = [(r.subject, r.relation, r.object) for r in self.relations
                   if r.source not in CONFIG.kg_excluded_sources]
        self.kg_embedder = KGEmbeddingModule()
        try:
            self.kg_embedder.train(triples)
            if verbose: print(f"    ✓ Trained embeddings on {len(triples)} triples")
        except Exception as e:
            logger.warning(f"KG embedding training skipped: {e}")

        if verbose:
            print("\n" + "=" * 70)
            print("EXTRACTION COMPLETE")
            print("=" * 70)

        return {
            'concepts': self.concepts,
            'relations': self.relations,
            'propositions': self.propositions,
            'frames': [],  # kept for backward-compatible consumers/eval harness
            'patterns': patterns,
            'similarity_hypotheses': self.similarity_hypotheses,
            'induced_relation_types': sorted(set(self.relation_inducer.path_to_label.values())),
            'relation_schema': self.relation_schema,
        }

    # ------------------------------------------------------------------------
    # induction <-> active-learning glue
    # ------------------------------------------------------------------------
    def _induced_examples_from_props(self, props: List[Proposition]) -> List[Dict]:
        """Weak positive labels for the learner, distilled from induced relations
        (never from a negated proposition)."""
        examples = []
        for p in props:
            if p.negated:
                continue
            label = p.induced_label or p.readable_path()
            examples.append({
                'entity1': p.arg1, 'entity2': p.arg2,
                'relation': label,
                # natural context is filled in from the matching candidate pair
                # during seeding; never the label itself (avoids feature leakage)
                'context': '',
                'frame_agreement': label,
                'confidence': self._raw_induced_conf(p.path_key),
                'type1': p.type1, 'type2': p.type2,
            })
        return examples

    def _annotate_induced_relation(self, pairs: List[Dict], props: List[Proposition]):
        """Tag each candidate pair (undirected) with the induced relation a
        proposition supports for it, plus that relation's raw confidence. The tag
        goes ONLY into 'frame_agreement' (which the classifier excludes from its
        features) and 'confidence'; the natural between-entity 'context' is left
        untouched so the model learns real linguistic evidence rather than
        memorizing the label. Mutates in place."""
        idx = {}
        for p in props:
            idx[(p.arg1, p.arg2)] = p
            idx.setdefault((p.arg2, p.arg1), p)
        for pr in pairs:
            prop = idx.get((pr['entity1'], pr['entity2']))
            if prop is not None:
                pr['frame_agreement'] = prop.induced_label or prop.readable_path()
                pr['confidence'] = self._raw_induced_conf(prop.path_key)
            else:
                pr.setdefault('frame_agreement', 'none')

    def run_active_learning_loop(self, pairs: List[Dict], oracle: 'Oracle',
                                 rounds: int = 3, k: int = 5, verbose: bool = True) -> List[Dict]:
        self.active_learner.oracle = oracle
        history = []
        for r in range(1, rounds + 1):
            queried = self.active_learner.run_round(pairs, k=k)
            history.append({
                'round': r,
                'queried': len(queried),
                'total_labels': len(self.label_store.training_data()),
                'model_fitted': self.relation_model.is_fitted,
            })
            if verbose:
                print(f"  AL round {r}: queried={len(queried)} "
                      f"total_labels={history[-1]['total_labels']} "
                      f"fitted={history[-1]['model_fitted']}")
            if not queried:
                if verbose:
                    print("  Stopping: no remaining unlabeled candidates.")
                break
        self._update_calibration()   # recompute reliability from accumulated labels
        self.label_store.compact()
        return history

    def _extract_concepts(self, doc) -> List[ConceptExtract]:
        concepts_dict = {}
        for ent in doc.ents:
            concept_id = self.canonicalizer.canon(ent.text)
            if concept_id not in concepts_dict:
                c = ConceptExtract(concept_id, ent.label_, ent.text, confidence=CONFIG.conf_ner)
                c.sources.append('NER')
                concepts_dict[concept_id] = c
        for chunk in doc.noun_chunks:
            concept_id = chunk.lemma_.lower()
            if concept_id not in concepts_dict and len(concept_id.split()) > 1:
                c = ConceptExtract(concept_id, 'CONCEPT', chunk.text, confidence=CONFIG.conf_noun_chunk)
                c.sources.append('NOUN_CHUNK')
                concepts_dict[concept_id] = c
        return list(concepts_dict.values())

    def _raw_induced_conf(self, path_key: str) -> float:
        """Uncalibrated corpus-derived confidence: blend of induced-cluster
        cohesion and support relative to the corpus median. Used for calibration
        bucketing (so calibration maps this raw score to observed reliability)."""
        s = self.path_stats.support.get(path_key, 1)
        med = max(self.relation_inducer.corpus_median_support, 1.0)
        support_term = min(1.0, s / (s + med))
        coh = self.relation_inducer.cohesion_for(path_key)
        if coh is None:
            return round(support_term, 3)
        w = self.relation_inducer.cfg.conf_cohesion_weight
        return round(w * coh + (1.0 - w) * support_term, 3)

    def _induced_conf(self, path_key: str) -> float:
        """Calibrated confidence: the raw corpus-derived score mapped through the
        empirical reliability learned from human-confirmed edges (identity until
        enough labels exist). Becomes more meaningful as the corpus/labels grow."""
        return self.relation_inducer.calibrated_confidence(self._raw_induced_conf(path_key))

    def _update_calibration(self):
        """Recompute confidence calibration from the label store: every human
        ('oracle') label on an edge that carried an induced label is a (raw
        confidence, was-it-correct) sample. Symbolic frequency counting only."""
        samples = []
        for r in self.label_store.training_data():
            if r.get('source') == 'oracle' and r.get('frame_agreement', 'none') != 'none':
                samples.append((float(r.get('confidence', 0.0)),
                                r.get('label') == r.get('frame_agreement')))
        if samples:
            self.relation_inducer.set_calibration(samples)
            self.relation_inducer.flush_state()

    def _consolidate_relations(self, props: List[Proposition], patterns,
                               pairs=None) -> List[RelationExtract]:
        relations_set = set()
        relations_list = []

        def _add(rel):
            if rel not in relations_set:
                relations_set.add(rel)
                relations_list.append(rel)

        # --- primary relations -------------------------------------------------
        # OpenIE + induction is the DEFAULT output. The classifier only takes over
        # once genuine human-labelled examples exist; weak self-supervised seeds
        # alone never replace the dependency-path relations (they only inform
        # query selection). This keeps "relations emerge from dependency paths"
        # true unless a human has actually corrected the labels.
        use_model = (pairs is not None
                     and self.active_learner.model.is_fitted
                     and self.active_learner.has_human_labels())
        if use_model:
            for pair in pairs:
                label, prob = self.active_learner.predict(pair)
                if label is None or label == Oracle.NO_RELATION:
                    continue
                rel = RelationExtract(pair['entity1'], label, pair['entity2'],
                                      confidence=self.active_learner.model.scaled_confidence(prob))
                rel.source = 'active_learned'
                rel.evidence = pair.get('context', '')
                _add(rel)
        else:
            for p in props:
                if p.negated and CONFIG.skip_negated:
                    continue
                neg_mult = CONFIG.negation_confidence_penalty if p.negated else 1.0
                label = p.induced_label or p.readable_path()
                conf = round(self._induced_conf(p.path_key) * neg_mult, 3)
                rel = RelationExtract(p.arg1, label, p.arg2, confidence=conf)
                rel.source = 'open_ie_induced' if not p.negated else 'open_ie_induced_negated'
                rel.evidence = p.sentence
                _add(rel)

        # --- distributional SIMILAR_TO hypotheses (kept separate) --------------
        self.similarity_hypotheses = []
        for pattern in patterns:
            ents = pattern['entities']
            if len(ents) >= 2:
                for i in range(len(ents)):
                    for j in range(i + 1, len(ents)):
                        rel = RelationExtract(ents[i], 'SIMILAR_TO', ents[j],
                                              confidence=CONFIG.conf_similarity)
                        rel.source = 'distributional_similarity'
                        rel.evidence = f"cluster {pattern['cluster_id']} signature={pattern['signature']}"
                        self.similarity_hypotheses.append(rel)
                        _add(rel)

        return relations_list

    # ------------------------------------------------------------------------
    # EXPORTS
    # ------------------------------------------------------------------------
    def export_to_csv(self, prefix: str = "ontology"):
        with open(f"{prefix}_concepts.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["concept", "type", "surface", "confidence", "sources"])
            writer.writeheader()
            writer.writerows([c.to_dict() for c in self.concepts])
        logger.info(f"✓ Exported {len(self.concepts)} concepts to {prefix}_concepts.csv")

        with open(f"{prefix}_relations.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["subject", "relation", "object", "confidence", "source", "evidence"])
            writer.writeheader()
            writer.writerows([r.to_dict() for r in self.relations])
        logger.info(f"✓ Exported {len(self.relations)} relations to {prefix}_relations.csv")

        # OpenIE propositions replace the old frame-slots export
        with open(f"{prefix}_propositions.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["arg1", "type1", "relation", "path_key", "arg2", "type2", "negated", "sentence"])
            writer.writeheader()
            for p in self.propositions:
                d = p.to_dict()
                writer.writerow({
                    'arg1': d['arg1'], 'type1': d['type1'], 'relation': d['relation'],
                    'path_key': d['path_key'], 'arg2': d['arg2'], 'type2': d['type2'],
                    'negated': d['negated'], 'sentence': d['sentence'],
                })
        logger.info(f"✓ Exported {len(self.propositions)} propositions to {prefix}_propositions.csv")

    def export_similarity_hypotheses_to_csv(self, prefix: str = "ontology"):
        with open(f"{prefix}_similarity_hypotheses.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["subject", "relation", "object", "confidence", "source", "evidence"])
            writer.writeheader()
            writer.writerows([r.to_dict() for r in self.similarity_hypotheses])
        logger.info(f"✓ Exported {len(self.similarity_hypotheses)} similarity hypotheses to "
                    f"{prefix}_similarity_hypotheses.csv")

    def export_embeddings_to_csv(self, prefix: str = "ontology"):
        if self.kg_embedder is None or not self.kg_embedder.is_trained:
            logger.warning("No trained KG embeddings available to export.")
            return
        self.kg_embedder.export_embeddings_to_csv(f"{prefix}_embeddings.csv")
        self.kg_embedder.export_link_predictions_to_csv(f"{prefix}_link_predictions.csv")
        self.kg_embedder.export_similarity_scores_to_csv(f"{prefix}_similarity_scores.csv")

    def print_summary(self):
        print("\n" + "=" * 70)
        print("ONTOLOGY SUMMARY")
        print("=" * 70)

        print(f"\n📦 CONCEPTS: {len(self.concepts)}")
        print("-" * 70)
        for i, c in enumerate(self.concepts[:10], 1):
            print(f"  {i:2}. {c.text:25} [{c.type:12}] conf={c.confidence:.2f}")
        if len(self.concepts) > 10:
            print(f"  ... and {len(self.concepts) - 10} more")

        print(f"\n🔗 RELATIONS: {len(self.relations)}")
        print("-" * 70)
        for i, r in enumerate(self.relations[:10], 1):
            print(f"  {i:2}. ({r.subject:20}) --[{r.relation:18}]--> ({r.object:20}) conf={r.confidence:.2f}")
        if len(self.relations) > 10:
            print(f"  ... and {len(self.relations) - 10} more")

        induced = sorted(set(self.relation_inducer.path_to_label.values()))
        print(f"\n🧭 INDUCED RELATION TYPES: {len(induced)}")
        print("-" * 70)
        for i, lbl in enumerate(induced[:12], 1):
            print(f"  {i:2}. {lbl}")
        if len(induced) > 12:
            print(f"  ... and {len(induced) - 12} more")

        print(f"\n🧩 PROPOSITIONS: {len(self.propositions)}")
        print("-" * 70)
        for i, p in enumerate(self.propositions[:6], 1):
            neg = " (NEGATED)" if p.negated else ""
            print(f"  {i}. ({p.arg1}) --[{p.induced_label}]--> ({p.arg2}){neg}")
        if len(self.propositions) > 6:
            print(f"  ... and {len(self.propositions) - 6} more")

        print("\n" + "=" * 70)


# ════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH EMBEDDINGS  (unchanged — operates only on final triples)
# ════════════════════════════════════════════════════════════════════════════
#
# NOTE: this reporting stage uses PyKEEN/RotatE, which is representation
# learning and orthogonal to the (symbolic) relation-extraction subsystem this
# refactor targets. It is optional: process() catches its failure and continues.
# If a strict "no neural methods anywhere" rule applies, swap it for a symbolic
# link-prediction method (e.g. rule mining over the triples); relation
# extraction does not depend on it.

class KGEmbeddingModule:
    def __init__(self, embedding_dim: int = 50, num_epochs: int = 50, use_complex: bool = False):
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.use_complex = use_complex
        self.is_trained = False
        self.evaluation_valid = False
        self.triples_factory = None
        self.rotate_result = None
        self.complex_result = None
        self._entity_to_id = {}
        self._id_to_entity = {}

    def train(self, triples: List[Tuple[str, str, str]]):
        from pykeen.triples import TriplesFactory
        from pykeen.pipeline import pipeline

        triples = [t for t in triples if t[0] and t[1] and t[2]]
        if len(triples) < 3:
            raise ValueError("Not enough triples to train KG embeddings (need >= 3).")

        triples_array = np.array(triples, dtype=str)
        self.triples_factory = TriplesFactory.from_labeled_triples(triples_array)
        self._entity_to_id = self.triples_factory.entity_to_id
        self._id_to_entity = {v: k for k, v in self._entity_to_id.items()}

        training_tf = self.triples_factory
        testing_tf = self.triples_factory
        if len(triples) >= CONFIG.min_triples_for_eval:
            try:
                training_tf, testing_tf = self.triples_factory.split(
                    [1.0 - CONFIG.kg_test_ratio, CONFIG.kg_test_ratio], random_state=42)
                self.evaluation_valid = True
            except Exception as e:
                logger.warning(f"Triple split failed ({e}); evaluation disabled.")
                training_tf = testing_tf = self.triples_factory
                self.evaluation_valid = False
        else:
            logger.info(
                f"Only {len(triples)} triples (< {CONFIG.min_triples_for_eval}); "
                f"Hits@K evaluation disabled to avoid train-set leakage.")
            self.evaluation_valid = False

        self.rotate_result = pipeline(
            training=training_tf,
            testing=testing_tf,
            model='RotatE',
            model_kwargs=dict(embedding_dim=self.embedding_dim),
            training_kwargs=dict(num_epochs=self.num_epochs, use_tqdm=False),
            random_seed=42,
        )

        if self.use_complex:
            self.complex_result = pipeline(
                training=training_tf,
                testing=testing_tf,
                model='ComplEx',
                model_kwargs=dict(embedding_dim=self.embedding_dim),
                training_kwargs=dict(num_epochs=self.num_epochs, use_tqdm=False),
                random_seed=42,
            )

        self.is_trained = True

    def predict_missing_links(self, top_k: int = 10):
        if not self.is_trained:
            raise RuntimeError("Call train() before predict_missing_links().")
        from pykeen.predict import predict_all
        predictions = predict_all(model=self.rotate_result.model, k=top_k)
        df = predictions.process(factory=self.triples_factory).df
        return df.head(top_k)

    def get_entity_similarity(self, entity: str, top_k: int = 5):
        if not self.is_trained:
            raise RuntimeError("Call train() before get_entity_similarity().")
        entity = entity.lower()
        if entity not in self._entity_to_id:
            return []
        entity_embeddings = self.rotate_result.model.entity_representations[0](indices=None).detach().cpu().numpy()
        if np.iscomplexobj(entity_embeddings):
            entity_embeddings = np.abs(entity_embeddings)
        target_idx = self._entity_to_id[entity]
        sims = cosine_similarity([entity_embeddings[target_idx]], entity_embeddings)[0]
        ranked = sorted(
            ((self._id_to_entity[i], float(s)) for i, s in enumerate(sims) if i != target_idx),
            key=lambda x: -x[1]
        )
        return ranked[:top_k]

    def export_embeddings_to_csv(self, filepath: str):
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["entity"] + [f"dim_{i}" for i in range(self.embedding_dim)])
            embeddings = self.rotate_result.model.entity_representations[0](indices=None).detach().cpu().numpy()
            if np.iscomplexobj(embeddings):
                embeddings = np.abs(embeddings)
            for entity, idx in self._entity_to_id.items():
                writer.writerow([entity] + list(embeddings[idx][:self.embedding_dim]))
        logger.info(f"✓ Exported entity embeddings to {filepath}")

    def export_link_predictions_to_csv(self, filepath: str, top_k: int = 20):
        try:
            df = self.predict_missing_links(top_k=top_k)
            df.to_csv(filepath, index=False)
            logger.info(f"✓ Exported link predictions to {filepath}")
        except Exception as e:
            logger.warning(f"Could not export link predictions: {e}")

    def export_similarity_scores_to_csv(self, filepath: str, top_k: int = 5):
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["entity", "similar_entity", "similarity"])
            for entity in self._entity_to_id:
                for similar_entity, score in self.get_entity_similarity(entity, top_k=top_k):
                    writer.writerow([entity, similar_entity, round(score, 4)])
        logger.info(f"✓ Exported similarity scores to {filepath}")


# ════════════════════════════════════════════════════════════════════════════
# EVALUATION METRICS  (unchanged, schema-agnostic)
# ════════════════════════════════════════════════════════════════════════════

def evaluate_coreference_improvement(raw_text: str, resolved_text: str) -> Dict:
    raw_doc, resolved_doc = _require_nlp()(raw_text), _require_nlp()(resolved_text)
    pronouns_before = sum(1 for t in raw_doc if t.text.lower() in CoreferenceResolver.PRONOUNS)
    pronouns_after = sum(1 for t in resolved_doc if t.text.lower() in CoreferenceResolver.PRONOUNS)
    resolved_count = max(pronouns_before - pronouns_after, 0)
    return {
        'pronouns_before': pronouns_before,
        'pronouns_after': pronouns_after,
        'pronouns_resolved': resolved_count,
        'resolution_rate': round(resolved_count / pronouns_before, 3) if pronouns_before else 0.0,
    }


def evaluate_relation_precision(relations_before: List['RelationExtract'],
                                 relations_after: List['RelationExtract'],
                                 sample_labels: Optional[Dict[Tuple[str, str, str], bool]] = None) -> Dict:
    def pronoun_free_ratio(rels):
        if not rels:
            return 0.0
        clean = sum(1 for r in rels if r.subject not in CoreferenceResolver.PRONOUNS
                    and r.object not in CoreferenceResolver.PRONOUNS)
        return round(clean / len(rels), 3)

    result = {
        'proxy_precision_before': pronoun_free_ratio(relations_before),
        'proxy_precision_after': pronoun_free_ratio(relations_after),
    }
    if sample_labels:
        def labeled_precision(rels):
            labeled = [(r.subject, r.relation, r.object) for r in rels
                       if (r.subject, r.relation, r.object) in sample_labels]
            if not labeled:
                return None
            correct = sum(1 for t in labeled if sample_labels[t])
            return round(correct / len(labeled), 3)
        result['labeled_precision_before'] = labeled_precision(relations_before)
        result['labeled_precision_after'] = labeled_precision(relations_after)
    return result


def evaluate_hits_at_k(kg_embedder: 'KGEmbeddingModule', k: int = 10) -> Optional[float]:
    if kg_embedder is None or not kg_embedder.is_trained:
        return None
    if not getattr(kg_embedder, 'evaluation_valid', False):
        logger.info(f"Hits@{k} suppressed: graph too small for a valid held-out evaluation.")
        return None
    try:
        metrics = kg_embedder.rotate_result.metric_results.to_dict()
        return metrics.get('both', {}).get('realistic', {}).get(f'hits_at_{k}')
    except Exception as e:
        logger.warning(f"Could not extract Hits@{k}: {e}")
        return None


def evaluate_entity_similarity_quality(kg_embedder: 'KGEmbeddingModule', sample_size: int = 10) -> Dict:
    if kg_embedder is None or not kg_embedder.is_trained:
        return {'avg_top1_similarity': None, 'sampled_entities': 0}
    entities = list(kg_embedder._entity_to_id.keys())[:sample_size]
    scores = []
    for e in entities:
        sims = kg_embedder.get_entity_similarity(e, top_k=1)
        if sims:
            scores.append(sims[0][1])
    return {
        'avg_top1_similarity': round(sum(scores) / len(scores), 3) if scores else None,
        'sampled_entities': len(scores),
    }


# ════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sample_text = """
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

    # Human labels are free-form strings; the model's class space grows to fit
    # them — nothing here declares a fixed relation schema.
    gold = {
        ('dave', 'general electric'): 'WORKS_FOR',
        ('dave', 'ge'): 'WORKS_FOR',
        ('dave', 'stanford'): 'STUDIED_AT',
        ('bill', 'stanford'): 'STUDIED_AT',
    }
    oracle = DictOracle(gold)

    learner = BOOFSOntologyLearner()
    results = learner.process(sample_text, use_active_learning=True, oracle=oracle, verbose=True)

    print("\n" + "=" * 70)
    print("ACTIVE LEARNING — MULTI-ROUND LOOP")
    print("=" * 70)
    learner.run_active_learning_loop(learner.distant_supervisor.entity_pairs,
                                     oracle, rounds=3, k=3)
    print("Label store sources:", learner.label_store.stats())

    learner.export_to_csv("boofs_results")
    learner.export_similarity_hypotheses_to_csv("boofs_results")
    learner.export_embeddings_to_csv("boofs_results")

    learner.print_summary()

    print("\n" + "=" * 70)
    print("EVALUATION METRICS")
    print("=" * 70)
    print(evaluate_coreference_improvement(learner.raw_text, learner.resolved_text))

    # Genuine before/after: extract WITHOUT coreference on a throwaway in-memory
    # learner, then compare against the WITH-coreference relations above. (Passing
    # the same set twice would have compared it to itself and told us nothing.)
    _no_coref = BOOFSOntologyLearner.for_evaluation()
    _no_coref.process(sample_text, use_active_learning=False, resolve_coreference=False,
                      verbose=False, persist_path_stats=False, enable_relation_model=False)
    print("coref off vs on:",
          evaluate_relation_precision(_no_coref.relations, learner.relations))
    print({'hits_at_10': evaluate_hits_at_k(learner.kg_embedder, k=10)})
    print(evaluate_entity_similarity_quality(learner.kg_embedder))

    print("\n✅ Ontology learning complete (relations induced, not declared).")


# ════════════════════════════════════════════════════════════════════════════
# CHANGE LOG  (relative to the frame-based BOOFS)
# ════════════════════════════════════════════════════════════════════════════
#
# REMOVED (all hardcoded / domain-specific):
#   - UNIVERSAL_FRAMES, CANONICAL_FRAME_RELATIONS, PLAUSIBLE_NER_PAIRS,
#     _plausible_ner_type_pairs  (declared relation schema + trigger lists +
#     evidence specs + slot definitions)
#   - FrameSlottingModule, FrameInstance  (predefined slot filling)
#   - SEMANTIC_ROLE_MAPPING as a frame-role table
#   - EntityCanonicalizer.DEFAULT_ALIASES, CORP_SUFFIXES  (static alias tables)
#   - RelationValidityModel's fixed canonical class space
#
# ADDED (symbolic, unsupervised, corpus-driven):
#   - PropositionExtractor: OpenIE over LCA dependency paths; the path IS the
#     relation. No triggers, no frames, no slots, no domain assumptions.
#   - PathStatsStore: persistent per-path filler distributions -> self-growing
#     corpus memory across runs.
#   - RelationInductionModule: DIRT-style path clustering to discover relation
#     TYPES; each cluster labeled by its most representative path.
#   - EntityCanonicalizer: dynamic, document-derived acronym/gloss detection only.
#
# PRESERVED (unchanged or lightly retargeted):
#   - CoreferenceResolver, DistantSupervisionModule (now a candidate generator),
#     UnsupervisedRelationDiscoveryModule (SIMILAR_TO), the full active-learning
#     loop (now seeded by induced relations), KGEmbeddingModule, all exports,
#     and every evaluation function. process() return keys stay backward
#     compatible ('frames' kept as []).
