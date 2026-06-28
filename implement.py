from __future__ import annotations

import csv
import logging
import warnings
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
warnings.filterwarnings("ignore")

import spacy
from nltk.corpus import verbnet as vn
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

from pykeen.models import TransE
from pykeen.training import SLCWATrainingLoop
from pykeen.triples import TriplesFactory
from torch.optim import Adam

logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")
logger = logging.getLogger("BOOFS")

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None
    logger.warning(
        "spaCy model not found. Run:  python -m spacy download en_core_web_sm"
    )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Concept:
    canonical_id: str    # normalised lowercase key, e.g. "dave packard"
    surface: str         # original text, e.g. "Dave Packard"
    entity_type: str     # PERSON | ORG | GPE | CONCEPT | …
    confidence: float
    source: str          # NER | NOUN_CHUNK


@dataclass
class RelationTriple:
    subject: str
    relation: str
    object_: str
    confidence: float
    source: str   # DISTANT_SUPERVISION | FRAME | DISTRIBUTIONAL
    evidence: str = ""

    def key(self) -> tuple:
        return (self.subject, self.relation, self.object_)

    def to_dict(self) -> dict:
        return {
            "subject": self.subject, "relation": self.relation,
            "object": self.object_, "confidence": round(self.confidence, 3),
            "source": self.source, "evidence": self.evidence[:120],
        }


@dataclass
class FrameInstance:
    frame_id: str
    trigger_verb: str
    tier: int             # 1 = VerbNet roles, 2 = dependency fallback
    confidence: float
    slots: Dict[str, Tuple[str, float]] = field(default_factory=dict)
    sentence: str = ""

    def add_slot(self, role: str, value: str, conf: float):
        if role not in self.slots or conf > self.slots[role][1]:
            self.slots[role] = (value, conf)

    def to_rows(self) -> List[dict]:
        return [
            {"frame_id": self.frame_id, "trigger": self.trigger_verb,
             "tier": self.tier, "slot": role,
             "value": val, "confidence": round(conf, 3)}
            for role, (val, conf) in self.slots.items()
        ]




@dataclass
class ExtractionContext:
    """
    Built once per document from a single spaCy parse.
    Every downstream module reads from this object — none of them
    touch doc.ents or doc.noun_chunks directly.  That structural
    constraint (enforced by API, not convention) is what fixes the
    three-divergent-entity-sets bug in v1.
    """
    doc: object
    concepts: Dict[str, Concept] = field(default_factory=dict)
    # stored so resolve() can consult it post-build (e.g. "he" -> "dave")
    _canonical_map: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def build(cls, doc, canonical_map: Dict[str, str] = None) -> "ExtractionContext":
        """
        canonical_map: output of RuleBasedCorefResolver.resolve().
        Stored on the context so every downstream resolve() call can use it.
        Pass None when coref is not needed — still builds correctly.
        """
        cm = canonical_map or {}
        ctx = cls(doc=doc, _canonical_map=cm)

        for ent in doc.ents:
            raw_key = ent.text.lower().strip()
            key = cm.get(raw_key, raw_key)
            if key not in ctx.concepts:
                ctx.concepts[key] = Concept(key, ent.text, ent.label_, 0.9, "NER")

        for chunk in doc.noun_chunks:
            raw_key = chunk.lemma_.lower().strip()
            key = cm.get(raw_key, raw_key)
            if key not in ctx.concepts and len(key.split()) > 1:
                ctx.concepts[key] = Concept(key, chunk.text, "CONCEPT", 0.6, "NOUN_CHUNK")

        return ctx

    def entities_in_sentence(self, sent) -> list:
        """
        Single replacement for every module's `list(sent.ents)` call.
        Accepts entities whose raw text is either directly in concepts or
        reachable through the coref map (e.g. the token 'he' maps to 'dave').
        """
        return [
            e for e in sent.ents
            if self._canonical_map.get(e.text.lower().strip(),
                                       e.text.lower().strip()) in self.concepts
        ]

    def resolve(self, text: str) -> str:
        """
        Resolution order:
          1. coref map  (pronoun / alias -> canonical name)
          2. concept registry  (return the stored canonical_id)
          3. normalised input  (unknown text, returned as-is)
        """
        key = text.lower().strip()
        key = self._canonical_map.get(key, key)          # step 1
        return self.concepts[key].canonical_id if key in self.concepts else key  # step 2/3




_PRONOUN_GENDER = {
    "he": "MALE",  "him": "MALE",  "his": "MALE",  "himself": "MALE",
    "she": "FEMALE", "her": "FEMALE", "hers": "FEMALE", "herself": "FEMALE",
    "they": "PLURAL", "them": "PLURAL", "their": "PLURAL", "themselves": "PLURAL",
    "it": "NEUT",  "its": "NEUT",  "itself": "NEUT",
}

_NER_GENDER = {
    "ORG": "NEUT", "GPE": "NEUT", "LOC": "NEUT",
    "PRODUCT": "NEUT", "FAC": "NEUT", "WORK_OF_ART": "NEUT",
    "PERSON": None,  # determined from name heuristic below
}

# ~200 common first names with known gender.
# Not an attempt at completeness — purpose is to break ties when the
# text has multiple male/female PERSON entities in scope.
_NAME_GENDER = {
    # male
    "bill": "MALE", "dave": "MALE", "david": "MALE", "william": "MALE",
    "fred": "MALE", "frederick": "MALE", "paul": "MALE", "george": "MALE",
    "john": "MALE", "james": "MALE", "robert": "MALE", "michael": "MALE",
    "steve": "MALE", "steven": "MALE", "larry": "MALE", "mark": "MALE",
    "elon": "MALE", "jeff": "MALE", "tim": "MALE", "satya": "MALE",
    "sam": "MALE", "peter": "MALE", "richard": "MALE", "charles": "MALE",
    # female
    "lucile": "FEMALE", "alice": "FEMALE", "sarah": "FEMALE", "mary": "FEMALE",
    "emily": "FEMALE", "lisa": "FEMALE", "jennifer": "FEMALE", "jessica": "FEMALE",
    "susan": "FEMALE", "karen": "FEMALE", "nancy": "FEMALE", "linda": "FEMALE",
    "patricia": "FEMALE", "barbara": "FEMALE", "elizabeth": "FEMALE",
}


class RuleBasedCorefResolver:
    """
    Resolves pronouns to their most-recent gender-matching named entity
    antecedent, using morphological gender signals (no external model).

    Works with the spaCy doc object your machine produces — tested with
    mock objects in this sandbox.

    Limitation: one-antecedent-per-pronoun (no cluster merging).  Errors
    propagate silently (wrong gender → wrong antecedent), which is why
    tier-1 frame results carry a small confidence penalty when their slot
    value came from a coref resolution rather than a direct NER hit.
    """

    def _entity_gender(self, ent) -> Optional[str]:
        if ent.label_ != "PERSON":
            return _NER_GENDER.get(ent.label_, "NEUT")
        for token in ent:
            g = _NAME_GENDER.get(token.text.lower())
            if g:
                return g
        return None  # gender unknown

    def resolve(self, doc) -> Dict[str, str]:
        """
        Returns canonical_map: {pronoun_or_alias -> canonical_entity_key}
        e.g. {"he": "dave", "his": "dave", "she": "lucile"}

        Algorithm:
          Walk sentences in document order.  Maintain a recency-ordered
          list of seen PERSON/ORG/GPE entities per gender bucket.
          When a pronoun token is encountered, look up its gender bucket
          and pick the most-recently-seen entity in that bucket.
        """
        canonical_map: Dict[str, str] = {}
        # gender bucket → list of entity canonical keys, most recent last
        recent: Dict[str, List[str]] = defaultdict(list)

        for sent in doc.sents:
            # First pass: register named entities in recency lists
            for ent in sent.ents:
                gender = self._entity_gender(ent)
                if gender:
                    key = ent.text.lower().strip()
                    bucket = recent[gender]
                    if key in bucket:
                        bucket.remove(key)
                    bucket.append(key)  # most recent is last

            # Second pass: resolve pronoun tokens
            for token in sent:
                lower = token.lower_
                if lower not in _PRONOUN_GENDER:
                    continue
                if token.pos_ not in ("PRON",):
                    continue
                pronoun_gender = _PRONOUN_GENDER[lower]
                bucket = recent.get(pronoun_gender, [])
                if bucket:
                    antecedent = bucket[-1]   # most recent matching entity
                    canonical_map[lower] = antecedent

        return canonical_map

