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