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



_VN_ROLE_TO_SLOT = {
    "Agent": "AGENT",      "Experiencer": "AGENT", "Actor": "AGENT",
    "Actor1": "AGENT",     "Co-Agent": "AGENT",
    "Patient": "PATIENT",  "Patient1": "PATIENT",  "Patient2": "PATIENT",
    "Theme": "PATIENT",    "Theme1": "PATIENT",    "Theme2": "PATIENT",
    "Stimulus": "PATIENT", "Topic": "PATIENT",     "Result": "PATIENT",
    "Recipient": "BENEFICIARY", "Beneficiary": "BENEFICIARY",
    "Location": "LOCATION", "Source": "LOCATION",  "Destination": "LOCATION",
    "Initial_Location": "LOCATION", "Trajectory": "LOCATION",
    "Time": "TEMPORAL",    "Duration": "TEMPORAL",
    "Instrument": "INSTRUMENT", "Co-Patient": "PATIENT",
}

# Tier-2 fallback: dependency label → universal slot.
# Only active when VerbNet has no class for a given verb lemma.
_DEP_TO_SLOT = {
    "nsubj": "AGENT", "nsubjpass": "PATIENT",
    "dobj": "PATIENT", "attr": "PATIENT", "oprd": "PATIENT",
}
_PREP_LOCATION = {"at", "in", "near", "outside", "inside", "beside", "by"}
_PREP_TEMPORAL = {"during", "before", "after", "since", "until", "when"}


class VerbNetFrameInducer:
    """
    For each verb lemma:
      Tier 1 — look up VerbNet, map thematic roles to universal slots.
               Covers ~3,626 verbs.  Zero words written by us.
      Tier 2 — generic dependency-based slot assignment.  Never fails.
               Confidence set to 0.55 so callers can filter.

    Honest measured coverage: 57% Tier-1 / 43% Tier-2 on an unselected
    28-verb sample spanning general, medical, legal, and business domains.
    See conversation for the test output.
    """
    def __init__(self):
        self._cache: Dict[str, list] = {}

    def induce(self, verb_lemma: str) -> dict:
        if verb_lemma not in self._cache:
            self._cache[verb_lemma] = vn.classids(verb_lemma)
        classids = self._cache[verb_lemma]

        for class_id in classids:
            mapped = {}
            for r in vn.themroles(class_id):
                slot = _VN_ROLE_TO_SLOT.get(r["type"])
                if slot:
                    mapped[slot] = r["type"]     # keep VN name for traceability
            if mapped:
                return {"frame_id": class_id, "roles": mapped,
                        "tier": 1, "confidence": 0.85}

        # Tier 2
        return {
            "frame_id": f"GENERIC[{verb_lemma}]",
            "roles": {"AGENT": "nsubj", "PATIENT": "dobj",
                      "LOCATION": "pobj+prep_loc", "TEMPORAL": "pobj+prep_temp"},
            "tier": 2, "confidence": 0.55,
        }






class DistantSupervisionModule:
    """
    Finds entity pairs that co-occur in the same sentence, extracts the
    context between them, then clusters contexts via DBSCAN to discover
    relation types with no predefined schema.
    """

    def __init__(self, max_char_gap: int = 150):
        self.max_char_gap = max_char_gap
        self._pairs: List[dict] = []

    def extract_pairs(self, doc, context: ExtractionContext) -> List[dict]:
        pairs = []
        for sent in doc.sents:
            ents = context.entities_in_sentence(sent)
            for i in range(len(ents)):
                for j in range(i + 1, len(ents)):
                    e1, e2 = ents[i], ents[j]
                    gap = e2.start_char - e1.end_char
                    if gap > self.max_char_gap:
                        continue
                    ctx_text = sent.text[
                        e1.end_char - sent.start_char:
                        e2.start_char - sent.start_char
                    ].strip()
                    ctx_doc = nlp(ctx_text) if nlp else None
                    ctx_lemmas = (
                        " ".join(t.lemma_ for t in ctx_doc
                                 if not t.is_stop and not t.is_punct)
                        if ctx_doc else ctx_text
                    )
                    pairs.append({
                        "entity1": context.resolve(e1.text),
                        "type1": e1.label_,
                        "entity2": context.resolve(e2.text),
                        "type2": e2.label_,
                        "context": ctx_lemmas,
                        "sentence": sent.text,
                    })
        self._pairs = pairs
        logger.info(f"[DistantSupervision] {len(pairs)} entity pairs found")
        return pairs

    def discover_relations(self) -> Dict[str, List[dict]]:
        if len(self._pairs) < 3:
            return {}
        by_sig: Dict[str, List] = defaultdict(list)
        for p in self._pairs:
            by_sig[f"{p['type1']}-{p['type2']}"].append(p)

        discovered: Dict[str, List] = {}
        for sig, group in by_sig.items():
            if len(group) < 2:
                continue
            contexts = [g["context"] for g in group if g["context"]]
            if len(set(contexts)) < 2:
                continue
            try:
                vec = TfidfVectorizer(max_features=50, min_df=1)
                X = vec.fit_transform(contexts).toarray()
                labels = DBSCAN(eps=0.35, min_samples=1,
                                metric="cosine").fit_predict(X)
                for cid in set(labels):
                    if cid == -1:
                        continue
                    cluster = [group[i] for i, l in enumerate(labels) if l == cid]
                    words = " ".join(p["context"] for p in cluster).split()
                    name = Counter(words).most_common(1)[0][0].upper() if words else sig
                    discovered[f"REL_{name}"] = cluster
            except Exception:
                continue
        logger.info(f"[DistantSupervision] {len(discovered)} relation types discovered")
        return discovered


class FrameSlottingModule:
    """
    Iterates every VERB token in every sentence.  VerbNetFrameInducer
    maps it to a frame with thematic roles.  Dependency parse assigns
    entities in the sentence to those roles.
    """

    def __init__(self):
        self._inducer = VerbNetFrameInducer()

    def detect_and_fill(self, doc, context: ExtractionContext) -> List[FrameInstance]:
        instances: List[FrameInstance] = []
        for sent in doc.sents:
            ent_by_root = {e.root: e for e in context.entities_in_sentence(sent)}
            for token in sent:
                if token.pos_ != "VERB":
                    continue
                fd = self._inducer.induce(token.lemma_)
                inst = FrameInstance(
                    frame_id=fd["frame_id"], trigger_verb=token.lemma_,
                    tier=fd["tier"], confidence=fd["confidence"],
                    sentence=sent.text,
                )
                for ent_root, ent in ent_by_root.items():
                    role = self._dep_role(ent_root, token, fd)
                    if role:
                        val = context.resolve(ent.text)
                        # slight confidence penalty when value came from coref
                        is_coref = ent.text.lower().strip() != val
                        inst.add_slot(role, val,
                                      fd["confidence"] * (0.9 if is_coref else 1.0))
                if inst.slots:
                    instances.append(inst)
        logger.info(f"[FrameSlotting] {len(instances)} frames filled")
        return instances

    @staticmethod
    def _dep_role(ent_tok, trigger_tok, fd: dict) -> Optional[str]:
        current = ent_tok
        for _ in range(5):
            if current == trigger_tok:
                return "AGENT"
            dep = current.dep_
            if dep in _DEP_TO_SLOT:
                role = _DEP_TO_SLOT[dep]
                return role if fd["tier"] == 2 or role in fd["roles"] else role
            if dep == "pobj":
                prep = current.head.lower_ if current.head else ""
                if prep in _PREP_LOCATION:
                    return "LOCATION"
                if prep in _PREP_TEMPORAL:
                    return "TEMPORAL"
                return "LOCATION"
            current = current.head
            if current == current.head:
                break
        return None


class UnsupervisedRelationDiscovery:
    """
    Builds a distributional context profile for every entity, clusters
    by TF-IDF cosine similarity via DBSCAN, then interprets each cluster
    as a grouping of entities that share a latent relation.
    """

    def __init__(self):
        self._entity_contexts: Dict[str, List[str]] = defaultdict(list)

    def build_profiles(self, doc, context: ExtractionContext):
        for sent in doc.sents:
            ents = context.entities_in_sentence(sent)
            for ent in ents:
                key = context.resolve(ent.text)
                context_words = [
                    t.lemma_ for t in sent
                    if not t.is_stop and not t.is_punct and t != ent.root
                ]
                self._entity_contexts[key].extend(context_words)

    def discover(self, min_cluster: int = 2) -> List[dict]:
        keys = list(self._entity_contexts.keys())
        if len(keys) < 3:
            return []
        feature_vecs = [
            " ".join(self._entity_contexts[k]) or "EMPTY" for k in keys
        ]
        try:
            X = TfidfVectorizer(max_features=50).fit_transform(feature_vecs).toarray()
            labels = DBSCAN(eps=0.4, min_samples=min_cluster,
                            metric="cosine").fit_predict(X)
            patterns = []
            for cid in set(labels):
                if cid == -1:
                    continue
                members = [keys[i] for i, l in enumerate(labels) if l == cid]
                all_words = []
                for m in members:
                    all_words.extend(self._entity_contexts[m])
                sig = [w for w, _ in Counter(all_words).most_common(3)]
                patterns.append({"cluster_id": int(cid),
                                  "entities": members, "signature": sig})
            logger.info(f"[UnsupervisedDiscovery] {len(patterns)} patterns found")
            return patterns
        except Exception:
            return []
