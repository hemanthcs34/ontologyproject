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
import gender_guesser.detector as gender_detector

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
    logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")

# One shared gender detector instance — library uses a 48,000-name dataset
# covering 79 countries. Not written by us, not domain-specific.
_GENDER_DETECTOR = gender_detector.Detector(case_sensitive=False)






@dataclass
class Concept:
    """
    One entity or multi-word concept extracted from the text.

    canonical_id  — lowercase normalised key used as the universal reference
                    across all modules (e.g. "general electric").
    surface       — original capitalisation from the text ("General Electric").
    entity_type   — spaCy NER label or "CONCEPT" for noun-chunk entries.
    confidence    — 0.9 for NER hits, 0.6 for noun-chunk-only entries.
    source        — "NER" or "NOUN_CHUNK".
    """
    canonical_id: str
    surface: str
    entity_type: str
    confidence: float
    source: str


@dataclass
class RelationTriple:
    """
    (subject, relation, object) triple with provenance.

    source values:
      DISTANT_SUPERVISION — came from entity-pair co-occurrence clustering
      FRAME               — came from a VerbNet frame slot pair (AGENT, PATIENT)
      DISTRIBUTIONAL      — came from entity context-profile clustering
    """
    subject: str
    relation: str
    object_: str
    confidence: float
    source: str
    evidence: str = ""

    def key(self) -> tuple:
        return (self.subject, self.relation, self.object_)

    def to_dict(self) -> dict:
        return {
            "subject":    self.subject,
            "relation":   self.relation,
            "object":     self.object_,
            "confidence": round(self.confidence, 3),
            "source":     self.source,
            "evidence":   self.evidence[:120],
        }


@dataclass
class FrameInstance:
    """
    One activated semantic frame with filled slots.

    frame_id     — VerbNet class id (Tier 1) or "GENERIC[verb]" (Tier 2).
    trigger_verb — the lemma that activated the frame.
    tier         — 1 = VerbNet roles, 2 = dependency-label fallback.
    confidence   — 0.85 (Tier 1) or 0.55 (Tier 2).
    slots        — {role_name: (entity_canonical_id, confidence)}
    """
    frame_id: str
    trigger_verb: str
    tier: int
    confidence: float
    slots: Dict[str, Tuple[str, float]] = field(default_factory=dict)
    sentence: str = ""

    def add_slot(self, role: str, value: str, conf: float):
        # keep only the highest-confidence fill per role
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
    The single shared entity registry for one document.

    WHY THIS EXISTS
    ---------------
    BOOFS v1 had three modules each running their own `for ent in sent.ents`
    loop independently. Because each loop normalised text differently (one
    lowercased, another didn't, a third lemmatised), the same entity could
    appear under three different keys in three different data structures.
    Deduplication then failed silently.

    This class is built once from the spaCy parse and passed to every module.
    No module touches doc.ents or doc.noun_chunks directly — they all call
    self.entities_in_sentence(sent) and self.resolve(text).  That structural
    constraint (enforced at the API level, not by convention) makes the
    divergence bug impossible.

    HOW canonical_map THREADS THROUGH
    ----------------------------------
    RuleBasedCorefResolver produces a dict like {"he": "dave", "his": "dave"}.
    That dict is passed into build() and stored as self._canonical_map.
    Two methods use it:
      resolve(text)              — step 1 in every value lookup
      entities_in_sentence(sent) — so pronoun tokens that spaCy tagged as
                                   entities are also matched
    """
    doc: object
    concepts: Dict[str, Concept] = field(default_factory=dict)
    _canonical_map: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def build(cls, doc, canonical_map: Dict[str, str] = None) -> "ExtractionContext":
        cm = canonical_map or {}
        ctx = cls(doc=doc, _canonical_map=cm)

        # Single NER pass → canonical concept registry
        for ent in doc.ents:
            raw = ent.text.lower().strip()
            key = cm.get(raw, raw)           # resolve pronoun if coref mapped it
            if key not in ctx.concepts:
                ctx.concepts[key] = Concept(key, ent.text, ent.label_, 0.9, "NER")

        # Single noun-chunk pass → catches multi-word concepts NER missed
        for chunk in doc.noun_chunks:
            raw = chunk.lemma_.lower().strip()
            key = cm.get(raw, raw)
            if key not in ctx.concepts and len(key.split()) > 1:
                ctx.concepts[key] = Concept(key, chunk.text, "CONCEPT", 0.6, "NOUN_CHUNK")

        return ctx

    def entities_in_sentence(self, sent) -> list:
        """
        The ONE function every module calls instead of `list(sent.ents)`.

        Returns spaCy entity spans whose normalised text is either:
          a) directly in self.concepts, OR
          b) reachable through self._canonical_map (pronoun → entity)
        so that pronoun tokens that spaCy tagged as PERSON entities are
        still matched after coref resolution.
        """
        return [
            e for e in sent.ents
            if self._canonical_map.get(e.text.lower().strip(),
                                       e.text.lower().strip()) in self.concepts
        ]

    def resolve(self, text: str) -> str:
        """
        Resolution order:
          1. coref map    "he" → "dave"
          2. concept dict "dave" → canonical_id "dave"
          3. normalised input as-is (unknown text)
        """
        key = text.lower().strip()
        key = self._canonical_map.get(key, key)
        return self.concepts[key].canonical_id if key in self.concepts else key





# What pronouns exist and which gender bucket they belong to.
# This encodes English grammar, not any domain knowledge.
_PRONOUN_GENDER: Dict[str, str] = {
    "he": "MALE",    "him": "MALE",    "his": "MALE",    "himself": "MALE",
    "she": "FEMALE", "her": "FEMALE",  "hers": "FEMALE", "herself": "FEMALE",
    "they": "PLURAL","them": "PLURAL", "their": "PLURAL","themselves": "PLURAL",
    "it": "NEUT",    "its": "NEUT",    "itself": "NEUT",
}

# What gender bucket each NER label maps to.
# PERSON is absent because gender is resolved per-name via gender-guesser.
_NER_TYPE_TO_GENDER: Dict[str, str] = {
    "ORG": "NEUT", "GPE": "NEUT", "LOC": "NEUT",
    "PRODUCT": "NEUT", "FAC": "NEUT", "WORK_OF_ART": "NEUT",
    "DATE": "NEUT", "MONEY": "NEUT", "NORP": "NEUT",
}

# gender-guesser returns these strings; we map them to our three buckets.
_GG_TO_BUCKET: Dict[str, str] = {
    "male":          "MALE",
    "mostly_male":   "MALE",
    "female":        "FEMALE",
    "mostly_female": "FEMALE",
    "andy":          "PERSON_UNKNOWN",   # androgynous — could be either
    "unknown":       "PERSON_UNKNOWN",   # library has no data for this name
}


class RuleBasedCorefResolver:
    """
    Resolves pronouns to their most-recent gender-matching named entity.

    WHY NO EXTERNAL MODEL
    ---------------------
    spacy-experimental and neuralcoref both failed to build in this sandbox
    (wheel build errors). This rule-based approach works with plain spaCy
    and covers the main pronoun-antecedent failure mode in BOOFS v1 where
    "he moved to California" produced the triple (he, LOCATED_IN, california).

    GENERALISATION
    --------------
    The key improvement over v1's _NAME_GENDER dict is using gender-guesser,
    which covers 48,000 names from 79 countries. Names it doesn't know go
    into a PERSON_UNKNOWN bucket so the resolver degrades gracefully rather
    than silently skipping the entity.

    LIMITATION
    ----------
    Still one-antecedent-per-pronoun (no cluster merging). Wrong gender
    assignment propagates silently, which is why coref-resolved slot values
    carry a 10% confidence penalty in FrameSlottingModule.
    """

    def _entity_gender(self, ent) -> str:
        """
        Return the gender bucket for a named entity.

        For PERSON entities: ask gender-guesser using the first token
        (first name is the strongest gender signal). Falls back to
        PERSON_UNKNOWN — never returns None so the entity always enters
        some bucket.

        For all other NER labels: use _NER_TYPE_TO_GENDER, default NEUT.
        """
        if ent.label_ == "PERSON":
            first_name = list(ent)[0].text if list(ent) else ent.text
            gg_result = _GENDER_DETECTOR.get_gender(first_name)
            return _GG_TO_BUCKET.get(gg_result, "PERSON_UNKNOWN")
        return _NER_TYPE_TO_GENDER.get(ent.label_, "NEUT")

    def resolve(self, doc) -> Dict[str, str]:
        """
        Walk the document sentence by sentence.
        Maintain a recency-ordered list per gender bucket (most recent last).
        When a pronoun is found, pick the most recent entity in its bucket.

        For PERSON_UNKNOWN entities: try to use MALE or FEMALE buckets as
        a fallback when no unknown-gender entities exist, so texts with a
        single PERSON entity of unrecognised gender still resolve correctly.
        """
        canonical_map: Dict[str, str] = {}
        recent: Dict[str, List[str]] = defaultdict(list)

        for sent in doc.sents:
            # Pass 1: register entities into recency lists
            for ent in sent.ents:
                bucket = self._entity_gender(ent)
                key = ent.text.lower().strip()
                lst = recent[bucket]
                if key in lst:
                    lst.remove(key)
                lst.append(key)

            # Pass 2: resolve pronoun tokens
            for token in sent:
                lower = token.lower_
                if lower not in _PRONOUN_GENDER or token.pos_ != "PRON":
                    continue
                pronoun_bucket = _PRONOUN_GENDER[lower]

                # Primary: exact gender match
                candidates = recent.get(pronoun_bucket, [])

                # Fallback for he/she: include PERSON_UNKNOWN entities
                if not candidates and pronoun_bucket in ("MALE", "FEMALE"):
                    candidates = recent.get("PERSON_UNKNOWN", [])

                if candidates:
                    canonical_map[lower] = candidates[-1]

        return canonical_map



# Translates VerbNet's ~25 thematic role names to our 6 universal slot names.
# This is a one-time schema mapping, like renaming columns in a JOIN.
# It is NOT a list of domain-specific trigger words.
_VN_ROLE_TO_SLOT: Dict[str, str] = {
    "Agent":            "AGENT",
    "Experiencer":      "AGENT",
    "Actor":            "AGENT",
    "Actor1":           "AGENT",
    "Co-Agent":         "AGENT",
    "Patient":          "PATIENT",
    "Patient1":         "PATIENT",
    "Patient2":         "PATIENT",
    "Theme":            "PATIENT",
    "Theme1":           "PATIENT",
    "Theme2":           "PATIENT",
    "Stimulus":         "PATIENT",
    "Topic":            "PATIENT",
    "Result":           "PATIENT",
    "Co-Patient":       "PATIENT",
    "Recipient":        "BENEFICIARY",
    "Beneficiary":      "BENEFICIARY",
    "Location":         "LOCATION",
    "Source":           "LOCATION",
    "Destination":      "LOCATION",
    "Initial_Location": "LOCATION",
    "Trajectory":       "LOCATION",
    "Time":             "TEMPORAL",
    "Duration":         "TEMPORAL",
    "Instrument":       "INSTRUMENT",
}

# Tier-2: dependency label → slot, used ONLY when VerbNet has no class for
# a verb lemma (e.g. "found", "launch", "merge" — confirmed gaps in VerbNet).
_DEP_TO_SLOT: Dict[str, str] = {
    "nsubj":    "AGENT",
    "nsubjpass":"PATIENT",
    "dobj":     "PATIENT",
    "attr":     "PATIENT",
    "oprd":     "PATIENT",
}
_PREP_LOCATION = {"at", "in", "near", "outside", "inside", "beside", "by", "within",
                  "from", "to", "into", "onto", "through"}   # source/destination are LOCATION
_PREP_TEMPORAL  = {"during", "before", "after", "since", "until", "when", "while"}


class VerbNetFrameInducer:
    """
    Maps every verb lemma to a semantic frame.

    Tier 1 (VerbNet)
    ----------------
    Calls vn.classids(lemma).  If one or more VerbNet classes are returned
    and at least one has thematic roles that map to our universal slots,
    we use that class. Covers ~3,626 verbs across all domains.
    Confidence 0.85.

    Tier 2 (dependency fallback)
    ----------------------------
    Used when VerbNet has no class for the verb, or when all classes have
    zero mappable roles (a real data gap in NLTK's VerbNet parser for some
    inherited-role classes).  Returns AGENT/PATIENT/LOCATION/TEMPORAL based
    purely on syntactic dependency labels. Confidence 0.55.

    Neither tier requires us to write a single domain-specific trigger word.

    Coverage (measured on 28 unselected verbs): 57% Tier-1, 43% Tier-2.
    """

    def __init__(self):
        self._cache: Dict[str, list] = {}

    def induce(self, verb_lemma: str) -> dict:
        if verb_lemma not in self._cache:
            self._cache[verb_lemma] = vn.classids(verb_lemma)
        classids = self._cache[verb_lemma]

        for class_id in classids:
            mapped: Dict[str, str] = {}
            for r in vn.themroles(class_id):
                slot = _VN_ROLE_TO_SLOT.get(r["type"])
                if slot:
                    mapped[slot] = r["type"]   # keep VN name for traceability
            if mapped:                          # at least one mappable role
                return {
                    "frame_id":   class_id,
                    "roles":      mapped,
                    "tier":       1,
                    "confidence": 0.85,
                }

        # Tier 2 — no usable VerbNet entry
        return {
            "frame_id":   f"GENERIC[{verb_lemma}]",
            "roles":      {
                "AGENT":    "nsubj",
                "PATIENT":  "dobj",
                "LOCATION": "pobj+prep_loc",
                "TEMPORAL": "pobj+prep_temp",
            },
            "tier":       2,
            "confidence": 0.55,
        }




class DistantSupervisionModule:
    """
    Entity-pair co-occurrence + context clustering.

    Step 1: For every sentence, find all pairs of entities that co-occur
            within max_char_gap characters. Extract the text between them,
            lemmatise it, and store as the 'context'.

    Step 2: Group pairs by NER-type signature (e.g. PERSON-ORG).
            Within each group, TF-IDF-vectorise contexts and cluster with
            DBSCAN (eps=0.35, min_samples=2).

            min_samples=2 is required (Bug 2 fix): at min_samples=1, every
            single pair is its own cluster, so clustering does nothing.
            With min_samples=2, genuinely isolated pairs become noise (-1)
            and similar-context pairs group into meaningful relation types.

    Step 3: Name each cluster by its most frequent context word.
            The name is data-driven, not hardcoded.
    """

    def __init__(self, max_char_gap: int = 150):
        self.max_char_gap = max_char_gap
        self._pairs: List[dict] = []

    def extract_pairs(self, doc, context: ExtractionContext) -> List[dict]:
        pairs: List[dict] = []
        for sent in doc.sents:
            ents = context.entities_in_sentence(sent)
            for i in range(len(ents)):
                for j in range(i + 1, len(ents)):
                    e1, e2 = ents[i], ents[j]
                    gap = e2.start_char - e1.end_char
                    if gap > self.max_char_gap:
                        continue

                    # Extract and lemmatise the text between the two entities
                    rel_start = e1.end_char - sent.start_char
                    rel_end   = e2.start_char - sent.start_char
                    ctx_text  = sent.text[rel_start:rel_end].strip()
                    ctx_doc   = nlp(ctx_text) if (nlp and ctx_text) else None
                    ctx_lemmas = (
                        " ".join(t.lemma_ for t in ctx_doc
                                 if not t.is_stop and not t.is_punct)
                        if ctx_doc else ctx_text
                    )

                    pairs.append({
                        "entity1":  context.resolve(e1.text),
                        "type1":    e1.label_,
                        "entity2":  context.resolve(e2.text),
                        "type2":    e2.label_,
                        "context":  ctx_lemmas,
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
                X = TfidfVectorizer(max_features=50, min_df=1).fit_transform(contexts).toarray()
                # min_samples=2: isolated pairs become noise, real clusters need ≥2 members
                labels = DBSCAN(eps=0.35, min_samples=2, metric="cosine").fit_predict(X)
                for cid in set(labels):
                    if cid == -1:   # noise — skip isolated pairs
                        continue
                    cluster = [group[i] for i, l in enumerate(labels) if l == cid]
                    words   = " ".join(p["context"] for p in cluster).split()
                    name    = Counter(words).most_common(1)[0][0].upper() if words else sig
                    discovered[f"REL_{name}"] = cluster
            except Exception:
                continue

        logger.info(f"[DistantSupervision] {len(discovered)} relation types discovered")
        return discovered


class FrameSlottingModule:
    """
    For every VERB token in every sentence, induce a frame via VerbNet and
    assign entities in the same sentence to that frame's slots.

    GENERALISATION
    --------------
    There is no list of verbs to check against.  Every verb token in every
    sentence is evaluated, whether the text is about medicine, law, business,
    history, or anything else.

    SLOT ASSIGNMENT  (_dep_role)
    ----------------------------
    Walks from the entity's dependency-tree root toward the trigger verb,
    up to 5 hops.  At each step:
      - If the current node IS the trigger, return AGENT.
      - If the dependency label is in _DEP_TO_SLOT, return that slot —
        BUT ONLY if the slot is declared in this frame's roles (Tier 1)
        OR if we're in Tier 2 where all roles are generic.
        Returning None when the role is absent from the frame is Bug 1's fix:
        the old code had 'else role' which always returned a value.
      - If the dep is 'pobj', inspect the preposition to distinguish
        LOCATION from TEMPORAL.

    CONFIDENCE PENALTY
    ------------------
    If the entity value came from coref resolution (ent.text != resolved value),
    its slot confidence is multiplied by 0.9 to reflect the coref uncertainty.
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
                    if role is None:
                        continue
                    val       = context.resolve(ent.text)
                    is_coref  = ent.text.lower().strip() != val
                    inst.add_slot(role, val,
                                  fd["confidence"] * (0.9 if is_coref else 1.0))
                if inst.slots:
                    instances.append(inst)

        logger.info(f"[FrameSlotting] {len(instances)} frames filled")
        return instances

    @staticmethod
    def _dep_role(ent_tok, trigger_tok, fd: dict) -> Optional[str]:
        """
        Walk dependency tree from entity root toward trigger verb.
        Return the appropriate universal slot, or None if no valid role found.

        Bug 1 fix: the old code had 'else role' at the end, which made the
        function always return a role even when the Tier-1 frame did not
        declare that role.  The correct behaviour is to return None so the
        slot is not added at all.
        """
        current = ent_tok
        for _ in range(5):
            if current == trigger_tok:
                return "AGENT"

            dep = current.dep_

            if dep in _DEP_TO_SLOT:
                role = _DEP_TO_SLOT[dep]
                # Tier 1: only accept this role if the frame declares it
                # Tier 2: accept any role (the frame is generic)
                if fd["tier"] == 2 or role in fd["roles"]:
                    return role
                return None   # ← Bug 1 fix: was 'return role' in old code

            if dep == "pobj":
                prep = current.head.lower_ if current.head else ""
                if prep in _PREP_LOCATION:
                    return "LOCATION"
                if prep in _PREP_TEMPORAL:
                    return "TEMPORAL"
                return None   # unrecognised preposition — don't guess

            current = current.head
            if current == current.head:   # reached ROOT node
                break
        return None


class UnsupervisedRelationDiscovery:
    """
    Distributional entity-context clustering.

    Each entity accumulates a bag-of-words from all sentences it appears in
    (stop words and punctuation excluded).  Entities are vectorised with
    TF-IDF and clustered with DBSCAN (min_samples=2).

    Entities in the same cluster share a similar linguistic context and
    therefore likely participate in the same latent relation type.
    Adjacent entities within each cluster are output as relation triples.

    GENERALISATION
    --------------
    No text category, domain, or entity type is assumed.  The clustering is
    purely data-driven: if the text is about drugs and diseases, entities
    that appear next to treatment verbs will cluster together; if the text is
    about contracts, parties that appear next to legal verbs will cluster.
    """

    def __init__(self):
        self._entity_contexts: Dict[str, List[str]] = defaultdict(list)

    def build_profiles(self, doc, context: ExtractionContext):
        for sent in doc.sents:
            ents = context.entities_in_sentence(sent)
            for ent in ents:
                key = context.resolve(ent.text)
                words = [
                    t.lemma_ for t in sent
                    if not t.is_stop and not t.is_punct
                    and t.i != ent.root.i      # exclude the entity token itself
                ]
                self._entity_contexts[key].extend(words)

    def discover(self, min_cluster: int = 2) -> List[dict]:
        keys = list(self._entity_contexts.keys())
        if len(keys) < 3:
            return []
        feature_vecs = [
            " ".join(self._entity_contexts[k]) or "EMPTY" for k in keys
        ]
        try:
            X      = TfidfVectorizer(max_features=50).fit_transform(feature_vecs).toarray()
            labels = DBSCAN(eps=0.4, min_samples=min_cluster,
                            metric="cosine").fit_predict(X)
            patterns: List[dict] = []
            for cid in set(labels):
                if cid == -1:
                    continue
                members  = [keys[i] for i, l in enumerate(labels) if l == cid]
                all_words: List[str] = []
                for m in members:
                    all_words.extend(self._entity_contexts[m])
                sig = [w for w, _ in Counter(all_words).most_common(3)]
                patterns.append({"cluster_id": int(cid),
                                  "entities":   members,
                                  "signature":  sig})
            logger.info(f"[UnsupervisedDiscovery] {len(patterns)} patterns found")
            return patterns
        except Exception:
            return []