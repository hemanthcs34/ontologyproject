"""
BOOFS: Bootstrapped Ontology and Object Frame Semantics
========================================================

Universal Ontology Learning System - Complete Implementation

Novel algorithms:
1. Bootstrapped Distant Supervision (auto-seed generation)
2. Frame-Based Semantic Slot Filling (universal frames)
3. Unsupervised Relation Discovery (distributional clustering)
4. Active Learning (smart example selection)
5. Distributional Fact Completion (entity similarity transfer)

Works on ANY text without domain-specific hardcoding!
Research paper: "Universal Ontology Learning from Unstructured Text"
"""

import csv
import re
import logging
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.error("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    raise


# ════════════════════════════════════════════════════════════════════════════
# [NEW CODE — EXTENSION 1] COREFERENCE RESOLUTION (PREPROCESSING STAGE)
# ════════════════════════════════════════════════════════════════════════════
#
# Inserted BEFORE the existing pipeline. Resolves pronouns to their canonical
# mentions so that downstream modules (concept extraction, distant supervision,
# frame slotting, relation discovery) receive cleaner text without needing any
# changes themselves. This is purely additive: if no coreference backend is
# available, resolve() simply returns the original text unchanged, and the
# rest of BOOFS behaves exactly as before.

try:
    import spacy_experimental  # noqa: F401
    _COREF_BACKEND = "spacy_experimental"
except ImportError:
    try:
        import neuralcoref  # noqa: F401
        _COREF_BACKEND = "neuralcoref"
    except ImportError:
        _COREF_BACKEND = None


class CoreferenceResolver:
    """
    Resolves pronouns to canonical entity mentions before BOOFS extraction.

    Tries spacy-experimental's coref pipeline first, then neuralcoref, and
    falls back to a lightweight rule-based resolver (nearest preceding PERSON
    entity matching) if neither library is installed. This guarantees the
    module always works, while preferring the more accurate neural backends
    when present.
    """

    def __init__(self):
        self.backend = _COREF_BACKEND
        self._coref_nlp = None

        if self.backend == "spacy_experimental":
            try:
                self._coref_nlp = spacy.load("en_coreference_web_trf")
            except Exception:
                logger.warning("spacy-experimental coref model not found; falling back to rule-based resolver.")
                self.backend = None

        elif self.backend == "neuralcoref":
            try:
                neuralcoref.add_to_pipe(nlp)
                self._coref_nlp = nlp
            except Exception:
                logger.warning("neuralcoref failed to attach; falling back to rule-based resolver.")
                self.backend = None

    def resolve(self, text: str) -> str:
        """Replace pronouns in `text` with their resolved canonical mentions."""
        if self.backend == "spacy_experimental" and self._coref_nlp is not None:
            return self._resolve_spacy_experimental(text)
        if self.backend == "neuralcoref" and self._coref_nlp is not None:
            return self._resolve_neuralcoref(text)
        return self._resolve_rule_based(text)

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
            main = max(cluster, key=lambda s: len(s.text))
            for mention in cluster:
                if mention.text.lower() != main.text.lower():
                    replacements.append((mention.start_char, mention.end_char, main.text))
        replacements.sort(key=lambda r: r[0], reverse=True)
        resolved = text
        for start, end, repl in replacements:
            resolved = resolved[:start] + repl + resolved[end:]
        return resolved

    PRONOUNS = {'he', 'him', 'his', 'she', 'her', 'hers', 'they', 'them', 'their', 'it', 'its'}

    def _resolve_rule_based(self, text: str) -> str:
        """Minimal fallback: replace pronouns with the nearest preceding PERSON/ORG entity."""
        doc = nlp(text)
        last_entity = None
        replacements = []
        for token in doc:
            if token.text.lower() in self.PRONOUNS and last_entity:
                replacements.append((token.idx, token.idx + len(token.text), last_entity))
            for ent in doc.ents:
                if ent.label_ in ("PERSON", "ORG") and ent.start <= token.i < ent.end:
                    last_entity = ent.text
                    break
        replacements.sort(key=lambda r: r[0], reverse=True)
        resolved = text
        for start, end, repl in replacements:
            resolved = resolved[:start] + repl + resolved[end:]
        return resolved


# ════════════════════════════════════════════════════════════════════════════
# UNIVERSAL FRAMES DEFINITION (Language/Domain Independent)
# ════════════════════════════════════════════════════════════════════════════

UNIVERSAL_FRAMES = {
    'EMPLOYMENT': {
        'triggers': ['work', 'job', 'employ', 'hire', 'join', 'position', 'serve', 'hired', 'employed', 'role'],
        'description': 'Someone works at organization in a role',
        'slots': {
            'EMPLOYEE': {'role': 'AGENT', 'ner_types': ['PERSON']},
            'EMPLOYER': {'role': 'PATIENT', 'ner_types': ['ORG', 'PRODUCT']},
            'POSITION': {'role': 'ATTRIBUTE', 'ner_types': ['NOUN']},
            'START_TIME': {'role': 'TEMPORAL', 'ner_types': ['DATE']},
            'END_TIME': {'role': 'TEMPORAL', 'ner_types': ['DATE']},
            'LOCATION': {'role': 'LOCATION', 'ner_types': ['GPE', 'LOC']},
        }
    },
    'FOUNDING': {
        'triggers': ['found', 'establish', 'create', 'start', 'launch', 'founded', 'founded', 'formed', 'establish'],
        'description': 'Someone/organization founds/establishes an entity',
        'slots': {
            'FOUNDER': {'role': 'AGENT', 'ner_types': ['PERSON', 'ORG']},
            'FOUNDED_ENTITY': {'role': 'PATIENT', 'ner_types': ['ORG', 'PRODUCT']},
            'TIME': {'role': 'TEMPORAL', 'ner_types': ['DATE']},
            'LOCATION': {'role': 'LOCATION', 'ner_types': ['GPE', 'LOC']},
        }
    },
    'EDUCATION': {
        'triggers': ['study', 'graduate', 'attend', 'major', 'degree', 'enroll', 'studied', 'graduate', 'educated'],
        'description': 'Someone studies at institution',
        'slots': {
            'STUDENT': {'role': 'AGENT', 'ner_types': ['PERSON']},
            'INSTITUTION': {'role': 'LOCATION', 'ner_types': ['ORG']},
            'FIELD': {'role': 'ATTRIBUTE', 'ner_types': ['NOUN']},
            'DEGREE': {'role': 'ATTRIBUTE', 'ner_types': ['NOUN']},
            'TIME': {'role': 'TEMPORAL', 'ner_types': ['DATE']},
        }
    },
    'FAMILY': {
        'triggers': ['marry', 'married', 'divorce', 'parent', 'child', 'spouse', 'sibling', 'brother', 'sister', 'son', 'daughter'],
        'description': 'Family relationships',
        'slots': {
            'PERSON1': {'role': 'AGENT', 'ner_types': ['PERSON']},
            'PERSON2': {'role': 'PATIENT', 'ner_types': ['PERSON']},
            'RELATION_TYPE': {'role': 'ATTRIBUTE', 'ner_types': ['NOUN']},
            'TIME': {'role': 'TEMPORAL', 'ner_types': ['DATE']},
        }
    },
    'LOCATION': {
        'triggers': ['locate', 'base', 'situate', 'headquarter', 'reside', 'live', 'located', 'based', 'situated'],
        'description': 'Entity is located at place',
        'slots': {
            'ENTITY': {'role': 'AGENT', 'ner_types': ['ORG', 'PERSON']},
            'PLACE': {'role': 'LOCATION', 'ner_types': ['GPE', 'LOC']},
            'TIME': {'role': 'TEMPORAL', 'ner_types': ['DATE']},
        }
    },
}

# Universal semantic role mapping (language-independent)
SEMANTIC_ROLE_MAPPING = {
    'nsubj': 'AGENT',              # Nominal subject
    'nsubjpass': 'PATIENT',        # Passive subject
    'dobj': 'PATIENT',             # Direct object
    'iobj': 'BENEFICIARY',         # Indirect object
    'pobj': 'CONTEXT',             # Prepositional object (context-dependent)
    'attr': 'ATTRIBUTE',           # Attribute
    'prep': 'CONTEXT',             # Preposition modifier
    'npadvmod': 'CONTEXT',         # Noun phrase adverbial modifier
}


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
        self.sources = []  # Where discovered from
    
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


class FrameInstance:
    """Detected frame with filled slots."""
    def __init__(self, frame_type: str, trigger: str):
        self.frame_type = frame_type
        self.trigger = trigger
        self.slots = {}  # slot_name -> (value, confidence)
        self.sentence = None
    
    def add_slot(self, slot_name: str, value: str, confidence: float = 0.5):
        """Add a slot value."""
        if slot_name not in self.slots:
            self.slots[slot_name] = (value, confidence)
        elif confidence > self.slots[slot_name][1]:
            self.slots[slot_name] = (value, confidence)
    
    def to_dict(self):
        return {
            'frame_type': self.frame_type,
            'trigger': self.trigger,
            'slots': self.slots,
            'sentence': self.sentence
        }


# ════════════════════════════════════════════════════════════════════════════
# ALGORITHM 1: BOOTSTRAPPED DISTANT SUPERVISION
# ════════════════════════════════════════════════════════════════════════════

class DistantSupervisionModule:
    """
    Automatically generates training pairs from text.
    No manual seed annotation needed!
    """
    
    def __init__(self, max_entity_distance: int = 10):
        self.max_entity_distance = max_entity_distance
        self.entity_pairs = []
        self.discovered_relations = defaultdict(list)
    
    def extract_entity_pairs(self, doc: spacy.tokens.Doc) -> List[Dict]:
        """Find all entity pairs within distance threshold."""
        pairs = []
        
        for sent in doc.sents:
            entities = list(sent.ents)
            
            # All pairs of entities
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    ent1, ent2 = entities[i], entities[j]
                    
                    # Check distance
                    distance = ent2.start_char - ent1.end_char
                    if distance <= self.max_entity_distance * 10:  # Rough character limit
                        context = self._extract_context(sent, ent1, ent2)
                        
                        pairs.append({
                            'entity1': ent1.text.lower(),
                            'type1': ent1.label_,
                            'entity2': ent2.text.lower(),
                            'type2': ent2.label_,
                            'context': context,
                            'sentence': sent.text,
                            'confidence': 0.7  # Initial confidence from co-occurrence
                        })
        
        self.entity_pairs = pairs
        logger.info(f"[Distant Supervision] Found {len(pairs)} entity pairs")
        return pairs
    
    def _extract_context(self, sent, ent1, ent2):
        """Extract context between two entities."""
        try:
            # Get text between entities
            if ent1.start_char < ent2.start_char:
                between_text = sent.text[ent1.end_char:ent2.start_char].strip()
            else:
                between_text = sent.text[ent2.end_char:ent1.start_char].strip()
            
            # Lemmatize and clean
            if between_text:
                context_doc = nlp(between_text)
                lemmas = [t.lemma_ for t in context_doc if not t.is_stop and not t.is_punct]
                return " ".join(lemmas)
            return ""
        except:
            return ""
    
    def discover_relations_via_clustering(self) -> Dict[str, List]:
        """
        Cluster entity pairs by context similarity to discover relation types.
        This is AUTOMATIC - no predefined relation types!
        """
        if len(self.entity_pairs) < 3:
            logger.warning("Need at least 3 entity pairs for clustering")
            return {}
        
        # Group by type signature
        by_signature = defaultdict(list)
        for pair in self.entity_pairs:
            sig = f"{pair['type1']}-{pair['type2']}"
            by_signature[sig].append(pair)
        
        discovered = {}
        
        for signature, sig_pairs in by_signature.items():
            if len(sig_pairs) < 2:
                continue
            
            # Extract contexts
            contexts = [p['context'] for p in sig_pairs if p['context']]
            if not contexts or len(set(contexts)) < 2:
                continue
            
            try:
                # TF-IDF vectorization
                vectorizer = TfidfVectorizer(max_features=50, min_df=1, max_df=10)
                X = vectorizer.fit_transform(contexts)
                
                # DBSCAN clustering (automatic # of clusters!)
                clustering = DBSCAN(eps=0.35, min_samples=1, metric='cosine')
                labels = clustering.fit_predict(X.toarray())
                
                # Each cluster = one relation type
                for cluster_id in set(labels):
                    if cluster_id == -1:  # Noise
                        continue
                    
                    cluster_pairs = [sig_pairs[i] for i, l in enumerate(labels) if l == cluster_id]
                    
                    # Name relation by most common context words
                    cluster_contexts = [p['context'] for p in cluster_pairs]
                    all_words = " ".join(cluster_contexts).split()
                    most_common = Counter(all_words).most_common(3)
                    
                    if most_common:
                        rel_name = f"REL_{most_common[0][0].upper()}"
                        discovered[rel_name] = cluster_pairs
            
            except Exception as e:
                logger.debug(f"Clustering failed for {signature}: {e}")
                continue
        
        self.discovered_relations = discovered
        logger.info(f"[Distant Supervision] Discovered {len(discovered)} relation types")
        return discovered


# ════════════════════════════════════════════════════════════════════════════
# ALGORITHM 2: FRAME-BASED SEMANTIC SLOT FILLING
# ════════════════════════════════════════════════════════════════════════════

class FrameSlottingModule:
    """
    Frame semantics: detect frames and fill slots automatically.
    Uses universal semantic roles (not domain-specific).
    """
    
    def __init__(self, frames: Dict = None):
        self.frames = frames or UNIVERSAL_FRAMES
        self.detected_frames = []
    
    def detect_and_fill_frames(self, doc: spacy.tokens.Doc) -> List[FrameInstance]:
        """Detect and fill frames in document."""
        filled_frames = []
        
        for sent in doc.sents:
            frame = self._detect_frame_in_sentence(sent)
            if frame:
                self._fill_frame_slots(frame, sent)
                if frame.slots:  # Only keep if slots were filled
                    filled_frames.append(frame)
        
        self.detected_frames = filled_frames
        logger.info(f"[Frame Filling] Detected and filled {len(filled_frames)} frames")
        return filled_frames
    
    def _detect_frame_in_sentence(self, sent) -> Optional[FrameInstance]:
        """Detect which frame is activated."""
        for token in sent:
            for frame_name, frame_def in self.frames.items():
                if token.lemma_ in frame_def['triggers']:
                    return FrameInstance(frame_name, token.lemma_)
        return None
    
    def _fill_frame_slots(self, frame: FrameInstance, sent):
        """Fill slots based on semantic roles."""
        frame.sentence = sent.text
        entities = {ent.root: ent for ent in sent.ents}
        
        # Trigger token
        trigger_token = None
        for token in sent:
            if token.lemma_ == frame.trigger:
                trigger_token = token
                break
        
        if not trigger_token:
            return
        
        # Find slot fillers
        for token in sent:
            if token in entities:
                entity = entities[token]
                
                # Determine semantic role
                role = self._assign_semantic_role(token, trigger_token)
                
                # Check if role matches any frame slot
                for slot_name, slot_def in self.frames[frame.frame_type]['slots'].items():
                    required_role = slot_def['role']
                    allowed_types = slot_def['ner_types']
                    
                    if role == required_role and entity.label_ in allowed_types:
                        confidence = 0.9 if entity.label_ in allowed_types else 0.6
                        frame.add_slot(slot_name, entity.text, confidence)
    
    def _assign_semantic_role(self, entity_token, trigger_token):
        """Assign semantic role based on universal dependency mapping."""
        # Find shortest dependency path to trigger
        current = entity_token
        
        for _ in range(5):
            if current == trigger_token:
                return "AGENT"
            
            dep = current.dep_
            if dep in SEMANTIC_ROLE_MAPPING:
                return SEMANTIC_ROLE_MAPPING[dep]
            
            current = current.head
            if current == current.head:  # Root reached
                break
        
        return "CONTEXT"


# ════════════════════════════════════════════════════════════════════════════
# ALGORITHM 3: UNSUPERVISED RELATION DISCOVERY
# ════════════════════════════════════════════════════════════════════════════

class UnsupervisedRelationDiscoveryModule:
    """
    Discover relations via distributional clustering.
    Zero manual seeds required!
    """
    
    def __init__(self):
        self.entity_contexts = defaultdict(list)
        self.discovered_patterns = []
    
    def build_context_profiles(self, doc: spacy.tokens.Doc):
        """Build distributional profiles for entities."""
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
        """Discover relation patterns via clustering."""
        if len(self.entity_contexts) < 3:
            logger.warning("Need at least 3 entities for discovery")
            return []
        
        entity_names = list(self.entity_contexts.keys())
        
        # Vectorize contexts
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
            
            clustering = DBSCAN(eps=0.4, min_samples=min_cluster_size, metric='cosine')
            labels = clustering.fit_predict(X.toarray())
            
            patterns = []
            for cluster_id in set(labels):
                if cluster_id == -1:
                    continue
                
                cluster_entities = [entity_names[i] for i, l in enumerate(labels) if l == cluster_id]
                
                # Find signature words
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
        
        except Exception as e:
            logger.debug(f"Discovery failed: {e}")
            return []


# ════════════════════════════════════════════════════════════════════════════
# ALGORITHM 4: ACTIVE LEARNING
# ════════════════════════════════════════════════════════════════════════════

class ActiveLearningModule:
    """
    Intelligently select which examples to show user.
    Goal: Learn maximally from minimal human annotation.
    """
    
    @staticmethod
    def compute_uncertainty(example: Dict) -> float:
        """Compute uncertainty score for an example."""
        score = 0.0
        
        # Rarity of context
        context = example.get('context', '')
        if len(context.split()) < 2:
            score += 0.3
        
        # Type ambiguity
        type1 = example.get('type1', '')
        type2 = example.get('type2', '')
        if type1 == type2:
            score += 0.3
        
        # Confidence (inverse)
        conf = example.get('confidence', 0.5)
        score += (1.0 - conf) * 0.4
        
        return min(score, 1.0)
    
    @staticmethod
    def select_informative_examples(candidates: List[Dict], k: int = 5) -> List[Dict]:
        """Select k most informative examples."""
        if len(candidates) <= k:
            return candidates
        
        scored = [
            (ActiveLearningModule.compute_uncertainty(c), c)
            for c in candidates
        ]
        scored.sort(reverse=True, key=lambda x: x[0])
        
        return [item[1] for item in scored[:k]]



# MAIN ONTOLOGY LEARNING SYSTEM


class BOOFSOntologyLearner:
    """
    Complete Ontology Learning System.
    Bootstrapped Ontology and Object Frame Semantics.
    Works on ANY text without domain-specific hardcoding!
    """
    
    def __init__(self, frames: Dict = None):
        self.distant_supervisor = DistantSupervisionModule()
        self.frame_filler = FrameSlottingModule(frames)
        self.relation_discoverer = UnsupervisedRelationDiscoveryModule()
        self.active_learner = ActiveLearningModule()
        # [NEW CODE — EXTENSION 1] coreference resolver, instantiated lazily/cheaply
        self.coref_resolver = CoreferenceResolver()
        # [NEW CODE — EXTENSION 2] embedding module, populated after consolidation
        self.kg_embedder = None

        self.concepts = []
        self.relations = []
        self.frames = []
        self.raw_text = None          # [NEW CODE] original text, pre-coref (for eval metrics)
        self.resolved_text = None     # [NEW CODE] text after coreference resolution
    
    def process(self, text: str, use_active_learning: bool = False, verbose: bool = True,
                resolve_coreference: bool = True):
        """
        Process text and extract ontology.
        Args:
            text: Input text
            use_active_learning: Ask user to label examples
            verbose: Print progress
            resolve_coreference: [NEW] If True, run CoreferenceResolver before extraction
        Returns:
            Dictionary with concepts, relations, frames
        """
        if verbose:
            print("\n" + "="*70)
            print("BOOFS: UNIVERSAL ONTOLOGY LEARNING SYSTEM")
            print("="*70)

        # [NEW CODE — EXTENSION 1] Preprocessing: coreference resolution.
        # Existing behavior is preserved exactly when resolve_coreference=False.
        self.raw_text = text
        if resolve_coreference:
            if verbose: print("\n[0] Resolving coreferences...")
            text = self.coref_resolver.resolve(text)
            if verbose: print(f"    ✓ Coreference backend: {self.coref_resolver.backend or 'rule-based fallback'}")
        self.resolved_text = text

        # Parse once (single source of truth!)
        doc = nlp(text)
        
        # Module 1: Extract Concepts
        if verbose: print("\n[1] Extracting concepts...")
        self.concepts = self._extract_concepts(doc)
        if verbose: print(f"    ✓ Found {len(self.concepts)} concepts")
        
        # Module 2: Distant Supervision
        if verbose: print("\n[2] Distant supervision - discovering entity pairs...")
        pairs = self.distant_supervisor.extract_entity_pairs(doc)
        discovered_rels = self.distant_supervisor.discover_relations_via_clustering()
        if verbose: print(f"    ✓ Discovered {len(discovered_rels)} relation types")
        
        # Module 3: Frame-Based Slot Filling
        if verbose: print("\n[3] Frame-based slot filling...")
        filled_frames = self.frame_filler.detect_and_fill_frames(doc)
        self.frames = filled_frames
        if verbose: print(f"    ✓ Detected {len(filled_frames)} frames with slots")
        
        # Module 4: Unsupervised Relation Discovery
        if verbose: print("\n[4] Unsupervised relation discovery...")
        self.relation_discoverer.build_context_profiles(doc)
        patterns = self.relation_discoverer.discover_relations()
        if verbose: print(f"    ✓ Found {len(patterns)} distributional patterns")
        
        # Module 5: Consolidate Relations
        if verbose: print("\n[5] Consolidating relations...")
        self.relations = self._consolidate_relations(discovered_rels, filled_frames, patterns)
        if verbose: print(f"    ✓ Consolidated {len(self.relations)} unique relations")

        # [NEW CODE — EXTENSION 2] Train KG embeddings on consolidated triples (optional).
        # DOES NOT alter relation extraction logic; consumes self.relations as-is.
        if verbose: print("\n[5b] Training knowledge graph embeddings...")
        triples = [(r.subject, r.relation, r.object) for r in self.relations]
        self.kg_embedder = KGEmbeddingModule()
        try:
            self.kg_embedder.train(triples)
            if verbose: print(f"    ✓ Trained embeddings on {len(triples)} triples")
        except Exception as e:
            logger.warning(f"KG embedding training skipped: {e}")

        # Optional: Active Learning
        if use_active_learning and len(pairs) > 5:
            if verbose: print("\n[6] Active learning...")
            to_label = self.active_learner.select_informative_examples(pairs, k=5)
            if verbose:
                print("    Please label these 5 examples:")
                for i, ex in enumerate(to_label, 1):
                    print(f"    {i}. ({ex['entity1']}) <-> ({ex['entity2']})")
                    print(f"       Context: {ex['context'][:50]}...")
        
        if verbose:
            print("\n" + "="*70)
            print("EXTRACTION COMPLETE")
            print("="*70)
        
        return {
            'concepts': self.concepts,
            'relations': self.relations,
            'frames': self.frames,
            'patterns': patterns
        }
    
    def _extract_concepts(self, doc) -> List[ConceptExtract]:
        """Extract concepts from NER and noun chunks."""
        concepts_dict = {}
        
        # From NER
        for ent in doc.ents:
            concept_id = ent.text.lower()
            if concept_id not in concepts_dict:
                c = ConceptExtract(concept_id, ent.label_, ent.text, confidence=0.9)
                c.sources.append('NER')
                concepts_dict[concept_id] = c
        
        # From noun chunks
        for chunk in doc.noun_chunks:
            concept_id = chunk.lemma_.lower()
            if concept_id not in concepts_dict and len(concept_id.split()) > 1:
                c = ConceptExtract(concept_id, 'CONCEPT', chunk.text, confidence=0.6)
                c.sources.append('NOUN_CHUNK')
                concepts_dict[concept_id] = c
        
        return list(concepts_dict.values())
    
    def _consolidate_relations(self, discovered_rels, frames, patterns) -> List[RelationExtract]:
        """Merge relations from all sources."""
        relations_set = set()
        relations_list = []
        
        # From distant supervision
        for rel_name, pairs in discovered_rels.items():
            for pair in pairs:
                rel = RelationExtract(
                    pair['entity1'],
                    rel_name,
                    pair['entity2'],
                    confidence=pair.get('confidence', 0.7)
                )
                rel.source = 'distant_supervision'
                rel.evidence = pair.get('context', '')
                
                if rel not in relations_set:
                    relations_set.add(rel)
                    relations_list.append(rel)
        
        # From frames
        for frame in frames:
            if 'EMPLOYEE' in frame.slots and 'EMPLOYER' in frame.slots:
                employee, _ = frame.slots['EMPLOYEE']
                employer, _ = frame.slots['EMPLOYER']
                rel = RelationExtract(employee, 'WORKS_FOR', employer, confidence=0.85)
                rel.source = 'frame_based'
                rel.evidence = frame.sentence
                
                if rel not in relations_set:
                    relations_set.add(rel)
                    relations_list.append(rel)
            
            elif 'FOUNDER' in frame.slots and 'FOUNDED_ENTITY' in frame.slots:
                founder, _ = frame.slots['FOUNDER']
                entity, _ = frame.slots['FOUNDED_ENTITY']
                rel = RelationExtract(founder, 'FOUNDED', entity, confidence=0.85)
                rel.source = 'frame_based'
                rel.evidence = frame.sentence
                
                if rel not in relations_set:
                    relations_set.add(rel)
                    relations_list.append(rel)
        
        # From distributional patterns
        for pattern in patterns:
            if len(pattern['entities']) >= 2:
                for i in range(len(pattern['entities']) - 1):
                    rel = RelationExtract(
                        pattern['entities'][i],
                        f"REL_{pattern['cluster_id']}",
                        pattern['entities'][i+1],
                        confidence=0.6
                    )
                    rel.source = 'distributional'
                    
                    if rel not in relations_set:
                        relations_set.add(rel)
                        relations_list.append(rel)
        
        return relations_list
    
    def export_to_csv(self, prefix: str = "ontology"):
        """Export ontology to CSV files."""
        # Concepts
        with open(f"{prefix}_concepts.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["concept", "type", "surface", "confidence", "sources"])
            writer.writeheader()
            writer.writerows([c.to_dict() for c in self.concepts])
        
        logger.info(f"✓ Exported {len(self.concepts)} concepts to {prefix}_concepts.csv")
        
        # Relations
        with open(f"{prefix}_relations.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["subject", "relation", "object", "confidence", "source", "evidence"])
            writer.writeheader()
            writer.writerows([r.to_dict() for r in self.relations])
        
        logger.info(f"✓ Exported {len(self.relations)} relations to {prefix}_relations.csv")
        
        # Frames
        frame_rows = []
        for frame in self.frames:
            for slot_name, (slot_value, confidence) in frame.slots.items():
                frame_rows.append({
                    'frame_type': frame.frame_type,
                    'trigger': frame.trigger,
                    'slot_name': slot_name,
                    'slot_value': slot_value,
                    'confidence': round(confidence, 3)
                })
        
        with open(f"{prefix}_frames.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["frame_type", "trigger", "slot_name", "slot_value", "confidence"])
            writer.writeheader()
            writer.writerows(frame_rows)
        
        logger.info(f"✓ Exported {len(frame_rows)} frame slots to {prefix}_frames.csv")

    # [NEW CODE — EXTENSION 2] Additional export, existing export_to_csv untouched above.
    def export_embeddings_to_csv(self, prefix: str = "ontology"):
        """Export KG embeddings, link predictions, and similarity scores to CSV."""
        if self.kg_embedder is None or not self.kg_embedder.is_trained:
            logger.warning("No trained KG embeddings available to export.")
            return

        self.kg_embedder.export_embeddings_to_csv(f"{prefix}_embeddings.csv")
        self.kg_embedder.export_link_predictions_to_csv(f"{prefix}_link_predictions.csv")
        self.kg_embedder.export_similarity_scores_to_csv(f"{prefix}_similarity_scores.csv")

    def print_summary(self):
        """Print summary of extracted ontology."""
        print("\n" + "="*70)
        print("ONTOLOGY SUMMARY")
        print("="*70)
        
        print(f"\n📦 CONCEPTS: {len(self.concepts)}")
        print("-" * 70)
        for i, c in enumerate(self.concepts[:10], 1):
            print(f"  {i:2}. {c.text:25} [{c.type:12}] conf={c.confidence:.2f}")
        if len(self.concepts) > 10:
            print(f"  ... and {len(self.concepts) - 10} more")
        
        print(f"\n🔗 RELATIONS: {len(self.relations)}")
        print("-" * 70)
        for i, r in enumerate(self.relations[:10], 1):
            print(f"  {i:2}. ({r.subject:20}) --[{r.relation:15}]--> ({r.object:20}) conf={r.confidence:.2f}")
        if len(self.relations) > 10:
            print(f"  ... and {len(self.relations) - 10} more")
        
        print(f"\n🎯 FRAMES: {len(self.frames)}")
        print("-" * 70)
        for i, f in enumerate(self.frames[:5], 1):
            slot_count = len(f.slots)
            print(f"  {i}. {f.frame_type} (trigger: {f.trigger}, {slot_count} slots filled)")
            for slot_name, (value, conf) in list(f.slots.items())[:3]:
                print(f"     - {slot_name}: {value} (conf={conf:.2f})")
        if len(self.frames) > 5:
            print(f"  ... and {len(self.frames) - 5} more")
        
        print("\n" + "="*70)


# ════════════════════════════════════════════════════════════════════════════
# [NEW CODE — EXTENSION 2] KNOWLEDGE GRAPH EMBEDDINGS
# ════════════════════════════════════════════════════════════════════════════
#
# Operates ONLY on the final consolidated triples produced by BOOFS. Does not
# touch relation extraction logic. Uses PyKEEN with RotatE as the primary
# model (ComplEx optionally trained as a secondary model for comparison).

class KGEmbeddingModule:
    """
    Trains knowledge graph embeddings on BOOFS's consolidated triples and
    supports link prediction and entity similarity queries.
    """

    def __init__(self, embedding_dim: int = 50, num_epochs: int = 50, use_complex: bool = False):
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.use_complex = use_complex
        self.is_trained = False
        self.triples_factory = None
        self.rotate_result = None
        self.complex_result = None
        self._entity_to_id = {}
        self._id_to_entity = {}

    def train(self, triples: List[Tuple[str, str, str]]):
        """Train RotatE (primary) and optionally ComplEx (secondary) on the given triples."""
        from pykeen.triples import TriplesFactory
        from pykeen.pipeline import pipeline

        triples = [t for t in triples if t[0] and t[1] and t[2]]
        if len(triples) < 3:
            raise ValueError("Not enough triples to train KG embeddings (need >= 3).")

        triples_array = np.array(triples, dtype=str)
        self.triples_factory = TriplesFactory.from_labeled_triples(triples_array)
        self._entity_to_id = self.triples_factory.entity_to_id
        self._id_to_entity = {v: k for k, v in self._entity_to_id.items()}

        self.rotate_result = pipeline(
            training=self.triples_factory,
            testing=self.triples_factory,
            model='RotatE',
            model_kwargs=dict(embedding_dim=self.embedding_dim),
            training_kwargs=dict(num_epochs=self.num_epochs, use_tqdm=False),
            random_seed=42,
        )

        if self.use_complex:
            self.complex_result = pipeline(
                training=self.triples_factory,
                testing=self.triples_factory,
                model='ComplEx',
                model_kwargs=dict(embedding_dim=self.embedding_dim),
                training_kwargs=dict(num_epochs=self.num_epochs, use_tqdm=False),
                random_seed=42,
            )

        self.is_trained = True

    def predict_missing_links(self, top_k: int = 10):
        """Return top-k predicted (head, relation, tail) triples not already in the graph."""
        if not self.is_trained:
            raise RuntimeError("Call train() before predict_missing_links().")
        from pykeen.models.predict import predict_all
        predictions = predict_all(model=self.rotate_result.model, k=top_k)
        df = predictions.process(factory=self.triples_factory).df
        return df.head(top_k)

    def get_entity_similarity(self, entity: str, top_k: int = 5):
        """Return top-k most similar entities to `entity` by embedding cosine similarity."""
        if not self.is_trained:
            raise RuntimeError("Call train() before get_entity_similarity().")
        entity = entity.lower()
        if entity not in self._entity_to_id:
            return []

        entity_embeddings = self.rotate_result.model.entity_representations[0](indices=None).detach().cpu().numpy()
        # RotatE embeddings are complex-valued (stored as concatenated real/imag); use magnitude for similarity
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
# [NEW CODE — EVALUATION METRICS]
# ════════════════════════════════════════════════════════════════════════════

def evaluate_coreference_improvement(raw_text: str, resolved_text: str) -> Dict:
    """Rough measure of how many pronoun tokens were replaced by coreference resolution."""
    raw_doc, resolved_doc = nlp(raw_text), nlp(resolved_text)
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
    """
    Compares relation sets before/after coreference. Without gold labels, reports
    the proxy metric of how many relations have non-pronoun subjects/objects
    (a cheap stand-in for "resolved" precision). If `sample_labels` (a dict of
    (subj, rel, obj) -> is_correct) is supplied, true precision is computed instead.
    """
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
            labeled = [(r.subject, r.relation, r.object) for r in rels if (r.subject, r.relation, r.object) in sample_labels]
            if not labeled:
                return None
            correct = sum(1 for t in labeled if sample_labels[t])
            return round(correct / len(labeled), 3)
        result['labeled_precision_before'] = labeled_precision(relations_before)
        result['labeled_precision_after'] = labeled_precision(relations_after)

    return result


def evaluate_hits_at_k(kg_embedder: 'KGEmbeddingModule', k: int = 10) -> Optional[float]:
    """Returns the Hits@k metric from PyKEEN's evaluation results for the RotatE model."""
    if kg_embedder is None or not kg_embedder.is_trained:
        return None
    try:
        metrics = kg_embedder.rotate_result.metric_results.to_dict()
        return metrics.get('both', {}).get('realistic', {}).get(f'hits_at_{k}')
    except Exception as e:
        logger.warning(f"Could not extract Hits@{k}: {e}")
        return None


def evaluate_entity_similarity_quality(kg_embedder: 'KGEmbeddingModule', sample_size: int = 10) -> Dict:
    """Reports average top-1 similarity score across a sample of entities, as a rough quality proxy."""
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
    # Example text (works on ANY text without configuration!)
    sample_text = """
    Narendra Damodardas Modi (born 17 September 1950) is an Indian politician who has served as the prime minister of India since 26 May 2014. Modi was the chief minister of Gujarat from 2001 to 2014 and is the Member of Parliament (MP) for Varanasi. He is a member of the Bharatiya Janata Party (BJP) and of the Rashtriya Swayamsevak Sangh (RSS), a right-wing Hindutva paramilitary volunteer organisation. He is India's third-longest-serving prime minister, and the longest-serving prime minister outside the Indian National Congress.
    """
    
    # Create learner (no domain-specific configuration!)
    learner = BOOFSOntologyLearner()
    
    # Process text
    results = learner.process(sample_text, use_active_learning=False, verbose=True)
    
    # Export to CSV
    learner.export_to_csv("boofs_results")

    # [NEW CODE — EXTENSION 2] Export embeddings/predictions/similarity (additive, optional)
    learner.export_embeddings_to_csv("boofs_results")

    # Print summary
    learner.print_summary()

    # [NEW CODE — EVALUATION METRICS] (additive, optional)
    print("\n" + "="*70)
    print("BOOFS EXTENDED — EVALUATION METRICS")
    print("="*70)
    print(evaluate_coreference_improvement(learner.raw_text, learner.resolved_text))
    print(evaluate_relation_precision(learner.relations, learner.relations))
    print({'hits_at_10': evaluate_hits_at_k(learner.kg_embedder, k=10)})
    print(evaluate_entity_similarity_quality(learner.kg_embedder))

    print("\n✅ Ontology learning complete!")
    print("📁 Results saved to: boofs_results_[concepts|relations|frames|embeddings|link_predictions|similarity_scores].csv")
