# P0 audit for BOOFS

This audit is based on the implementation in [../boofs.py](../boofs.py), [../boofs_eval.py](../boofs_eval.py), [../server.py](../server.py), and [../README.md](../README.md). It records the actual code paths, config defaults, and algorithmic triggers used by the repository.

## 1. RelationInductionModule: incremental path assignment and full-pass trigger

- Exact code: `RelationInductionModule.fit`, `RelationInductionModule._full_cluster`, `RelationInductionModule._incremental_update`, `RelationInductionModule._new_singletons_since_full`, `RelationInductionModule._fits_since_full`.
- Line refs: [../boofs.py](../boofs.py#L900-L1450)
- Answer: The module keeps a per-label medoid and a block threshold; changed paths are assigned incrementally in `RelationInductionModule._incremental_update` at [../boofs.py](../boofs.py#L1360-L1389). For each changed path, it computes the dominant type signature for the current block, looks up the block threshold, and compares the path to each existing medoid in that same block. If the pair distance is within the threshold, the path joins the nearest medoid; otherwise it becomes a new singleton. Unchanged paths keep their existing label and are not revisited.
- Full-pass trigger: the trigger logic is in `RelationInductionModule.fit` at [../boofs.py](../boofs.py#L1225-L1247). A full pass occurs when the module is empty, when the fit count exceeds `recluster_every` (`CONFIG.recluster_every = 25`), when the number of changed paths is larger than half of the candidate set, when a new type-signature block appears, or when the singleton-drift signal fires. The drift test is `self._new_singletons_since_full > max(2, len(self._medoids) // 10)`, and it is intentionally corpus-relative rather than a fixed absolute count.
- Normalization of drift: the threshold is normalized by the current number of induced medoids, not by total candidate count; the code explicitly states this is a corpus-relative drift signal in [../boofs.py](../boofs.py#L1235-L1245).

## 2. Adaptive threshold: largest gap, EMA, and sparse-block reference distance

- Exact code: `RelationInductionModule._block_threshold`, `RelationInductionModule._ema_threshold`, `RelationInductionModule._update_merge_reference`, `BOOFSConfig.adaptive_threshold_min`, `BOOFSConfig.adaptive_threshold_max`, `BOOFSConfig.threshold_ema_beta`, `BOOFSConfig.min_block_for_gap`.
- Line refs: [../boofs.py](../boofs.py#L90-L115), [../boofs.py](../boofs.py#L1111-L1160), [../boofs.py](../boofs.py#L1264-L1271)
- Answer: The largest gap is computed in `RelationInductionModule._block_threshold` at [../boofs.py](../boofs.py#L1117-L1143). The code forms the non-zero pairwise distances in the block, sorts them, and uses `np.argmax(np.diff(nz))` to find the widest adjacent gap; the threshold is the midpoint of the two neighboring values, `float((nz[cut] + nz[cut + 1]) / 2.0)`. This is the “largest gap” rule.
- EMA: the EMA is implemented in `RelationInductionModule._ema_threshold` at [../boofs.py](../boofs.py#L1264-L1271). The code sets `beta = self.cfg.threshold_ema_beta` and then computes `thr = new_thr if prev is None else (beta * prev + (1.0 - beta) * new_thr)`. The previous threshold is the term weighted by `beta`.
- Sparse-block fallback: when the block is too small for a meaningful gap (`D.shape[0] < min_block_for_gap`), the method uses the learned corpus reference distance `self._merge_ref_dist` if present, otherwise the minimum threshold. The reference is learned in `RelationInductionModule._update_merge_reference` at [../boofs.py](../boofs.py#L1145-L1160): it computes the mean intra-cluster distance of merged members and updates the reference by a running mean, `self._merge_ref_dist = d if self._merge_ref_dist is None else 0.5 * self._merge_ref_dist + 0.5 * d`.

## 3. Type-signature blocking

- Exact code: `RelationInductionModule._dominant`, `RelationInductionModule._dominant_sig`, `RelationInductionModule._full_cluster`, `BOOFSConfig.type_signature_blocking`, `BOOFSConfig.max_block_size`.
- Line refs: [../boofs.py](../boofs.py#L90-L105), [../boofs.py](../boofs.py#L1041-L1065), [../boofs.py](../boofs.py#L1281-L1355)
- Answer: The block signature is defined by the dominant argument-type pair for each path. `RelationInductionModule._dominant_sig` returns `(dominant(typeX[path]), dominant(typeY[path]))` using the dominant label from each argument-type counter; the helper `_dominant` picks the label with the highest count, breaking ties lexicographically. This is implemented in [../boofs.py](../boofs.py#L1039-L1065).
- Max-block enforcement: in `RelationInductionModule._full_cluster`, the block is split into `head = full_block[:self.cfg.max_block_size]` and `tail = full_block[self.cfg.max_block_size:]` at [../boofs.py](../boofs.py#L1298-L1307). The head is clustered; the tail is then assigned to the nearest in-block medoid or seeded as a new singleton if no close medoid exists. This means oversized blocks are not discarded; they are deferred into a second assignment pass.
- `type_signature_blocking` is enabled by default in [../boofs.py](../boofs.py#L90-L105) with `type_signature_blocking: bool = True`.

## 4. `type_smoothing_max_alpha`

- Exact code: `BOOFSConfig.type_smoothing_max_alpha`, `RelationInductionModule._smooth_slot`.
- Line refs: [../boofs.py](../boofs.py#L90-L105), [../boofs.py](../boofs.py#L1052-L1068)
- Answer: Smoothing is only applied when `alpha > 0`. In `RelationInductionModule._smooth_slot`, the code computes `hapax = sum(n for f, n in c.items() if global_freq[f] <= 1)`, then `alpha = min(self.cfg.type_smoothing_max_alpha, hapax / total)`. If `alpha > 0`, the method mixes in a proportion of the path’s argument-type distribution as pseudo-token mass. This is a real smoothing step, but it is inactive by default because `type_smoothing_max_alpha` is defined as `0.0` in [../boofs.py](../boofs.py#L90-L105). The code does not apply any smoothing when the value is zero.

## 5. DIRT similarity: PMI, clipping, and slot combination

- Exact code: `RelationInductionModule._slot_mi`, `RelationInductionModule._slot_sim`, `RelationInductionModule._global_mi`, `RelationInductionModule._block_distance`.
- Line refs: [../boofs.py](../boofs.py#L1000-L1051), [../boofs.py](../boofs.py#L1068-L1108)
- Answer: The DIRT-style path similarity is implemented as a PMI-weighted overlap over filler distributions. In `_slot_mi`, the per-path word mass is computed as `val = log((n * N) / denom)`, and the code only keeps a word if `val > 0`; this is effectively a positive-PMI restriction. The mass term is `mass[path] = sum(val)` over retained words, and `_slot_sim` forms the shared-word overlap as `num = sum(dp[w] + dq[w] for w in shared)` divided by `mass[p] + mass[q]`.
- Clipping and smoothing: the code clips negative PMI to zero at the word level because it ignores non-positive values. There is no extra negative-clipping step beyond that; once the PMI contribution is non-positive, it is dropped. For path-to-path similarity, `RelationInductionModule._block_distance` computes `sim = sqrt(sx * sy)` when both slot similarities are positive, and then sets distance to `1.0 - sim`. This is the same geometric-mean combination described in the class docstring.

## 6. Label assignment and carry-over across runs

- Exact code: `RelationInductionModule._carryover_label`, `RelationInductionModule._load_state`, `RelationInductionModule._save_state`, `BOOFSConfig.label_carryover_jaccard`.
- Line refs: [../boofs.py](../boofs.py#L90-L110), [../boofs.py](../boofs.py#L951-L988), [../boofs.py](../boofs.py#L1190-L1207)
- Answer: Label carry-over is handled in `_carryover_label` at [../boofs.py](../boofs.py#L1190-L1207). It computes a Jaccard overlap between the current cluster member set and the previous-run cluster member set, uses `best_j = inter / len(mset | pset)`, and reuses the previous label if `best_j >= label_carryover_jaccard` and the label is not already taken. The default threshold is `0.5`, as set in [../boofs.py](../boofs.py#L90-L110).
- Persistence: previous clusters are loaded from the stored state in `_load_state` at [../boofs.py](../boofs.py#L953-L980), and the state is written in `_save_state` at [../boofs.py](../boofs.py#L969-L988). The stored payload includes `clusters`, `path_to_label`, `merge_ref_dist`, `block_thr_ema`, and `calib_bins`. That is the mechanism by which labels and calibration drift persist across runs.

## 7. Relation subsumption

- Exact code: `RelationInductionModule.induce_ontology`, `BOOFSConfig.subsumption_min_containment`.
- Line refs: [../boofs.py](../boofs.py#L90-L115), [../boofs.py](../boofs.py#L1450-L1515)
- Answer: Subsumption is derived in `induce_ontology` at [../boofs.py](../boofs.py#L1450-L1515). Domain and range are the dominant NER-type statistics for each induced label, from `agg[L]['dx']` and `agg[L]['dy']`, and the code checks only labels whose domain and range match exactly.
- Comparison rule: for each candidate parent `B`, the method computes the argument-set containment ratios `cont_a_in_b` and `cont_b_in_a` using the intersection-over-set size of the X and Y filler sets, then tests `cont_a_in_b >= subsumption_min_containment` and `cont_b_in_a < cont_a_in_b`. The default threshold is `0.6` from [../boofs.py](../boofs.py#L90-L115). The chosen parent is the nearest broader candidate (smallest broader set), so the code builds a multi-level hierarchy rather than only direct one-hop parent links.

## 8. ActiveLearningModule and RelationValidityModel

- Exact code: `RelationValidityModel`, `ActiveLearningModule.seed_from_induction`, `ActiveLearningModule.select_queries`, `ActiveLearningModule.run_round`, `BOOFSConfig.al_human_weight`, `BOOFSConfig.al_seed_weight`, `BOOFSConfig.al_confidence_shrinkage`, `PropositionExtractor.structural_negatives`.
- Line refs: [../boofs.py](../boofs.py#L90-L145), [../boofs.py](../boofs.py#L1600-L1715), [../boofs.py](../boofs.py#L1711-L1852), [../boofs.py](../boofs.py#L630-L662)
- Answer: `RelationValidityModel` is a shallow SGD logistic-regression model built with `HashingVectorizer(n_features=2**14)` and `SGDClassifier(loss="log_loss")` at [../boofs.py](../boofs.py#L1711-L1767). Its feature string is computed by `_features` as `context + TYPE1=... TYPE2=...`; the induced label itself is excluded to avoid leakage. The class space is initially `NO_RELATION` and grows as induced and human labels appear.
- Seeds: `ActiveLearningModule.seed_from_induction` at [../boofs.py](../boofs.py#L1848-L1879) creates positives from induced_examples by matching candidate pairs in either entity order and writing weak labels with `source="induced_seed_pos"`. It creates negatives from `PropositionExtractor.structural_negatives`, which were gathered during dependency-path extraction when a short path had no content predicate; those examples are recorded as `source="induced_seed_neg"`.
- Weights: `RelationValidityModel._weight` at [../boofs.py](../boofs.py#L1726-L1735) returns `CONFIG.al_human_weight` for oracle-labeled examples and `CONFIG.al_seed_weight` for weak-seed examples (`1.0` and `0.3` by default).
- Query strategy: `ActiveLearningModule.select_queries` at [../boofs.py](../boofs.py#L1819-L1837) computes uncertainty either from model uncertainty or `compute_uncertainty`, then returns the top-k examples. `run_round` at [../boofs.py](../boofs.py#L1831-L1846) retrains, chooses the queries, gets oracle labels when an oracle is attached, and retrains again.

## 9. Confidence: cohesion, support, negation penalty, and calibration

- Exact code: `RelationInductionModule._raw_induced_conf`, `RelationInductionModule._induced_conf`, `RelationInductionModule.set_calibration`, `RelationInductionModule.calibrated_confidence`, `BOOFSConfig.conf_cohesion_weight`, `BOOFSConfig.calib_bins_count`, `BOOFSConfig.calib_min_evidence`, `BOOFSConfig.negation_confidence_penalty`.
- Line refs: [../boofs.py](../boofs.py#L90-L145), [../boofs.py](../boofs.py#L2135-L2198)
- Answer: The raw induced confidence is computed in `RelationInductionModule._raw_induced_conf` at [../boofs.py](../boofs.py#L2135-L2158). It blends support and cluster cohesion as `support_term = min(1.0, s / (s + med))` and `w = self.relation_inducer.cfg.conf_cohesion_weight`, then returns `w * coh + (1.0 - w) * support_term`. The cohesion term comes from the per-label cluster cohesion recorded during induction.
- Negation penalty: during relation consolidation in `BOOFSOntologyLearner._consolidate_relations`, the code multiplies the confidence by `CONFIG.negation_confidence_penalty` when the proposition is negated, otherwise it uses `1.0` at [../boofs.py](../boofs.py#L2230-L2255).
- Calibration: `RelationInductionModule.set_calibration` at [../boofs.py](../boofs.py#L1404-L1418) bins `(confidence, correct)` samples and `RelationInductionModule.calibrated_confidence` at [../boofs.py](../boofs.py#L1419-L1437) maps the raw score onto an empirical reliability estimate using Laplace smoothing `emp = (c + 1) / (t + 2)` and a weight `w = t / (t + calib_min_evidence)`. The calibration is updated from human labels recorded in the label store and persisted through `flush_state` in the same class.

## 10. Disable switches for passive voice, blocking, drift trigger, and coreference

- Exact code: `BOOFSConfig.type_signature_blocking`, `BOOFSConfig.recluster_every`, `BOOFSConfig.min_block_for_gap`, `BOOFSOntologyLearner.process`, `CoreferenceResolver.resolve`.
- Line refs: [../boofs.py](../boofs.py#L90-L105), [../boofs.py](../boofs.py#L1980-L2005), [../boofs.py](../boofs.py#L211-L285)
- Answer:
  - Passive-voice direction handling: not available as a separate config flag. The code hard-codes the passive-voice logic in `PropositionExtractor._dependency_path` at [../boofs.py](../boofs.py#L725-L754), using `PASSIVE_SUBJ_DEPS`, `AGENT_DEPS`, `SUBJECT_DEPS`, and `OBJECT_DEPS`; it is algorithmic, not a toggle.
  - Blocking: available as a config switch. `BOOFSConfig.type_signature_blocking` is `True` by default at [../boofs.py](../boofs.py#L90-L105), and the induction logic checks `if self.cfg.type_signature_blocking` in [../boofs.py](../boofs.py#L1281-L1289) and [../boofs.py](../boofs.py#L1360-L1368). This is the direct disable switch for the blocking policy.
  - Drift trigger: no dedicated disable flag is exposed. The trigger is hard-coded in `RelationInductionModule.fit` at [../boofs.py](../boofs.py#L1235-L1247) as a combination of `recluster_every`, a size-based singleton drift test, and a new-block test. `recluster_every` is configurable, but there is no boolean switch for “turn off drift trigger.”
  - Coreference: yes, it is directly switchable. `BOOFSOntologyLearner.process` takes a `resolve_coreference` boolean at [../boofs.py](../boofs.py#L1980-L2005), and the code does `if resolve_coreference: ...` before the extraction stages. This is the explicit off switch.

## 11. Bugs, dead code, or unsupported behavior

- Exact code: `RelationInductionModule.full_pass_partition`, `BOOFSOntologyLearner.for_evaluation`, `PathStatsStore.IN_MEMORY`, `LabelStore.IN_MEMORY`, `BOOFSOntologyLearner.process`, `server.py` startup behavior.
- Line refs: [../boofs.py](../boofs.py#L1439-L1450), [../boofs.py](../boofs.py#L1936-L2090), [../boofs.py](../boofs.py#L750-L810), [../server.py](../server.py#L1-L200)
- Answer: The clearest deliberate design choice is that the evaluation pipeline deliberately avoids mutating the real persistent state by using in-memory stores (`PathStatsStore.IN_MEMORY` and `LabelStore.IN_MEMORY`) in `BOOFSOntologyLearner.for_evaluation` at [../boofs.py](../boofs.py#L1965-L1969). This is a robust evaluation mechanism, not a bug.
- A real operational issue is environmental rather than algorithmic: the FastAPI app in [../server.py](../server.py#L1-L200) binds to port 8000, and a stale listener there causes the standard socket reuse error (`WinError 10048` / “only one usage of each socket address”) when the app is restarted. That is a runtime-port conflict, not a BOOFS logic error.
- There is no evidence of a hidden relation schema being baked into the source; the explicit docstring and config at [../boofs.py](../boofs.py#L1-L145) state that the relation inventory is not declared in source and is learned from corpus statistics. The repository therefore implements the intended “schema-free” behavior, with the caveat that runtime port conflicts must be cleared before local execution.

## Short takeaway

The implementation is a schema-free OpenIE-plus-DIRT pipeline with explicit corpus memory, incremental clustering, active learning, and empirical calibration. The key mechanism is that dependency paths are treated as raw relation evidence, then path clusters are induced and stabilized using type-aware blocking, adaptive thresholds, drift-driven full passes, and label carry-over across runs. The project is designed to remain domain-independent by not hard-coding a relation ontology; the relationship inventory emerges from corpus statistics and the data that the parser sees.
