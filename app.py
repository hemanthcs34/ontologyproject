
import io
import time
import traceback

import pandas as pd
import streamlit as st

import aarya_boofs as boofs

st.set_page_config(page_title="BOOFS Ontology Learner", layout="wide")

EXAMPLE_TEXT = """Bill and Dave became friends when they were both engineering students at Stanford.
After graduation, Dave took a job with General Electric and moved to Schenectady,
New York, where he married his college sweetheart Lucile Salter in 1938.
But he and Bill stayed in touch. The two were encouraged by their former professor Fred Terman
to start a technology company of their own.
Taking a leave of absence from his job at GE, Dave and his new bride drove to California with
a used drill press (an important piece of equipment for the new venture) in the rumble seat.
Bill scouted for places where the newlyweds could live. He found the ideal rental at 367 Addison
Avenue in Palo Alto for $45 per month. Dave and Lucile would live in the downstairs flat,
while Bill would bunk in a tiny backyard shed where there was indoor plumbing and just enough
room for a cot. But what made the property truly perfect for their needs was the
small garage that the landlady told them they could use as a workshop."""


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


# ── Session state ────────────────────────────────────────────────────────────
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "results" not in st.session_state:
    st.session_state.results = None

# ── Sidebar: pipeline options ───────────────────────────────────────────────
st.sidebar.header("Pipeline options")
resolve_coref = st.sidebar.checkbox("Resolve coreference", value=True)
enable_relation_model = st.sidebar.checkbox(
    "Enable learned relation model", value=True,
    help="Seeds a lightweight classifier from BOOFS's own frames and uses it "
         "to filter/label candidate relations during consolidation."
)
verbose_logs = st.sidebar.checkbox("Show pipeline log output", value=False)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Active learning (human-in-the-loop labeling) is left off in this UI — "
    "it requires an interactive oracle. Everything else in the 7-stage "
    "pipeline runs as normal."
)

# ── Main layout ──────────────────────────────────────────────────────────────
st.title("BOOFS: Ontology & Object Frame Semantics")
st.caption("Paste text below, then run the pipeline to extract concepts, relations, and frames.")

col_a, col_b = st.columns([1, 1])
with col_a:
    if st.button("Load example text"):
        st.session_state.input_text = EXAMPLE_TEXT
with col_b:
    if st.button("Clear"):
        st.session_state.input_text = ""
        st.session_state.results = None

text_input = st.text_area(
    "Sample text",
    value=st.session_state.input_text,
    height=260,
    placeholder="Paste your text here...",
    key="input_text",
)

run_clicked = st.button("▶ Run BOOFS pipeline", type="primary", disabled=not text_input.strip())

if run_clicked:
    log_box = st.empty()
    with st.spinner("Running the 7-stage pipeline (this can take a little while, especially "
                     "the first run while spaCy / KG embedding models warm up)..."):
        try:
            learner = boofs.BOOFSOntologyLearner()
            start = time.time()

            if verbose_logs:
                import contextlib
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    results = learner.process(
                        text_input,
                        use_active_learning=False,
                        verbose=True,
                        resolve_coreference=resolve_coref,
                        enable_relation_model=enable_relation_model,
                    )
                log_box.code(buf.getvalue())
            else:
                results = learner.process(
                    text_input,
                    use_active_learning=False,
                    verbose=False,
                    resolve_coreference=resolve_coref,
                    enable_relation_model=enable_relation_model,
                )

            elapsed = time.time() - start
            st.session_state.results = {
                "learner": learner,
                "results": results,
                "elapsed": elapsed,
            }
            st.success(f"Done in {elapsed:.1f}s")
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.code(traceback.format_exc())
            st.session_state.results = None

# ── Results ──────────────────────────────────────────────────────────────────
if st.session_state.results:
    learner = st.session_state.results["learner"]
    results = st.session_state.results["results"]

    concepts_df = pd.DataFrame([c.to_dict() for c in learner.concepts])
    relations_df = pd.DataFrame([r.to_dict() for r in learner.relations])
    sim_df = pd.DataFrame([r.to_dict() for r in learner.similarity_hypotheses])

    frame_rows = []
    for frame in learner.frames:
        for slot_name, (slot_value, confidence) in frame.slots.items():
            frame_rows.append({
                "frame_type": frame.frame_type,
                "trigger": frame.trigger,
                "slot_name": slot_name,
                "slot_value": slot_value,
                "confidence": round(confidence, 3),
                "negated": frame.negated,
            })
    frames_df = pd.DataFrame(frame_rows)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Concepts", len(learner.concepts))
    m2.metric("Relations", len(learner.relations))
    m3.metric("Frames", len(learner.frames))
    m4.metric("Similarity hypotheses", len(learner.similarity_hypotheses))

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Concepts", "Relations", "Frames", "Similarity hypotheses", "KG embeddings"]
    )

    with tab1:
        st.dataframe(concepts_df, use_container_width=True)
        if not concepts_df.empty:
            st.download_button("Download concepts.csv", df_to_csv_bytes(concepts_df),
                                "boofs_concepts.csv", "text/csv")

    with tab2:
        st.dataframe(relations_df, use_container_width=True)
        if not relations_df.empty:
            st.download_button("Download relations.csv", df_to_csv_bytes(relations_df),
                                "boofs_relations.csv", "text/csv")

    with tab3:
        st.dataframe(frames_df, use_container_width=True)
        if not frames_df.empty:
            st.download_button("Download frames.csv", df_to_csv_bytes(frames_df),
                                "boofs_frames.csv", "text/csv")

    with tab4:
        st.dataframe(sim_df, use_container_width=True)
        if not sim_df.empty:
            st.download_button("Download similarity_hypotheses.csv", df_to_csv_bytes(sim_df),
                                "boofs_similarity_hypotheses.csv", "text/csv")

    with tab5:
        kg = learner.kg_embedder
        if kg is not None and getattr(kg, "is_trained", False):
            st.write("Knowledge-graph embeddings trained successfully.")
            hits10 = boofs.evaluate_hits_at_k(kg, k=10)
            st.write(f"Hits@10: {hits10 if hits10 is not None else 'suppressed (graph too small for a valid held-out split)'}")
            sim_quality = boofs.evaluate_entity_similarity_quality(kg)
            st.json(sim_quality)
        else:
            st.info("No trained KG embeddings for this run (e.g. not enough triples, or training was skipped).")

    st.markdown("---")
    with st.expander("Coreference resolution: before vs after"):
        colx, coly = st.columns(2)
        colx.text_area("Raw text", learner.raw_text or "", height=200, disabled=True)
        coly.text_area("Resolved text", learner.resolved_text or "", height=200, disabled=True)
