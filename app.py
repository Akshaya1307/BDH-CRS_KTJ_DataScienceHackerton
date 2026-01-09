import streamlit as st
import google.generativeai as genai
import pandas as pd
from reasoner import run_bdh_pipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Track B – BDH Continuous Reasoner",
    layout="wide"
)

# ---------------- SAFETY CHECK ----------------
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ GEMINI_API_KEY not configured in Streamlit Secrets.")
    st.stop()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style='text-align:center;'>🐉 Track B: BDH-Driven Continuous Reasoner</h1>
    <p style='text-align:center; color: #9ca3af;'>
    Persistent belief-state reasoning over narrative evidence
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------- INPUT ----------------
col1, col2 = st.columns(2)

with col1:
    narrative = st.text_area(
        "Narrative Evidence",
        height=260,
        placeholder="Paste narrative text (events, actions, decisions)…"
    )

with col2:
    backstory = st.text_area(
        "Hypothetical Backstory",
        height=260,
        placeholder="Describe character beliefs, motivations, commitments…"
    )

run = st.button("🚀 Execute BDH Continuous Reasoning", use_container_width=True)

# ---------------- EXECUTION ----------------
if run:
    if not narrative or not backstory:
        st.warning("⚠️ Both inputs are required.")
    else:
        with st.spinner("Running BDH belief-state updates…"):
            prediction, state = run_bdh_pipeline(model, narrative, backstory)

        st.divider()

        # ---- FINAL VERDICT ----
        st.subheader("✅ Final Consistency Judgment")
        if prediction == 1:
            st.success("CONSISTENT (1)")
        else:
            st.error("CONTRADICT (0)")

        # ---- TRAJECTORY ----
        st.subheader("📈 Belief-State Confidence Trajectory")
        if state.trajectory:
            df = pd.DataFrame(state.trajectory)
            df["step"] = range(1, len(df) + 1)
            st.line_chart(df.set_index("step")["current_score"])
        else:
            st.info("No significant belief updates recorded.")

        # ---- STATE UPDATES ----
        st.subheader("🧠 Incremental State Updates (BDH-style)")
        for i, t in enumerate(state.trajectory, start=1):
            with st.expander(f"Chunk {i}: {t['claim']}"):
                st.markdown(f"""
                **Signal:** `{t['signal']}`  
                **Updated Score:** `{t['current_score']}`
                """)

        # ---- BELIEF NODES ----
        st.subheader("🧩 Final Belief Nodes")
        if state.nodes:
            rows = []
            for claim, node in state.nodes.items():
                rows.append({
                    "Claim": claim,
                    "Support": node.support,
                    "Conflict": node.conflict,
                    "Final Score": node.score
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No belief nodes formed.")

st.divider()
st.caption(
    "Track B Submission • BDH-Inspired Continuous Reasoning • "
    "Decision derived from persistent internal state dynamics"
)
