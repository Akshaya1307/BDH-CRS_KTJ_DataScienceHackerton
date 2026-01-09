import streamlit as st
import google.generativeai as genai
from reasoner import run_bdh_pipeline

st.set_page_config(layout="wide")
st.title("🐉 Track B: BDH-Driven Continuous Reasoner")

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

narrative = st.text_area("Narrative Text (excerpt)", height=300)
backstory = st.text_area("Hypothetical Backstory", height=300)

if st.button("Run BDH Reasoning"):
    if not narrative or not backstory:
        st.error("Inputs required.")
    else:
        with st.spinner("Running BDH-CRS pipeline..."):
            pred, state = run_bdh_pipeline(model, narrative, backstory)

        st.subheader("Consistency Judgment")
        st.success("CONSISTENT (1)" if pred else "CONTRADICT (0)")

        st.subheader("State Trajectory (BDH-style)")
        st.json(state.trajectory)

        st.subheader("Belief Nodes")
        for k, v in state.nodes.items():
            st.write(k, "→ score:", v.score)
