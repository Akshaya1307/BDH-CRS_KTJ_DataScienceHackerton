import streamlit as st
import google.generativeai as genai
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime
from reasoner import run_bdh_pipeline
import json

# --------------------- PAGE CONFIG ---------------------
st.set_page_config(
    page_title="Track B – BDH Continuous Reasoner",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# --------------------- CUSTOM CSS ---------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    .prediction-consistent {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #c3e6cb;
    }
    .prediction-contradict {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #f5c6cb;
    }
    .prediction-uncertain {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

# --------------------- SAFETY CHECK ---------------------
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ GEMINI_API_KEY not configured in Streamlit Secrets.")
    st.stop()

# --------------------- INITIALIZE MODEL ---------------------
@st.cache_resource
def load_model():
    """Load and cache the Gemini model."""
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        return genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

model = load_model()

# --------------------- SIDEBAR ---------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Analysis parameters
    st.markdown("### Analysis Parameters")
    min_chunk_length = st.slider(
        "Minimum chunk length (words)",
        min_value=1,
        max_value=10,
        value=1,  # Changed from 3 to 1
        help="Minimum number of words per chunk to process"
    )
    
    signal_threshold = st.slider(
        "Signal threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.01,  # Changed from 0.1 to 0.01
        step=0.01,
        help="Minimum absolute signal value to update beliefs (recommended: 0.01)"
    )
    
    decision_threshold = st.slider(
        "Decision threshold",
        min_value=0.0,
        max_value=0.3,
        value=0.05,
        step=0.01,
        help="Threshold for final decision (higher = more conservative)"
    )
    
    use_caching = st.checkbox(
        "Use caching",
        value=True,
        help="Cache analyses for faster repeated runs"
    )
    
    # Example selector
    st.markdown("### Examples")
    example_choice = st.selectbox(
        "Load example",
        ["None", "Loyal Friend", "Suspicious Behavior", "Career Change", "Simple Test"],
        index=4  # Default to Simple Test
    )
    
    # Info section
    with st.expander("ℹ️ About BDH"):
        st.markdown("""
        **BDH (Belief-Dynamics Hybrid) Reasoning**:
        
        - **Continuous**: Updates beliefs incrementally as evidence arrives
        - **Sparse**: Only significant evidence updates belief nodes
        - **Persistent**: Maintains internal state across narrative
        - **Explainable**: Tracks reasoning trajectory
        
        Each narrative chunk is evaluated for consistency with the backstory,
        with scores from -1 (contradiction) to +1 (alignment).
        """)
    
    st.divider()
    st.caption(f"BDH Reasoner v1.3 • {datetime.now().strftime('%Y-%m-%d')}")

# --------------------- HEADER ---------------------
st.markdown('<h1 class="main-header">🧠 BDH Continuous Reasoner</h1>', unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center; color: #6b7280; font-size: 1.1rem;'>
Track B: Belief-Dynamics Hybrid reasoning with persistent internal state
</p>
""", unsafe_allow_html=True)

st.divider()

# --------------------- EXAMPLE DATA ---------------------
examples = {
    "None": {"narrative": "", "backstory": ""},
    "Loyal Friend": {
        "narrative": """John received an urgent call from his best friend Mike. 
        Mike was stranded downtown with a flat tire. John immediately left his dinner party, 
        drove 20 miles through heavy rain, and helped Mike change the tire. 
        He then drove Mike home safely, even though it made him miss an important work deadline.""",
        "backstory": "John is a loyal friend who always prioritizes friendships over personal convenience."
    },
    "Suspicious Behavior": {
        "narrative": """Sarah noticed her colleague Tom frequently staying late at the office. 
        He would often close his laptop when others approached. Last week, company financial data 
        was found on an unsecured server. Security logs show Tom accessed the server at unusual hours. 
        Yesterday, Tom abruptly resigned without notice.""",
        "backstory": "Tom is planning to steal company secrets for a competitor."
    },
    "Career Change": {
        "narrative": """Maria had been a successful lawyer for 15 years. Last month, 
        she started taking evening art classes. She converted her home office into a studio. 
        She recently declined a promotion at her firm. Yesterday, she registered a business 
        license for an art gallery.""",
        "backstory": "Maria is preparing to leave her legal career to become a full-time artist."
    },
    "Simple Test": {
        "narrative": "John helped his friend. This shows loyalty. He prioritized friendship.",
        "backstory": "John is a loyal friend."
    }
}

# --------------------- INPUT SECTION ---------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📖 Narrative Evidence")
    narrative = st.text_area(
        "Enter narrative text (events, actions, decisions):",
        height=280,
        placeholder="""Example: John helped his friend. This shows loyalty. He prioritized friendship.""",
        value=examples[example_choice]["narrative"],
        help="Enter the sequence of events or facts to analyze"
    )
    
    # Narrative stats
    if narrative:
        word_count = len(narrative.split())
        sentence_count = len([s for s in narrative.split('.') if s.strip()])
        st.caption(f"📊 {word_count} words, ~{sentence_count} sentences")

with col2:
    st.subheader("🎭 Hypothetical Backstory")
    backstory = st.text_area(
        "Describe character beliefs, motivations, or context:",
        height=280,
        placeholder="""Example: John is a loyal friend.""",
        value=examples[example_choice]["backstory"],
        help="Enter the hypothetical context to evaluate the narrative against"
    )
    
    # Backstory stats
    if backstory:
        word_count = len(backstory.split())
        st.caption(f"📊 {word_count} words")

st.divider()

# --------------------- CONTROL BUTTONS ---------------------
col_run, col_clear, col_demo = st.columns([2, 1, 1])

with col_run:
    run_button = st.button(
        "🚀 Execute BDH Continuous Reasoning",
        use_container_width=True,
        type="primary"
    )

with col_clear:
    if st.button("🧹 Clear Results", use_container_width=True):
        st.session_state.clear()

with col_demo:
    if st.button("🎯 Quick Demo", use_container_width=True):
        example = examples["Simple Test"]
        narrative = example["narrative"]
        backstory = example["backstory"]
        run_button = True  # Trigger execution

# --------------------- EXECUTION SECTION ---------------------
if run_button and narrative.strip() and backstory.strip():
    
    # Initialize session state for results
    if 'results' not in st.session_state:
        st.session_state.results = {}
    
    # Create progress container
    progress_container = st.container()
    
    with progress_container:
        # Progress bars
        st.markdown("### ⏳ Processing...")
        main_progress = st.progress(0, text="Initializing analysis...")
        status_text = st.empty()
        
        # Performance tracking
        start_time = time.time()
        
        try:
            # Update status
            status_text.info("📝 Splitting narrative into chunks...")
            main_progress.progress(10, text="Preprocessing narrative...")
            
            # Run the pipeline
            status_text.info("🧠 Analyzing chunks with BDH reasoning...")
            main_progress.progress(30, text="Analyzing evidence chunks...")
            
            prediction, state, metadata = run_bdh_pipeline(
                model=model,
                narrative=narrative,
                backstory=backstory,
                min_chunk_length=min_chunk_length,
                signal_threshold=signal_threshold,
                decision_threshold=decision_threshold,
                use_caching=use_caching
            )
            
            # Complete progress
            main_progress.progress(100, text="Analysis complete!")
            processing_time = time.time() - start_time
            
            # Update metadata
            metadata["processing_time"] = processing_time
            metadata["timestamp"] = datetime.now().isoformat()
            
            # Store results
            st.session_state.results = {
                "prediction": prediction,
                "state": state,
                "metadata": metadata
            }
            
            status_text.success(f"✅ Analysis completed in {processing_time:.2f} seconds!")
            
        except ValueError as e:
            st.error(f"❌ Input error: {e}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Analysis failed: {e}")
            st.exception(e)
            st.stop()
    
    st.divider()
    
    # --------------------- RESULTS DISPLAY ---------------------
    if 'results' in st.session_state and st.session_state.results:
        results = st.session_state.results
        prediction = results["prediction"]
        state = results["state"]
        metadata = results["metadata"]
        
        # ---- FINAL VERDICT ----
        st.subheader("🎯 Final Consistency Judgment")
        
        col_verdict, col_score, col_confidence = st.columns(3)
        
        with col_verdict:
            if prediction == 1:
                st.markdown('<div class="prediction-consistent">', unsafe_allow_html=True)
                st.markdown("### ✅ CONSISTENT")
                st.markdown("Narrative aligns with backstory")
                st.markdown('</div>', unsafe_allow_html=True)
            elif prediction == 0:
                st.markdown('<div class="prediction-contradict">', unsafe_allow_html=True)
                st.markdown("### ❌ CONTRADICT")
                st.markdown("Narrative contradicts backstory")
                st.markdown('</div>', unsafe_allow_html=True)
            else:  # prediction == 0.5
                st.markdown('<div class="prediction-uncertain">', unsafe_allow_html=True)
                st.markdown("### ⚖️ UNCERTAIN")
                st.markdown("Evidence is ambiguous or neutral")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col_score:
            st.metric(
                "Normalized Score",
                f"{metadata['normalized_score']:.3f}",
                delta=f"Threshold: {decision_threshold}",
                delta_color="off"
            )
        
        with col_confidence:
            st.metric(
                "Confidence",
                f"{metadata['confidence_score']:.3f}",
                help="Weighted confidence based on evidence volume"
            )
        
        # ---- METRICS DASHBOARD ----
        st.subheader("📊 Analysis Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Processing Time", f"{metadata['processing_time']:.2f}s")
        
        with col2:
            st.metric("Chunks Processed", f"{metadata['processed_chunks']}/{metadata['total_chunks']}")
        
        with col3:
            density = metadata.get('belief_density', 0)
            st.metric("Belief Density", f"{density:.1%}")
        
        # ---- TRAJECTORY VISUALIZATION ----
        st.subheader("📈 Belief-State Trajectory")
        
        if state.trajectory and len(state.trajectory) > 0:
            df_traj = pd.DataFrame(state.trajectory)
            
            # Create interactive plot with Plotly
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Global Belief Score', 'Individual Signals'),
                vertical_spacing=0.15,
                row_heights=[0.6, 0.4]
            )
            
            # Global score line
            fig.add_trace(
                go.Scatter(
                    x=df_traj['step'],
                    y=df_traj['current_score'],
                    mode='lines+markers',
                    name='Global Score',
                    line=dict(color='#667eea', width=3),
                    marker=dict(size=8)
                ),
                row=1, col=1
            )
            
            # Individual signals as bars
            colors = ['#10b981' if x >= 0 else '#ef4444' for x in df_traj['signal']]
            fig.add_trace(
                go.Bar(
                    x=df_traj['step'],
                    y=df_traj['signal'],
                    name='Chunk Signal',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=600,
                showlegend=True,
                hovermode='x unified',
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            fig.update_xaxes(title_text="Processing Step", row=2, col=1)
            fig.update_yaxes(title_text="Score", row=1, col=1)
            fig.update_yaxes(title_text="Signal", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show trajectory table
            with st.expander("📋 View Detailed Trajectory"):
                st.dataframe(
                    df_traj[['step', 'claim', 'signal', 'current_score']],
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("No significant belief updates recorded.")
        
        # ---- BELIEF NODES ANALYSIS ----
        st.subheader("🧩 Belief Nodes Analysis")
        
        if state.nodes:
            # Create nodes dataframe
            nodes_data = []
            for claim, node in state.nodes.items():
                nodes_data.append({
                    "Claim": claim[:100] + "..." if len(claim) > 100 else claim,
                    "Full Claim": claim,
                    "Support": node.support,
                    "Conflict": node.conflict,
                    "Score": node.score,
                    "Evidence Count": node.evidence_count,
                    "Confidence": f"{node.confidence:.2%}"
                })
            
            df_nodes = pd.DataFrame(nodes_data)
            
            # Display interactive table
            st.dataframe(
                df_nodes[["Claim", "Support", "Conflict", "Score", "Evidence Count", "Confidence"]],
                use_container_width=True,
                hide_index=True
            )
            
            # Node visualization
            with st.expander("📊 Visualize Belief Network"):
                if len(df_nodes) > 0:
                    # Create bar chart of node scores
                    fig_nodes = px.bar(
                        df_nodes.nlargest(10, 'Score'),
                        x='Claim',
                        y='Score',
                        color='Score',
                        color_continuous_scale='RdYlGn',
                        title='Top 10 Belief Nodes by Score'
                    )
                    fig_nodes.update_layout(height=400)
                    st.plotly_chart(fig_nodes, use_container_width=True)
        else:
            st.info("No belief nodes formed.")
        
        # ---- RAW DATA EXPORT ----
        st.subheader("💾 Export Results")
        
        col_export1, col_export2, col_export3 = st.columns(3)
        
        with col_export1:
            if st.button("📥 Download JSON", use_container_width=True):
                # Clean metadata for export
                clean_metadata = {k: v for k, v in metadata.items() 
                                if k not in ['timestamp']}
                
                download_data = {
                    "metadata": clean_metadata,
                    "prediction": prediction,
                    "trajectory": state.trajectory,
                    "nodes": {k: v.__dict__ for k, v in state.nodes.items()}
                }
                
                # Convert to JSON
                json_str = json.dumps(download_data, indent=2)
                
                # Create download button
                st.download_button(
                    label="Click to download",
                    data=json_str,
                    file_name=f"bdh_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        with col_export2:
            if st.button("📊 Export to CSV", use_container_width=True):
                if state.trajectory:
                    df_export = pd.DataFrame(state.trajectory)
                    csv = df_export.to_csv(index=False)
                    st.download_button(
                        label="Click to download",
                        data=csv,
                        file_name=f"bdh_trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        with col_export3:
            if st.button("📋 Copy Summary", use_container_width=True):
                summary = f"""
BDH Analysis Summary:
- Final Judgment: {'CONSISTENT' if prediction == 1 else 'CONTRADICT' if prediction == 0 else 'UNCERTAIN'}
- Normalized Score: {metadata['normalized_score']:.3f}
- Confidence: {metadata['confidence_score']:.3f}
- Processing Time: {metadata['processing_time']:.2f}s
- Chunks Analyzed: {metadata['total_chunks']} (processed: {metadata['processed_chunks']})
- Belief Nodes: {metadata['belief_nodes']}
- Belief Density: {metadata.get('belief_density', 0):.1%}
                """
                st.code(summary)
        
        # ---- DEBUG INFO ----
        with st.expander("🔍 Debug Information"):
            st.json(metadata)
            
            st.markdown("### Processing Log")
            st.code(f"""
Input Validation: ✓
Chunk Splitting: {metadata['total_chunks']} chunks
Signal Threshold: {signal_threshold}
Decision Threshold: {decision_threshold}
Final Decision: {prediction} ({metadata['decision_reason']})
            """)
        
elif run_button:
    st.warning("⚠️ Please provide both narrative and backstory to begin analysis.")

# --------------------- FOOTER ---------------------
st.divider()
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 1rem;'>
    <p>BDH Continuous Reasoner • Track B Submission • Belief-Dynamics Hybrid Architecture</p>
    <p>Decision derived from persistent internal state dynamics with explainable trajectory</p>
</div>
""", unsafe_allow_html=True)

# --------------------- SESSION STATE INIT ---------------------
if 'results' not in st.session_state:
    st.session_state.results = {}
