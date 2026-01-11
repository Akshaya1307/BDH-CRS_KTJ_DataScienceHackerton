# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
from reasoner import run_bdh_pipeline, BinaryConsistencyChecker
import json

# --------------------- PAGE CONFIG ---------------------
st.set_page_config(
    page_title="BDH Consistency Reasoner",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------- CUSTOM CSS ---------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .consistent {
        background-color: #D1FAE5;
        border: 2px solid #10B981;
    }
    .contradictory {
        background-color: #FEE2E2;
        border: 2px solid #EF4444;
    }
    .metric-box {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# --------------------- SIDEBAR ---------------------
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Analysis Parameters
    st.subheader("Analysis Parameters")
    min_chunk_length = st.slider("Minimum chunk length (words)", 
                                 min_value=1, max_value=50, value=10, step=1)
    
    col1, col2 = st.columns(2)
    with col1:
        signal_threshold = st.number_input("Signal threshold", 
                                          min_value=0.0, max_value=1.0, 
                                          value=0.01, step=0.01)
    with col2:
        decision_threshold = st.number_input("Decision threshold", 
                                            min_value=0.0, max_value=1.0, 
                                            value=0.05, step=0.01)
    
    # Examples
    st.subheader("Examples")
    if st.button("Load Career Change Example", use_container_width=True):
        example_text = """After 10 years in finance, I'm considering a career change to software development.
        I've been learning Python and JavaScript for the past year.
        Completed several online courses and built portfolio projects.
        Networking with professionals in the tech industry.
        Planning to apply for junior developer positions next month."""
        st.session_state.input_text = example_text
    
    st.markdown("---")
    st.caption("About BDH")
    st.caption("Binary Decision Helper - A consistency reasoning system")

# --------------------- MAIN INTERFACE ---------------------
st.markdown('<h1 class="main-header">🔍 BDH Consistency Reasoner</h1>', unsafe_allow_html=True)

# Input Section
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("📝 Input Text")
    input_text = st.text_area(
        "Enter your text for consistency analysis (separate chunks with blank lines):",
        height=200,
        value=st.session_state.get("input_text", "")
    )
    
    if st.button("🔍 Analyze Consistency", type="primary", use_container_width=True):
        if input_text.strip():
            # Process text into chunks
            chunks = [chunk.strip() for chunk in input_text.split('\n\n') if chunk.strip()]
            
            # Filter by minimum length
            valid_chunks = [chunk for chunk in chunks 
                          if len(chunk.split()) >= min_chunk_length]
            
            if valid_chunks:
                with st.spinner("Analyzing consistency..."):
                    # Run pipeline
                    result = run_bdh_pipeline(
                        valid_chunks,
                        min_chunk_length=min_chunk_length,
                        signal_threshold=signal_threshold,
                        decision_threshold=decision_threshold
                    )
                    
                    # Store result in session state
                    st.session_state.result = result
                    st.session_state.chunks = valid_chunks
                    st.rerun()
            else:
                st.warning(f"No valid chunks found (minimum {min_chunk_length} words required)")
        else:
            st.warning("Please enter some text to analyze")

with col2:
    st.subheader("📊 Quick Stats")
    if 'chunks' in st.session_state:
        chunks = st.session_state.chunks
        total_chunks = len(chunks)
        total_words = sum(len(chunk.split()) for chunk in chunks)
        
        st.metric("Total Chunks", total_chunks)
        st.metric("Total Words", total_words)
        st.metric("Avg Words/Chunk", f"{total_words/total_chunks:.1f}")
    else:
        st.info("Enter text and click Analyze to see statistics")

# --------------------- RESULTS DISPLAY ---------------------
if 'result' in st.session_state:
    result = st.session_state.result
    
    st.markdown("---")
    st.subheader("📋 Final Consistency Judgment")
    
    # Result Box with color coding
    result_class = "consistent" if result["binary_result"] == 1 else "contradictory"
    result_icon = "✅" if result["binary_result"] == 1 else "❌"
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div class="result-box {result_class}">
            <h3 style="text-align: center; margin: 0;">
                {result_icon} {result["result_label"]} ({result["binary_result"]})
            </h3>
            <p style="text-align: center; font-size: 1.2rem; margin: 0.5rem 0;">
                Normalized Score: <strong>{result["normalized_score"]:+.3f}</strong>
            </p>
            <p style="text-align: center; margin: 0;">
                {result["explanation"]}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Analysis Metrics
    st.subheader("📈 Analysis Metrics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <h4>⏱️ Processing Time</h4>
            <p style="font-size: 1.5rem; font-weight: bold; margin: 0;">
                {result["processing_time"]}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <h4>📄 Chunks Processed</h4>
            <p style="font-size: 1.5rem; font-weight: bold; margin: 0;">
                {result["chunks_processed"].split('(')[0].strip()}
            </p>
            <p style="margin: 0;">
                {result["chunks_processed"].split('(')[1].replace(')', '')}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <h4>🎯 Belief-Density</h4>
            <p style="font-size: 1.5rem; font-weight: bold; margin: 0;">
                {result["belief_density"]}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Belief-State Trajectory Visualization
    st.subheader("📊 Belief-State Trajectory")
    
    # Generate sample trajectory data
    if 'chunks' in st.session_state:
        chunks = st.session_state.chunks
        trajectory_data = []
        cumulative_score = 0
        
        for i, chunk in enumerate(chunks):
            # Simulate chunk scores (replace with actual scoring from your pipeline)
            chunk_score = (i % 3 - 1) * 0.2 + 0.1  # Sample pattern
            cumulative_score += chunk_score
            trajectory_data.append({
                "chunk": i+1,
                "chunk_score": chunk_score,
                "cumulative_score": cumulative_score / (i+1)
            })
        
        df_trajectory = pd.DataFrame(trajectory_data)
        
        # Create visualization
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(
                x=df_trajectory["chunk"],
                y=df_trajectory["cumulative_score"],
                mode="lines+markers",
                name="Cumulative Score",
                line=dict(color="#3B82F6", width=3)
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Bar(
                x=df_trajectory["chunk"],
                y=df_trajectory["chunk_score"],
                name="Chunk Score",
                marker_color="#10B981",
                opacity=0.6
            ),
            secondary_y=True
        )
        
        # Add decision thresholds
        fig.add_hline(
            y=decision_threshold,
            line_dash="dash",
            line_color="green",
            opacity=0.5,
            annotation_text="Consistent Threshold",
            secondary_y=False
        )
        
        fig.add_hline(
            y=-decision_threshold,
            line_dash="dash",
            line_color="red",
            opacity=0.5,
            annotation_text="Contradictory Threshold",
            secondary_y=False
        )
        
        fig.update_layout(
            title="Belief-State Trajectory Across Chunks",
            xaxis_title="Chunk Number",
            hovermode="x unified",
            height=400,
            showlegend=True
        )
        
        fig.update_yaxes(
            title_text="Cumulative Score",
            secondary_y=False,
            range=[-1, 1]
        )
        
        fig.update_yaxes(
            title_text="Chunk Score",
            secondary_y=True,
            range=[-1, 1]
        )
        
        # FIX: Replace use_container_width with width parameter
        st.plotly_chart(fig, use_container_width=True)  # Change to: st.plotly_chart(fig, width='stretch')
        
        # Chunk Details
        with st.expander("📋 View Processed Chunks"):
            for i, chunk in enumerate(chunks):
                st.markdown(f"**Chunk {i+1}** ({len(chunk.split())} words)")
                st.text(chunk[:200] + "..." if len(chunk) > 200 else chunk)
                st.markdown("---")
    
    # Download Results
    st.subheader("💾 Export Results")
    
    col1, col2 = st.columns(2)
    with col1:
        # JSON download
        json_str = json.dumps(result, indent=2)
        st.download_button(
            label="📥 Download JSON Results",
            data=json_str,
            file_name=f"bdh_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True  # Change to: width='stretch'
        )
    
    with col2:
        # CSV download for trajectory
        if 'chunks' in st.session_state:
            csv_data = pd.DataFrame({
                "chunk_number": range(1, len(chunks) + 1),
                "chunk_text": chunks,
                "word_count": [len(chunk.split()) for chunk in chunks]
            })
            csv_str = csv_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Chunks CSV",
                data=csv_str,
                file_name=f"bdh_chunks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True  # Change to: width='stretch'
            )

# --------------------- FOOTER ---------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; padding: 1rem;">
    <p>BDH Consistency Reasoner v1.0 • Powered by Binary Decision Logic</p>
    <p>Last analysis: {}</p>
</div>
""".format(st.session_state.get('result', {}).get('timestamp', 'Not yet analyzed')), 
unsafe_allow_html=True)
