import streamlit as st
import os
import pandas as pd
from src.complaint_agent import ComplaintResolutionAgent
from src.llm import LLMClient
from src.prompts import PROMPT_NAMES
from src.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    EMBEDDING_MODEL,
    RETRIEVAL_TOP_K
)
from evaluation.evaluate import run_benchmark

# Page Configuration
st.set_page_config(
    page_title="AI Complaint Resolution Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End SaaS Dark Theme CSS
st.markdown("""
<style>
    /* Dark Theme Core */
    .stApp {
        background-color: #0B0F17;
        color: #E2E8F0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Global Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }
    
    /* Top Header Bar */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 20px;
        background: linear-gradient(90deg, #111827 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
    }
    .top-nav-left h1 {
        font-size: 1.35rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 0;
        letter-spacing: -0.01em;
    }
    .top-nav-left p {
        font-size: 0.8rem;
        color: #94A3B8;
        margin: 2px 0 0 0;
    }
    .top-nav-right {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .status-badge-online {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border: 1px solid rgba(52, 211, 153, 0.25);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        background-color: #34D399;
        border-radius: 50%;
        box-shadow: 0 0 8px #34D399;
    }
    .chip-badge {
        background: rgba(56, 189, 248, 0.12);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.25);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }

    /* Hero Section */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFFFFF;
        margin-bottom: 4px;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.0rem;
        color: #94A3B8;
        margin-bottom: 20px;
        line-height: 1.5;
    }

    /* KPI Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    .kpi-card {
        background: #131D31;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #38BDF8, #2563EB);
    }
    .kpi-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #64748B;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.45rem;
        font-weight: 800;
        color: #F8FAFC;
    }
    .kpi-sub {
        font-size: 0.75rem;
        color: #38BDF8;
        font-weight: 500;
    }

    /* Modern SaaS Container Cards */
    .saas-card {
        background: #131D31;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.28);
    }
    .card-header-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #1E293B;
        padding-bottom: 12px;
        margin-bottom: 16px;
    }
    .card-title-text {
        font-size: 0.88rem;
        font-weight: 700;
        color: #E2E8F0;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Pipeline Visualization */
    .pipeline-wrapper {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #0B132B;
        border: 1px solid #1E2D4D;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 24px;
        overflow-x: auto;
    }
    .pipeline-step {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        min-width: 80px;
    }
    .step-circle {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .step-active {
        background: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        border: 1.5px solid #38BDF8;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.35);
    }
    .step-done {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1.5px solid #34D399;
    }
    .step-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #94A3B8;
        letter-spacing: 0.02em;
    }
    .pipeline-arrow {
        color: #334155;
        font-weight: 700;
        font-size: 1.05rem;
        margin: 0 4px;
        margin-bottom: 14px;
    }

    /* Metric Badges for Category, Sentiment, Priority */
    .badge-card {
        background: #111C33;
        border: 1px solid #1E2E4E;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .badge-card-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #64748B;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .badge-pill {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.95rem;
        font-weight: 700;
    }
    .pill-category { background: rgba(56, 189, 248, 0.12); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.3); }
    .pill-sentiment-pos { background: rgba(52, 211, 153, 0.12); color: #34D399; border: 1px solid rgba(52, 211, 153, 0.3); }
    .pill-sentiment-neu { background: rgba(148, 163, 184, 0.12); color: #CBD5E1; border: 1px solid rgba(148, 163, 184, 0.3); }
    .pill-sentiment-neg { background: rgba(248, 113, 113, 0.12); color: #F87171; border: 1px solid rgba(248, 113, 113, 0.3); }
    
    .pill-priority-high { background: rgba(239, 68, 68, 0.15); color: #F87171; border: 1px solid #EF4444; box-shadow: 0 0 10px rgba(239, 68, 68, 0.2); }
    .pill-priority-med { background: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid #F59E0B; }
    .pill-priority-low { background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid #10B981; }

    /* Policy Context Box */
    .policy-box {
        background: #0B1325;
        border: 1px solid #1E2D4A;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
        font-size: 0.85rem;
        line-height: 1.55;
        color: #CBD5E1;
    }
    .policy-tag {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        color: #818CF8;
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 8px;
    }

    /* Response / Resolution Boxes */
    .resolution-box {
        background: #0D1B2A;
        border-left: 3px solid #38BDF8;
        padding: 16px;
        border-radius: 0 8px 8px 0;
        color: #E2E8F0;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    .response-box {
        background: #0D211A;
        border-left: 3px solid #34D399;
        padding: 18px;
        border-radius: 0 8px 8px 0;
        color: #E2E8F0;
        font-size: 0.94rem;
        line-height: 1.65;
        margin-bottom: 12px;
    }

    /* Status Cards in Sidebar */
    .status-list {
        background: #0B1222;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 12px 14px;
        margin-top: 10px;
    }
    .status-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.75rem;
        padding: 4px 0;
        border-bottom: 1px solid #152238;
    }
    .status-row:last-child {
        border-bottom: none;
    }
    .status-lbl {
        color: #94A3B8;
        font-weight: 500;
    }
    .status-val-ready {
        color: #34D399;
        font-weight: 700;
    }
    .status-val-live {
        color: #38BDF8;
        font-weight: 700;
    }
    .status-val-fallback {
        color: #FBBF24;
        font-weight: 700;
    }

    /* Mode Banners */
    .banner-live {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(52, 211, 153, 0.25);
        color: #34D399;
        padding: 8px 14px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .banner-fallback {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.25);
        color: #FBBF24;
        padding: 8px 14px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 1. SIDEBAR REDESIGN
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding-bottom: 8px; border-bottom: 1px solid #1E293B; margin-bottom: 16px;">
        <div style="font-size: 0.7rem; font-weight: 800; color: #38BDF8; letter-spacing: 0.1em; text-transform: uppercase;">AI SUPPORT</div>
        <div style="font-size: 1.15rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.01em;">INTELLIGENCE</div>
    </div>
    """, unsafe_allow_html=True)

    # Section: Configuration
    st.markdown('<div style="font-size: 0.72rem; font-weight: 700; color: #64748B; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 8px;">CONFIGURATION</div>', unsafe_allow_html=True)
    with st.expander("LLM Provider", expanded=False):
        sidebar_api_key = st.text_input(
            "API Key",
            value=os.getenv("LLM_API_KEY", ""),
            type="password",
            help="OpenAI-compatible API key. Leave blank for Local Fallback mode."
        )
        sidebar_base_url = st.text_input(
            "Base URL",
            value=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        )
        sidebar_model = st.text_input(
            "Model",
            value=os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        )

    # Section: Prompt Engineering
    st.markdown('<div style="font-size: 0.72rem; font-weight: 700; color: #64748B; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 14px; margin-bottom: 8px;">PROMPT ENGINEERING</div>', unsafe_allow_html=True)
    selected_prompt = st.selectbox(
        "Prompt Template",
        options=PROMPT_NAMES,
        index=0,
        help="Select prompt template. 'optimized' includes anti-hallucination guardrails."
    )
    st.markdown(f'<div style="font-size: 0.72rem; color: #94A3B8; margin-top: -6px; margin-bottom: 10px;">Active: <span style="color:#38BDF8; font-weight:600;">{selected_prompt}</span></div>', unsafe_allow_html=True)

    # Section: Retrieval
    st.markdown('<div style="font-size: 0.72rem; font-weight: 700; color: #64748B; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 14px; margin-bottom: 8px;">RETRIEVAL</div>', unsafe_allow_html=True)
    top_k = st.slider("Top-K Chunks", min_value=1, max_value=5, value=RETRIEVAL_TOP_K)
    st.markdown(f'<div style="font-size: 0.72rem; color: #64748B; margin-top: -6px; margin-bottom: 10px;">Embedder: <code>{EMBEDDING_MODEL}</code></div>', unsafe_allow_html=True)

    # Section: Demo
    st.markdown('<div style="font-size: 0.72rem; font-weight: 700; color: #64748B; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 14px; margin-bottom: 8px;">DEMO COMPLAINTS</div>', unsafe_allow_html=True)
    sample_complaints = {
        "Select a category demo...": "",
        "Refund": "I cancelled my order five days ago but I still haven't received my refund.",
        "Return": "Can I return these shoes within the allowed return period?",
        "Delivery": "My order was supposed to arrive yesterday. When will it be delivered?",
        "Damaged Product": "My headphones arrived damaged yesterday and I want a replacement.",
        "Warranty": "My laptop stopped working after two months. Is it covered under warranty?",
        "Payment": "My payment was deducted but my order was not confirmed.",
        "General Support": "I need help changing the address associated with my account."
    }
    chosen_sample = st.selectbox("Load Sample", list(sample_complaints.keys()), label_visibility="collapsed")

    # Section: System Status
    st.markdown('<div style="font-size: 0.72rem; font-weight: 700; color: #64748B; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 16px; margin-bottom: 6px;">SYSTEM STATUS</div>', unsafe_allow_html=True)
    is_live = bool((sidebar_api_key or os.getenv("LLM_API_KEY", "")).strip() and (sidebar_api_key or os.getenv("LLM_API_KEY", "")) != "your_api_key_here")
    llm_status_val = '<span class="status-val-live">LIVE</span>' if is_live else '<span class="status-val-fallback">FALLBACK</span>'

    st.markdown(f"""
    <div class="status-list">
        <div class="status-row">
            <span class="status-lbl">● RAG Engine</span>
            <span class="status-val-ready">READY</span>
        </div>
        <div class="status-row">
            <span class="status-lbl">● Vector Store</span>
            <span class="status-val-ready">READY</span>
        </div>
        <div class="status-row">
            <span class="status-lbl">● Classifier</span>
            <span class="status-val-ready">READY</span>
        </div>
        <div class="status-row">
            <span class="status-lbl">● LLM Engine</span>
            {llm_status_val}
        </div>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------
# 2. TOP NAVIGATION / HEADER
# -------------------------------------------------------------
st.markdown("""
<div class="top-nav">
    <div class="top-nav-left">
        <h1>AI Complaint Resolution Agent</h1>
        <p>RAG-powered Customer Support Intelligence</p>
    </div>
    <div class="top-nav-right">
        <div class="status-badge-online">
            <div class="status-dot"></div>
            SYSTEM ONLINE
        </div>
        <div class="chip-badge">RAG + LLM</div>
    </div>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# 3. HERO & KPI METRICS
# -------------------------------------------------------------
st.markdown('<div class="hero-title">Resolve customer complaints with AI.</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Retrieve trusted policy context, analyze complaints, and generate policy-grounded responses in real-time.</div>', unsafe_allow_html=True)

# 3 KPI Cards
st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">RAG Status</div>
        <div class="kpi-value" style="color: #34D399;">READY</div>
        <div class="kpi-sub">FAISS Cosine Similarity Active</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Policy Documents</div>
        <div class="kpi-value">7</div>
        <div class="kpi-sub">Fully Chunked & Indexed</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Retrieval Top-K</div>
        <div class="kpi-value">{top_k}</div>
        <div class="kpi-sub">MiniLM-L6 Embeddings</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Mode Banner
if is_live:
    st.markdown('<div class="banner-live"><span>🟢</span> <b>LIVE AI MODE:</b> Policy-grounded resolution generation is active via OpenAI-compatible endpoint.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="banner-fallback"><span>🟡</span> <b>LOCAL FALLBACK MODE:</b> RAG retrieval and local complaint classification are active. Connect an LLM API to enable live AI-generated resolutions.</div>', unsafe_allow_html=True)


# -------------------------------------------------------------
# 4. COMPLAINT INPUT SECTION
# -------------------------------------------------------------
initial_text = sample_complaints.get(chosen_sample, "") if chosen_sample != "Select a category demo..." else ""

st.markdown("""
<div style="background: #131D31; border: 1px solid #1E293B; border-radius: 12px 12px 0 0; padding: 14px 20px 6px 20px;">
    <div style="font-size: 0.82rem; font-weight: 700; color: #94A3B8; letter-spacing: 0.06em; text-transform: uppercase;">
        CUSTOMER COMPLAINT
    </div>
</div>
""", unsafe_allow_html=True)

complaint_input = st.text_area(
    "Customer Complaint Input",
    value=initial_text,
    height=125,
    placeholder="Describe the customer's issue...",
    label_visibility="collapsed"
)

char_count = len(complaint_input.strip())
col_chars, col_res_btn = st.columns([3, 1])
with col_chars:
    st.markdown(f'<div style="font-size: 0.78rem; color: #64748B; padding-top: 8px;">Length: <b>{char_count}</b> characters</div>', unsafe_allow_html=True)
with col_res_btn:
    resolve_clicked = st.button("⚡ Resolve Complaint", type="primary", use_container_width=True)


# -------------------------------------------------------------
# 5. RESOLUTION WORKFLOW & DASHBOARD RESULTS
# -------------------------------------------------------------
if resolve_clicked:
    if not complaint_input.strip():
        st.warning("Please enter a customer complaint to analyze.")
    else:
        with st.spinner("Executing RAG semantic search and policy-grounded resolution..."):
            try:
                llm = LLMClient(
                    api_key=sidebar_api_key or os.getenv("LLM_API_KEY"),
                    base_url=sidebar_base_url,
                    model=sidebar_model
                )
                agent = ComplaintResolutionAgent(llm_client=llm)
                agent.retriever.top_k = top_k
                result = agent.resolve(complaint_input, prompt_type=selected_prompt)

                category = result.get("category", "General Support")
                sentiment = result.get("sentiment", "Neutral")
                priority = result.get("priority", "Medium")
                chunks = result.get("retrieved_chunks", [])
                top_doc = chunks[0]["source"] if chunks else "N/A"
                top_score = chunks[0]["score"] if chunks else 0.0

                # ---------------------------------------------------------
                # 5. ANALYSIS CARDS
                # ---------------------------------------------------------
                st.markdown('<div style="font-size: 0.88rem; font-weight: 700; color: #E2E8F0; letter-spacing: 0.05em; text-transform: uppercase; margin: 24px 0 12px 0;">COMPLAINT ANALYSIS</div>', unsafe_allow_html=True)
                
                # Priority Pill styling
                p_lower = priority.lower()
                if p_lower == "high":
                    priority_html = f'<span class="badge-pill pill-priority-high">● {priority.upper()}</span>'
                elif p_lower == "medium":
                    priority_html = f'<span class="badge-pill pill-priority-med">● {priority.upper()}</span>'
                else:
                    priority_html = f'<span class="badge-pill pill-priority-low">● {priority.upper()}</span>'

                # Sentiment Pill styling
                s_lower = sentiment.lower()
                if "pos" in s_lower:
                    sentiment_html = f'<span class="badge-pill pill-sentiment-pos">{sentiment}</span>'
                elif "neg" in s_lower:
                    sentiment_html = f'<span class="badge-pill pill-sentiment-neg">{sentiment}</span>'
                else:
                    sentiment_html = f'<span class="badge-pill pill-sentiment-neu">{sentiment}</span>'

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"""
                    <div class="badge-card">
                        <div class="badge-card-label">Category</div>
                        <span class="badge-pill pill-category">{category}</span>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="badge-card">
                        <div class="badge-card-label">Sentiment</div>
                        {sentiment_html}
                    </div>
                    """, unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""
                    <div class="badge-card">
                        <div class="badge-card-label">Priority</div>
                        {priority_html}
                    </div>
                    """, unsafe_allow_html=True)

                # ---------------------------------------------------------
                # 6. AI RESOLUTION PIPELINE VISUALIZATION
                # ---------------------------------------------------------
                st.markdown('<div style="font-size: 0.88rem; font-weight: 700; color: #E2E8F0; letter-spacing: 0.05em; text-transform: uppercase; margin: 24px 0 12px 0;">AI RESOLUTION PIPELINE</div>', unsafe_allow_html=True)
                st.markdown("""
                <div class="pipeline-wrapper">
                    <div class="pipeline-step">
                        <div class="step-circle step-done">✓</div>
                        <div class="step-label">Complaint</div>
                    </div>
                    <div class="pipeline-arrow">→</div>
                    <div class="pipeline-step">
                        <div class="step-circle step-done">✓</div>
                        <div class="step-label">Analyze</div>
                    </div>
                    <div class="pipeline-arrow">→</div>
                    <div class="pipeline-step">
                        <div class="step-circle step-done">✓</div>
                        <div class="step-label">Retrieve</div>
                    </div>
                    <div class="pipeline-arrow">→</div>
                    <div class="pipeline-step">
                        <div class="step-circle step-done">✓</div>
                        <div class="step-label">Ground</div>
                    </div>
                    <div class="pipeline-arrow">→</div>
                    <div class="pipeline-step">
                        <div class="step-circle step-done">✓</div>
                        <div class="step-label">Generate</div>
                    </div>
                    <div class="pipeline-arrow">→</div>
                    <div class="pipeline-step">
                        <div class="step-circle step-active">⚡</div>
                        <div class="step-label">Response</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ---------------------------------------------------------
                # 7. RETRIEVED POLICY SECTION
                # ---------------------------------------------------------
                st.markdown(f"""
                <div class="saas-card">
                    <div class="card-header-bar">
                        <div class="card-title-text">
                            <span>📄</span> RETRIEVED POLICY CONTEXT
                        </div>
                        <div style="font-size: 0.78rem; color: #94A3B8;">
                            Primary Source: <code style="color:#38BDF8;">{top_doc}</code> | Cosine Similarity: <b style="color:#34D399;">{top_score:.3f}</b>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                if not chunks:
                    st.write("No matching policy chunks found.")
                else:
                    for i, chunk in enumerate(chunks, 1):
                        source = chunk.get("source", "Unknown Document")
                        score = chunk.get("score", 0.0)
                        text = chunk.get("text", "")
                        with st.expander(f"Chunk {i}: {source} (Similarity: {score:.3f})", expanded=(i == 1)):
                            st.markdown(f'<span class="policy-tag">RAG SOURCE: {source}</span>', unsafe_allow_html=True)
                            st.markdown(f'<div class="policy-box">{text}</div>', unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

                # ---------------------------------------------------------
                # 8. AI-GENERATED RESOLUTION
                # ---------------------------------------------------------
                resolution_text = result.get("resolution", "No resolution generated.")
                st.markdown(f"""
                <div class="saas-card">
                    <div class="card-header-bar">
                        <div class="card-title-text">
                            <span>🛡️</span> AI-GENERATED RESOLUTION
                        </div>
                        <span class="chip-badge" style="border-color:#38BDF8; color:#38BDF8;">POLICY GROUNDED</span>
                    </div>
                    <div class="resolution-box">
                        {resolution_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ---------------------------------------------------------
                # 9. CUSTOMER RESPONSE
                # ---------------------------------------------------------
                cust_response = result.get("customer_response", "No customer response generated.")
                st.markdown(f"""
                <div class="saas-card">
                    <div class="card-header-bar">
                        <div class="card-title-text">
                            <span>💬</span> CUSTOMER RESPONSE
                        </div>
                        <span style="font-size: 0.75rem; color: #64748B;">Ready for dispatch</span>
                    </div>
                    <div class="response-box">
                        {cust_response}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Copy Helper / Raw text view
                with st.expander("Copy Response Text", expanded=False):
                    st.code(cust_response, language=None)

                # ---------------------------------------------------------
                # 10. TECHNICAL DETAILS (EXPANDABLE)
                # ---------------------------------------------------------
                with st.expander("TECHNICAL DETAILS", expanded=False):
                    tech = result.get("technical_details", {})
                    st.markdown(f"""
                    - **Embedding Model:** `{tech.get('embedding_model', EMBEDDING_MODEL)}`
                    - **Vector Store:** `FAISS (IndexFlatIP Cosine Similarity)`
                    - **Retrieved Chunks:** `{tech.get('retrieved_chunks_count', len(chunks))}`
                    - **Prompt Template:** `{tech.get('prompt_type', selected_prompt)}`
                    - **LLM Model:** `{tech.get('llm_model', sidebar_model)}`
                    - **Execution Mode:** `{tech.get('execution_mode', 'Local Fallback')}`
                    """)

            except Exception as e:
                st.error(f"Error processing complaint: {str(e)}")


# -------------------------------------------------------------
# 11. PROMPT ENGINEERING SECTION (EXPANDABLE)
# -------------------------------------------------------------
with st.expander("PROMPT ENGINEERING DETAILS", expanded=False):
    st.markdown(f"**Currently Active Template:** `{selected_prompt}`")
    prompt_explanations = {
        "optimized": "Optimized prompt combines role instructions, retrieved policy context, pre-classification metadata, structured JSON output requirements, and strict anti-hallucination constraints.",
        "structured_rag": "Structured RAG directly couples policy documents with strict JSON schema instructions.",
        "few_shot": "Few-shot prompt supplies domain demonstration examples of complaint-to-JSON resolutions.",
        "role_based": "Role-based prompt establishes an empathetic Senior Customer Support Manager persona.",
        "zero_shot": "Zero-shot direct instruction prompt without demonstrations or persona framing."
    }
    st.info(prompt_explanations.get(selected_prompt, "Standard resolution prompt template."))


# -------------------------------------------------------------
# 12. MODEL & PROMPT EVALUATION SECTION
# -------------------------------------------------------------
with st.expander("MODEL & PROMPT EVALUATION", expanded=False):
    st.markdown("Run automated evaluation benchmark over the 21 representative test complaints covering all 7 categories.")
    
    col_bench_btn, _ = st.columns([1, 2])
    with col_bench_btn:
        run_eval_clicked = st.button("Run Benchmark Evaluation", use_container_width=True)

    if run_eval_clicked:
        with st.spinner(f"Running benchmark on 21 test complaints with template '{selected_prompt}'..."):
            llm_eval = LLMClient(
                api_key=sidebar_api_key or os.getenv("LLM_API_KEY"),
                base_url=sidebar_base_url,
                model=sidebar_model
            )
            agent_eval = ComplaintResolutionAgent(llm_client=llm_eval)
            agent_eval.retriever.top_k = top_k
            bench_results = run_benchmark(agent=agent_eval, prompt_type=selected_prompt)

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Category Match Rate", f"{bench_results['category_match_rate']:.1f}%", f"{bench_results['category_matches']}/{bench_results['total_complaints']}")
            with m2:
                st.metric("Retrieval Match Rate", f"{bench_results['retrieval_match_rate']:.1f}%", f"{bench_results['retrieval_matches']}/{bench_results['total_complaints']}")
            with m3:
                st.metric("Valid Structured JSON Rate", f"{bench_results['valid_json_rate']:.1f}%", f"{bench_results['valid_json_count']}/{bench_results['total_complaints']}")

            # Detailed table
            df = pd.DataFrame(bench_results["detailed_results"])
            st.dataframe(df[["id", "complaint", "expected_category", "predicted_category", "category_match", "expected_policy", "retrieved_policy", "policy_match"]], use_container_width=True)
    else:
        st.markdown("*Click 'Run Benchmark Evaluation' to run the 21-case test dataset.*")
