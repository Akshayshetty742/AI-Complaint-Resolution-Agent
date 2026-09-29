# AI Complaint Resolution Agent Using RAG and Prompt Optimization

An intelligent, policy-grounded AI agent that analyzes customer complaints, retrieves relevant company policies via semantic vector search (FAISS + Sentence Transformers), and generates policy-compliant resolutions and customer responses without hallucination.

---

## 🌟 Key Features

1. **Complaint Analysis**: Classifies complaints into categories (*Refund, Return, Delivery, Damaged Product, Warranty, Payment, General Support*), detects sentiment (*Positive, Neutral, Negative*), and priority level (*Low, Medium, High*).
2. **RAG Knowledge Base**: 7 fictional company policies covering refunds, returns, deliveries, damaged items, warranty, billing, and customer support.
3. **FAISS Vector Retrieval**: Local semantic search powered by `sentence-transformers` (`all-MiniLM-L6-v2`) with cosine similarity.
4. **Prompt Optimization**: Supports 5 prompt engineering patterns (`zero_shot`, `role_based`, `few_shot`, `structured_rag`, `optimized`) with strict policy grounding and anti-hallucination guardrails.
5. **Interactive Streamlit UI**: Professional interface displaying complaint breakdown, retrieved policy chunks, resolutions, customer-facing messages, and technical metrics.

---

## 🏗️ Architecture

```
Customer Complaint
        │
        ▼
[Complaint Analysis] ──► Category / Sentiment / Priority
        │
        ▼
[FAISS Vector Store] ──► Top-K Policy Chunks (Local Embeddings)
        │
        ▼
[Prompt Optimization] ──► Context + Complaint + Grounding Rules
        │
        ▼
[OpenAI-Compatible LLM] ──► Structured Output (Resolution + Customer Response)
        │
        ▼
[Streamlit Interface] ──► Interactive UI & Technical Details
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)
Copy `.env.example` to `.env` and add your API key:
```bash
cp .env.example .env
```
Configure your credentials in `.env`:
```env
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-3.5-turbo
```
*(Note: If no API key is provided, the application operates gracefully in offline/retrieval mode with local classifications.)*

### 3. Run Tests
```bash
python tests/test_retrieval.py
python tests/test_basic.py
```

### 4. Launch Streamlit Application
```bash
streamlit run app.py
```
