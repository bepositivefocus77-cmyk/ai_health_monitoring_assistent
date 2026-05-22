# 🏥 AI Health Monitoring Assistant — Streamlit App

A production-ready upgrade of the Jupyter notebook, with:
- **Streamlit UI** — clean sidebar config, multi-tab results
- **PDF Report Analysis** — upload lab results / discharge summaries; text is chunked and indexed into Pinecone
- **Proper RAG** — semantic retrieval over both past symptom history *and* PDF content, filtered per patient

---

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file (or enter keys in the sidebar):

```
GROQ_API_KEY=gsk_...
SERPER_API_KEY=...
PINECONE_API_KEY=pcsk_...
```

Run the app:

```bash
streamlit run app.py
```

---

## Architecture

```
User Input (symptoms + optional PDF)
        │
        ▼
[PDF Ingestion] ─── pdfminer extract ──► chunk ──► embed ──► Pinecone upsert
        │
        ▼
[RAG Retrieval] ─── embed query ──► Pinecone query (patient_id filter)
        │           returns: past symptoms + PDF chunks (ranked by cosine similarity)
        ▼
[Web Search] ─── Serper API ──► top-3 medical articles
        │
        ▼
[Emergency Detection] ─── keyword scan
        │
        ▼
[Risk Analysis] ─── Groq LLM (llama-3.3-70b)
        │
        ▼
[Summary Agent] ─── Groq LLM with full context
        │
        ▼
[Memory Store] ─── embed & upsert interaction to Pinecone
```

---

## RAG Design

| Chunk source | Metadata `type` | Stored when |
|---|---|---|
| PDF upload | `pdf_report` | PDF uploaded + Analyse clicked |
| Symptom interaction | `symptom_history` | Every successful analysis |

All vectors filtered by `patient_id` so each patient only retrieves their own data.

Embedding model: `all-MiniLM-L6-v2` (384-dim, cosine similarity)
