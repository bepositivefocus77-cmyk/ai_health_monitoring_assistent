import os
import uuid
import requests
import streamlit as st
from datetime import datetime
from typing import TypedDict
import tempfile

# ─── PAGE CONFIG ────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Health Monitoring Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .risk-low    { background: #d4edda; border-left: 5px solid #28a745; padding: 1rem; border-radius: 8px; }
    .risk-medium { background: #fff3cd; border-left: 5px solid #ffc107; padding: 1rem; border-radius: 8px; }
    .risk-high   { background: #f8d7da; border-left: 5px solid #dc3545; padding: 1rem; border-radius: 8px; }
    .emergency-box {
        background: #dc3545;
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        font-weight: bold;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.7} }
    .step-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 6px rgba(0,0,0,.06);
    }
    .source-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: .75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: .4rem 0;
        font-size: .9rem;
    }
    .pdf-chunk {
        background: #e8f4fd;
        border: 1px solid #bee5f5;
        border-radius: 8px;
        padding: .75rem 1rem;
        margin: .3rem 0;
        font-size: .88rem;
    }
    .stButton>button {
        background: linear-gradient(135deg,#667eea,#764ba2);
        color: white; border: none;
        border-radius: 8px; padding: .6rem 1.8rem;
        font-weight: 600; width: 100%;
    }
    .stButton>button:hover { opacity: .9; }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ─────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏥 AI Health Monitoring Assistant</h1>
    <p>Powered by LangGraph · RAG Memory · PDF Medical Reports</p>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR – API KEYS & PATIENT ───────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    with st.expander("🔑 API Keys", expanded=True):
        groq_key     = st.text_input("Groq API Key",     type="password", value=os.getenv("GROQ_API_KEY", ""))
        serper_key   = st.text_input("Serper API Key",   type="password", value=os.getenv("SERPER_API_KEY", ""))
        pinecone_key = st.text_input("Pinecone API Key", type="password", value=os.getenv("PINECONE_API_KEY", ""))

    st.divider()
    st.header("👤 Patient")
    patient_id = st.text_input("Patient ID", value="patient_001")

    st.divider()
    st.header("📄 Upload Medical PDF")
    uploaded_pdf = st.file_uploader("Medical report / lab result", type=["pdf"])

    if uploaded_pdf:
        st.success(f"✅ {uploaded_pdf.name} loaded")

    st.divider()
    st.markdown("### 💡 Tips")
    st.info("Upload previous lab results or discharge summaries to enhance RAG context.")
    st.warning("⚠️ For informational use only. Always consult a doctor.")

# ─── LAZY-LOAD HEAVY DEPS ───────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Connecting to Pinecone…")
def get_pinecone_index(_pinecone_key):
    from pinecone import Pinecone, ServerlessSpec
    pc = Pinecone(api_key=_pinecone_key)
    index_name = "health-memory"
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name, dimension=384, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(index_name)

def get_llm(groq_key):
    from langchain_groq import ChatGroq
    return ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_key)

# ─── PDF EXTRACTION & CHUNKING ──────────────────────────────────────
def extract_pdf_text(pdf_file) -> str:
    """Extract text from uploaded PDF using pdfminer."""
    try:
        import pdfminer.high_level as pdfhl
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            tmp_path = tmp.name
        text = pdfhl.extract_text(tmp_path)
        os.unlink(tmp_path)
        return text or ""
    except Exception as e:
        return f"[PDF extraction failed: {e}]"

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 50]

# ─── RAG: EMBED & STORE PDF CHUNKS ──────────────────────────────────
def store_pdf_chunks(pdf_text: str, patient_id: str, pdf_name: str,
                     embedding_model, index) -> int:
    chunks = chunk_text(pdf_text)
    vectors = []
    for chunk in chunks:
        emb = embedding_model.encode(chunk).tolist()
        vectors.append({
           "id": str(uuid.uuid4()),
           "values": emb,
           "metadata": {"text": chunk, "patient_id": patient_id,
                        "source": pdf_name, "type": "pdf_report",
                        "time": str(datetime.now())}
})
    if vectors:
        index.upsert(vectors)
    return len(vectors)

# ─── RAG: RETRIEVE RELEVANT CONTEXT ─────────────────────────────────
def retrieve_context(query: str, patient_id: str, embedding_model, index,
                     top_k: int = 5) -> tuple[str, list[dict]]:
    q_emb = embedding_model.encode(query).tolist()
    results = index.query(
        vector=q_emb, top_k=top_k, include_metadata=True,
        filter={"patient_id": patient_id}
    )
    memories, sources = [], []
    for match in results.get("matches", []):
        meta = match.get("metadata", {})
        if "text" in meta:
            memories.append(meta["text"])
            sources.append({
                "text": meta["text"][:200] + "…",
                "source": meta.get("source", "memory"),
                "type": meta.get("type", "symptom_history"),
                "score": round(match.get("score", 0), 3)
            })
    return "\n\n---\n\n".join(memories) if memories else "No history found.", sources

# ─── WEB SEARCH ─────────────────────────────────────────────────────
def web_search(symptoms: str, serper_key: str) -> tuple[str, list[dict]]:
    try:
        url = "https://google.serper.dev/search"
        payload = {"q": symptoms + " medical symptoms causes treatment"}
        headers = {"X-API-KEY": serper_key, "Content-Type": "application/json"}
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        data = resp.json()
        search_text, search_sources = "", []
        for item in data.get("organic", [])[:3]:
            title   = item.get("title", "")
            snippet = item.get("snippet", "")
            link    = item.get("link", "")
            search_text += f"\nTITLE: {title}\nINFO: {snippet}\nSOURCE: {link}\n---\n"
            search_sources.append({"title": title, "snippet": snippet, "link": link})
        return search_text, search_sources
    except Exception as e:
        return f"Search failed: {e}", []

# ─── EMERGENCY DETECTION ─────────────────────────────────────────────
EMERGENCY_KEYWORDS = [
    "chest pain", "difficulty breathing", "blood vomiting", "unconscious",
    "stroke", "heart attack", "seizure", "suicidal", "severe bleeding", "fainting"
]

def check_emergency(symptoms: str) -> tuple[bool, str]:
    lower = symptoms.lower()
    for kw in EMERGENCY_KEYWORDS:
        if kw in lower:
            return True, f"🚨 EMERGENCY: '{kw}' detected — seek immediate medical help!"
    return False, "No emergency symptoms detected."

# ─── STORE INTERACTION IN PINECONE ──────────────────────────────────
def store_interaction(patient_id: str, symptoms: str, response: str,
                      embedding_model, index):
    text = f"DATE: {datetime.now()}\nPATIENT: {patient_id}\nSYMPTOMS: {symptoms}\nRESPONSE: {response}"
    emb = embedding_model.encode(text).tolist()
    index.upsert(vectors=[{
        "id": str(uuid.uuid4()),
        "values": emb,
        "metadata": {"text": text, "patient_id": patient_id,
                     "type": "symptom_history", "time": str(datetime.now())}
}])

# ─── MAIN ANALYSIS FUNCTION ──────────────────────────────────────────
def run_health_analysis(symptoms, patient_id, groq_key, serper_key,
                        pinecone_key, pdf_text=None, pdf_name=None):

    steps = []

    # Load resources
    try:
        embedding_model = load_embedding_model()
        index           = get_pinecone_index(pinecone_key)
        llm             = get_llm(groq_key)
    except Exception as e:
        st.error(f"Initialisation error: {e}")
        return None

    # Step 1 – PDF ingestion
    pdf_chunks_stored = 0
    if pdf_text and pdf_name:
        with st.spinner("📄 Indexing PDF into RAG…"):
            pdf_chunks_stored = store_pdf_chunks(pdf_text, patient_id,
                                                  pdf_name, embedding_model, index)
        steps.append(("📄 PDF Ingestion", f"Stored {pdf_chunks_stored} chunks from '{pdf_name}'"))

    # Step 2 – RAG retrieval
    with st.spinner("🔍 Retrieving memory & PDF context…"):
        rag_context, rag_sources = retrieve_context(symptoms, patient_id,
                                                     embedding_model, index)
    steps.append(("🧠 RAG Retrieval", f"Retrieved {len(rag_sources)} relevant records"))

    # Step 3 – Web search
    with st.spinner("🌐 Searching medical knowledge…"):
        search_text, search_sources = web_search(symptoms, serper_key)
    steps.append(("🌐 Web Search", f"Found {len(search_sources)} medical sources"))

    # Step 4 – Emergency check
    is_emergency, emergency_msg = check_emergency(symptoms)
    steps.append(("🚨 Emergency Check", emergency_msg))

    # Step 5 – Risk analysis
    with st.spinner("📊 Analysing risk level…"):
        risk_prompt = f"""Analyze these symptoms carefully.

Symptoms: {symptoms}

Previous History & PDF Context:
{rag_context}

Classify risk level as exactly one of: LOW / MEDIUM / HIGH
Then explain briefly in 2–3 sentences."""
        risk_response = llm.invoke(risk_prompt)
        risk_level = risk_response.content

    risk_label = "MEDIUM"
    for lvl in ["HIGH", "LOW", "MEDIUM"]:
        if lvl in risk_level.upper():
            risk_label = lvl
            break
    steps.append(("📊 Risk Analysis", f"Risk Level: **{risk_label}**"))

    # Step 6 – Final summary
    with st.spinner("💊 Generating health summary…"):
        summary_prompt = f"""You are an AI Health Monitoring Assistant.

RETRIEVED MEDICAL HISTORY & PDF REPORTS (RAG):
{rag_context}

CURRENT SYMPTOMS:
{symptoms}

MEDICAL SEARCH RESULTS:
{search_text}

EMERGENCY STATUS:
{emergency_msg}

RISK ANALYSIS:
{risk_level}

Provide a comprehensive but clearly structured response with:
1. **Possible Causes** – list 3–5 potential diagnoses
2. **Risk Explanation** – explain the risk level
3. **Health Safety Advice** – key precautions
4. **Home Care Suggestions** – practical remedies
5. **Doctor Recommendation** – when / which specialist
6. **Emergency Recommendation** – if applicable

End with: *This is not professional medical advice.*"""
        summary_response = llm.invoke(summary_prompt)
        final_response = summary_response.content

    steps.append(("✅ Summary Generated", "Health analysis complete"))

    # Step 7 – Store interaction
    with st.spinner("💾 Saving to memory…"):
        store_interaction(patient_id, symptoms, final_response,
                          embedding_model, index)

    return {
        "steps": steps,
        "rag_sources": rag_sources,
        "search_sources": search_sources,
        "is_emergency": is_emergency,
        "emergency_msg": emergency_msg,
        "risk_label": risk_label,
        "risk_level": risk_level,
        "final_response": final_response,
        "pdf_chunks": pdf_chunks_stored,
    }

# ─── MAIN UI ─────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("🩺 Enter Symptoms")
    symptoms_input = st.text_area(
        "Describe your symptoms in detail:",
        placeholder="e.g. fever, headache, sore throat, body aches since 2 days…",
        height=160,
    )

    st.subheader("📋 Additional Context")
    age        = st.number_input("Age", 1, 120, 30)
    gender     = st.selectbox("Gender", ["Not specified", "Male", "Female", "Other"])
    conditions = st.text_input("Known conditions (optional)", placeholder="e.g. diabetes, hypertension")
    medications = st.text_input("Current medications (optional)")

    if conditions or medications or age:
        extra = f"\nPatient: {age}y {gender}"
        if conditions:  extra += f", known: {conditions}"
        if medications: extra += f", meds: {medications}"
    else:
        extra = ""

    analyze_btn = st.button("🔬 Analyse Symptoms", use_container_width=True)

with col2:
    st.subheader("ℹ️ How It Works")
    st.markdown("""
    **This assistant runs a 7-step pipeline:**

    | Step | Action |
    |------|--------|
    | 📄 | Ingest uploaded PDF into RAG |
    | 🧠 | Retrieve past history from Pinecone |
    | 🌐 | Search current medical information |
    | 🚨 | Detect emergency symptoms |
    | 📊 | LLM risk classification |
    | 💊 | Generate health summary |
    | 💾 | Store interaction for future memory |
    """)

    if not all([groq_key, serper_key, pinecone_key]):
        st.warning("⚠️ Please fill in all API keys in the sidebar to run analysis.")

st.divider()

# ─── ANALYSIS TRIGGER ────────────────────────────────────────────────
if analyze_btn:
    if not symptoms_input.strip():
        st.error("Please describe your symptoms first.")
    elif not all([groq_key, serper_key, pinecone_key]):
        st.error("Please enter all API keys in the sidebar.")
    else:
        # Handle PDF
        pdf_text, pdf_name = None, None
        if uploaded_pdf:
            with st.spinner("Extracting PDF text…"):
                uploaded_pdf.seek(0)
                pdf_text = extract_pdf_text(uploaded_pdf)
                pdf_name = uploaded_pdf.name

        full_symptoms = symptoms_input + extra

        result = run_health_analysis(
            full_symptoms, patient_id,
            groq_key, serper_key, pinecone_key,
            pdf_text, pdf_name
        )

        if result:
            # ── EMERGENCY ALERT ──────────────────────────────────────
            if result["is_emergency"]:
                st.markdown(f'<div class="emergency-box">🚨 {result["emergency_msg"]}</div>',
                            unsafe_allow_html=True)
                st.markdown("")

            # ── PIPELINE STEPS ────────────────────────────────────────
            with st.expander("⚙️ Pipeline Execution Steps", expanded=False):
                for icon_title, detail in result["steps"]:
                    st.markdown(f'<div class="step-card"><b>{icon_title}</b><br>{detail}</div>',
                                unsafe_allow_html=True)

            # ── RISK BADGE ────────────────────────────────────────────
            risk_css = {"LOW": "risk-low", "MEDIUM": "risk-medium", "HIGH": "risk-high"}
            risk_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
            css_class = risk_css.get(result["risk_label"], "risk-medium")
            emoji     = risk_emoji.get(result["risk_label"], "🟡")
            st.markdown(
                f'<div class="{css_class}"><h4>{emoji} Risk Level: {result["risk_label"]}</h4>'
                f'<p>{result["risk_level"]}</p></div>',
                unsafe_allow_html=True
            )
            st.markdown("")

            # ── TABS: Summary / RAG / Web Sources ────────────────────
            tab1, tab2, tab3 = st.tabs(["💊 Health Summary", "🧠 RAG Context", "🌐 Web Sources"])

            with tab1:
                st.markdown(result["final_response"])
                st.download_button(
                    "📥 Download Report",
                    data=f"AI HEALTH REPORT\nPatient: {patient_id}\nDate: {datetime.now()}\n\n"
                         f"SYMPTOMS:\n{full_symptoms}\n\nRISK: {result['risk_label']}\n\n"
                         f"{result['final_response']}",
                    file_name=f"health_report_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                )

            with tab2:
                st.subheader("Retrieved RAG Records")
                if result["rag_sources"]:
                    for src in result["rag_sources"]:
                        badge = "📄" if src["type"] == "pdf_report" else "💬"
                        st.markdown(
                            f'<div class="pdf-chunk">'
                            f'<b>{badge} [{src["type"]}]</b> '
                            f'<span style="color:#888;font-size:.8rem">score: {src["score"]}</span><br>'
                            f'{src["text"]}<br>'
                            f'<i style="font-size:.8rem;color:#666">source: {src["source"]}</i>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No previous records found for this patient.")

                if result["pdf_chunks"] > 0:
                    st.success(f"✅ {result['pdf_chunks']} PDF chunks indexed into Pinecone")

            with tab3:
                st.subheader("Medical Web Sources")
                for src in result["search_sources"]:
                    st.markdown(
                        f'<div class="source-card">'
                        f'<b>{src["title"]}</b><br>'
                        f'{src["snippet"]}<br>'
                        f'<a href="{src["link"]}" target="_blank">{src["link"]}</a>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
