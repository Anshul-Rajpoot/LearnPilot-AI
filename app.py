from __future__ import annotations

import html

import streamlit as st

from agent.graph import LearnPilotAI
from config import DEFAULT_TOP_K, EMBEDDINGS_PATH, MAX_TOP_K
from llm.ollama import OllamaError, generate, is_available, stream
from retrieval.retriever import Retriever


st.set_page_config(page_title="LearnPilot AI", page_icon="🎓", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --bg: #07111f;
        --panel: rgba(15, 23, 42, 0.7);
        --panel-2: rgba(15, 23, 42, 0.92);
        --soft: rgba(148, 163, 184, 0.12);
        --card: rgba(15, 23, 42, 0.78);
        --line: rgba(148, 163, 184, 0.18);
        --text: #e2e8f0;
        --muted: #94a3b8;
        --brand: #7dd3fc;
        --brand-2: #a78bfa;
        --success: #34d399;
        --warning: #fbbf24;
        --shadow: 0 20px 45px rgba(15, 23, 42, 0.38);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(125, 211, 252, 0.16), transparent 26%),
            radial-gradient(circle at bottom right, rgba(167, 139, 250, 0.14), transparent 28%),
            var(--bg);
        color: var(--text);
    }

    .main .block-container {
        max-width: 1240px;
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: rgba(8, 15, 29, 0.9);
        border-right: 1px solid var(--line);
    }

    .sidebar-title {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin-bottom: 0.15rem;
    }

    .sidebar-subtitle {
        color: var(--muted);
        font-size: 0.9rem;
    }

    .glass-card {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.78), rgba(15, 23, 42, 0.66));
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 1.2rem 1.1rem;
        box-shadow: var(--shadow);
        backdrop-filter: blur(12px);
    }

    .hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, rgba(125, 211, 252, 0.12), rgba(167, 139, 250, 0.16));
        border: 1px solid rgba(125, 211, 252, 0.15);
        border-radius: 24px;
        padding: 1.5rem 1.5rem 1.15rem;
        margin-bottom: 1.2rem;
    }

    .hero::after {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(120deg, transparent, rgba(255,255,255,0.03), transparent);
        pointer-events: none;
    }

    .hero h1 {
        margin: 0;
        font-size: clamp(2.2rem, 3vw, 3.2rem);
        letter-spacing: -0.06em;
        line-height: 1.06;
        color: white;
        font-weight: 800;
    }

    .hero p {
        color: var(--muted);
        margin: 0.45rem 0 0;
        font-size: 1rem;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0.8rem;
        border-radius: 999px;
        background: rgba(125, 211, 252, 0.12);
        border: 1px solid rgba(125, 211, 252, 0.22);
        color: #d8f3ff;
        font-size: 0.8rem;
        font-weight: 700;
    }

    .stat-card {
        background: linear-gradient(180deg, rgba(15,23,42,0.8), rgba(15,23,42,0.75));
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1rem 1rem 0.9rem;
        box-shadow: var(--shadow);
        height: 100%;
    }

    .stat-card .stMetric {
        background: transparent;
        border: none;
        padding: 0;
    }

    .prompt-button {
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.75);
        color: var(--text);
        padding: 0.55rem 0.85rem;
        margin: 0.2rem 0.35rem 0.2rem 0;
        transition: 0.2s ease;
    }

    .prompt-button:hover {
        border-color: rgba(125, 211, 252, 0.4);
        transform: translateY(-1px);
        box-shadow: 0 8px 22px rgba(125, 211, 252, 0.18);
    }

    .source {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.92), rgba(15, 23, 42, 0.72));
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1rem 1rem 0.9rem;
        margin: 0.8rem 0;
        box-shadow: 0 12px 24px rgba(15, 23, 42, 0.14);
    }

    .source-title {
        font-weight: 700;
        color: #f8fafc;
        font-size: 1rem;
        margin-bottom: 0.2rem;
    }

    .source-meta {
        font-size: 0.78rem;
        color: var(--muted);
        margin-bottom: 0.45rem;
    }

    .source-text {
        color: #dfe7f5;
        line-height: 1.65;
        white-space: pre-wrap;
    }

    .assistant {
        background: linear-gradient(135deg, rgba(125, 211, 252, 0.12), rgba(167, 139, 250, 0.08));
        border: 1px solid rgba(125, 211, 252, 0.2);
        border-radius: 20px;
        padding: 1rem 1rem 0.8rem;
        margin: 0.8rem 0;
    }

    .user {
        background: rgba(15, 23, 42, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 20px;
        padding: 0.9rem 1rem;
        margin: 0.8rem 0;
    }

    [data-testid="stChatMessage"] {
        background: transparent;
    }

    .stButton > button {
        background: rgba(15, 23, 42, 0.8);
        color: var(--text);
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 0.6rem 0.9rem;
        font-weight: 600;
    }

    .stButton > button:hover {
        border-color: rgba(125, 211, 252, 0.4);
        box-shadow: 0 12px 24px rgba(125, 211, 252, 0.12);
        transform: translateY(-1px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def seconds_to_hms(value: object) -> str:
    try:
        seconds = int(float(value))
    except (TypeError, ValueError):
        return "--:--"
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}" if seconds >= 3600 else f"{seconds // 60:02d}:{seconds % 60:02d}"


@st.cache_resource
def load_retriever() -> Retriever:
    return Retriever(EMBEDDINGS_PATH)


def generator(prompt: str) -> str:
    return generate(prompt)


def stream_generator(prompt: str):
    return stream(prompt)


def render_source_card(item: dict, index: int) -> None:
    title = html.escape(str(item.get("title", "Untitled")))
    number = html.escape(str(item.get("number", "?")))
    start = seconds_to_hms(item.get("start"))
    end = seconds_to_hms(item.get("end"))
    text = html.escape(str(item.get("text", "")))
    score = item.get("score")
    score_text = f"<span>· score {float(score):.3f}</span>" if isinstance(score, (int, float)) else ""
    st.markdown(
        f"""
        <div class="source">
            <div class="source-title">[{index + 1}] Video {number} — {title}</div>
            <div class="source-meta">{start}–{end} {score_text}</div>
            <div class="source-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown('<div class="sidebar-title">🎓 LearnPilot AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Agentic learning assistant</div>', unsafe_allow_html=True)
    st.divider()

    st.sidebar.markdown("""
    ### 👨‍💻 Developer

    **Anshul Rajpoot**  
    📘 Scholar No: `2311401168`  

    🎓 Electronics & Communication Engineering  
    🏛️ MANIT Bhopal
    
    """)
    st.divider()

    top_k = st.slider("Retrieved sources", 1, MAX_TOP_K, DEFAULT_TOP_K)
    use_ollama = st.checkbox("Use Ollama", value=True)
    stream_response = st.checkbox("Stream response", value=True)
    show_sources = st.checkbox("Show retrieved sources", value=True)

    st.divider()
    st.markdown("**Quick prompts**")
    quick_prompts = [
        "Summarize the HTML course",
        "Give me a quiz on CSS selectors",
        "Explain semantic HTML tags",
        "Evaluate this answer about forms",
    ]
    for prompt in quick_prompts:
        if st.button(prompt, use_container_width=True, key=f"prompt_{prompt}"):
            st.session_state["pending_query"] = prompt

    st.divider()
    st.markdown("**Capabilities**")
    st.markdown("- 📚 Grounded RAG answers\n- 🧠 Tool-based routing\n- 📝 Quiz generation\n- 📌 Summarization\n- ✅ Answer evaluation")


st.markdown(
    """
    <div class="hero">
      <h1>LearnPilot AI</h1>
      <p>Agentic AI learning assistant grounded in your course transcripts.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    retriever = load_retriever()
except Exception as exc:
    st.error(str(exc))
    st.info("If embeddings.joblib is tracked with Git LFS, run `git lfs pull`. Otherwise rebuild it with `python preprocess_json.py`.")
    st.stop()

ollama_ready = use_ollama and is_available()
if use_ollama and not ollama_ready:
    st.warning("Ollama is not available. Start Ollama to enable semantic retrieval and LLM responses. TF-IDF retrieval remains available.")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="stat-card"><div class="stMetric">Videos</div><div style="font-size: 2rem; font-weight: 700; margin-top: 0.25rem;">{}</div></div>'.format(retriever.video_count), unsafe_allow_html=True)
with col2:
    st.markdown('<div class="stat-card"><div class="stMetric">Knowledge chunks</div><div style="font-size: 2rem; font-weight: 700; margin-top: 0.25rem;">{}</div></div>'.format(retriever.chunk_count), unsafe_allow_html=True)
with col3:
    st.markdown('<div class="stat-card"><div class="stMetric">LLM</div><div style="font-size: 1.7rem; font-weight: 700; margin-top: 0.25rem;">{}</div></div>'.format("Ollama" if ollama_ready else "Unavailable"), unsafe_allow_html=True)

if "pending_query" in st.session_state and st.session_state["pending_query"]:
    query = st.session_state.pop("pending_query")
else:
    query = st.chat_input("Ask about your course, request a quiz, summarize a topic, or evaluate an answer...")

if query:
    st.markdown(f'<div class="user">{html.escape(query)}</div>', unsafe_allow_html=True)

    if not ollama_ready:
        result = retriever.search(query, top_k=top_k, prefer_semantic=False)
        context = retriever.records(result)
        st.markdown('<span class="badge">Tool: RAG Retrieval</span>', unsafe_allow_html=True)
        st.info("LLM generation is unavailable, so LearnPilot AI is showing the retrieved course material.")
        if show_sources:
            st.subheader("Relevant course material")
            for index, item in enumerate(context):
                render_source_card(item, index)
    else:
        agent = LearnPilotAI(retriever, generator)
        with st.spinner("Agent is selecting a tool and retrieving course context..."):
            state = agent.prepare(query, top_k=top_k, prefer_semantic=True)

        if state.error:
            st.error(state.error)
            st.stop()

        intent = state.intent
        context = state.retrieved_context
        labels = {"rag": "RAG Retrieval", "quiz": "Quiz Generator", "summary": "Summarization", "evaluate": "Answer Evaluation"}
        method = context[0].get("retrieval_method", "unknown") if context else "none"
        st.markdown(f'<div class="assistant"><span class="badge">{html.escape(labels[intent])}</span><div style="margin-top: 0.7rem; color: var(--muted);">Intent: {html.escape(intent)} · Retrieval: {html.escape(method)}</div></div>', unsafe_allow_html=True)

        if stream_response:
            try:
                response_text = ""
                status = st.status("Generating answer...", expanded=True)
                with status:
                    placeholder = st.empty()
                    for token in stream_generator(state.tool_result["prompt"]):
                        response_text += token
                        placeholder.markdown(
                            f'<div class="assistant">{response_text if response_text else "Generating..."}</div>',
                            unsafe_allow_html=True,
                        )
                    status.update(label="Answer ready", state="complete")
                    placeholder.markdown(
                        f'<div class="assistant">{response_text}</div>',
                        unsafe_allow_html=True,
                    )
            except OllamaError as exc:
                st.error(str(exc))
        else:
            try:
                with st.status("Generating answer...", expanded=True) as status:
                    response_text = generator(state.tool_result["prompt"])
                    status.update(label="Answer ready", state="complete")
                st.markdown(f'<div class="assistant">{response_text}</div>', unsafe_allow_html=True)
            except OllamaError as exc:
                st.error(str(exc))

        if show_sources:
            st.subheader("Retrieved course sources")
            if not context:
                st.info("No source chunks were retrieved.")
            else:
                for index, item in enumerate(context):
                    render_source_card(item, index)
