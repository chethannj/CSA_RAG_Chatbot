import os
import requests
import streamlit as st
from dotenv import load_dotenv

# ----------------------------
# Load .env from project root
# ----------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

# ----------------------------
# API Config
# ----------------------------
CHAT_API_URL = os.getenv("RAG_CHAT_API_URL", "http://127.0.0.1:8000/chat")
INGEST_API_URL = os.getenv("RAG_INGEST_API_URL", "http://127.0.0.1:8000/ingest")

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(page_title="Customer RAG Chatbot", layout="wide")
st.title("📄 Customer RAG Chatbot (Groq + LangChain + ChromaDB)")
st.caption("Ask questions from documents stored in the Data/ folder. Answers include sources.")

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("⚙️ Controls")

   

    st.markdown("---")

    # ✅ Ingest Button
    if st.button("📥 Ingest / Rebuild Knowledge Base", use_container_width=True):
        with st.spinner("Ingestion running... Reading Data/ folder and updating ChromaDB..."):
            try:
                resp = requests.post(INGEST_API_URL, timeout=600)  # allow long ingestion
                resp.raise_for_status()
                data = resp.json()

                if data.get("status") == "success":
                    st.success("✅ Ingestion completed successfully!")
                else:
                    st.error(f"❌ Ingestion failed: {data.get('message', 'Unknown error')}")
            except Exception as e:
                st.error(f"❌ API Error while ingesting: {e}")

    st.markdown("---")

    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ----------------------------
# Session chat history
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# Display chat history
# ----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📌 Sources"):
                for idx, s in enumerate(msg["sources"], start=1):
                    src = s.get("source", "unknown")
                    page = s.get("page", None)
                    path = s.get("path", "")
                    snippet = s.get("snippet", "")

                    title = f"{idx}. {src}"
                    if page is not None:
                        title += f" (page {page + 1})"

                    st.markdown(f"**{title}**")
                    if path:
                        st.caption(path)
                    if snippet:
                        st.write(snippet)

# ----------------------------
# Chat input
# ----------------------------
question = st.chat_input("Ask something from your documents...")

if question:
    # user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(CHAT_API_URL, json={"question": question}, timeout=180)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                st.error(f"❌ Failed to connect to backend API.\n\n{e}")
                st.stop()

            answer = data.get("answer", "No answer received.")
            sources = data.get("sources", [])

            st.markdown(answer)

            if sources:
                with st.expander("📌 Sources"):
                    for idx, s in enumerate(sources, start=1):
                        src = s.get("source", "unknown")
                        page = s.get("page", None)
                        path = s.get("path", "")
                        snippet = s.get("snippet", "")

                        title = f"{idx}. {src}"
                        if page is not None:
                            title += f" (page {page + 1})"

                        st.markdown(f"**{title}**")
                        if path:
                            st.caption(path)
                        if snippet:
                            st.write(snippet)

    # save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
