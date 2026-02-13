from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import PERSIST_DIR, GROQ_API_KEY

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful customer support assistant.
Answer the user's question strictly using the provided context.
If the answer is not present in the context, say exactly:
"I don't have that information in the provided documents."

<context>
{context}
</context>

Question: {question}
""".strip())


def get_embeddings():
    """Create embeddings object (must match ingestion embeddings)."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def format_docs(docs):
    """Formats retrieved docs into a single context string."""
    parts = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)

        if page is not None:
            src_display = f"{src} (page {page + 1})"
        else:
            src_display = src

        parts.append(f"[{i}] Source: {src_display}\n{d.page_content}")

    return "\n\n".join(parts)


def deduplicate_sources(docs):
    """Deduplicate sources so same file/page does not appear multiple times."""
    sources_map = {}

    for d in docs:
        source = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        path = d.metadata.get("path", "")

        key = (source, page)

        if key not in sources_map:
            sources_map[key] = {
                "source": source,
                "path": path,
                "page": page,
                "snippet": d.page_content[:300]
            }

    return list(sources_map.values())


def build_chain():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=GROQ_API_KEY
    )

    embeddings = get_embeddings()

    # ✅ Use direct DB access to allow scoring
    db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    # Safety check
    if db._collection.count() == 0:
        raise ValueError("ChromaDB is empty. Please run ingestion first (/ingest).")

    parser = StrOutputParser()

    def rag_answer(question: str):
        # ✅ retrieve more candidates and filter by relevance score
        results = db.similarity_search_with_score(question, k=6)

        # ✅ score is distance: smaller = better match
        # Tune threshold based on your data:
        # try: 0.6 strict, 0.8 normal, 1.0 loose
        THRESHOLD = 0.8

        filtered_docs = []
        for doc, score in results:
            if score <= THRESHOLD:
                filtered_docs.append(doc)

        # fallback: if everything filtered out, use top-2 anyway
        if not filtered_docs:
            filtered_docs = [doc for doc, score in results[:2]]

        context = format_docs(filtered_docs)

        answer = (RAG_PROMPT | llm | parser).invoke({
            "context": context,
            "question": question
        })

        if "i don't have that information in the provided documents" in answer.lower():
            return {"answer": answer, "sources": []}

        sources = deduplicate_sources(filtered_docs)

        return {"answer": answer, "sources": sources}

    return rag_answer
