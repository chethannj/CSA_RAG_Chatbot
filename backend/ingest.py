# built-in python libraries
import os  # for file/folder operations
from pathlib import Path  # easier way to work with file paths

# LangChain document loaders (to read different file formats)
from langchain_community.document_loaders import (
    PyPDFLoader,                     # loads PDF files
    TextLoader,                      # loads plain text files
    UnstructuredWordDocumentLoader,  # loads Word (.docx) files
    UnstructuredMarkdownLoader,      # loads Markdown (.md) files
    CSVLoader,                       # loads CSV files
)

# LangChain splitter (to split large documents into chunks)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain embeddings (converts text into vectors for similarity search)
from langchain_community.embeddings import HuggingFaceEmbeddings

# Chroma vector database (stores vectors + text)
from langchain_community.vectorstores import Chroma

# importing project paths from config.py
from config import DATA_DIR, PERSIST_DIR


# ✅ A map that decides which loader to use for each file extension
# Example: ".pdf" → use PyPDFLoader
LOADER_MAP = {
    ".pdf": PyPDFLoader,                               # PDF loader
    ".txt": lambda p: TextLoader(p, encoding="utf-8"), # TXT loader with encoding
    ".docx": UnstructuredWordDocumentLoader,           # DOCX loader
    ".md": UnstructuredMarkdownLoader,                 # Markdown loader
    ".csv": CSVLoader,                                 # CSV loader
}


def get_embeddings():
    """
    Creates embedding model.
    Must match the embedding model used in rag_chain.py
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"  # small, fast & accurate
    )


def load_documents():
    """
    Loads all supported files inside Data/ folder and converts them into LangChain docs.
    """
    docs = []  # list that stores all documents

    # Create a Path object pointing to Data folder
    data_path = Path(DATA_DIR)

    # If Data folder does not exist, stop program
    if not data_path.exists():
        raise FileNotFoundError(f"Data folder not found: {DATA_DIR}")

    # rglob("*") scans all files recursively inside folder/subfolders
    for file_path in data_path.rglob("*"):

        # skip folders, only keep files
        if not file_path.is_file():
            continue

        # get extension like ".pdf", ".txt"
        ext = file_path.suffix.lower()

        # if extension not supported, skip the file
        if ext not in LOADER_MAP:
            continue

        # pick correct loader function/class based on file type
        loader_fn = LOADER_MAP[ext]

        # create the loader instance (some are classes, some are lambdas)
        loader = loader_fn(str(file_path))

        # load the file content into LangChain Document objects
        file_docs = loader.load()

        # add metadata to each document (useful for citations/sources)
        for d in file_docs:
            d.metadata["source"] = file_path.name   # filename only
            d.metadata["path"] = str(file_path)     # full path

        # add documents from this file into overall docs list
        docs.extend(file_docs)

    # return all loaded documents
    return docs


def ingest():
    """
    Main ingestion function:
    Data folder → load docs → chunk → embed → store in ChromaDB
    """
    print(f"📂 Reading files from: {DATA_DIR}")

    # load all documents from Data folder
    documents = load_documents()

    # If no docs found, stop here
    if not documents:
        print("⚠️ No supported documents found in Data folder.")
        return

    # show number of loaded documents/pages
    print(f"✅ Loaded {len(documents)} documents/pages")

    # Create a text splitter to chunk documents into smaller pieces
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,   # max characters per chunk
        chunk_overlap=150  # overlapping text for better context
    )

    # split all documents into chunks
    chunks = splitter.split_documents(documents)

    # print number of chunks
    print(f"✂️ Split into {len(chunks)} chunks")

    # Create Chroma vector database and store documents
    vectordb = Chroma.from_documents(
        documents=chunks,               # chunks to store
        embedding=get_embeddings(),     # embedding model
        persist_directory=PERSIST_DIR   # folder location where db will be saved
    )

    # Save database to disk (persistent)
    vectordb.persist()

    print(f"✅ Stored in ChromaDB at: {PERSIST_DIR}")


# if this file is run directly, ingestion will start
if __name__ == "__main__":
    ingest()
