import os
from dotenv import load_dotenv
from rag_pipeline import load_documents, split_documents, build_or_load_vectorstore

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

if __name__ == "__main__":
    print("[Ingest] Loading documents…")
    docs = load_documents(DATA_DIR)
    print(f"[Ingest] Loaded {len(docs)} raw docs")

    print("[Ingest] Splitting…")
    chunks = split_documents(docs)
    print(f"[Ingest] Created {len(chunks)} chunks")

    print("[Ingest] Building vector store…")
    _ = build_or_load_vectorstore(chunks)

    print("[Ingest] Done. Vector DB is ready.")