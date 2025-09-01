from backend.rag_pipeline import build_or_load_vectorstore
vs = build_or_load_vectorstore(chunks=None)
docs = vs.similarity_search("when can the clerk of the course interrupt practice?", k=2)
for d in docs:
    print(d.page_content[:500])
