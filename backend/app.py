import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_pipeline import build_or_load_vectorstore, build_rag_chain

load_dotenv()

app = Flask(__name__)
CORS(app)

_vectorstore = None
_chain = None

def _ensure_chain():
    global _vectorstore, _chain
    if _vectorstore is None:
        _vectorstore = build_or_load_vectorstore(chunks = None)
    
    if _chain is None:
        _chain = build_rag_chain(_vectorstore)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat():
    _ensure_chain()
    data = request.get_json(force = True)
    question = (data or {}).get("message", "").strip()
    if not question:
        return jsonify({"error": "message is required"}), 400
    
    try:
        # Invoke Langchain RAG
        result = _chain.invoke({"question": question})

        # The prompt instructs LLM to cite
        return jsonify(result)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"erorr": str(e)}), 500
    
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5001))

    app.run(host = host, port=port, debug = True)