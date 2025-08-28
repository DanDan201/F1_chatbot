# F1 Racing RAG Chatbot (Flask + Ollama + JS Frontend)

A minimal Retrieval-Augmented Generation (RAG) app focused on Formula 1. It uses:

- **Backend**: Flask (Python) + LangChain
- **LLM**: **Ollama** (local), default model `llama3.1`
- **Embeddings**: `nomic-embed-text` via Ollama
- **Vector DB**: Chroma (local on disk)
- **Frontend**: Single-page JavaScript (no framework)

> You can swap models in `.env`. This setup runs fully offline after you load data.

---

## Project Structure

```
f1-rag-chatbot/
├─ backend/
│  ├─ app.py
│  ├─ rag_pipeline.py
│  ├─ ingest.py
│  ├─ requirements.txt
│  ├─ .env.example
│  └─ data/
│     ├─ sample_f1_rules.txt
│     └─ (put your PDFs/TXTs here)
└─ frontend/
   ├─ index.html
   ├─ styles.css
   └─ app.js
```

---

## 1) Backend Code

### `backend/requirements.txt`
```
flask==3.0.3
flask-cors==4.0.1
langchain==0.2.12
langchain-community==0.2.11
langchain-text-splitters==0.2.2
chromadb==0.5.4
python-dotenv==1.0.1
pypdf==4.3.1
waitress==3.0.0
```

### `backend/.env.example`
```
# Ollama server (default local)
OLLAMA_HOST=http://localhost:11434

# Model names you have pulled with `ollama pull <model>`
LLM_MODEL=llama3.1
EMBED_MODEL=nomic-embed-text

# Retrieval parameters
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
TOP_K=5

# Where Chroma stores the vector DB
CHROMA_DIR=./chroma_db

# Bind address for Flask
HOST=0.0.0.0
PORT=5001
```

> Copy to `.env` and edit if needed.

### `backend/rag_pipeline.py`
```python
import os
import glob
from typing import List, Tuple
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1200))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
TOP_K = int(os.getenv("TOP_K", 5))

# ---- Loaders ---------------------------------------------------------------

def load_documents(data_dir: str) -> List:
    docs = []
    # PDFs
    for path in glob.glob(os.path.join(data_dir, "**", "*.pdf"), recursive=True):
        try:
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        except Exception as e:
            print(f"[WARN] PDF load failed {path}: {e}")
    # TXTs
    for path in glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True):
        try:
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())
        except Exception as e:
            print(f"[WARN] TXT load failed {path}: {e}")
    return docs

# ---- Splitter --------------------------------------------------------------

def split_documents(docs: List):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", ".", " "]
    )
    return splitter.split_documents(docs)

# ---- Embeddings & VectorStore ---------------------------------------------

def get_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_HOST)


def build_or_load_vectorstore(chunks):
    os.makedirs(CHROMA_DIR, exist_ok=True)
    vectorstore = Chroma(
        collection_name="f1_rag",
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )
    if chunks:
        vectorstore.add_documents(chunks)
        vectorstore.persist()
    return vectorstore

# ---- LLM + Prompt ---------------------------------------------------------

def get_llm():
    return Ollama(model=LLM_MODEL, base_url=OLLAMA_HOST, temperature=0.1)

SYSTEM_PROMPT = (
    "You are an expert Formula 1 assistant. Answer using retrieved context. "
    "If the answer isn't in context, say you don't know. Be concise and cite sources as [S1], [S2]."
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:")
])

# ---- RAG Chains -----------------------------------------------------------

def make_retriever(vectorstore):
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})


def format_docs(docs: List) -> Tuple[str, List[dict]]:
    blocks = []
    sources = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        source = meta.get("source", "unknown")
        page = meta.get("page", None)
        tag = f"[S{i}] {os.path.basename(source)}" + (f" p.{page+1}" if page is not None else "")
        blocks.append(f"{tag}\n{d.page_content}")
        sources.append({"tag": f"S{i}", "source": source, "page": (None if page is None else page+1)})
    return "\n\n---\n\n".join(blocks), sources


def build_rag_chain(vectorstore):
    retriever = make_retriever(vectorstore)

    def _context_fn(q):
        docs = retriever.get_relevant_documents(q)
        ctx, sources = format_docs(docs)
        return {"context": ctx, "sources": sources}

    chain = (
        {"question": RunnablePassthrough()} |
        (lambda q: {"question": q, **_context_fn(q)}) |
        PROMPT |
        get_llm()
    )
    return chain
```

### `backend/ingest.py`
```python
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
```

### `backend/app.py`
```python
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

from rag_pipeline import build_or_load_vectorstore, build_rag_chain

load_dotenv()

app = Flask(__name__)
CORS(app)

# Lazy load vectorstore / chain on first request
_vectorstore = None
_chain = None


def _ensure_chain():
    global _vectorstore, _chain
    if _vectorstore is None:
        _vectorstore = build_or_load_vectorstore(chunks=None)
    if _chain is None:
        _chain = build_rag_chain(_vectorstore)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat():
    _ensure_chain()
    data = request.get_json(force=True)
    question = (data or {}).get("message", "").strip()
    if not question:
        return jsonify({"error": "message is required"}), 400

    try:
        # Invoke LangChain RAG
        answer = _chain.invoke(question)
        # The prompt instructs LLM to cite [S*]; return raw + vector sources
        return jsonify({"answer": str(answer)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5001))

    # For production on Windows, use waitress; dev can use Flask dev server
    try:
        from waitress import serve
        print(f"Serving with waitress on http://{host}:{port}")
        serve(app, host=host, port=port)
    except Exception:
        app.run(host=host, port=port, debug=True)
```

### `backend/data/sample_f1_rules.txt`
```
Formula 1 races adhere to regulations defined by the FIA. A typical Grand Prix weekend may include practice sessions, qualifying, and the race. Tyre compounds, pit stop rules, safety car procedures, and penalties are defined by the sporting regulations. (Placeholder sample text.)
```

---

## 2) Frontend (Vanilla JS)

### `frontend/index.html`
```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>F1 RAG Chatbot</title>
    <link rel="stylesheet" href="styles.css" />
  </head>
  <body>
    <div class="app">
      <header>
        <h1>F1 RAG Chatbot</h1>
        <p>Ask anything about Formula 1 rules, history, and races.</p>
      </header>

      <main>
        <div id="chat" class="chat"></div>
        <form id="chat-form">
          <input id="input" type="text" placeholder="e.g., How do F1 tyre compounds work?" autofocus />
          <button type="submit">Send</button>
        </form>
      </main>

      <footer>
        <small>Powered by Flask + Ollama + Chroma</small>
      </footer>
    </div>

    <script src="app.js"></script>
  </body>
</html>
```

### `frontend/styles.css`
```css
:root { font-family: system-ui, Arial, sans-serif; }
body { margin: 0; background: #0b0b0f; color: #f0f3f7; }
.app { max-width: 900px; margin: 0 auto; padding: 24px; }
header { margin-bottom: 12px; }
header h1 { margin: 0 0 4px; }
.chat { background: #14151c; border: 1px solid #242634; border-radius: 12px; padding: 16px; height: 65vh; overflow: auto; }
.msg { margin: 8px 0; padding: 12px 14px; border-radius: 10px; line-height: 1.4; }
.msg.user { background: #243b5a; }
.msg.bot  { background: #1e222d; }
form { display: flex; gap: 8px; margin-top: 12px; }
input { flex: 1; padding: 12px; border-radius: 10px; border: 1px solid #2c2f3c; background: #0f1117; color: #e7ecf3; }
button { padding: 12px 16px; border: none; border-radius: 10px; background: #e10600; color: #fff; font-weight: 600; cursor: pointer; }
button:hover { filter: brightness(1.1); }
footer { opacity: 0.7; margin-top: 16px; }
.code { font-family: ui-monospace, Menlo, Consolas, monospace; white-space: pre-wrap; }
```

### `frontend/app.js`
```javascript
const API_BASE = (localStorage.getItem('api_base') || 'http://localhost:5001');

const chatEl = document.getElementById('chat');
const formEl = document.getElementById('chat-form');
const inputEl = document.getElementById('input');

function addMsg(text, who='bot') {
  const div = document.createElement('div');
  div.className = `msg ${who}`;
  div.textContent = text;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
}

formEl.addEventListener('submit', async (e) => {
  e.preventDefault();
  const q = inputEl.value.trim();
  if (!q) return;
  addMsg(q, 'user');
  inputEl.value = '';
  addMsg('Thinking…');

  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: q })
    });
    const data = await res.json();
    const last = chatEl.querySelector('.msg.bot:last-child');
    last.textContent = data.answer || data.error || 'Error';
  } catch (err) {
    const last = chatEl.querySelector('.msg.bot:last-child');
    last.textContent = 'Network error';
  }
});
```

---

## 3) Setup & Run

### A) Start Ollama & Pull Models
```bash
# Install Ollama: https://ollama.com/download
ollama serve

# In another terminal, pull models
ollama pull llama3.1
ollama pull nomic-embed-text
```

### B) Prepare Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
```

### C) Add F1 Knowledge
- Put your PDFs/TXTs into `backend/data/`.
- Example sources you might add: FIA Sporting Regulations PDFs, team/race guides you are allowed to use, personal notes, etc.

Then build the vector DB:
```bash
python ingest.py
```

### D) Run the API
```bash
python app.py
# or production-ish
python -c "from app import app; from waitress import serve; serve(app, host='0.0.0.0', port=5001)"
```

### E) Open the Frontend
```bash
# any static server works; simplest is Python's http.server
cd ../frontend
python -m http.server 8080
# visit http://localhost:8080
```

If your backend is on a different host/port, open the browser console and run:
```js
localStorage.setItem('api_base', 'http://YOUR_HOST:5001')
```
Reload the page.

---

## 4) Customization Tips

- **Change models**: Edit `.env` → `LLM_MODEL` (e.g., `llama3`, `phi3`, etc.). Ensure you `ollama pull` first.
- **Tune retrieval**: Adjust `TOP_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP` in `.env`.
- **More file types**: Add loaders in `rag_pipeline.load_documents` (e.g., Markdown, HTML) using LangChain community loaders.
- **Citations**: The prompt formats sources as `[S1]`, `[S2]` with filenames and pages. You can render them nicely on the frontend if desired.
- **Streaming**: Switch to Flask SSE or WebSocket; on Python side, iterate over `llm.stream()` from LangChain.
- **Auth**: Put a reverse proxy (Nginx/Caddy) in front; add an API key header check in `/chat`.

---

## 5) Smoke Test Prompts

- *“Summarize the difference between a Safety Car and a Virtual Safety Car.”*
- *“How are grid penalties applied after qualifying?”*
- *“Explain tyre compound allocation on a Sprint weekend.”*

---

## 6) Troubleshooting

- **`ModuleNotFoundError`**: Ensure the venv is active and requirements installed.
- **`connection refused to http://localhost:11434`**: Start `ollama serve` and pull models.
- **No answers / hallucinations**: Verify ingestion ran and `chroma_db/` has content; increase `TOP_K`.
- **Windows & long paths**: If Chroma path errors appear, move the project to a shorter path (e.g., `C:\f1rag`).

---

## 7) License
MIT (adjust as you wish).

