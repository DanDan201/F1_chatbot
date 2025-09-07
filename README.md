# F1 Racing RAG Chatbot

This repository contains a Retrieval-Augmented Generation (RAG) chatbot for Formula 1 racing, built with a Flask backend and a JavaScript frontend. The chatbot leverages local document storage and a vector database to answer questions about F1 regulations and related topics.

## Features
- **Chatbot UI**: Simple web interface for chatting about F1.
- **RAG Pipeline**: Uses document ingestion and vector search to provide accurate answers.
- **PDF Ingestion**: Supports ingesting F1 sporting regulations and other documents.
- **Local Database**: Stores user data and chat history.

## Project Structure
```
backend/         # Flask backend, RAG pipeline, ingestion scripts
frontend/        # JavaScript frontend, HTML/CSS
chroma_db/       # Vector database files
instance/        # Local SQLite databases
```

## Getting Started
### Prerequisites
- Python 3.11+

### A) Start Ollama & Pull Models
```bash
# Install Ollama: https://ollama.com/download
ollama serve

# In another terminal, pull models
ollama pull qwen3:4b
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
# This process will take a while
```


### D) Run the API
```bash
python app.py
```

### E) Open the Frontend
```bash
cd ../frontend
python -m http.server 8080
# visit http://localhost:8080
```
## Usage
- Ask questions about F1 regulations and get answers powered by RAG.
- Ingest new documents via the backend to expand chatbot knowledge.

## File Overview
- `backend/app.py`: Main Flask server
- `backend/ingest.py`: Document ingestion script
- `backend/rag_pipeline.py`: RAG logic
- `frontend/app.js`: Frontend logic
- `frontend/index.html`: Chatbot UI

## License
MIT

## Author
DanDan201