import os
import glob
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from ollama import chat

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST")
LLM_MODEL = os.getenv("LLM_MODEL")
EMBED_MODEL = os.getenv("EMBED_MODEL")
CHROMA_DIR = os.getenv("EMBED_MODEL")
CHROMA_DIR = os.getenv("CHROMA_DIR")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))
TOP_K = int(os.getenv("TOP_K"))

# Load documents function
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
            loader = TextLoader(path, encoding = "utf-8")
            docs.extend(loader.load())
        except Exception as e:
            print("f[WARN] txt load failed {path}: {e}")
    return docs

# Split documents function
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", ".", " "]
    )
    return splitter.split_documents(docs)

# Input embeddings
def get_embeddings():
    return OllamaEmbeddings(model = EMBED_MODEL, base_url=OLLAMA_HOST)

# Build vector db for documents
def build_or_load_vectorstore(chunks):
    os.makedirs(CHROMA_DIR, exist_ok = True)
    vectorstore = Chroma(
        collection_name="f1_rag",
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR
    )
    if chunks:
        vectorstore.add_documents(chunks)
    return vectorstore

# LLM + Prompt
def get_llm():
    return Ollama(model = LLM_MODEL, base_url=OLLAMA_HOST, temperature = 0.1)

SYSTEM_PROMPT = (
    "You are an expert Formula 1 assistant. Answer using retrieved context. "
    "If the answer isn't in context, say you don't know. Be concise and cite sources as [S1], [S2]."
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Question: {question}\n\nContext:\n\n:Answer")
])

# RAG chain
def make_retriever(vectorstore):
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})

def format_docs(docs: List) -> Tuple[str, List[dict]]:
    blocks = []
    sources = []
    for i,d in enumerate(docs, start=1):
        meta = d.metadata or {}
        source = meta.get("source", "unknown")
        page = meta.get("page", None)
        tag = f"[S{i}] {os.path.basename(source)}" + (f" p.{page+1}" if page is not None 
                                                      else "")
        blocks.append(f"{tag}\n{d.page_content}")
        sources.append({"tag": f"S{i}", "source": source, "page": (None if page is None 
                                                                   else page+1)})
    return "\n\n---\n\n".join(blocks), sources

def build_rag_chain(vectorstore):
    retriever = make_retriever(vectorstore)

    memory = ConversationBufferMemory(
        memory_key = "chat_history",
        return_messages=True
    )

    def _context_fn(inputs):
        q = inputs["question"]
        docs = retriever.invoke(q)
        ctx, sources = format_docs(docs)

        history = memory.load_memory_variables({})["chat_history"]
        return {"context": ctx, 
                "sources": sources, 
                "question": q,
                "chat_history": history}

    chain = (
        {"question": RunnablePassthrough()}
        | RunnableLambda(_context_fn)
        | PROMPT
        | get_llm()
    )

    def chain_with_memory(inputs):
        # run chain
        answer = chain.invoke(inputs)
        # save interaction
        memory.save_context({"input": inputs["question"]}, 
                            {"output": str(answer)})
        return {"answer": str(answer)}
    
    return RunnableLambda(chain_with_memory)