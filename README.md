# LangChain RAG

A Retrieval-Augmented Generation (RAG) question-answering system built with LangChain, supporting PDF file upload, document vector-based retrieval, and integration with large language models to generate context-aware answers.

## 🌟 Highlights

- ✅ Built on LangChain v0.3 architecture
- 📄 Supports PDF document upload and content embedding
- 🧠 **Intent Detection**: LLM-based intent classification to route questions to appropriate handlers (knowledge_base vs chitchat)
- 🔍 Integrated with FAISS for efficient document retrieval
- 🎯 **Multi-Channel Retrieval**: Combines BM25 (keyword search) + FAISS (semantic search) via EnsembleRetriever for improved recall accuracy
- 🏆 **Rerank with Flashrank**: Deep semantic re-ranking of retrieved documents using cross-encoder model for better relevance
- 🤖 Compatible with OpenAI / DeepSeek and other LLM providers
- 🌐 FastAPI backend for easy deployment and integration
- 📊 Integrated with LangSmith for tracing and debugging of RAG pipelines
- 📥 Uses [rlm/rag-prompt](https://smith.langchain.com/hub/rlm/rag-prompt) from LangChain Hub as the RAG prompt template, instead of manually crafting prompts from raw messages

---

## 📁 Project Structure

```
langchain_rag/
│
├── app/
│ ├── main.py # FastAPI entry point
│ ├── ragbot.py # RAG class with Multi-Channel Retrieval (BM25 + FAISS) + Rerank
│ ├── config.py # Load project environment variables
│ ├── utils.py # PDF processing and document splitting
│ ├── in_memory.py # In-memory chat history implementation
│
├── .env # Project environment variables
├── requirements.txt
└── README.md
```

---

## 🚀 Fast start

### 1. Create environment

```bash
python -m venv venv
source ./venv/bin/activate
```

### 2. install dependencies

```bash
pip install -r requirements.txt
mv .env.template .env
# edit .env and fill in your LLM's api key and base url
```

### 3. Start service

```bash
uvicorn app.main:app --reload
```

### 4. upload

Upload Documents and Ask Questions
Use the /upload-pdf/ endpoint to upload a PDF file with field name 'file', then use the /query/ endpoint to ask questions. The system will generate answers based on the document content. For example:
http://localhost:8000/query?question=hi

### 🧩 Tech Stack

LangChain v0.3+

**Query Processing Pipeline**:
1. **Intent Detection**: LLM-based classification (knowledge_base / chitchat)
2. **Multi-Channel Retrieval** (for knowledge_base):
   - BM25 Retriever (keyword-based search)
   - FAISS vector store (semantic search)
   - EnsembleRetriever (combines BM25 + FAISS with weighted scoring)
3. **Rerank**:
   - FlashrankRerank (cross-encoder based semantic re-ranking)
   - ContextualCompressionRetriever (wraps reranker with base retriever)

Sentence-Transformers embedding models

FastAPI + Uvicorn

Pydantic + Python 3.10+
