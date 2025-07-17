# LangChain RAG

A Retrieval-Augmented Generation (RAG) question-answering system built with LangChain, supporting PDF file upload, document vector-based retrieval, and integration with large language models to generate context-aware answers.

## 🌟 Highlights

- ✅ Built on LangChain v0.3 architecture
- 📄 Supports PDF document upload and content embedding
- 🔍 Integrated with FAISS for efficient document retrieval
- 🤖 Compatible with OpenAI / DeepSeek and other LLM providers
- 🌐 FastAPI backend for easy deployment and integration

---

## 📁 Project Structure

```
langchain_rag/
│
├── app/
│ ├── main.py # FastAPI entry point
│ ├── loader.py # read and process pdf file
│ ├── embedder.py # Embedding model and vector store integration
│ ├── config.py # Load project environment variables
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
uvicorn main:app --reload
```

### 4. upload

Upload Documents and Ask Questions
Use the /upload-pdf/ endpoint to upload a PDF file with field name 'file', then use the /query/ endpoint to ask questions. The system will generate answers based on the document content. For example:
http://localhost:8000/query?question=hi

### 🧩 Tech Stack

LangChain v0.3+

FAISS vector store

Sentence-Transformers embedding models

FastAPI + Uvicorn

Pydantic + Python 3.10+
