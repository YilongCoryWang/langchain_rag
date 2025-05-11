from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
def embed_texts(splitted_docs: list[str]) -> list[list[float]]:
    db = FAISS.from_documents(splitted_docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    return retriever
