from app.utils import extract_text_from_pdf, split_docs
from app.embedder import embed_texts


def get_retriever(contents: bytes):
    raw_text = extract_text_from_pdf(contents)
    splitted_docs = split_docs(raw_text)
    retriever = embed_texts(splitted_docs)
    return retriever
