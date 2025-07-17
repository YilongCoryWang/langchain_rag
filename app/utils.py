from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile


def extract_text_from_pdf(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp.flush()  # have to flush，PyPDFLoader reads it

        loader = PyPDFLoader(tmp.name)
        docs = loader.load()
        return docs


def split_docs(raw_text) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splitted_docs = splitter.split_documents(raw_text)
    return splitted_docs
