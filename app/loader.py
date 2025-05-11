from langchain_community.document_loaders import PyPDFLoader
import tempfile

def extract_text_from_pdf(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp.flush()  # have to flush，PyPDFLoader reads it

        loader = PyPDFLoader(tmp.name)
        docs = loader.load()
        # print(docs)
        return docs