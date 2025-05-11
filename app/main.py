from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from app.loader import extract_text_from_pdf
# from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.embedder import embed_texts
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from app.config import LLM_API_KEY
from app.config import LLM_BASE_URL
# from langchain.schema import HumanMessage

app = FastAPI()

qa_chain = None

@app.get("/")
def hello():
    return {"message": "RAG backend is running."}

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        raw_text = extract_text_from_pdf(contents)
        # print('raw_text', raw_text)

        global qa_chain
        # split by paragraph
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splitted_docs = splitter.split_documents(raw_text)

        # embed texts
        retriever = embed_texts(splitted_docs)

        # build qa chain
        llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0,
            openai_api_key=LLM_API_KEY,
            openai_api_base=LLM_BASE_URL,
        )
        # response = llm([HumanMessage(content="Hello, who are you?")])
        # print(response.content)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
        return {"msg": "PDF uploaded and qa chain created", "chunks": len(splitted_docs)}
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )
    
@app.get("/query/")
async def query_pdf(question: str):
    global qa_chain
    # print(query)
    response = qa_chain.invoke({"query": question})
    answer = response["result"]

    # print("Answer:", answer)
    # for doc in response["source_documents"]:
    #     print("--- Source Chunk ---")
    #     print(doc.page_content[:200])
    return {"answer": answer}