from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from app.ragbot import Ragbot
from app.retriever import get_retriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate


# global rag instance
rag_bot = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_bot
    print("🚀 Server starting up...")
    rag_bot = Ragbot()

    yield
    print("🛑 Server shutting down...")


app = FastAPI(lifespan=lifespan)


@app.get("/")
def hello():
    return {"message": "RAG backend is running."}


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        retriever = get_retriever(contents)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                ("user", "{context}\n\nQuestion: {question}"),
            ]
        )

        # Chain in LCEL style
        global rag_bot
        rag_bot.qa_chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough(),
            }
            | prompt
            | rag_bot.llm
        )

        return {"msg": "PDF uploaded and qa chain created"}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/query/")
async def query_pdf(question: str):
    global rag_bot
    response = rag_bot.run(question)
    if isinstance(response, str):
        answer = response
    elif hasattr(response, "content"):  # 如果是 AIMessage 或类似对象
        answer = response.content
    elif isinstance(response, dict):  # 如果是字典
        answer = response.get("content", "No content found")
    else:
        answer = str(response)

    print("Answer:", answer)
    return {"answer": answer}
