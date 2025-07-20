from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from app.ragbot import Ragbot, rag_bot


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Server starting up...")
    global rag_bot
    rag_bot = Ragbot()

    yield
    print("🛑 Server shutting down...")


app = FastAPI(lifespan=lifespan)


from langchain_community.chat_message_histories import ChatMessageHistory


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        rag_bot.add_documents(contents)
        return {"msg": "PDF uploaded and qa chain created"}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/query/")
async def query(question: str):
    answer = rag_bot.run(question)

    return {"answer": answer}
