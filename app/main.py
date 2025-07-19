from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from app.ragbot import Ragbot
from app.retriever import get_retriever
from langchain import hub
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from app.in_memory import InMemoryHistory

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


from langchain_community.chat_message_histories import ChatMessageHistory


# Here we use a global variable to store the chat message history.
# This will make it easier to inspect it to see the underlying results.
store = {}


def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        retriever = get_retriever(contents)
        prompt = hub.pull("rlm/rag-prompt")

        # Chain in LCEL style
        global rag_bot
        qa_chain = (
            {
                # "context": retriever,
                # "question": RunnablePassthrough(),
                "question": lambda x: x["question"],
                "context": lambda x: retriever.invoke(x["question"]),
            }
            | prompt
            | rag_bot.llm
        )

        rag_bot.qa_chain = RunnableWithMessageHistory(
            qa_chain,
            get_session_history=get_by_session_id,
            input_messages_key="question",
            history_messages_key="chat_history",
        )

        return {"msg": "PDF uploaded and qa chain created"}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/query/")
async def query(question: str):
    global rag_bot
    answer = rag_bot.run(question)

    return {"answer": answer}
