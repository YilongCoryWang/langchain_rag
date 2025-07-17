from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from app.config import LLM_API_KEY, LLM_BASE_URL


class Ragbot:
    name: str
    llm = None
    qa_chain: Runnable = None

    def __init__(self):
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0,
            openai_api_key=LLM_API_KEY,
            openai_api_base=LLM_BASE_URL,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant. Be precise, if you don't know, just say it.",
                ),
                ("user", "Question: {question}"),
            ]
        )

        self.qa_chain = (
            {
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
        )
        pass

    def run(self, question):
        if self.qa_chain == None:
            print("RAG qa chain is not initialized.")
            return ""
        else:
            response = self.qa_chain.invoke(question)
            return response
