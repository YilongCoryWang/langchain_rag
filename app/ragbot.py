from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from app.config import LLM_API_KEY, LLM_BASE_URL
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory


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

        prompt = ChatPromptTemplate(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template(
                    "You are a helpful assistant. User's question: {question}"
                ),
            ]
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, input_key="question"
        )

        self.qa_chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            memory=memory,
        )

    def run(self, question, session_id="default"):
        if self.qa_chain == None:
            print("RAG qa chain is not initialized.")
            return ""
        else:
            response = self.qa_chain.invoke(
                {"question": question},
                config={"configurable": {"session_id": session_id}},
            )

            if isinstance(response, str):
                answer = response
            elif hasattr(response, "content"):  # 如果是 AIMessage 或类似对象
                answer = response.content
            elif isinstance(response, dict):  # 如果是字典
                answer = response.get("text", "No content found")
            else:
                answer = str(response)

            return answer
