from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from app.config import LLM_API_KEY, LLM_BASE_URL
from langchain import hub
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from app.in_memory import InMemoryHistory
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from app.utils import extract_text_from_pdf, split_docs
from langchain_huggingface import HuggingFaceEmbeddings

# global rag instance
rag_bot = None


# Here we use a global variable to store the chat message history.
# This will make it easier to inspect it to see the underlying results.
store = {}


def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]


class Ragbot:
    name: str
    llm = None
    vector_store = None
    retriever = None
    qa_chain: Runnable = None

    def __init__(self):
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0,
            openai_api_key=LLM_API_KEY,
            openai_api_base=LLM_BASE_URL,
        )
        self.init_documents()
        self.init_qa_chain()

    def init_documents(self):
        doc_placeholder = [
            Document(page_content="placeholder", metadata={"source": "init"})
        ]
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = FAISS.from_documents(
            doc_placeholder, embedding=embedding_model
        )
        self.retriever = self.vector_store.as_retriever()

    def add_documents(self, contents: bytes = b""):
        raw_text = extract_text_from_pdf(contents)
        splitted_docs = split_docs(raw_text)
        self.vector_store.add_documents(splitted_docs)

    # Chain in LCEL style
    def init_qa_chain(self):
        prompt = hub.pull("rlm/rag-prompt")
        qa_chain = (
            {
                "question": lambda x: x["question"],
                "context": lambda x: self.retriever.invoke(x["question"]),
            }
            | prompt
            | self.llm
        )

        self.qa_chain = RunnableWithMessageHistory(
            qa_chain,
            get_session_history=get_by_session_id,
            input_messages_key="question",
            history_messages_key="chat_history",
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
