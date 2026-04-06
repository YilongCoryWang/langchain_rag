from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from app.config import LLM_API_KEY, LLM_BASE_URL
from langchain import hub
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from app.in_memory import InMemoryHistory
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
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
    bm25_retriever = None
    ensemble_retriever = None
    compression_retriever = None
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
        
        # Initialize Vector Store (FAISS) - Semantic Search
        self.vector_store = FAISS.from_documents(
            doc_placeholder, embedding=embedding_model
        )
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Initialize BM25 Retriever - Keyword Search
        self.bm25_retriever = BM25Retriever.from_documents(doc_placeholder)
        self.bm25_retriever.k = 3
        
        # Create Ensemble Retriever (Multi-Channel Retrieval)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, vector_retriever],
            weights=[0.4, 0.6]  # BM25: 40%, Vector: 60%
        )
        
        # Wrap with Flashrank Rerank for semantic re-ranking
        compressor = FlashrankRerank()
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=self.ensemble_retriever
        )
        
        # Use compression retriever as the main retriever
        self.retriever = self.compression_retriever

    def add_documents(self, contents: bytes = b""):
        raw_text = extract_text_from_pdf(contents)
        splitted_docs = split_docs(raw_text)
        self.vector_store.add_documents(splitted_docs)
        
        # Update BM25 retriever with new documents
        if len(splitted_docs) > 0:
            existing_docs = self.bm25_retriever.docs
            all_docs = existing_docs + splitted_docs
            self.bm25_retriever = BM25Retriever.from_documents(all_docs)
            self.bm25_retriever.k = 3
            
            # Recreate ensemble retriever with updated BM25
            vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, vector_retriever],
                weights=[0.4, 0.6]
            )
            
            # Recreate compression retriever with updated ensemble retriever
            compressor = FlashrankRerank()
            self.compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=self.ensemble_retriever
            )
            self.retriever = self.compression_retriever

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
