from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
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


class IntentSchema(BaseModel):
    """Identify user's real intent"""
    intent: Literal["knowledge_base", "chitchat"] = Field(
        description="knowledge_base: questions about document content, technical knowledge that require retrieval to answer; "
                    "chitchat: greetings, casual chat or meaningless conversation."
    )


class Ragbot:
    name: str
    llm = None
    vector_store = None
    retriever = None
    bm25_retriever = None
    ensemble_retriever = None
    compression_retriever = None
    qa_chain: Runnable = None
    intent_chain = None

    def __init__(self):
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0,
            openai_api_key=LLM_API_KEY,
            openai_api_base=LLM_BASE_URL,
        )
        self.init_documents()
        self.init_qa_chain()
        self.init_intent_chain()

    def init_intent_chain(self):
        structured_llm = self.llm.with_structured_output(IntentSchema)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an intent detection expert. Analyze the user input and classify it."),
            ("human", "{input}")
        ])
        self.intent_chain = prompt | structured_llm

    def detect_intent(self, question: str) -> str:
        result = self.intent_chain.invoke({"input": question})
        return result.intent

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
        
        intent = self.detect_intent(question)
        
        if intent == "chitchat":
            chitchat_prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个友好的助手。请简洁地回复用户的问候或闲聊。"),
                ("human", "{input}")
            ])
            chitchat_chain = chitchat_prompt | self.llm
            response = chitchat_chain.invoke({"input": question})
            return response.content if hasattr(response, "content") else str(response)
        
        response = self.qa_chain.invoke(
            {"question": question},
            config={"configurable": {"session_id": session_id}},
        )

        if isinstance(response, str):
            answer = response
        elif hasattr(response, "content"):
            answer = response.content
        elif isinstance(response, dict):
            answer = response.get("text", "No content found")
        else:
            answer = str(response)

        return answer
