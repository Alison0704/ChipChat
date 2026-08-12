from pathlib import Path

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config.settings import CHAT_MODEL, PROMPT_PATH
from services.document_service import DocumentService
from services.vector_store_service import VectorStoreService


class RagService:
    def __init__(self, prompt_path: Path = PROMPT_PATH):
        self.prompt_path = prompt_path
        self.document_service = DocumentService()
        self.vector_store_service = VectorStoreService()

    def load_system_prompt(self) -> str:
        if not self.prompt_path.exists():
            raise FileNotFoundError(
                f"System prompt not found: {self.prompt_path}"
            )

        return self.prompt_path.read_text(encoding="utf-8")

    def create_prompt(self) -> ChatPromptTemplate:
        system_prompt = self.load_system_prompt()

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

    def create_chain(self):
        chunks = self.document_service.load_and_split()

        vector_store = (
            self.vector_store_service.create_vector_store(chunks)
        )

        retriever = (
            self.vector_store_service.create_retriever(vector_store)
        )

        llm = ChatOpenAI(
            model=CHAT_MODEL,
            temperature=0,
        )

        question_answer_chain = create_stuff_documents_chain(
            llm,
            self.create_prompt(),
        )

        return create_retrieval_chain(
            retriever,
            question_answer_chain,
        )