from pathlib import Path

from langchain_classic.chains import (
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
)
from langchain_openai import ChatOpenAI

from config.settings import (
    CHAT_MODEL,
    PROMPT_PATH,
)
from services.document_service import (
    DocumentService,
)
from services.vector_store_service import (
    VectorStoreService,
)
from services.web_document_service import (
    WebDocumentService,
)


class RagService:
    def __init__(
        self,
        prompt_path: Path = PROMPT_PATH,
    ):
        self.prompt_path = prompt_path

        self.document_service = (
            DocumentService()
        )

        self.web_document_service = (
            WebDocumentService()
        )

        self.vector_store_service = (
            VectorStoreService()
        )

    def load_system_prompt(self) -> str:
        if not self.prompt_path.exists():
            raise FileNotFoundError(
                f"System prompt not found: "
                f"{self.prompt_path}"
            )

        return self.prompt_path.read_text(
            encoding="utf-8"
        )

    def create_prompt(
        self,
    ) -> ChatPromptTemplate:
        system_prompt = (
            self.load_system_prompt()
        )

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

    @staticmethod
    def create_document_prompt():
        return PromptTemplate.from_template(
            """
            Source: {source}
            Type: {document_type}

            Content:
            {page_content}
            """.strip()
                    )

    def load_all_documents(self):
        pdf_documents = (
            self.document_service.load_documents()
        )

        web_documents = (
            self.web_document_service.load_documents()
        )

        documents = (
            pdf_documents
            + web_documents
        )

        if not documents:
            raise FileNotFoundError(
                "No PDFs or web documents "
                "were successfully loaded."
            )

        return documents

    def create_chain(self):
        documents = self.load_all_documents()

        chunks = (
            self.document_service
            .split_documents(documents)
        )

        print(
            f"\nTotal chunks: {len(chunks)}"
        )

        web_chunks = [
            chunk
            for chunk in chunks
            if chunk.metadata.get(
                "document_type"
            ) == "web"
        ]

        print(
            f"Web chunks: {len(web_chunks)}"
        )

        vector_store = (
            self.vector_store_service
            .create_vector_store(chunks)
        )


        retriever = (
            self.vector_store_service
            .create_retriever(
                vector_store
            )
        )

        llm = ChatOpenAI(
            model=CHAT_MODEL,
            temperature=0,
        )

        question_answer_chain = (
            create_stuff_documents_chain(
                llm,
                self.create_prompt(),
                document_prompt=(
                    self.create_document_prompt()
                ),
            )
        )

        return create_retrieval_chain(
            retriever,
            question_answer_chain,
        )