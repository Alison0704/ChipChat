from pathlib import Path

import chromadb

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config.settings import (
    CHROMA_DIRECTORY,
    EMBEDDING_MODEL,
    RETRIEVAL_COUNT,
)


class VectorStoreService:
    COLLECTION_NAME = "asic_documents"

    def __init__(
        self,
        persist_directory: Path = CHROMA_DIRECTORY,
    ):
        self.persist_directory = persist_directory

        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL
        )

    def create_vector_store(
        self,
        chunks: list[Document],
    ) -> Chroma:

        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        collection_names = [
            collection.name
            for collection
            in client.list_collections()
        ]

        if self.COLLECTION_NAME in collection_names:
            client.delete_collection(
                self.COLLECTION_NAME
            )

        return Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.COLLECTION_NAME,
            persist_directory=(
                str(self.persist_directory)
            ),
        )

    def create_retriever(
        self,
        vector_store: Chroma,
    ):
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": RETRIEVAL_COUNT,
            },
        )