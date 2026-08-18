from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from config.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_DIRECTORY,
)


class DocumentService:
    def __init__(
        self,
        documents_directory: Path = DOCUMENTS_DIRECTORY,
    ):
        self.documents_directory = documents_directory

    def find_pdf_files(self) -> list[Path]:
        if not self.documents_directory.exists():
            raise FileNotFoundError(
                f"Documents directory does not exist: "
                f"{self.documents_directory}"
            )

        return sorted(
            path
            for path in self.documents_directory.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower() == ".pdf"
            )
        )

    def load_documents(self) -> list[Document]:
        documents: list[Document] = []

        for pdf_path in self.find_pdf_files():
            loader = PyPDFLoader(
                str(pdf_path)
            )

            loaded_pages = loader.load()

            for page in loaded_pages:
                page.metadata["source"] = (
                    str(pdf_path)
                )

                page.metadata["document_name"] = (
                    pdf_path.name
                )

                page.metadata["document_path"] = (
                    str(pdf_path)
                )

                page.metadata["document_type"] = (
                    "pdf"
                )

            documents.extend(loaded_pages)

        return documents

    def split_documents(
        self,
        documents: list[Document],
    ) -> list[Document]:
        text_splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=[
                    "\n\n",
                    "\n",
                    ". ",
                    " ",
                    "",
                ],
            )
        )

        return text_splitter.split_documents(
            documents
        )