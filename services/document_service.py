from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_DIRECTORY,
    WEB_SOURCES,
)


class DocumentService:

    def __init__(
        self,
        documents_directory: Path = DOCUMENTS_DIRECTORY,
    ):
        self.documents_directory = documents_directory

    def find_pdf_files(self) -> list[Path]:
        if not self.documents_directory.exists():
            return []

        return sorted(
            path
            for path in self.documents_directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() == ".pdf"
        )

    def load_pdf_documents(self) -> list[Document]:
        documents = []

        for pdf_path in self.find_pdf_files():

            loader = PyPDFLoader(
                str(pdf_path)
            )

            loaded_pages = loader.load()

            for page_number, page in enumerate(
                loaded_pages,
                start=1,
            ):
                page.metadata["document_name"] = (
                    pdf_path.name
                )

                page.metadata["document_path"] = (
                    str(pdf_path)
                )

                page.metadata["document_type"] = "pdf"

                page.metadata["page_number"] = (
                    page_number
                )

            documents.extend(loaded_pages)

        return documents

    def load_web_documents(self) -> list[Document]:
        documents = []

        for url in WEB_SOURCES:

            loader = WebBaseLoader(
                web_paths=(url,)
            )

            loaded_pages = loader.load()

            for page in loaded_pages:
                page.metadata["document_type"] = "website"
                page.metadata["url"] = url

            documents.extend(loaded_pages)

        return documents

    def load_documents(self) -> list[Document]:

        documents = []

        documents.extend(
            self.load_pdf_documents()
        )

        documents.extend(
            self.load_web_documents()
        )

        if not documents:
            raise FileNotFoundError(
                "No PDF or website documents were loaded."
            )

        return documents

    def split_documents(
        self,
        documents: list[Document],
    ) -> list[Document]:

        text_splitter = RecursiveCharacterTextSplitter(
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

        return text_splitter.split_documents(
            documents
        )

    def load_and_split(self) -> list[Document]:

        documents = self.load_documents()

        return self.split_documents(
            documents
        )