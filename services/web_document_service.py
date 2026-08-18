import logging
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from config.settings import (
    WEB_REQUEST_TIMEOUT,
    WEB_URLS_PATH,
    WEB_USER_AGENT,
)


logger = logging.getLogger(__name__)


class WebDocumentService:
    def load_urls(self) -> list[str]:
        if not WEB_URLS_PATH.exists():
            return []

        urls = []

        for line in WEB_URLS_PATH.read_text(
            encoding="utf-8"
        ).splitlines():
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            if self.is_valid_url(line):
                urls.append(line)
            else:
                logger.warning(
                    "Ignoring invalid URL: %s",
                    line,
                )

        return urls

    @staticmethod
    def is_valid_url(url: str) -> bool:
        parsed = urlparse(url)

        return (
            parsed.scheme in {"http", "https"}
            and bool(parsed.netloc)
        )

    @staticmethod
    def extract_text(html: str) -> str:
        soup = BeautifulSoup(
            html,
            "html.parser",
        )

        for element in soup(
            [
                "script",
                "style",
                "noscript",
                "svg",
            ]
        ):
            element.decompose()

        return soup.get_text(
            separator="\n",
            strip=True,
        )

    def load_url(
        self,
        url: str,
    ) -> Document | None:
        try:
            response = requests.get(
                url,
                timeout=WEB_REQUEST_TIMEOUT,
                headers={
                    "User-Agent": WEB_USER_AGENT,
                },
            )

            response.raise_for_status()

        except requests.RequestException as error:
            logger.warning(
                "Could not load %s: %s",
                url,
                error,
            )

            return None

        content_type = response.headers.get(
            "Content-Type",
            "",
        ).lower()

        if "text/html" in content_type:
            content = self.extract_text(
                response.text
            )

        elif (
            "text/plain" in content_type
            or "text/markdown" in content_type
        ):
            content = response.text

        else:
            logger.warning(
                "Unsupported web content type "
                "for %s: %s",
                url,
                content_type,
            )

            return None

        if not content.strip():
            logger.warning(
                "No readable text found at %s",
                url,
            )

            return None

        return Document(
            page_content=content,
            metadata={
                "source": url,
                "document_name": url,
                "document_type": "web",
                "url": url,
            },
        )

    def load_documents(
        self,
    ) -> list[Document]:
        documents = []

        for url in self.load_urls():
            document = self.load_url(url)

            if document is not None:
                documents.append(document)

        return documents