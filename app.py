from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from config.settings import ENV_PATH
from services.rag_service import RagService


load_dotenv(ENV_PATH)


st.set_page_config(
    page_title="ChipChat: ASIC Assistant",
    page_icon="assets/assistant.png",
    layout="centered",
)


st.title("ChipChat: ASIC RAG Assistant")

st.markdown(
    "Ask questions about your hardware specifications, "
    "protocols, PDFs, and approved web documentation."
)


@st.cache_resource(show_spinner=False)
def initialize_rag_pipeline():
    rag_service = RagService()
    return rag_service.create_chain()


def initialize_chat_history() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I am your ASIC assistant. "
                    "I have loaded your local documents and "
                    "approved web documentation. "
                    "What would you like to know?"
                ),
                "sources": [],
            }
        ]


def get_pdf_page(metadata: dict) -> str:
    """
    Return a readable PDF page number.

    PyPDFLoader normally stores:
        page = 0, 1, 2, ...

    so +1 is used for human-readable page numbering.
    """

    page_label = metadata.get("page_label")

    if page_label is not None:
        return str(page_label)

    page = metadata.get("page")

    if isinstance(page, int):
        return str(page + 1)

    if page is not None:
        return str(page)

    return "Unknown"


def format_sources(context) -> list[str]:
    """
    Convert retrieved LangChain documents into
    readable source labels.

    Supports:
    - PDFs
    - Web URLs
    """

    sources = []
    seen = set()

    for document in context:
        metadata = document.metadata or {}

        source = metadata.get("source", "")
        document_type = metadata.get(
            "document_type",
            "",
        ).lower()

        url = metadata.get("url")

        # Detect URL even if document_type is missing
        if not url and isinstance(source, str):
            if source.startswith(
                ("http://", "https://")
            ):
                url = source

        # -----------------------------
        # Web source
        # -----------------------------
        if document_type == "web" or url:
            if not url:
                continue

            source_key = (
                "web",
                url,
            )

            if source_key in seen:
                continue

            seen.add(source_key)

            sources.append(
                f"🌐 [{url}]({url})"
            )

            continue

        # -----------------------------
        # PDF source
        # -----------------------------
        document_name = metadata.get(
            "document_name"
        )

        if not document_name and source:
            document_name = Path(
                str(source)
            ).name

        if not document_name:
            document_name = "Unknown PDF"

        page = get_pdf_page(metadata)

        section = metadata.get("section")

        source_key = (
            "pdf",
            document_name,
            page,
            section,
        )

        if source_key in seen:
            continue

        seen.add(source_key)

        label = (
            f"📄 {document_name} "
            f"— Page {page}"
        )

        if section:
            label += f" — {section}"

        sources.append(label)

    return sources


def display_sources(
    sources: list[str],
) -> None:
    if not sources:
        return

    with st.expander("Sources"):
        for source in sources:
            st.markdown(
                f"- {source}"
            )


def display_chat_history() -> None:
    for message in st.session_state.messages:

        avatar = (
            "assets/user.png"
            if message["role"] == "user"
            else "assets/assistant.png"
        )

        with st.chat_message(
            message["role"],
            avatar=avatar,
        ):
            st.markdown(
                message["content"]
            )

            display_sources(
                message.get(
                    "sources",
                    [],
                )
            )


def process_user_question(
    user_question: str,
    rag_chain,
) -> None:

    # Save user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question,
            "sources": [],
        }
    )

    with st.chat_message(
        "user",
        avatar="assets/user.png",
    ):
        st.markdown(user_question)

    # Run RAG

    with st.chat_message(
        "assistant",
        avatar="assets/assistant.png",
    ):

        with st.spinner(
            "Searching available sources..."
        ):

            try:
                response = rag_chain.invoke(
                    {
                        "input": user_question
                    }
                )

                answer = response.get(
                    "answer",
                    "No answer was generated.",
                )

                context = response.get(
                    "context",
                    [],
                )
                    
                sources = format_sources(
                    context
                )

                st.markdown(answer)

                display_sources(
                    sources
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as error:
                st.error(
                    f"An error occurred: {error}"
                )


def main() -> None:

    initialize_chat_history()

    with st.spinner(
        "Initializing knowledge base..."
    ):
        rag_chain = (
            initialize_rag_pipeline()
        )

    display_chat_history()

    user_question = st.chat_input(
        "Ask about your documents "
        "or indexed web sources."
    )

    if user_question:
        process_user_question(
            user_question,
            rag_chain,
        )


if __name__ == "__main__":
    main()