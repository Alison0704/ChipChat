from urllib import response

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
    "Ask questions about your hardware specifications and protocols."
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
                    "I have loaded the documents in the documents "
                    "directory. What would you like to know?"
                ),
            }
        ]


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
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.markdown(f"- {source}")


def process_user_question(
    user_question: str,
    rag_chain,
) -> None:

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question,
        }
    )

    # Display user message
    with st.chat_message(
        "user",
        avatar="assets/user.png",
    ):
        st.markdown(user_question)

    
    with st.chat_message(
        "assistant",
        avatar="assets/assistant.png",
    ):
        with st.spinner("Searching specifications..."):
            try:
                response = rag_chain.invoke(
                    {"input": user_question}
                )

                answer = response["answer"]
                documents = response["context"]

                sources = format_sources(documents)

                # Display answer
                st.markdown(answer)

                # Display sources
                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            st.markdown(f"- {source}")

                # Save assistant message
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
# def format_sources(documents) -> list[str]:
#     sources = []

#     for document in documents:
#         name = document.metadata.get(
#             "document_name",
#             "Unknown document",
#         )

#         page = document.metadata.get(
#             "page_number",
#             "Unknown page",
#         )

#         citation = f"{name} — Page {page}"

#         if citation not in sources:
#             sources.append(citation)

#     return sources

def format_sources(documents) -> list[str]:
    sources = []

    for document in documents:

        document_type = document.metadata.get(
            "document_type"
        )

        if document_type == "pdf":

            name = document.metadata.get(
                "document_name",
                "Unknown PDF",
            )

            page = document.metadata.get(
                "page_number",
                "Unknown page",
            )

            citation = (
                f"📄 {name} — Page {page}"
            )

        elif document_type == "website":

            title = document.metadata.get(
                "title",
                "Website",
            )

            url = document.metadata.get(
                "url",
                document.metadata.get(
                    "source",
                    "",
                ),
            )

            citation = (
                f"🌐 {title} — {url}"
            )

        else:
            continue

        if citation not in sources:
            sources.append(citation)

    return sources


def main() -> None:
    initialize_chat_history()

    with st.spinner("Initializing document database..."):
        rag_chain = initialize_rag_pipeline()

    display_chat_history()

    user_question = st.chat_input(
        "ask away! I have your documents loaded."
    )

    if user_question:
        process_user_question(
            user_question,
            rag_chain,
        )


if __name__ == "__main__":
    main()