import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ------------ Load Environment Variables ------------ 
load_dotenv()

st.set_page_config(page_title="ChipChat: ASIC Assistant", page_icon="⚙️", layout="centered")

st.title("⚙️ ChipChat: ASIC RAG Assistant")
st.markdown("Ask questions about your hardware specifications and protocols.")

@st.cache_resource(show_spinner=False)
def initialize_rag_pipeline():
    """Initializes the document loader, vector store, and LLM chain."""
    # 1. Load Document
    # Make sure AMBA_AXI_Protocol_Specification.pdf is in the same folder!
    loader = PyPDFLoader("AMBA_AXI_Protocol_Specification.pdf")
    docs = loader.load()

    # 2. Split Document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    # 3. Create/Load Vector Database (OpenAI Embeddings)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="./asic_chroma_db"
    )

    # 4. Initialize OpenAI LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 5. Create Prompts & Chain
    system_prompt = (
        "You are an expert ASIC Design and Verification Engineer. "
        "Use the provided pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use precise hardware terminology (e.g., clock cycles, RTL, signals). "
        "\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(
        vector_db.as_retriever(search_kwargs={"k": 4}), 
        question_answer_chain
    )
    
    return rag_chain

with st.spinner("Initializing Vector Database and OpenAI LLM... This may take a moment on the first run."):
    rag_chain = initialize_rag_pipeline()

# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add a welcome message
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Hello! I am your local ASIC assistant. I have loaded the AXI Protocol spec. What would you like to know?"
    })

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("E.g., What is the relationship between AWVALID and AWREADY?"):
    # 1. Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate response using the RAG chain
    with st.chat_message("assistant"):
        with st.spinner("Searching datasheets and thinking..."):
            try:
                response = rag_chain.invoke({"input": prompt})
                answer = response['answer']
                st.markdown(answer)
                # Save assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"An error occurred: {e}")