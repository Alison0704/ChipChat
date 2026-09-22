# ChipChat
The AI assistant using RAG achitechture with LLM configuration to help Hardware DV engineering during development and testing of their design.
The aim is to help DV engineers to navigate their log and spec files generated from OpenROAD and help with code debugging.

## Project Implementation steps
[ ] RAG architecture implementation.<br>
[ ] Data ingestion pipeline: PDF collection and linking the knowledge base to log and specs files.<br>
[ ] Custom tokenization optimization adapted to RTL coding syntax.<br>
[ ] LLM system integration with the streamlit UI for chatbot interface. <br>
[ ] Retrieval quality testing (chunking and retrieval effectiveness). <br>
[ ] GTKwave command linked to streamlit UI.<br>
[ ] End-to-end iterative testing on how effectively this system helps with RTL code debugging.<br>
[ ] Elevenlabs setup for text-to-speech.<br>
[ ] Docker setup for easier deployment on other devices. <br>

## RAG Architecture
ChipChat follows a standard retrieval-augmented generation pipeline, implemented in [services/rag.py](services/rag.py):

1. **Ingest** — every `.pdf`, `.log`, and `.txt` file under `data/` is picked up automatically: PDFs go through `langchain_unstructured`'s `UnstructuredLoader`, logs/specs are read as plain text.
2. **Chunk** — documents are split with `RecursiveCharacterTextSplitter` (chunk size 1200, overlap 300) using RTL/UVM-aware separators (`module`/`endmodule`, `class`/`endclass`, `task`/`endtask`, `function`/`endfunction`, etc.) so a chunk doesn't get cut mid-construct, falling back to paragraph/line/word boundaries.
3. **Embed** — chunks are embedded locally via Ollama's `nomic-embed-text` model (`OllamaEmbeddings`).
4. **Store** — embeddings are persisted to disk in a `Chroma` vector database (`data/chroma_db/`, `simple-rag` collection), so the app doesn't re-embed on every restart. Use the "Rebuild knowledge base" button in the sidebar after adding new files.
5. **Retrieve** — a `MultiQueryRetriever` rewrites the user's question into several variants to pull more relevant chunks out of the vector store than a single similarity search would.
6. **Generate** — retrieved chunks are stuffed into a prompt and answered by `ChatOllama` running the `chipchat` model (see [Modelfile](Modelfile)), a `llama3.2` variant with a system prompt tuned for RTL/DV question answering and grounded, cited responses.

[app.py](app.py) is a Streamlit chat UI over this pipeline (`services/rag.py`), with a sidebar to list indexed sources, trigger a rebuild, launch [GTKWave](services/gtkwave.py) on a waveform file, and optionally read replies aloud via [ElevenLabs](services/tts.py).


## Why Ollama?
- **Keeps design data local.** DV artifacts — RTL, specs, regression logs — are often confidential; Ollama runs both the embedding model and the LLM entirely on-device, so nothing is sent to a third-party API.
- **No per-token cost for iterative use.** Debugging a testbench often means asking many small follow-up questions; a local, open-weight model avoids per-request API billing.
- **Simple model lifecycle.** `ollama pull`/`ollama run` plus a [Modelfile](Modelfile) are enough to define and version a custom system prompt and inference parameters (temperature, context window) for the `chipchat` model.
- **One local runtime for both halves of RAG.** Ollama serves the `nomic-embed-text` embedding model and the `llama3.2` chat model, so the whole pipeline in [pdf-rag.py](pdf-rag.py) has a single dependency instead of separate embedding and inference providers.
- **First-class LangChain support.** `langchain-ollama`integrates directly with `OllamaEmbeddings`/`ChatOllama`, matching the rest of the LangChain-based pipeline.