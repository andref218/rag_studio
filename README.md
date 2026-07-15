# RAG Studio

A Retrieval-Augmented Generation (RAG) application for semantic question answering over PDF documents using a local Large Language Model (LLM).

## Overview

RAG Studio allows users to upload PDF documents, index their contents into a vector database and ask natural language questions about the uploaded documents.

Instead of relying on the model's internal knowledge, answers are generated using the information retrieved from the indexed documents, reducing hallucinations and improving factual accuracy.

---

## Features

- Upload one or multiple PDF documents.
- Automatic document ingestion and indexing.
- Document chunking for semantic retrieval.
- Embedding generation using Hugging Face embedding models.
- Vector storage with ChromaDB.
- Semantic search over indexed documents.
- Context-aware answer generation using a local LLM.
- Conversational interface with chat history.
- Display of retrieved document chunks used to generate each answer.

---

## Architecture

```text
                  Document Indexing
┌─────────────────────────────────────────────────────┐
│                                                     │
│  PDF Documents                                      │
│        │                                            │
│        ▼                                            │
│  Document Loader                                    │
│        │                                            │
│        ▼                                            │
│     Chunking                                        │
│        │                                            │
│        ▼                                            │
│  Embedding Model                                    │
│        │                                            │
│        ▼                                            │
│  Vector Database (ChromaDB)                         │
│                                                     │
└─────────────────────────────────────────────────────┘


                  Question Answering
┌─────────────────────────────────────────────────────┐
│                                                     │
│  User Question                                      │
│        │                                            │
│        ▼                                            │
│  Embedding Model                                    │
│        │                                            │
│        ▼                                            │
│  Semantic Retrieval (ChromaDB)                      │
│        │                                            │
│        ▼                                            │
│  Retrieved Context                                  │
│        │                                            │
│        ▼                                            │
│  Local LLM (Llama 3.2)                              │
│        │                                            │
│        ▼                                            │
│  Final Answer                                       │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## Technologies

- Python
- LangChain
- Ollama
- Llama 3.2
- ChromaDB
- Hugging Face Embeddings
- Gradio

---

## Project Structure

```text
rag_studio/
│
├── app.py
├── pyproject.toml
├── uv.lock
├── .python-version
├── README.md
│
├── data/
│   ├── uploaded_documents/
│   └── vector_store/
│
└── src/
    ├── answer.py
    ├── chunking.py
    ├── embeddings.py
    ├── file_manager.py
    ├── ingestion.py
    ├── loaders.py
    ├── prompts.py
    ├── retrieval.py
    └── utils.py
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/andref218/rag_studio.git
cd rag_studio
```

Install the dependencies:

```bash
uv sync
```

Install Ollama:

https://ollama.com

Download the language model:

```bash
ollama pull llama3.2
```

Run the application:

```bash
uv run app.py
```

---

## Configuration

Optionally, create a `.env` file based on `.env.example` and set your Hugging Face access token to avoid download rate limits when retrieving embedding models.

```env
HF_TOKEN=your_huggingface_token
```

---

## Usage

1. Upload one or more PDF documents.
2. Click **Index Documents**.
3. Wait until indexing is complete.
4. Ask questions about the uploaded documents.
5. Inspect the retrieved document chunks used to generate the answer.

---

## Future Improvements

The project currently implements the complete Retrieval-Augmented Generation (RAG) pipeline, including document ingestion, semantic retrieval and context-aware answer generation.

Future work includes evaluating the retrieval component using standard Information Retrieval metrics to measure and improve the effectiveness of the semantic search process.
