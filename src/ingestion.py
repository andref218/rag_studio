"""
Document ingestion pipeline.

Coordinates the document indexing process by loading PDFs,
creating chunks and generating the vector database.
"""

from pathlib import Path

from src.loaders import load_documents
from src.chunking import create_chunks
from src.embeddings import create_vector_store


def ingest_documents(pdf_directory: Path):
    """
    Run the complete document ingestion pipeline.

    Pipeline:
        1. Load PDF documents.
        2. Split documents into chunks.
        3. Generate embeddings.
        4. Store embeddings in the Chroma vector database.

    Args:
        pdf_directory: Directory containing PDF files.
    """

    # Load PDF documents.
    yield 25, "📄 Loading PDF documents..."

    documents = load_documents(pdf_directory)

    # Split documents into smaller chunks.
    yield 50, "✂️ Creating document chunks..."

    chunks = create_chunks(documents)

    # Create the vector database.
    yield 75, "🧠 Generating embeddings..."

    create_vector_store(chunks)

    print("\nDocument ingestion completed successfully.")


if __name__ == "__main__":

    pdf_directory = Path("data/uploaded_documents")

    ingest_documents(pdf_directory)