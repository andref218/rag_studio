from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def create_chunks(documents: list[Document]) -> list[Document]:
    """
    Split documents into smaller chunks.

    Args:
        documents: List of LangChain Document objects.

    Returns:
        A list of chunked LangChain Document objects.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = text_splitter.split_documents(documents)

    return chunks

if __name__ == "__main__":
    from pathlib import Path

    from loaders import load_documents

    pdf_directory = Path("data/uploaded_documents")

    documents = load_documents(pdf_directory)

    chunks = create_chunks(documents)

    print(f"\nDocuments: {len(documents)}")
    print(f"Chunks: {len(chunks)}")

    print("\nFirst Chunk\n")
    print(chunks[0].page_content)

    print("\nMetadata\n")
    print(chunks[0].metadata)
    
    print(f"\nAverage chunks per document: {len(chunks) / len(documents):.2f}")