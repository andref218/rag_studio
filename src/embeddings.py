from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

VECTOR_DB_PATH = "data/vector_db"


def create_vector_store(chunks: list[Document]) -> Chroma:
    """
    Create a Chroma vector database from a list of document chunks.

    Args:
        chunks: List of LangChain Document chunks.

    Returns:
        A Chroma vector store.
    """

    # Initialize the embedding model used to convert text into vectors.
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    db_path = Path(VECTOR_DB_PATH)

    # Remove the existing vector database to ensure it only contains
    # embeddings for the current set of uploaded documents.
    if db_path.exists():
        Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings).delete_collection()

    # Generate embeddings for all chunks and store them in Chroma.
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH,
    )

    # Access the Chroma collection to inspect its contents.
    collection = vectorstore._collection

    # Total number of stored vectors.
    count = collection.count()

    # Retrieve one embedding to determine the embedding dimensionality.
    sample_embedding = collection.get(
        limit=1,
        include=["embeddings"]
    )["embeddings"][0]

    dimensions = len(sample_embedding)

    print(f"\nVector Store created successfully.")
    print(f"Vectors: {count}")
    print(f"Embedding dimensions: {dimensions}")
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")

    return vectorstore


if __name__ == "__main__":
    from pathlib import Path

    from loaders import load_documents
    from chunking import create_chunks

    pdf_directory = Path("data/uploaded_documents")

    documents = load_documents(pdf_directory)

    chunks = create_chunks(documents)

    create_vector_store(chunks)