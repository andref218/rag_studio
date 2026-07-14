from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.config import TOP_K

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

VECTOR_DB_PATH = "data/vector_db"

# Load the embedding model once when the application starts.
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

def load_vector_store() -> Chroma:
    """
    Load the existing Chroma vector database.
    """
  
    return Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings,
    )

def create_retriever() -> BaseRetriever:
    """
    Create a retriever from the vector store.
    """

    vectorstore = load_vector_store()

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": TOP_K,
            "fetch_k": 30,
        },
    )

    return retriever

def retrieve(question: str) -> list[Document]:
    """
    Retrieve the most relevant chunks for a given question.

    Args:
        question: User question.

    Returns:
        List of retrieved documents.
    """

    retriever = create_retriever()

    return retriever.invoke(question)


if __name__ == "__main__":

    question = input("Question: ")

    documents = retrieve(question)

    print(f"\nQuestion: {question}")

    print(f"\nRetrieved {len(documents)} chunks\n")

    for rank, document in enumerate(documents, start=1):

        print("=" * 80)
        print(f"Rank #{rank}")
        print(f"Source: {document.metadata['source']}")
        print(f"Page: {document.metadata['page'] + 1}")

        print("\nChunk:\n")
        print(document.page_content)