"""
Document loading.

Loads PDF documents from the uploaded documents directory
and converts each page into LangChain Document objects.
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_documents(pdf_directory: Path) -> list[Document]:
    """
    Load all PDF documents from the given directory.

    Args:
        pdf_directory: Directory containing PDF files.

    Returns:
        A list of LangChain Document objects.
    """

    documents: list[Document] = []

    pdf_files = sorted(pdf_directory.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_directory}")

    for pdf_path in pdf_files:
        print(f"\nLoading: {pdf_path.name}")

        loader = PyPDFLoader(str(pdf_path))
        pdf_documents = loader.load()

        print(f"Pages loaded: {len(pdf_documents)}")

        documents.extend(pdf_documents)
    
    print(f"\nTotal Documents: {len(documents)}")

    return documents


if __name__ == "__main__":
    # Manual test
    pdf_directory = Path("data/uploaded_documents")

    documents = load_documents(pdf_directory)

    print(f"\nLoaded {len(documents)} documents.\n")

    print("First document:\n")
    print(documents[0].page_content[:250])

    print("\nMetadata:\n")
    print(documents[0].metadata)
