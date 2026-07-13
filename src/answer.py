from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from prompts import SYSTEM_PROMPT
from retrieval import retrieve

MODEL_NAME = "llama3.2:latest"  # Ollama model

def load_llm() -> ChatOllama:
    """
    Load the Ollama language model.
    """

    return ChatOllama(
        model=MODEL_NAME,
        temperature=0,
    )

# Load the language model once when the application starts.
llm = load_llm()


def fetch_context(question: str) -> list[Document]:
    """
    Retrieve the most relevant documents for the user's question.
    """

    return retrieve(question)


def answer_question(question: str) -> tuple[str, list[Document]]:
    """
    Answer a question using Retrieval-Augmented Generation (RAG).

    Args:
        question: User question.

    Returns:
        Tuple containing:
            - Generated answer.
            - Retrieved documents.
    """

    # Retrieve the most relevant documents from the vector database.
    documents = fetch_context(question)

    # Combine all retrieved documents into a single context string to pass to the language model.
    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # Insert the retrieved context into the system prompt.
    system_prompt = SYSTEM_PROMPT.format(
        context=context
    )


    # Build the conversation sent to the language model.
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ]

    # Generate the final answer.
    response = llm.invoke(messages)

    return response.content, documents


if __name__ == "__main__":

    question = input("Question: ")

    answer, documents = answer_question(question)

    print("\nQuestion:\n")
    print(question)

    print("\nAnswer:\n")
    print(answer)

    print("\nRetrieved Documents:\n")

    for rank, document in enumerate(documents, start=1):

        print("=" * 80)
        print(f"Rank #{rank}")
        print(f"Source: {document.metadata['source']}")
        print(f"Page: {document.metadata['page'] + 1}")

        print("\nChunk:\n")
        print(document.page_content)