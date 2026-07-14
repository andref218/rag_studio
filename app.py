"""
Gradio application entry point.

Provides the user interface for uploading PDF documents,
indexing them into a vector database and chatting with the RAG system.
"""

import gradio as gr

from src.answer import answer_question
from src.file_manager import save_uploaded_files
from src.ingestion import ingest_documents


def format_context(documents):
    """
    Format retrieved documents for display.
    """

    if not documents:
        return (
            "## Retrieved Documents 📚\n\n"
            "*No documents were retrieved for this message.*"
        )

    result = "<h2 style='color:#ff7800;'>📚 Retrieved Documents</h2><br>"

    for rank, document in enumerate(documents, start=1):

        result += f"<h3>Rank {rank}</h3>"

        result += (
            f"<span style='color:#ff7800;'>"
            f"<b>Source:</b> {document.metadata['source']}"
            f"</span><br>"
        )

        result += f"<b>Page:</b> {document.metadata['page'] + 1}<br><br>"

        result += (
            "<pre style='white-space: pre-wrap;"
            "padding:12px;"
            "border-radius:8px;"
            "background:#1f1f1f;"
            "color:white;'>"
        )

        result += document.page_content

        result += "</pre><br>"

    return result


def index_documents(files):
    """
    Save uploaded PDF files and create the vector database.
    """

    if not files:
        yield 0, "Please upload at least one PDF. ⚠️"
        return

    yield 10, "Saving uploaded PDF files... 💾 "

    pdf_directory = save_uploaded_files(files)

    for progress, message in ingest_documents(pdf_directory):
        yield progress, message

    yield (
        100,
        f"""
        Successfully indexed **{len(files)}** PDF document(s).

        The documents are now ready for semantic search. ✅
        """
    )

def chat(history):
    """
    Generate an answer for the latest user message.
    """

    question = history[-1]["content"]

    prior_history = history[:-1]

    answer, documents = answer_question(
        question,
        prior_history,
    )

    history.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    return history, format_context(documents)


def add_user_message(message, history):
    """
    Add the user's message to the conversation.
    """

    return "", history + [
        {
            "role": "user",
            "content": message,
        }
    ]


def main():

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="blue",
        neutral_hue="slate",
        font=["Inter", "system-ui", "sans-serif"],
    )

    with gr.Blocks(
        title="RAG Studio",
        theme=theme,
    ) as demo:

        gr.Markdown(
            """
            # RAG Studio

            Upload PDF documents, index them into a vector database and ask questions using a local LLM.
            """
        )

        with gr.Group():

            gr.Markdown("## Document Ingestion 📄")

           

            upload = gr.File(
                label="Upload PDF Documents",
                file_types=[".pdf"],
                file_count="multiple",
                scale=4,
            )

            index_button = gr.Button(
                "Index Documents",
                variant="primary",
                scale=1,
                )

            progress_bar = gr.Slider(
                minimum=0,
                maximum=100,
                value=0,
                step=1,
                interactive=False,
                label="Indexing Progress",
        )

            status = gr.Markdown(
                "No documents indexed."
            )

        gr.Markdown("---")

        with gr.Row(equal_height=True):

            with gr.Column(scale=3):

                chatbot = gr.Chatbot(
                    label="Conversation 💬",
                    type="messages",
                    height=650,
                    show_copy_button=True,
                )

                message = gr.Textbox(
                    placeholder="Ask a question about your documents...",
                    show_label=False,
                )

            with gr.Column(scale=2):

                context = gr.Markdown(
                    value="Retrieved documents will appear here.",
                    label="Retrieved Documents 📚",
                    container=True,
                    height=650,
                )

        index_button.click(
            fn=index_documents,
            inputs=upload,
            outputs=[
                progress_bar,
                status,
            ],
        )

        message.submit(
            fn=add_user_message,
            inputs=[message, chatbot],
            outputs=[message, chatbot],
        ).then(
            fn=chat,
            inputs=chatbot,
            outputs=[chatbot, context],
        )

    demo.launch(
        inbrowser=True,
    )


if __name__ == "__main__":
    main()