"""
File management utilities.

Handles uploaded PDF files by saving them to the project
directory and preparing them for document ingestion.
"""

from pathlib import Path
import shutil

UPLOAD_DIRECTORY = Path("data/uploaded_documents")


def save_uploaded_files(files) -> Path:
    """
    Save uploaded PDF files to the upload directory.

    Existing PDF files are removed before saving the new ones.

    Args:
        files: List of uploaded files from Gradio.

    Returns:
        Path to the upload directory.
    """

    # Create the upload directory if it does not exist.
    UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # Remove existing PDF files.
    for pdf_file in UPLOAD_DIRECTORY.glob("*.pdf"):
        pdf_file.unlink()

    # Copy uploaded files into the upload directory.
    for file in files:

        destination = UPLOAD_DIRECTORY / Path(file.name).name

        shutil.copy(file.name, destination)

    return UPLOAD_DIRECTORY