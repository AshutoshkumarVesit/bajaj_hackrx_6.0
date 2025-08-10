# src/utils/text_splitter.py

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text_into_chunks(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[str]:
    """
    Splits a large text into smaller, overlapping chunks using RecursiveCharacterTextSplitter.

    Args:
        text (str): The full document text to split.
        chunk_size (int): The desired size of each chunk (in characters).
        chunk_overlap (int): The number of characters to overlap between consecutive chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,  # Use character length
        is_separator_regex=False, # Treat separators as plain strings
    )
    chunks = text_splitter.split_text(text)
    return chunks