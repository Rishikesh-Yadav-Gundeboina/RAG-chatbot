import pymupdf as fitz
import os


class InvalidPDFError(Exception):
    pass


def validate_pdf(content: bytes) -> None:
    try:
        document = fitz.open(stream=content, filetype="pdf")
        document.close()
    except Exception as exc:
        raise InvalidPDFError("File could not be parsed as a valid PDF") from exc


def extract_pages(file_path: str):
    # Debug: Check if the file actually exists
    if not os.path.exists(file_path):
        print(f"ERROR: File not found at: {file_path}")
        return []

    print(f"SUCCESS: File found! Opening: {file_path}")


    document = fitz.open(file_path)
    pages= []
    for page_number,page in enumerate(document):
        text = page.get_text()
        pages.append({
            "page_number":page_number+1,
            "text":text
        })
    document.close()
    return pages


