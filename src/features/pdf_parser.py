import logging
from pypdf import PdfReader
import io

logger = logging.getLogger("PDFParser")

def extract_text_from_pdf(pdf_file) -> str:
    """
    Extracts and cleans text from a PDF file object.
    """
    try:
        reader = PdfReader(pdf_file)
        text = ""
        page_count = len(reader.pages)
        
        if page_count == 0:
            return "Error: The uploaded PDF has no pages."

        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            except Exception as e:
                logger.warning(f"Failed to extract text from page {i+1}: {e}")
        
        if not text.strip():
            return "Error: Could not extract any readable text from the PDF. It might be an image-only scan."

        # Basic cleanup: remove excessive whitespace
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Failed to parse PDF: {e}")
        return f"Error parsing PDF: {e}"