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
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "
"
        
        # Basic cleanup: remove excessive whitespace
        lines = [line.strip() for line in text.split('
') if line.strip()]
        return "
".join(lines)
    except Exception as e:
        logger.error(f"Failed to parse PDF: {e}")
        return f"Error parsing PDF: {e}"
