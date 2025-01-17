import logging
from pathlib import Path

def load_text(file_path: str) -> str:
    """Load text from file."""
    logger = logging.getLogger(__name__)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        logger.info(f"Successfully loaded text from {file_path}")
        return text
    except Exception as e:
        logger.error(f"Error loading text: {str(e)}")
        raise

def calculate_compression_ratio(original_text: str, encoded_ids: list) -> float:
    """Calculate compression ratio."""
    original_size = len(original_text.encode('utf-8'))
    encoded_size = len(encoded_ids) * 2  # Assuming 2 bytes per token ID
    return original_size / encoded_size 