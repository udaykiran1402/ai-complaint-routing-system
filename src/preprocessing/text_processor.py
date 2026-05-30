"""Text preprocessing and cleaning."""

import re
from typing import Optional


class TextProcessor:
    """Process and clean text complaints."""
    
    def __init__(self):
        """Initialize text processor."""
        pass
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text input
        
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def process(self, text: str, language: Optional[str] = None) -> dict:
        """
        Process text complaint.
        
        Args:
            text: Raw text input
            language: Optional language hint
        
        Returns:
            Dictionary with processed text and metadata
        """
        cleaned_text = self.clean_text(text)
        
        return {
            "text": cleaned_text,
            "language": language or "unknown",
            "original_length": len(text),
            "cleaned_length": len(cleaned_text)
        }
    
    def process_batch(self, texts: list[str]) -> list[dict]:
        """Process multiple text complaints."""
        return [self.process(text) for text in texts]


if __name__ == "__main__":
    # Test text processor
    processor = TextProcessor()
    sample = "My water supply has been   disrupted!!! Please help..."
    result = processor.process(sample)
    print(f"Original: {sample}")
    print(f"Cleaned: {result['text']}")
