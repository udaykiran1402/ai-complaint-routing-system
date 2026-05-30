"""Audio processing using Whisper for speech-to-text."""

import whisper
import torch
from pathlib import Path
from typing import Optional


class AudioProcessor:
    """Process audio files and convert to text using Whisper."""
    
    def __init__(self, model_name: str = "base"):
        """
        Initialize Whisper model.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
        """
        print(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
    
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> dict:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file
            language: Optional language code (e.g., 'en', 'hi', 'es')
        
        Returns:
            Dictionary with 'text', 'language', and 'segments'
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"Transcribing audio: {audio_path}")
        
        # Transcribe with Whisper
        result = self.model.transcribe(
            str(audio_path),
            language=language,
            fp16=False  # Use FP32 for CPU compatibility
        )
        
        return {
            "text": result["text"].strip(),
            "language": result.get("language", "unknown"),
            "segments": result.get("segments", [])
        }
    
    def transcribe_batch(self, audio_paths: list[str]) -> list[dict]:
        """Transcribe multiple audio files."""
        results = []
        for audio_path in audio_paths:
            try:
                result = self.transcribe(audio_path)
                results.append(result)
            except Exception as e:
                print(f"Error transcribing {audio_path}: {e}")
                results.append({"text": "", "language": "unknown", "error": str(e)})
        return results


if __name__ == "__main__":
    # Test audio processor
    processor = AudioProcessor(model_name="base")
    print("Audio processor initialized successfully")
