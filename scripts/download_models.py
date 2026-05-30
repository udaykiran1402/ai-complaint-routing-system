"""Download required models for offline use."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.config import SENTENCE_TRANSFORMER_MODEL, WHISPER_MODEL
from sentence_transformers import SentenceTransformer
import whisper


def download_sentence_transformer():
    """Download sentence transformer model."""
    print(f"Downloading sentence transformer: {SENTENCE_TRANSFORMER_MODEL}")
    model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    print(f"✓ Sentence transformer downloaded successfully")
    return model


def download_whisper():
    """Download Whisper model."""
    print(f"\nDownloading Whisper model: {WHISPER_MODEL}")
    model = whisper.load_model(WHISPER_MODEL)
    print(f"✓ Whisper model downloaded successfully")
    return model


def main():
    """Download all required models."""
    print("="*60)
    print("DOWNLOADING MODELS FOR OFFLINE USE")
    print("="*60)
    print("\nThis will download:")
    print(f"  1. Sentence Transformer: {SENTENCE_TRANSFORMER_MODEL}")
    print(f"  2. Whisper: {WHISPER_MODEL}")
    print("\nThis may take a few minutes depending on your internet speed...")
    print()
    
    try:
        # Download models
        download_sentence_transformer()
        download_whisper()
        
        print("\n" + "="*60)
        print("ALL MODELS DOWNLOADED SUCCESSFULLY!")
        print("="*60)
        print("\nYou can now run the system offline.")
        print("\nNext steps:")
        print("  1. Generate training data: python scripts/generate_data.py")
        print("  2. Train models: python scripts/train_pipeline.py")
        print("  3. Run inference: python app/cli.py --text 'Your complaint'")
        
    except Exception as e:
        print(f"\n✗ Error downloading models: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
