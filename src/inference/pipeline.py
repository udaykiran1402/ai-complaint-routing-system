"""Complete inference pipeline for complaint processing."""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import (
    OFFICER_ROUTER_PATH, PRIORITY_CLASSIFIER_PATH, ETA_REGRESSOR_PATH,
    FAISS_INDEX_PATH, COMPLAINT_METADATA_PATH, EMBEDDER_PATH, OFFICERS
)
from src.training.embedder import ComplaintEmbedder
from src.training.officer_router import OfficerRouter
from src.training.priority_classifier import PriorityClassifier
from src.training.eta_predictor import ETAPredictor
from src.training.similarity_search import SimilaritySearch
from src.preprocessing.text_processor import TextProcessor
from src.preprocessing.audio_processor import AudioProcessor
from src.preprocessing.video_processor import VideoProcessor


class ComplaintRoutingPipeline:
    """End-to-end pipeline for complaint processing and routing."""
    
    def __init__(self):
        """Initialize the pipeline with all models."""
        print("Loading complaint routing pipeline...")
        
        # Load models
        self.embedder = ComplaintEmbedder.load(str(EMBEDDER_PATH))
        self.officer_router = OfficerRouter.load(str(OFFICER_ROUTER_PATH))
        self.priority_classifier = PriorityClassifier.load(str(PRIORITY_CLASSIFIER_PATH))
        self.eta_predictor = ETAPredictor.load(str(ETA_REGRESSOR_PATH))
        self.similarity_search = SimilaritySearch.load(
            str(FAISS_INDEX_PATH),
            str(COMPLAINT_METADATA_PATH)
        )
        
        # Initialize processors
        self.text_processor = TextProcessor()
        self.audio_processor = None  # Lazy load
        self.video_processor = None  # Lazy load
        
        print("✓ Pipeline loaded successfully")
    
    def _get_audio_processor(self):
        """Lazy load audio processor."""
        if self.audio_processor is None:
            from src.config import WHISPER_MODEL
            self.audio_processor = AudioProcessor(model_name=WHISPER_MODEL)
        return self.audio_processor
    
    def _get_video_processor(self):
        """Lazy load video processor."""
        if self.video_processor is None:
            from src.config import WHISPER_MODEL
            self.video_processor = VideoProcessor(whisper_model=WHISPER_MODEL)
        return self.video_processor
    
    def process_text(self, text: str) -> str:
        """Process text complaint."""
        result = self.text_processor.process(text)
        return result['text']
    
    def process_audio(self, audio_path: str) -> str:
        """Process audio complaint."""
        processor = self._get_audio_processor()
        result = processor.transcribe(audio_path)
        return result['text']
    
    def process_video(self, video_path: str) -> str:
        """Process video complaint."""
        processor = self._get_video_processor()
        result = processor.process_video(video_path)
        return result['text']
    
    def predict(
        self,
        text: Optional[str] = None,
        audio_path: Optional[str] = None,
        video_path: Optional[str] = None
    ) -> Dict:
        """
        Process complaint and make predictions.
        
        Args:
            text: Text complaint (optional)
            audio_path: Path to audio file (optional)
            video_path: Path to video file (optional)
        
        Returns:
            Dictionary with all predictions
        """
        # Extract text from input
        if text:
            complaint_text = self.process_text(text)
            input_type = "text"
        elif audio_path:
            complaint_text = self.process_audio(audio_path)
            input_type = "audio"
        elif video_path:
            complaint_text = self.process_video(video_path)
            input_type = "video"
        else:
            raise ValueError("Must provide text, audio_path, or video_path")
        
        if not complaint_text:
            raise ValueError("No text extracted from input")
        
        # Generate embedding
        embedding = self.embedder.embed(complaint_text)
        
        # Make predictions
        officer_id = int(self.officer_router.predict(embedding)[0])
        officer_probs = self.officer_router.predict_proba(embedding)[0]
        
        priority_id = int(self.priority_classifier.predict(embedding)[0])
        priority_label = self.priority_classifier.priority_map[priority_id]
        priority_probs = self.priority_classifier.predict_proba(embedding)[0]
        
        eta_days = float(self.eta_predictor.predict(embedding)[0])
        
        similar_complaints = self.similarity_search.search(embedding, k=5)
        
        # Prepare result
        result = {
            "input_type": input_type,
            "complaint_text": complaint_text,
            "officer": {
                "id": officer_id,
                "name": OFFICERS[officer_id]["name"],
                "confidence": float(officer_probs[officer_id])
            },
            "priority": {
                "level": priority_label,
                "confidence": float(priority_probs[priority_id])
            },
            "eta": {
                "days": round(eta_days, 1),
                "description": self._format_eta(eta_days)
            },
            "similar_complaints": similar_complaints[:5]
        }
        
        return result
    
    def _format_eta(self, days: float) -> str:
        """Format ETA in human-readable form."""
        if days < 1:
            return "Less than 1 day"
        elif days < 2:
            return "1-2 days"
        elif days < 7:
            return f"{int(days)} days"
        elif days < 14:
            return "1-2 weeks"
        else:
            return f"{int(days/7)} weeks"


def main():
    """Test the pipeline."""
    pipeline = ComplaintRoutingPipeline()
    
    # Test with sample complaint
    test_complaint = "Water supply has been disrupted in my area for 3 days"
    print(f"\nTest complaint: {test_complaint}")
    
    result = pipeline.predict(text=test_complaint)
    
    print(f"\nResults:")
    print(f"  Officer: {result['officer']['name']} (confidence: {result['officer']['confidence']:.2%})")
    print(f"  Priority: {result['priority']['level']} (confidence: {result['priority']['confidence']:.2%})")
    print(f"  ETA: {result['eta']['days']} days ({result['eta']['description']})")
    print(f"  Similar complaints: {len(result['similar_complaints'])}")


if __name__ == "__main__":
    main()
