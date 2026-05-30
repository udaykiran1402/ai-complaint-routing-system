"""Video processing - extract audio and transcribe."""

import ffmpeg
import tempfile
from pathlib import Path
from typing import Optional
from .audio_processor import AudioProcessor


class VideoProcessor:
    """Process video files by extracting audio and transcribing."""
    
    def __init__(self, whisper_model: str = "base"):
        """Initialize with audio processor."""
        self.audio_processor = AudioProcessor(model_name=whisper_model)
    
    def extract_audio(self, video_path: str, output_path: Optional[str] = None) -> str:
        """
        Extract audio from video file.
        
        Args:
            video_path: Path to video file
            output_path: Optional output path for audio file
        
        Returns:
            Path to extracted audio file
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Create temporary file if output path not specified
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_path = temp_file.name
            temp_file.close()
        
        print(f"Extracting audio from video: {video_path}")
        
        try:
            # Extract audio using ffmpeg
            stream = ffmpeg.input(str(video_path))
            stream = ffmpeg.output(stream, output_path, acodec='pcm_s16le', ac=1, ar='16k')
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            
            return output_path
        except ffmpeg.Error as e:
            raise RuntimeError(f"Error extracting audio: {e.stderr.decode()}")
    
    def process_video(self, video_path: str, language: Optional[str] = None) -> dict:
        """
        Process video: extract audio and transcribe.
        
        Args:
            video_path: Path to video file
            language: Optional language code
        
        Returns:
            Dictionary with transcription results
        """
        # Extract audio
        audio_path = self.extract_audio(video_path)
        
        try:
            # Transcribe audio
            result = self.audio_processor.transcribe(audio_path, language=language)
            result["video_path"] = str(video_path)
            return result
        finally:
            # Clean up temporary audio file
            Path(audio_path).unlink(missing_ok=True)
    
    def process_batch(self, video_paths: list[str]) -> list[dict]:
        """Process multiple video files."""
        results = []
        for video_path in video_paths:
            try:
                result = self.process_video(video_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                results.append({"text": "", "language": "unknown", "error": str(e)})
        return results


if __name__ == "__main__":
    # Test video processor
    processor = VideoProcessor(whisper_model="base")
    print("Video processor initialized successfully")
