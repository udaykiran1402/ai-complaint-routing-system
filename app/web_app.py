"""Simple Flask web interface for complaint routing system."""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
from pathlib import Path
import tempfile
import os

sys.path.append(str(Path(__file__).parent.parent))

from src.inference.pipeline import ComplaintRoutingPipeline

app = Flask(__name__)
CORS(app)

# Initialize pipeline (lazy load)
pipeline = None


def get_pipeline():
    """Get or initialize pipeline."""
    global pipeline
    if pipeline is None:
        print("Initializing complaint routing pipeline...")
        pipeline = ComplaintRoutingPipeline()
    return pipeline


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/api/process', methods=['POST'])
def process_complaint():
    """Process complaint and return predictions."""
    import time
    temp_file = None
    
    try:
        pipe = get_pipeline()
        
        # Get input type
        if 'text' in request.form and request.form['text']:
            result = pipe.predict(text=request.form['text'])
        
        elif 'audio' in request.files and request.files['audio'].filename:
            audio_file = request.files['audio']
            original_filename = audio_file.filename
            file_ext = os.path.splitext(original_filename)[1].lower()
            
            # Validate audio extension
            valid_audio_exts = ['.wav', '.mp3', '.ogg', '.m4a', '.flac', '.aac', '.opus']
            if not file_ext or file_ext not in valid_audio_exts:
                return jsonify({'error': f'Unsupported audio format: {file_ext}. Please use WAV, MP3, OGG, or M4A'}), 400
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                temp_file = tmp.name
                audio_file.save(tmp.name)
            
            result = pipe.predict(audio_path=temp_file)
        
        elif 'video' in request.files and request.files['video'].filename:
            video_file = request.files['video']
            original_filename = video_file.filename
            file_ext = os.path.splitext(original_filename)[1].lower()
            
            # Validate video extension
            valid_video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v']
            if not file_ext or file_ext not in valid_video_exts:
                return jsonify({'error': f'Unsupported video format: {file_ext}. Please use MP4, AVI, or MOV'}), 400
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                temp_file = tmp.name
                video_file.save(tmp.name)
            
            result = pipe.predict(video_path=temp_file)
        
        else:
            return jsonify({'error': 'No input provided. Please enter text or upload audio/video file.'}), 400
        
        # Clean up temp file after a delay
        if temp_file:
            try:
                time.sleep(0.5)
                os.unlink(temp_file)
            except:
                pass
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error processing complaint: {error_details}")
        
        # Clean up temp file
        if temp_file:
            try:
                time.sleep(0.5)
                os.unlink(temp_file)
            except:
                pass
        
        # Return user-friendly error
        error_msg = str(e)
        if 'Failed to load audio' in error_msg or 'Invalid data' in error_msg:
            return jsonify({'error': 'Failed to process file. Please ensure it is a valid audio/video file.'}), 500
        
        return jsonify({'error': f'Processing error: {error_msg}'}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print("Starting Complaint Routing Web App...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)
