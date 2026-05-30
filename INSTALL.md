# Manual Installation Guide

## Step-by-Step Installation (Windows)

### 1. Upgrade pip and install setuptools first
```cmd
python -m pip install --upgrade pip
pip install setuptools>=65.0.0 wheel
```

### 2. Install core dependencies
```cmd
pip install numpy>=1.24.0,<2.0.0
pip install pandas>=2.0.0
```

### 3. Install PyTorch (CPU version)
```cmd
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install ML libraries
```cmd
pip install scikit-learn>=1.3.0
pip install sentence-transformers>=2.2.0
pip install faiss-cpu>=1.7.4
pip install imbalanced-learn>=0.11.0
pip install joblib>=1.3.0
```

### 5. Install Whisper (for audio/video support)
```cmd
pip install git+https://github.com/openai/whisper.git
```

**Alternative if git install fails:**
```cmd
pip install openai-whisper
```

### 6. Install audio processing libraries
```cmd
pip install librosa>=0.10.0
pip install soundfile>=0.12.1
```

### 7. Install FFmpeg for video processing

**Windows:**
- Download from: https://www.gyan.dev/ffmpeg/builds/
- Extract and add to PATH
- Or use: `pip install ffmpeg-python>=0.2.0` (Python wrapper only)

**Verify FFmpeg:**
```cmd
ffmpeg -version
```

### 8. Install remaining dependencies
```cmd
pip install nltk>=3.8.0
pip install langdetect>=1.0.9
pip install tqdm>=4.65.0
pip install pyyaml>=6.0
pip install faker>=19.0.0
```

### 9. Install web interface dependencies (optional)
```cmd
pip install flask>=2.3.0
pip install flask-cors>=4.0.0
```

## Minimal Installation (Text-only, no audio/video)

If you only need text processing and want to skip audio/video:

```cmd
pip install --upgrade pip setuptools wheel
pip install numpy>=1.24.0,<2.0.0 pandas>=2.0.0
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn>=1.3.0 sentence-transformers>=2.2.0
pip install faiss-cpu>=1.7.4 joblib>=1.3.0
pip install faker>=19.0.0 tqdm>=4.65.0
```

Then comment out audio/video imports in the code or skip those features.

## Verify Installation

```cmd
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import sentence_transformers; print('Sentence Transformers: OK')"
python -c "import sklearn; print('Scikit-learn:', sklearn.__version__)"
python -c "import faiss; print('FAISS: OK')"
python -c "import whisper; print('Whisper: OK')"
```

## Next Steps

After successful installation:

1. **Download models:**
```cmd
python scripts\download_models.py
```

2. **Generate training data:**
```cmd
python scripts\generate_data.py
```

3. **Train models:**
```cmd
python scripts\train_pipeline.py
```

4. **Test the system:**
```cmd
python app\cli.py --text "Water supply disrupted for 3 days"
```

## Troubleshooting

### Issue: Whisper installation fails
**Solution:** Skip Whisper for now, use text-only mode:
```cmd
pip install torch sentence-transformers scikit-learn faiss-cpu faker
```

### Issue: FAISS installation fails
**Solution:** Try conda instead:
```cmd
conda install -c conda-forge faiss-cpu
```

### Issue: NumPy version conflict
**Solution:** Install specific compatible version:
```cmd
pip install numpy==1.24.3
```

### Issue: Out of memory during model download
**Solution:** Download models one at a time in `scripts/download_models.py`

## System Requirements

- **Minimum:** 4GB RAM, 2GB disk space
- **Recommended:** 8GB RAM, 5GB disk space
- **Python:** 3.8, 3.9, 3.10, or 3.11
- **OS:** Windows 10/11, Linux, macOS

## Optional: GPU Support

For faster inference with CUDA GPU:

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Requires NVIDIA GPU with CUDA 11.8 or compatible drivers.
