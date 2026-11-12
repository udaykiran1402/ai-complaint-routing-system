AI-Powered Complaint Auto-Routing System

An end-to-end machine learning system that automatically processes public complaints, predicts their priority, routes them to the most relevant officers, estimates resolution time, and retrieves similar historical complaints.

Built with: Python · Machine Learning · NLP · Sentence Transformers · FAISS · Whisper · Scikit-learn

"GitHub Repository" (https://github.com/udaykiran1402/ai-complaint-routing-system)

---

Project Overview

Public grievance systems often receive complaints in different formats and languages. Manually reviewing each complaint, identifying the responsible department, assigning priority, and estimating resolution time can be time-consuming.

The AI-Powered Complaint Auto-Routing System aims to automate these tasks through machine learning and natural language processing.

The system accepts complaint text and audio input, extracts relevant information, predicts the responsible officer and complaint priority, estimates the expected resolution time, and retrieves similar complaints using semantic similarity search.

The project demonstrates how multiple machine learning components can be integrated into a practical complaint-processing workflow.

Key Features

- Multimodal complaint input: Supports text and audio-based complaints.
- Audio transcription: Uses OpenAI Whisper for converting speech into text.
- Automatic officer routing: Predicts the most relevant officer for a complaint.
- Priority prediction: Classifies complaints based on their urgency.
- Resolution time estimation: Predicts the expected time required to resolve a complaint.
- Similar complaint retrieval: Uses sentence embeddings and FAISS to find semantically similar complaints.
- CLI and web application: Provides interfaces for interacting with the system.
- Local inference: Supports local model inference after the required models and dependencies have been downloaded.

Project Screenshots

1. Complaint Routing Result

"Complaint Routing Result 1" (https://github.com/udaykiran1402/ai-complaint-routing-system/blob/main/results/WhatsApp%20Image%202026-05-30%20at%2012.31.16.jpeg?raw=true)

2. Officer and Priority Prediction

"Officer and Priority Prediction" (https://github.com/udaykiran1402/ai-complaint-routing-system/blob/main/results/WhatsApp%20Image%202026-05-30%20at%2012.31.23.jpeg?raw=true)

3. System Output

"System Output" (https://github.com/udaykiran1402/ai-complaint-routing-system/blob/main/results/WhatsApp%20Image%202026-05-30%20at%2012.31.28.jpeg?raw=true)

System Architecture

The system is organized into multiple processing stages.

flowchart TD
    A[Complaint Input] --> B{Input Type}
    B -->|Text| C[Text Processing]
    B -->|Audio| D[Whisper Speech-to-Text]
    D --> C
    C --> E[Text Embedding]
    E --> F[Officer Routing Model]
    E --> G[Priority Prediction Model]
    E --> H[ETA Prediction Model]
    E --> I[FAISS Similarity Search]
    F --> J[Complaint Results]
    G --> J
    H --> J
    I --> J

Processing Workflow

1. Input collection: The user submits a complaint through text or audio.
2. Audio transcription: Audio input is transcribed into text using Whisper.
3. Text processing: The complaint text is prepared for model inference.
4. Embedding generation: Sentence Transformers convert complaint text into numerical representations.
5. Officer routing: The routing model predicts the appropriate officer.
6. Priority classification: The priority model predicts the complaint's urgency.
7. ETA prediction: The regression model estimates the expected resolution time.
8. Similarity retrieval: FAISS searches for semantically similar complaints.
9. Result generation: The system presents the predictions and related complaints.

Machine Learning Components

Component| Approach| Purpose
Officer Routing| Sentence embeddings and classification| Identify the appropriate officer
Priority Prediction| Machine learning classification| Predict complaint priority
ETA Prediction| Regression| Estimate resolution time
Similarity Search| Sentence embeddings and FAISS| Retrieve similar complaints
Audio Processing| Whisper| Convert speech into text

Technology Stack

Programming Language

- Python

Machine Learning and Data Processing

- Scikit-learn
- NumPy
- Pandas

Natural Language Processing

- Sentence Transformers
- multilingual-MiniLM-L12-v2

Audio Processing

- OpenAI Whisper

Similarity Search

- FAISS

Application

- Python CLI
- Web application

Development and Testing

- Git
- GitHub
- Python testing tools

Model Evaluation

The following metrics are the project's reported final evaluation results.

Component| Evaluation Metric| Result
Officer Routing| Accuracy| 87.3%
Officer Routing| Macro F1-score| 0.85
Priority Prediction| Accuracy| 82.1%
Priority Prediction| Macro F1-score| 0.80
ETA Prediction| MAE| 1.8 days
ETA Prediction| RMSE| 2.4 days
ETA Prediction| R²| 0.76
Similarity Search| Recall@5| 0.78
Similarity Search| Recall@10| 0.89
Similarity Search| MRR| 0.71

These are reported project evaluation metrics, not claims of production deployment or real-world operational performance.

Dataset and Similarity Search

The project includes a synthetic complaint dataset for model development and demonstration.

The demo documentation reports:

- 1,000 training complaint samples.
- 200 test complaint samples.
- A FAISS index built using 1,000 complaint vectors.

The similarity search component uses sentence embeddings to represent complaints and FAISS to retrieve semantically similar records.

Real-world deployment would require validation using representative, appropriately governed complaint data.

Installation

Prerequisites

- Python installed on your system.
- Git.
- FFmpeg for audio processing.
- Required model files and dependencies.

1. Clone the repository

git clone https://github.com/udaykiran1402/ai-complaint-routing-system.git

cd ai-complaint-routing-system

2. Create a virtual environment

python -m venv venv

Activate it:

Windows

venv\Scripts\activate

Linux / macOS

source venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Download the required model

The project provides a model download link:

"Download the trained model" (https://drive.google.com/file/d/1NF-XJzIlY95hu7fnY73yRCsCkqTbsyJk/view?usp=sharing)

Place the downloaded model files in the directory structure expected by the project.

The repository includes a saved embedding model at:

models/saved/embedder/model.safetensors

5. Configure audio dependencies

Install FFmpeg and ensure it is available in your system PATH if you want to use audio transcription.

Refer to "INSTALL.md" (INSTALL.md) for detailed installation instructions and troubleshooting.

Running the Application

The repository provides both command-line and web application entry points.

Command-line interface

python app/cli.py

Web application

python app/web_app.py

Follow the application prompts to submit a complaint and view the generated results.

Refer to "DEMO.md" (DEMO.md) for demonstration examples and additional usage information.

Project Structure

ai-complaint-routing-system/
│
├── app/
│   ├── cli.py
│   └── web_app.py
│
├── src/
│   └── Core machine learning and processing modules
│
├── models/
│   └── saved/
│       └── embedder/
│
├── data/
│   └── Complaint datasets
│
├── results/
│   └── Project screenshots
│
├── scripts/
│   └── Supporting scripts
│
├── tests/
│   └── Test files
│
├── README.md
├── ARCHITECTURE.md
├── DEMO.md
├── INSTALL.md
└── requirements.txt

This is a high-level overview of the repository structure.

Engineering Considerations

Modular machine learning pipeline

The system separates complaint processing into different prediction and retrieval components. This makes individual components easier to test and improve.

Multilingual text representation

The project uses "multilingual-MiniLM-L12-v2" to generate sentence embeddings for complaint text.

Semantic retrieval

FAISS enables similarity-based retrieval using vector representations rather than relying only on exact keyword matching.

Audio and text input

Whisper supports audio transcription, allowing the system to process spoken complaints alongside written submissions.

Local model inference

The project supports local inference after the necessary models and dependencies have been downloaded.

Current Limitations

- The reported evaluation results should not be interpreted as production performance.
- Synthetic datasets may not represent the full complexity of real-world public complaints.
- Officer routing and priority prediction require validation against real operational data.
- ETA predictions may vary depending on complaint complexity and external factors.
- Audio transcription quality can vary with language, pronunciation, and recording conditions.
- Real-world deployment would require appropriate data privacy, security, monitoring, and model evaluation processes.

Future Improvements

- Evaluate the models using real-world complaint datasets.
- Improve multilingual complaint understanding.
- Add confidence scores to model predictions.
- Introduce human review for uncertain or high-impact predictions.
- Improve model monitoring and evaluation.
- Develop REST APIs for external system integration.
- Add authentication and role-based access control.
- Deploy the application using cloud infrastructure.
- Monitor prediction quality and data drift over time.

Documentation

- "Installation Guide" (INSTALL.md)
- "Demonstration Guide" (DEMO.md)
- "System Architecture" (ARCHITECTURE.md)

Project Objective

The objective of this project is to demonstrate how machine learning, natural language processing, speech recognition, and vector similarity search can work together to automate a practical complaint-management workflow.

It combines multiple independent AI components into a single application, with an emphasis on practical problem-solving and modular engineering.

Author

Uday Kiran Anagalla

B.Tech – Computer Science and Engineering (AI & ML)

Areas of Interest

- Artificial Intelligence
- Machine Learning
- Generative AI
- Natural Language Processing
- AI Engineering
- Forward Deployed Engineering

GitHub: "udaykiran1402" (https://github.com/udaykiran1402)

---

⭐ If you find this project interesting, consider starring the repository.