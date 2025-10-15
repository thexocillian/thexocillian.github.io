# AI-Powered Instrument Identification Project

This repository will guide the development of an AI system that listens to a passage of music and reports the instruments that are being played. The plan is broken down into digestible milestones so you can make steady progress while learning new skills along the way.

---

## 1. Project Overview
- **Objective:** Build a pipeline that ingests an audio clip and outputs the set of instruments present, along with confidence scores and a natural-language summary.
- **High-Level Architecture:**
  1. Audio preprocessing to clean, normalize, and convert audio into machine-learning-friendly representations (e.g., Mel spectrograms).
  2. Instrument recognition model (fine-tuned from a pretrained model or trained from scratch).
  3. Post-processing and aggregation logic that smooths predictions over time and produces user-facing explanations.
  4. Optional: Wrap the pipeline in an app or custom GPT that takes user input and returns polished responses.

---

## 2. Development Roadmap

### Milestone A — Environment & Data Familiarization
1. **Set up tooling**
   - Install Python 3.10+, `pipenv` or `conda`, and `ffmpeg` for audio handling.
   - Create a virtual environment and install key packages: `librosa`, `numpy`, `pandas`, `torch` (or `tensorflow`), and `jupyter`.
2. **Collect sample data**
   - Download small open datasets such as [IRMAS](https://github.com/MTG/irmas) or [MedleyDB](https://medleydb.weebly.com/).
   - Inspect audio files, label formats, and licensing terms.
3. **Exploratory analysis**
   - Use a notebook to load a few tracks, plot waveforms/spectrograms, and listen interactively.

### Milestone B — Baseline Instrument Classifier
1. **Feature extraction**
   - Write a script to convert audio clips into Mel spectrogram tensors saved on disk (e.g., `.npy`).
   - Implement normalization and augmentation (time masking, pitch shift) as stretch goals.
2. **Model selection**
   - Start with a pretrained model such as [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) or [YAMNet](https://tfhub.dev/google/yamnet/1) to avoid training from scratch.
   - Run inference to obtain instrument probabilities; evaluate precision/recall on a validation split.
3. **Evaluation metrics**
   - Implement scripts to compute metrics (accuracy, F1-score, confusion matrix) to understand strengths/weaknesses.

### Milestone C — Fine-Tuning & Improvement
1. **Fine-tuning strategy**
   - Freeze early layers, fine-tune higher layers on your curated instrument set.
   - Handle class imbalance via weighted loss or oversampling.
2. **Temporal smoothing**
   - Add simple smoothing (moving average) or sequence models (CRF/HMM) to stabilize frame-level predictions.
3. **Error analysis**
   - Listen to misclassified segments, annotate failure modes, and iterate with targeted data augmentation.

### Milestone D — User Experience Layer
1. **API/Service**
   - Expose the model through a FastAPI/Flask endpoint that accepts audio uploads and returns JSON with predictions and timestamps.
2. **Custom GPT integration**
   - Use OpenAI function-calling to let GPT request analysis results from your backend and transform them into conversational explanations.
3. **Front-end prototype**
   - Build a simple web UI (React/Vue/Svelte or plain HTML/JS) allowing uploads and showing detected instruments on a timeline.

---

## 3. Repository Structure (Suggested)
```
.
├── data/                # Datasets (git-ignored)
├── notebooks/           # Jupyter notebooks for exploration
├── src/
│   ├── preprocess/      # Audio preprocessing utilities
│   ├── models/          # Training & inference scripts
│   └── app/             # API or UI code
├── tests/               # Unit/integration tests
├── README.md            # You are here
└── requirements.txt     # Python dependencies
```

Feel free to create folders as you reach each milestone.

---

## 4. Immediate Next Steps
1. **Set up the development environment** following Milestone A, Step 1.
2. **Create a `requirements.txt`** (or `Pipfile`/`environment.yml`) capturing the core dependencies.
3. **Start a `notebooks/` directory** and make an initial exploration notebook that loads a sample dataset track and displays its spectrogram.

Once these steps are complete, commit your work. We can then move on to building the baseline classifier together.

---

## 5. Support Checklist
- [ ] Python environment created
- [ ] Sample dataset downloaded
- [ ] First exploratory notebook saved
- [ ] Baseline inference using a pretrained model
- [ ] Evaluation metrics implemented
- [ ] API or GPT wrapper prototype

Check these boxes as you progress; feel free to add new tasks or notes for questions.

---

_Questions or blockers? Document them in a `NOTES.md` file and push with your commits so we can address them in the next session._

