# Multi-Modal Sentiment Analysis of Earnings Calls  
**Author:** Regan Yin  
**Acknowledgment:** Special thanks to Prof. Yi Yang (HKUST ISOM) for supporting the FinBERT fine-tuning and providing valuable model insights.

---

## Project Overview

This repository presents a multi-modal analytical framework for evaluating executive communication during earnings calls. The system integrates audio processing, sentiment classification, and stock price analysis to uncover behavioral signals that may correlate with abnormal stock movements.

The pipeline incorporates:
- Audio transcription using Whisper.cpp
- Financial sentiment extraction using FinBERT (fine-tuned)
- Abnormal return computation via event study methodology
- Statistical and visual analysis via Dash

---

## Directory Structure

```
Sentiment_Analysis_Models/
├── earning_call_auto_downloader/
│   ├── auto_downloader.py
│   └── audio/
│
├── FinBERT_Project/
│   ├── analyze_calls.py
│   ├── dash_app.py
│   ├── metadata_update.py
│   ├── metadata.json
│   ├── transcript/
│   └── report/
│       ├── sentiment_summary.csv
│       ├── event_study.csv
│       ├── price_change.csv
│       ├── corr_heatmap.jpg
│       ├── scatter_with_reg.jpg
│       ├── event_study_plot.jpg
│       ├── classification_report.txt
│       └── granger.txt
│
├── whisper.cpp/
│   ├── build/bin/
│   │   ├── whisper-cli
│   │   └── batch_transcribe.sh
│   ├── audio_input/
│   ├── transcripts/
│   └── models/
│
├── README_Whisper.md
└── README.md
```

---

## Environment Setup

### Install Python dependencies

```bash
pip install -r requirements.txt
```

Required packages include:

- torch
- transformers
- pandas
- numpy
- yfinance
- statsmodels
- scikit-learn
- plotly
- dash

---

## Whisper.cpp Setup (macOS/Linux)

Install dependencies:

```bash
brew install cmake ffmpeg git
```

Clone and build:

```bash
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
cmake -B build
cmake --build build --config Release
```

Download the model:

```bash
sh ./models/download-ggml-model.sh small.en
```

Place audio files into `audio_input/`, then run:

```bash
cd build/bin
./batch_transcribe.sh
```

Output transcripts will be saved in `transcripts/`.

---

## Workflow Execution

### Step 1: Download Audio and Market Data

```bash
cd earning_call_auto_downloader
python auto_downloader.py
```

Downloads earnings call recordings (if available) and T-1 to T+1 historical prices.

### Step 2: Update Metadata

```bash
cd ../FinBERT_Project
python metadata_update.py
```

Automatically updates `metadata.json` using the transcript filenames.

### Step 3: Run Sentiment Analysis and Event Study

```bash
python analyze_calls.py
```

This script:
- Applies FinBERT to each transcript
- Outputs sentiment scores (`sentiment_summary.csv`)
- Computes CAR and returns (`event_study.csv`, `price_change.csv`)
- Outputs supporting plots and statistical results in `report/`

### Step 4: Launch Dashboard

```bash
python dash_app.py
```

Access the interactive dashboard via browser:
- Sentiment correlation scatter plots
- CAR trends by company
- Heatmaps and diagnostics

---

## Extending the Project

### Add More Companies

1. Place additional audio files in `audio_input/`
2. Transcribe with `batch_transcribe.sh`
3. Move transcript `.json` files to `FinBERT_Project/transcript/`
4. Run `metadata_update.py` to refresh metadata
5. Run `analyze_calls.py` to recompute sentiment and returns

### Full-Market Automation

- Add a list of tickers to `auto_downloader.py`
- Implement batch crawling scripts (e.g., using SEC EDGAR or APIs)
- Store metadata/results in a structured database for scaling

---

## Outputs

All results are saved in `FinBERT_Project/report/`:

- `sentiment_summary.csv` – FinBERT sentiment results
- `event_study.csv` – Event study abnormal returns
- `price_change.csv` – Daily changes (T+0, T+1)
- `classification_report.txt` – Logistic regression summary
- `granger.txt` – Granger causality test output
- `*.jpg` – Visualizations (correlation heatmap, CAR chart, scatter plot)

---

## Limitations

- Sample size is currently limited; expansion is required for significance
- Logistic regression is underpowered due to sparse training labels
- Sentiment detection may misclassify Q&A segments without speaker identification

---

## Future Improvements

- Integrate diarization via Whisper `-tdrz` models
- Extract audio features (pitch, pauses) using Librosa
- Ensemble FinBERT with other financial sentiment models
- Support parallel transcript processing with multiprocessing
- Export results to SQL or NoSQL backends

---

## Citation

```
Yin, R. (2025). Multi-Modal Sentiment Analysis on Earnings Calls.
```
