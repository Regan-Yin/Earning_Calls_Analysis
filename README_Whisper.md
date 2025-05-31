# Whisper.cpp (MacOS) Full Setup & Batch Transcription Guide (2025)

**Written by Regan Yin** 
*For ISOM 3350 Final Project*. 
This guide provides a complete step-by-step walkthrough to install, configure, and use Whisper.cpp on macOS to batch transcribe `.m4a`, `.mp3`, and `.wav` audio files into both `.txt` and `.json` transcript formats using speaker-aware models (if supported). 

---

## ✅ Prerequisites (Install These First)

You must install the following dependencies on macOS:

### 1. Homebrew (if not already installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Core Dependencies
```bash
brew install cmake ffmpeg git
```

> `cmake`: for building Whisper.cpp  
> `ffmpeg`: for converting audio to 16-bit mono WAV (required format)  
> `git`: to clone the whisper.cpp repo

---

## 📁 Step 1: Clone the Whisper.cpp Repository
```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
```

---

## 🧱 Step 2: Build Whisper.cpp
```bash
cmake -B build
cmake --build build --config Release
```

The CLI binary will be generated at: `./build/bin/whisper-cli`

---

## 📦 Step 3: Download a Model
```bash
sh ./models/download-ggml-model.sh small.en
```

> You can replace `small.en` with any of:
> `tiny.en`, `base.en`, `small`, `medium`, `large-v3`, `small.en-tdrz` (speaker-aware)

---

## 🧪 Step 4: Verify with Sample Audio
```bash
./build/bin/whisper-cli -m models/ggml-small.en.bin -f samples/jfk.wav
```

---

## 🔁 Step 5: Batch Transcription Script

Place your `.m4a`, `.mp3`, or `.wav` files in a folder named `audio_input/` (next to `build/` directory).

Save the following script as `batch_transcribe.sh` in `build/bin/`:

```bash
#!/bin/bash

MODEL_PATH="../../models/ggml-small.en.bin"
INPUT_DIR="../../audio_input"
OUTPUT_DIR="../../transcripts"

mkdir -p "$OUTPUT_DIR"

for AUDIO in "$INPUT_DIR"/*.{wav,mp3,m4a}; do
    [ -f "$AUDIO" ] || continue
    FILENAME=$(basename -- "$AUDIO")
    NAME="${FILENAME%.*}"

    echo "Processing $FILENAME..."

    TMP_WAV="/tmp/${NAME}.wav"
    ffmpeg -y -i "$AUDIO" -ar 16000 -ac 1 -c:a pcm_s16le "$TMP_WAV" > /dev/null 2>&1

    ./whisper-cli -m "$MODEL_PATH" -f "$TMP_WAV" \
      --output-txt \
      --output-json-full \
      --output-file "$OUTPUT_DIR/$NAME"

    echo "Done: $OUTPUT_DIR/$NAME.txt / .json"
done

echo "All files processed successfully."
```

### Make It Executable:
```bash
chmod +x batch_transcribe.sh
```

---

## 🚀 Step 6: Run the Batch Transcription
```bash
cd build/bin
./batch_transcribe.sh
```

---

## 🔄 Optional: Enable Speaker Diarization

If you downloaded a diarization-compatible model like `ggml-small.en-tdrz.bin`, change the script’s model path:

```bash
MODEL_PATH="../../models/ggml-small.en-tdrz.bin"
```

And add:
```bash
-tdrz
```
as an argument to `./whisper-cli` in the script, after `--output-json-full`.

---

## 📂 Folder Structure Summary

```
whisper.cpp/
├── models/
│   └── ggml-small.en.bin (or tdrz model)
├── audio_input/
│   └── your .m4a / .mp3 / .wav files
├── transcripts/
│   └── Output .txt and .json files
├── build/
│   └── bin/
│       └── whisper-cli, batch_transcribe.sh
```

---

## 🧼 (Optional) Clean Stopwords & Duplicates
Use `clean_transcripts.py` to remove stopwords and repeated words for better readability.

---

Let me know if you also need a GUI frontend or integration with Python-based diarization models like pyannote-audio.
