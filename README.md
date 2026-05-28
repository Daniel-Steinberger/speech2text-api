# speech2text-api

Kleine FastAPI, die Audiodateien per Upload entgegennimmt, lokal auf GPU
(NVIDIA RTX 5090 / Blackwell) per **WhisperX** (faster-whisper `large-v3` +
pyannote Speaker Diarization) transkribiert und ein Markdown- oder Text-
Transkript mit Sprechererkennung zurückliefert.

## Voraussetzungen

- NVIDIA-GPU mit aktuellem Treiber (für RTX 5090 / Blackwell ≥ CUDA 12.8)
- `uv` (https://docs.astral.sh/uv/)
- `ffmpeg` im `PATH` (wird von WhisperX zum Dekodieren verwendet)
  ```bash
  sudo apt install ffmpeg
  ```
- Hugging-Face-Account + Access-Token (s. u.)

## Hugging-Face-Token einrichten

pyannote-Diarization-Modelle sind „gated" — kostenlos, aber Lizenz muss
einmalig akzeptiert werden:

1. Account anlegen: https://huggingface.co
2. Token erzeugen (Scope **Read**): https://huggingface.co/settings/tokens
3. Lizenz für beide Modelle bestätigen (eingeloggt auf die Seite gehen und
   die Bedingungen akzeptieren):
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
4. `.env` anlegen und Token eintragen:
   ```bash
   cp .env.example .env
   # HF_TOKEN=hf_... in .env eintragen
   ```

## Installation

```bash
uv sync
```

`uv` zieht automatisch die PyTorch-cu128-Wheels von
`https://download.pytorch.org/whl/cu128` (für Blackwell-Support).
Der erste Start lädt zusätzlich das Whisper-Modell (~3 GB) und die
pyannote-Modelle herunter — beim ersten Request kommt nochmal das
Alignment-Modell für die erkannte Sprache dazu.

## Starten

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

Health-Check:
```bash
curl http://localhost:8000/health
```

## Verwendung

### Markdown-Transkript (Default)

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@meeting.m4a" \
  -F "language=de" \
  -o transkript.md
```

### Reines Text-Format

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@meeting.m4a" \
  -F "format=txt" \
  -o transkript.txt
```

### JSON mit Wort-Zeitstempeln

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@meeting.m4a" \
  -F "format=json" \
  -o transkript.json
```

### Sprecherzahl eingrenzen (verbessert Diarization)

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@interview.wav" \
  -F "min_speakers=2" \
  -F "max_speakers=2" \
  -o transkript.md
```

### Parameter (alle optional, Form-Felder)

| Feld           | Default        | Beschreibung                                   |
| -------------- | -------------- | ---------------------------------------------- |
| `file`         | —              | Audiodatei (wav/mp3/m4a/flac/ogg/opus/…)       |
| `language`     | `de` (via env) | ISO-Code, z. B. `de`, `en`. Leer = Auto-Detect |
| `min_speakers` | —              | Mindestanzahl Sprecher                         |
| `max_speakers` | —              | Maximalanzahl Sprecher                         |
| `format`       | `md`           | `md`, `txt` oder `json`                        |

## Konfiguration

Über `.env` (siehe `.env.example`):

| Variable           | Default     | Wirkung                                      |
| ------------------ | ----------- | -------------------------------------------- |
| `HF_TOKEN`         | —           | Pflicht für pyannote-Download                |
| `WHISPER_MODEL`    | `large-v3`  | Modellgröße                                  |
| `DEVICE`           | `cuda`      | `cuda` oder `cpu`                            |
| `COMPUTE_TYPE`     | `float16`   | `float16` (GPU), `int8`, `float32`           |
| `BATCH_SIZE`       | `16`        | Kleiner = weniger VRAM                       |
| `DEFAULT_LANGUAGE` | `de`        | Wird verwendet wenn Request keine angibt     |

## Format des Markdown-Outputs

```markdown
# Transkript

_Sprache: `de`_

### SPEAKER_00
_00:00:00 → 00:00:04_
Guten Morgen, willkommen zur Besprechung.

### SPEAKER_01
_00:00:05 → 00:00:09_
Danke, freut mich hier zu sein.
```

## Hinweise

- Modelle werden **einmal beim App-Start** in den GPU-Speicher geladen,
  nicht pro Request — der erste Start dauert daher etwas länger.
- Synchron: der HTTP-Request wartet bis die Transkription fertig ist.
  Für lange Dateien Client-Timeout entsprechend hochsetzen.
- VRAM-Bedarf liegt bei `large-v3` + `float16` + Diarization typischerweise
  bei ca. 8–10 GB — auf der 5090 (32 GB) entspannt.
