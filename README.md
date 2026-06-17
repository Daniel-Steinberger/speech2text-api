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
3. Lizenz für drei Modelle bestätigen (eingeloggt auf die Seiten gehen und
   die Bedingungen akzeptieren):
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/embedding (für Sprecher-Wiedererkennung)
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
uv run uvicorn main:app --host 0.0.0.0 --port 8002
```

Health-Check:
```bash
curl http://localhost:8002/health
```

## Web-UI

Unter `http://<host>:8002/` läuft eine schlanke Web-Oberfläche zum:

- Audio hochladen und Transkript anzeigen / kopieren / als `.md` herunterladen
- Bekannte Sprecher auflisten und löschen
- Offene Sitzungen (anonyme Cluster aus letzten Transkriptionen) Namen
  zuweisen — das gespeicherte Embedding wird damit dem Sprecher zugeordnet
  und beim nächsten Transkribieren wieder erkannt
- Sprecher manuell per Sample-Upload anlernen

## Verwendung (API)

### Markdown-Transkript (Default)

```bash
curl -X POST http://localhost:8002/transcribe \
  -F "file=@meeting.m4a" \
  -F "language=de" \
  -o transkript.md
```

### Reines Text-Format

```bash
curl -X POST http://localhost:8002/transcribe \
  -F "file=@meeting.m4a" \
  -F "format=txt" \
  -o transkript.txt
```

### JSON mit Wort-Zeitstempeln

```bash
curl -X POST http://localhost:8002/transcribe \
  -F "file=@meeting.m4a" \
  -F "format=json" \
  -o transkript.json
```

### Sprecherzahl eingrenzen (verbessert Diarization)

```bash
curl -X POST http://localhost:8002/transcribe \
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

## Sprecher-Wiedererkennung über Sitzungen hinweg

pyannote-Diarization vergibt **anonyme Labels** (`SPEAKER_00`, `SPEAKER_01`)
nur pro Datei stabil. Damit ein Sprecher datei-übergreifend als z. B.
„Daniel" wiedererkannt wird, extrahiert die API **Voice-Embeddings** (ECAPA-
TDNN, 512-dim) pro Cluster und vergleicht sie per Cosine-Similarity gegen
eine lokale SQLite-Datei (`speakers.db`).

### Zwei Wege Sprecher anzulernen

**1. Dediziertes Enrollment** — sauberer kurzer Mitschnitt (≥ 5 s) mit Namen:

```bash
curl -X POST http://localhost:8002/speakers \
  -F "name=Daniel" \
  -F "file=@daniel_sample.wav"
```

Mehrfacher Upload mit gleichem Namen fügt zusätzliche Samples zum bestehenden
Sprecher hinzu — verbessert die Robustheit über verschiedene Mikrofone /
Umgebungen.

**2. Nachträgliche Zuweisung** — nach einer Transkription Cluster-Labels
auf Namen mappen. `/transcribe` legt jede Sitzung mit Cluster-Embeddings
zwischen und gibt eine `session_id` zurück (HTTP-Header `X-Session-Id`,
in JSON-Antworten zusätzlich im Body):

```bash
SESSION=$(curl -sD - -X POST http://localhost:8002/transcribe \
  -F "file=@meeting.m4a" -o transkript.md \
  | awk -F': ' '/^X-Session-Id/ {print $2}' | tr -d '\r')

curl -X POST http://localhost:8002/sessions/$SESSION/assign \
  -H "Content-Type: application/json" \
  -d '{"SPEAKER_00": "Daniel", "SPEAKER_01": "Anna"}'
```

Beim nächsten Transkribieren werden Daniel und Anna automatisch im
Transkript benannt (statt `SPEAKER_00`/`01`), wenn ihre Stimmen wieder
auftauchen.

### Bekannte Sprecher verwalten

```bash
curl http://localhost:8002/speakers          # auflisten
curl -X DELETE http://localhost:8002/speakers/Daniel
```

### Caveats

- **Sample-Länge**: Enrollment-Samples ≥ 5 s sauberer Sprache, je mehr desto
  besser. Match-Cluster sammelt automatisch bis zu 30 s pro Sitzung.
- **Akustik zählt**: Selbe Mikro-/Raum-Umgebung wie beim Enrollment liefert
  bessere Match-Raten. Telefonqualität ↔ Headset ↔ Konferenzmikro können
  abweichen — mehrere Samples in verschiedenen Settings hochladen hilft.
- **Schwelle (`MATCH_THRESHOLD`)** in `.env` tunen: zu niedrig → falsche
  Treffer, zu hoch → ständig „SPEAKER_00".

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
| `SPEAKER_DB`       | `speakers.db` | SQLite-Pfad für Sprecher-Embeddings        |
| `MATCH_THRESHOLD`  | `0.50`      | Cosine-Schwelle Cluster ↔ bekannter Sprecher |

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
