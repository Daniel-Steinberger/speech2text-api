import gc
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import torch
import whisperx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
DEVICE = os.getenv("DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE") or None

models: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN ist nicht gesetzt. Siehe .env.example.")
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA nicht verfügbar — DEVICE=cpu setzen oder Treiber prüfen.")

    print(f"[startup] Lade WhisperX-Modell '{WHISPER_MODEL}' auf {DEVICE} ({COMPUTE_TYPE})...")
    models["asr"] = whisperx.load_model(
        WHISPER_MODEL,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        language=DEFAULT_LANGUAGE,
    )
    print("[startup] Lade Diarization-Pipeline (pyannote)...")
    models["diarize"] = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
    print("[startup] Bereit.")
    yield
    models.clear()
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


app = FastAPI(title="speech2text-api", lifespan=lifespan)


def _format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _to_markdown(segments: list[dict], language: str) -> str:
    lines = ["# Transkript", "", f"_Sprache: `{language}`_", ""]
    current_speaker = None
    for seg in segments:
        speaker = seg.get("speaker", "SPEAKER_?")
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        if speaker != current_speaker:
            lines.append("")
            lines.append(
                f"### {speaker}  \n_{_format_timestamp(seg['start'])} → {_format_timestamp(seg['end'])}_"
            )
            current_speaker = speaker
        lines.append(text)
    return "\n".join(lines).strip() + "\n"


def _to_text(segments: list[dict]) -> str:
    out = []
    current_speaker = None
    for seg in segments:
        speaker = seg.get("speaker", "SPEAKER_?")
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        if speaker != current_speaker:
            out.append(f"\n[{_format_timestamp(seg['start'])}] {speaker}: {text}")
            current_speaker = speaker
        else:
            out.append(" " + text)
    return "".join(out).strip() + "\n"


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model": WHISPER_MODEL,
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
    min_speakers: int | None = Form(default=None),
    max_speakers: int | None = Form(default=None),
    format: Literal["md", "txt", "json"] = Form(default="md"),
):
    if not file.filename:
        raise HTTPException(400, "Keine Datei übergeben.")

    suffix = Path(file.filename).suffix or ".audio"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        audio_path = tmp.name

    try:
        audio = whisperx.load_audio(audio_path)

        asr_result = models["asr"].transcribe(
            audio,
            batch_size=BATCH_SIZE,
            language=language or DEFAULT_LANGUAGE,
        )
        detected_language = asr_result["language"]

        align_model, metadata = whisperx.load_align_model(
            language_code=detected_language, device=DEVICE
        )
        aligned = whisperx.align(
            asr_result["segments"],
            align_model,
            metadata,
            audio,
            DEVICE,
            return_char_alignments=False,
        )
        del align_model
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        diarize_segments = models["diarize"](
            audio,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        result = whisperx.assign_word_speakers(diarize_segments, aligned)
        segments = result["segments"]

        if format == "json":
            return {"language": detected_language, "segments": segments}
        if format == "txt":
            return PlainTextResponse(
                _to_text(segments), media_type="text/plain; charset=utf-8"
            )
        return PlainTextResponse(
            _to_markdown(segments, detected_language),
            media_type="text/markdown; charset=utf-8",
        )
    finally:
        try:
            os.unlink(audio_path)
        except OSError:
            pass
