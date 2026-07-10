import gc
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import torch
import whisperx
from whisperx.diarize import DiarizationPipeline
from dotenv import load_dotenv
from fastapi import Body, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from speakers import EmbeddingExtractor, SpeakerStore, relabel_segments, to_opus_bytes

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
DEVICE = os.getenv("DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE") or None
SPEAKER_DB = os.getenv("SPEAKER_DB", "speakers.db")
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.50"))

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
    models["diarize"] = DiarizationPipeline(token=HF_TOKEN, device=DEVICE)
    print("[startup] Lade Embedding-Modell (pyannote/embedding)...")
    models["embed"] = EmbeddingExtractor(hf_token=HF_TOKEN, device=DEVICE)
    print(f"[startup] Speaker-DB: {SPEAKER_DB}")
    models["store"] = SpeakerStore(SPEAKER_DB)
    print("[startup] Bereit.")
    yield
    models.clear()
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


app = FastAPI(title="speech2text-api", lifespan=lifespan)

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(_STATIC_DIR / "index.html")


def _format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _to_markdown(segments: list[dict], language: str, session_id: str) -> str:
    lines = ["# Transkript", "", f"_Sprache: `{language}` · Session: `{session_id}`_", ""]
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
        "match_threshold": MATCH_THRESHOLD,
    }


# --- Speaker enrollment / verwaltung ---

@app.get("/speakers")
def list_speakers() -> dict:
    store: SpeakerStore = models["store"]
    return {"speakers": store.list_speakers()}


@app.post("/speakers", status_code=201)
async def enroll_speaker(
    name: str = Form(...),
    file: UploadFile = File(...),
):
    name = name.strip()
    if not name:
        raise HTTPException(400, "name darf nicht leer sein.")
    if not file.filename:
        raise HTTPException(400, "Keine Datei übergeben.")

    suffix = Path(file.filename).suffix or ".audio"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        audio_path = tmp.name
    try:
        audio = whisperx.load_audio(audio_path)
        if len(audio) < 16000 * 1.5:
            raise HTTPException(400, "Sample zu kurz (mindestens 1.5 Sekunden Sprache empfohlen).")
        embed: EmbeddingExtractor = models["embed"]
        store: SpeakerStore = models["store"]
        vec = embed.extract(audio)
        # Sample für spätere Wiedergabe abspeichern (max. 15 s, Opus)
        preview = audio[: 15 * 16000]
        try:
            sample_bytes = to_opus_bytes(preview)
        except Exception as e:
            print(f"[warn] Opus-Encoding beim Enrollment fehlgeschlagen: {e}")
            sample_bytes = None
        store.add_sample(name, vec, source=f"enroll:{file.filename}", audio_sample=sample_bytes)
        return {"name": name, "samples": next(
            (s["samples"] for s in store.list_speakers() if s["name"].lower() == name.lower()),
            1,
        )}
    finally:
        try:
            os.unlink(audio_path)
        except OSError:
            pass


@app.delete("/speakers/{name}")
def delete_speaker(name: str):
    store: SpeakerStore = models["store"]
    if not store.delete_speaker(name):
        raise HTTPException(404, f"Speaker '{name}' nicht gefunden.")
    return {"deleted": name}


@app.patch("/speakers/{name}")
def rename_speaker(name: str, new_name: str = Body(..., embed=True)):
    new_name = new_name.strip()
    if not new_name:
        raise HTTPException(400, "new_name darf nicht leer sein.")
    store: SpeakerStore = models["store"]
    result = store.rename_speaker(name, new_name)
    if result == "notfound":
        raise HTTPException(404, f"Speaker '{name}' nicht gefunden.")
    if result == "conflict":
        raise HTTPException(409, f"Ein Sprecher namens '{new_name}' existiert bereits.")
    return {"renamed": name, "name": new_name}


@app.get("/speakers/{name}/samples")
def list_speaker_samples(name: str):
    store: SpeakerStore = models["store"]
    return {"name": name, "samples": store.list_speaker_samples(name)}


@app.get("/speakers/samples/{sample_id}/audio")
def get_speaker_sample_audio(sample_id: int):
    store: SpeakerStore = models["store"]
    blob = store.get_speaker_sample_audio(sample_id)
    if not blob:
        raise HTTPException(404, "Kein Audio-Sample für diese Sample-ID vorhanden.")
    return Response(content=blob, media_type="audio/ogg")


@app.get("/sessions/pending")
def list_pending_sessions(unassigned_only: bool = True, limit: int = 50):
    store: SpeakerStore = models["store"]
    return {"sessions": store.list_pending(unassigned_only=unassigned_only, limit=limit)}


# --- Session-Assignment (nachträglich) ---

@app.get("/sessions/{session_id}/clusters/{cluster_label}/audio")
def get_cluster_audio(session_id: str, cluster_label: str):
    store: SpeakerStore = models["store"]
    blob = store.get_pending_audio(session_id, cluster_label)
    if not blob:
        raise HTTPException(404, "Kein Audio-Sample für diesen Cluster vorhanden.")
    return Response(content=blob, media_type="audio/ogg")


@app.delete("/sessions/{session_id}/clusters/{cluster_label}")
def delete_pending_cluster(session_id: str, cluster_label: str):
    store: SpeakerStore = models["store"]
    if not store.delete_pending_cluster(session_id, cluster_label):
        raise HTTPException(404, "Cluster nicht gefunden.")
    return {"deleted": {"session_id": session_id, "cluster_label": cluster_label}}


@app.post("/sessions/{session_id}/assign")
def assign_session(session_id: str, mapping: dict[str, str] = Body(...)):
    """Ordnet anonyme Cluster-Labels (z.B. 'SPEAKER_00') Namen zu und speichert deren Embeddings."""
    store: SpeakerStore = models["store"]
    assigned = store.assign_session(session_id, mapping)
    if not assigned:
        raise HTTPException(404, "Session unbekannt oder keine passenden Cluster.")
    return {"assigned": assigned}


# --- Transkription ---

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

        # Pro Cluster-Label Audio + Embedding -> Match gegen DB
        embed: EmbeddingExtractor = models["embed"]
        store: SpeakerStore = models["store"]
        cluster_chunks = embed.per_speaker_chunks(audio, segments)
        cluster_embeddings = {
            label: embed.extract(chunk) for label, chunk in cluster_chunks.items()
        }

        cluster_to_name: dict[str, str] = {}
        match_info: dict[str, dict] = {}
        for label, vec in cluster_embeddings.items():
            name, score = store.match(vec, threshold=MATCH_THRESHOLD)
            match_info[label] = {"matched": name, "score": round(score, 3)}
            if name:
                cluster_to_name[label] = name

        # Audio-Samples (Opus) für Wiedergabe in der UI – max. 15 s
        cluster_audio: dict[str, bytes] = {}
        for label, chunk in cluster_chunks.items():
            preview = chunk[: 15 * 16000]
            try:
                cluster_audio[label] = to_opus_bytes(preview)
            except Exception as e:  # ffmpeg-Encoding fehlgeschlagen — Sample weglassen
                print(f"[warn] Opus-Encoding für {label} fehlgeschlagen: {e}")

        # Zeitspanne pro Cluster (erster Start / letztes Ende im Original-Audio)
        time_ranges: dict[str, tuple[float, float]] = {}
        for seg in segments:
            spk = seg.get("speaker")
            if not spk:
                continue
            s, e = float(seg["start"]), float(seg["end"])
            if spk in time_ranges:
                time_ranges[spk] = (min(time_ranges[spk][0], s), max(time_ranges[spk][1], e))
            else:
                time_ranges[spk] = (s, e)

        # Session anlegen und alle Cluster-Embeddings + Audio ablegen
        session_id = store.new_session()
        store.store_pending(
            session_id,
            cluster_embeddings,
            cluster_to_name,
            audio=cluster_audio,
            source_filename=file.filename,
            time_ranges=time_ranges,
        )

        segments = relabel_segments(segments, cluster_to_name)

        headers = {"X-Session-Id": session_id}
        if format == "json":
            return JSONResponse(
                {
                    "language": detected_language,
                    "session_id": session_id,
                    "speakers": match_info,
                    "segments": segments,
                },
                headers=headers,
            )
        if format == "txt":
            return PlainTextResponse(
                _to_text(segments),
                media_type="text/plain; charset=utf-8",
                headers=headers,
            )
        return PlainTextResponse(
            _to_markdown(segments, detected_language, session_id),
            media_type="text/markdown; charset=utf-8",
            headers=headers,
        )
    finally:
        try:
            os.unlink(audio_path)
        except OSError:
            pass
