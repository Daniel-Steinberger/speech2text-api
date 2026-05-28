"""Speaker-Embeddings: Enrollment, Persistenz, Matching."""
from __future__ import annotations

import io
import sqlite3
import subprocess
import uuid
import wave
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from pyannote.audio import Inference, Model

SAMPLE_RATE = 16000  # whisperx.load_audio liefert 16 kHz Mono


_SCHEMA = """
CREATE TABLE IF NOT EXISTS speakers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL COLLATE NOCASE,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    speaker_id INTEGER NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
    vector BLOB NOT NULL,
    source TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS pending_clusters (
    session_id TEXT NOT NULL,
    cluster_label TEXT NOT NULL,
    vector BLOB NOT NULL,
    matched_name TEXT,
    audio_sample BLOB,
    source_filename TEXT,
    first_start REAL,
    last_end REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (session_id, cluster_label)
);
CREATE INDEX IF NOT EXISTS idx_pending_created ON pending_clusters(created_at);
"""


def _vec_to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def _blob_to_vec(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


def to_wav_bytes(audio_np: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Mono float32 [-1,1] -> 16-bit PCM WAV."""
    pcm = np.clip(audio_np, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def to_opus_bytes(audio_np: np.ndarray, sample_rate: int = 16000, bitrate_kbps: int = 24) -> bytes:
    """Mono float32 -> Opus in Ogg-Container (für Sprache ~24 kbps optimal)."""
    wav = to_wav_bytes(audio_np, sample_rate)
    proc = subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y",
            "-f", "wav", "-i", "pipe:0",
            "-c:a", "libopus", "-b:a", f"{bitrate_kbps}k",
            "-application", "voip",
            "-f", "ogg", "pipe:1",
        ],
        input=wav,
        capture_output=True,
        check=True,
    )
    return proc.stdout


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class EmbeddingExtractor:
    """Wrapper um pyannote/embedding. Extrahiert 512-dim Vektoren."""

    def __init__(self, hf_token: str, device: str = "cuda"):
        model = Model.from_pretrained("pyannote/embedding", use_auth_token=hf_token)
        self.inference = Inference(model, window="whole", device=torch.device(device))

    def extract(self, audio_np: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """Audio (mono, float32, 16kHz) -> 512-dim Embedding."""
        if audio_np.ndim == 1:
            waveform = torch.from_numpy(audio_np).unsqueeze(0)  # (1, time)
        else:
            waveform = torch.from_numpy(audio_np)
        emb = self.inference({"waveform": waveform, "sample_rate": sample_rate})
        return np.asarray(emb, dtype=np.float32).reshape(-1)

    def per_speaker_chunks(
        self,
        audio_np: np.ndarray,
        segments: list[dict],
        sample_rate: int = SAMPLE_RATE,
        min_seconds: float = 1.5,
        max_seconds: float = 30.0,
    ) -> dict[str, np.ndarray]:
        """Sammelt pro Sprecher-Label dessen Audio-Anteile (concat, gecappt)."""
        chunks: dict[str, list[np.ndarray]] = {}
        for seg in segments:
            spk = seg.get("speaker")
            if not spk:
                continue
            start = int(float(seg["start"]) * sample_rate)
            end = int(float(seg["end"]) * sample_rate)
            if end <= start:
                continue
            chunks.setdefault(spk, []).append(audio_np[start:end])

        out: dict[str, np.ndarray] = {}
        max_samples = int(max_seconds * sample_rate)
        min_samples = int(min_seconds * sample_rate)
        for spk, parts in chunks.items():
            concat = np.concatenate(parts)
            if len(concat) < min_samples:
                continue
            if len(concat) > max_samples:
                concat = concat[:max_samples]
            out[spk] = concat
        return out

    def per_speaker(
        self,
        audio_np: np.ndarray,
        segments: list[dict],
        sample_rate: int = SAMPLE_RATE,
        min_seconds: float = 1.5,
        max_seconds: float = 30.0,
    ) -> dict[str, np.ndarray]:
        """Convenience: Embedding pro Sprecher (ohne Audio-Chunks zurückzugeben)."""
        chunks = self.per_speaker_chunks(
            audio_np, segments, sample_rate, min_seconds, max_seconds
        )
        return {spk: self.extract(c, sample_rate) for spk, c in chunks.items()}


class SpeakerStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        with self._conn() as c:
            c.executescript(_SCHEMA)
            self._migrate(c)

    @staticmethod
    def _migrate(c: sqlite3.Connection) -> None:
        cols = {row[1] for row in c.execute("PRAGMA table_info(pending_clusters)")}
        for name, ddl in [
            ("audio_sample", "ALTER TABLE pending_clusters ADD COLUMN audio_sample BLOB"),
            ("source_filename", "ALTER TABLE pending_clusters ADD COLUMN source_filename TEXT"),
            ("first_start", "ALTER TABLE pending_clusters ADD COLUMN first_start REAL"),
            ("last_end", "ALTER TABLE pending_clusters ADD COLUMN last_end REAL"),
        ]:
            if name not in cols:
                c.execute(ddl)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- Enrollment ---

    def add_sample(self, name: str, embedding: np.ndarray, source: str | None = None) -> int:
        with self._conn() as c:
            c.execute("INSERT OR IGNORE INTO speakers(name) VALUES (?)", (name,))
            row = c.execute("SELECT id FROM speakers WHERE name = ?", (name,)).fetchone()
            speaker_id = row[0]
            c.execute(
                "INSERT INTO embeddings(speaker_id, vector, source) VALUES (?, ?, ?)",
                (speaker_id, _vec_to_blob(embedding), source),
            )
            return speaker_id

    def list_speakers(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT s.name, s.created_at, COUNT(e.id) "
                "FROM speakers s LEFT JOIN embeddings e ON e.speaker_id = s.id "
                "GROUP BY s.id ORDER BY s.name"
            ).fetchall()
        return [{"name": r[0], "created_at": r[1], "samples": r[2]} for r in rows]

    def delete_speaker(self, name: str) -> bool:
        with self._conn() as c:
            cur = c.execute("DELETE FROM speakers WHERE name = ?", (name,))
            return cur.rowcount > 0

    # --- Matching ---

    def _reference_embeddings(self) -> list[tuple[str, np.ndarray]]:
        """Mittelwert-Embedding pro bekanntem Sprecher."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT s.name, e.vector FROM speakers s "
                "JOIN embeddings e ON e.speaker_id = s.id"
            ).fetchall()
        by_name: dict[str, list[np.ndarray]] = {}
        for name, blob in rows:
            by_name.setdefault(name, []).append(_blob_to_vec(blob))
        return [(name, np.mean(np.stack(vs), axis=0)) for name, vs in by_name.items()]

    def match(self, embedding: np.ndarray, threshold: float) -> tuple[str | None, float]:
        """Beste Übereinstimmung oder (None, best_score) wenn unter Schwelle."""
        refs = self._reference_embeddings()
        if not refs:
            return None, 0.0
        scores = [(name, _cosine(embedding, ref)) for name, ref in refs]
        scores.sort(key=lambda x: x[1], reverse=True)
        best_name, best_score = scores[0]
        if best_score < threshold:
            return None, best_score
        return best_name, best_score

    # --- Sessions / nachträgliche Zuweisung ---

    def new_session(self) -> str:
        return uuid.uuid4().hex

    def store_pending(
        self,
        session_id: str,
        clusters: dict[str, np.ndarray],
        matched: dict[str, str | None],
        audio: dict[str, bytes] | None = None,
        source_filename: str | None = None,
        time_ranges: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        audio = audio or {}
        time_ranges = time_ranges or {}
        with self._conn() as c:
            for label, vec in clusters.items():
                start, end = time_ranges.get(label, (None, None))
                c.execute(
                    "INSERT OR REPLACE INTO pending_clusters"
                    "(session_id, cluster_label, vector, matched_name, audio_sample, "
                    " source_filename, first_start, last_end) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        session_id, label, _vec_to_blob(vec), matched.get(label),
                        audio.get(label), source_filename, start, end,
                    ),
                )

    def delete_pending_cluster(self, session_id: str, cluster_label: str) -> bool:
        with self._conn() as c:
            cur = c.execute(
                "DELETE FROM pending_clusters WHERE session_id = ? AND cluster_label = ?",
                (session_id, cluster_label),
            )
            return cur.rowcount > 0

    def get_pending_audio(self, session_id: str, cluster_label: str) -> bytes | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT audio_sample FROM pending_clusters "
                "WHERE session_id = ? AND cluster_label = ?",
                (session_id, cluster_label),
            ).fetchone()
        return row[0] if row and row[0] else None

    def assign_session(self, session_id: str, mapping: dict[str, str]) -> dict[str, str]:
        """Mapping {cluster_label: name} -> persistiert die Cluster-Embeddings unter den Namen."""
        assigned: dict[str, str] = {}
        with self._conn() as c:
            for label, name in mapping.items():
                row = c.execute(
                    "SELECT vector FROM pending_clusters WHERE session_id = ? AND cluster_label = ?",
                    (session_id, label),
                ).fetchone()
                if not row:
                    continue
                vec = _blob_to_vec(row[0])
                # add_sample inline (gleiche Connection):
                c.execute("INSERT OR IGNORE INTO speakers(name) VALUES (?)", (name,))
                sid = c.execute("SELECT id FROM speakers WHERE name = ?", (name,)).fetchone()[0]
                c.execute(
                    "INSERT INTO embeddings(speaker_id, vector, source) VALUES (?, ?, ?)",
                    (sid, _vec_to_blob(vec), f"session:{session_id}:{label}"),
                )
                c.execute(
                    "UPDATE pending_clusters SET matched_name = ? "
                    "WHERE session_id = ? AND cluster_label = ?",
                    (name, session_id, label),
                )
                assigned[label] = name
        return assigned

    def list_pending(self, unassigned_only: bool = True, limit: int = 50) -> list[dict]:
        """Offene Sitzungen mit ihren Cluster-Labels, gruppiert nach session_id."""
        where = "WHERE matched_name IS NULL" if unassigned_only else ""
        with self._conn() as c:
            rows = c.execute(
                f"SELECT session_id, cluster_label, matched_name, created_at, "
                f"  (audio_sample IS NOT NULL) AS has_audio, "
                f"  source_filename, first_start, last_end "
                f"FROM pending_clusters {where} ORDER BY created_at DESC, session_id, cluster_label"
            ).fetchall()
        sessions: dict[str, dict] = {}
        for sid, label, matched, created, has_audio, src, fstart, lend in rows:
            s = sessions.setdefault(
                sid,
                {"session_id": sid, "created_at": created, "source_filename": src, "clusters": []},
            )
            s["clusters"].append({
                "label": label,
                "matched_name": matched,
                "has_audio": bool(has_audio),
                "first_start": fstart,
                "last_end": lend,
            })
        return list(sessions.values())[:limit]

    def prune_pending(self, older_than_seconds: int = 7 * 24 * 3600) -> int:
        with self._conn() as c:
            cur = c.execute(
                "DELETE FROM pending_clusters "
                "WHERE created_at < datetime('now', ?)",
                (f"-{int(older_than_seconds)} seconds",),
            )
            return cur.rowcount


def relabel_segments(
    segments: list[dict],
    cluster_to_name: dict[str, str],
) -> list[dict]:
    """Ersetzt die Speaker-Labels in den Segmenten."""
    for seg in segments:
        spk = seg.get("speaker")
        if spk and spk in cluster_to_name:
            seg["speaker"] = cluster_to_name[spk]
    return segments
