"""Offline Whisper transcription with word-level timestamps."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def _load_model(model_name: str, device: str) -> Any:
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError(
            "Local Whisper is not installed. Run `python -m pip install -r requirements.txt` first."
        ) from exc
    try:
        return whisper.load_model(model_name, device=device)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load Whisper model '{model_name}'. Check the local model download/cache and try again."
        ) from exc


def probe_duration_seconds(video_path: str | Path) -> float:
    """Read the media duration with ffprobe, without loading it into memory."""
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file was not found: {path.name}")
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe was not found on PATH. Install FFmpeg before analyzing videos.")
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode:
        raise RuntimeError("ffprobe could not read this video. Check that it is a valid MP4/MOV file.")
    try:
        duration = float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("The video duration could not be determined.") from exc
    if duration <= 0:
        raise RuntimeError("The video has no readable duration.")
    return duration


def _word_items(raw_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for segment in raw_segments:
        start = float(segment.get("start", 0.0))
        end = max(start, float(segment.get("end", start)))
        raw_words = segment.get("words") or []
        if raw_words:
            for item in raw_words:
                token = str(item.get("word", ""))
                if not token.strip():
                    continue
                words.append({
                    "word": token,
                    "start": round(float(item.get("start", start)), 3),
                    "end": round(float(item.get("end", end)), 3),
                    "probability": round(float(item["probability"]), 4)
                    if isinstance(item.get("probability"), (float, int)) else None,
                })
            continue

        # Older Whisper builds may omit word timestamps. Preserve the heard words
        # and distribute them across Whisper's segment interval as a fallback.
        tokens = re.findall(r"\S+", str(segment.get("text", "")))
        span = max(0.0, end - start)
        for index, token in enumerate(tokens):
            word_start = start + span * index / max(1, len(tokens))
            word_end = start + span * (index + 1) / max(1, len(tokens))
            words.append({"word": token, "start": round(word_start, 3),
                          "end": round(word_end, 3), "probability": None})
    words.sort(key=lambda item: (item["start"], item["end"]))
    return words


def transcribe(video_path: str | Path, model_name: str = "base", language: str | None = None) -> dict[str, Any]:
    """Transcribe a video with Whisper's local model; `language=None` enables autodetection."""
    path = Path(video_path)
    duration = probe_duration_seconds(path)
    device = os.getenv("WHISPER_DEVICE", "cpu").strip() or "cpu"
    model = _load_model(model_name, device)
    try:
        result = model.transcribe(
            str(path),
            task="transcribe",
            language=language or None,
            word_timestamps=True,
            verbose=False,
            fp16=device.lower().startswith("cuda"),
        )
    except Exception as exc:
        raise RuntimeError(f"Whisper could not transcribe this video: {exc}") from exc

    raw_segments = result.get("segments") or []
    segments = [
        {"start": round(float(item.get("start", 0.0)), 3),
         "end": round(float(item.get("end", 0.0)), 3),
         "text": str(item.get("text", "")).strip()}
        for item in raw_segments
        if str(item.get("text", "")).strip()
    ]
    words = _word_items(raw_segments)
    return {
        "text": str(result.get("text", "")).strip(),
        "language": result.get("language") or language or "unknown",
        "model": model_name,
        "duration_seconds": round(duration, 3),
        "segments": segments,
        "words": words,
    }
