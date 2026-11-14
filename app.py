"""Viral Clip Studio — upload intake, style profiles, clip finder, and Phase 3 editor."""
from __future__ import annotations

import os
import json
import re
import shutil
import sqlite3
import uuid
from contextlib import asynccontextmanager, closing
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from clip_finder import find_clips
from editor import render_clip
from style_extractor import PROFILE_FIELDS, extract_style, validate_profile
from transcriber import probe_duration_seconds, transcribe

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "outputs"
DB_PATH = ROOT / "viral_clip_studio.sqlite3"
TEMPLATE_PATH = ROOT / "templates" / "index.html"
STYLE_DIR = ROOT / "styles"
load_dotenv(ROOT / ".env")
VIDEO_EXTENSIONS = {".mp4", ".mov"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024 * 1024  # 8 GiB per file
CHUNK_SIZE = 1024 * 1024

@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    status = ffmpeg_status()
    if status["available"]:
        print(f"[startup] FFmpeg OK: {status['version']}", flush=True)
    else:
        print(f"[startup] WARNING: {status['version']}", flush=True)
    yield


app = FastAPI(title="Viral Clip Studio", version="0.3.0-phase3", lifespan=lifespan)


class RenderRequest(BaseModel):
    source_id: str = Field(min_length=8, max_length=64)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    word_timestamps: list[dict[str, object]] = Field(default_factory=list)
    style_slug: str = Field(min_length=1, max_length=100)
    hook_line: str = Field(default="", max_length=500)
    music_upload_id: str | None = Field(default=None, max_length=64)


def ffmpeg_status() -> dict[str, str | bool]:
    executable = shutil.which("ffmpeg")
    if not executable:
        return {"available": False, "path": "", "version": "FFmpeg was not found on PATH"}
    try:
        import subprocess

        result = subprocess.run([executable, "-version"], capture_output=True, text=True, timeout=5)
        first_line = (result.stdout or result.stderr).splitlines()[0]
        return {"available": result.returncode == 0, "path": executable, "version": first_line}
    except (OSError, subprocess.SubprocessError, IndexError) as exc:
        return {"available": False, "path": executable, "version": str(exc)}


def init_db() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STYLE_DIR.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS uploads (
            id TEXT PRIMARY KEY,
            role TEXT NOT NULL,
            original_name TEXT NOT NULL,
            stored_path TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS rendered_clips (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            output_dir TEXT NOT NULL,
            files_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""")
        conn.commit()


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    if not TEMPLATE_PATH.is_file():
        raise HTTPException(status_code=500, detail="UI template is missing")
    return HTMLResponse(TEMPLATE_PATH.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "phase": 3, "ffmpeg": ffmpeg_status()})


async def save_upload(file: UploadFile, role: str) -> dict[str, str | int]:
    original_name = Path(file.filename or "video").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"{role}: only MP4 and MOV files are accepted")

    upload_id = uuid.uuid4().hex
    destination = UPLOAD_DIR / f"{upload_id}{suffix}"
    size = 0
    try:
        with destination.open("wb") as target:
            while chunk := await file.read(CHUNK_SIZE):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail=f"{role}: file exceeds 8 GiB limit")
                target.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail=f"{role}: uploaded file is empty")
        with closing(sqlite3.connect(DB_PATH)) as conn:
            conn.execute(
                "INSERT INTO uploads VALUES (?, ?, ?, ?, ?, ?)",
                (upload_id, role, original_name, str(destination.relative_to(ROOT)), size,
                 datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
        return {"id": upload_id, "role": role, "filename": original_name, "size_bytes": size}
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


async def save_audio_upload(file: UploadFile) -> dict[str, str | int]:
    original_name = Path(file.filename or "music").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}:
        raise HTTPException(status_code=415, detail="Music: upload MP3, WAV, M4A, AAC, OGG, or FLAC.")
    upload_id = uuid.uuid4().hex
    destination = UPLOAD_DIR / f"{upload_id}{suffix}"
    size = 0
    try:
        with destination.open("wb") as target:
            while chunk := await file.read(CHUNK_SIZE):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Music file exceeds 8 GiB limit.")
                target.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="Music upload is empty.")
        with closing(sqlite3.connect(DB_PATH)) as conn:
            conn.execute(
                "INSERT INTO uploads VALUES (?, ?, ?, ?, ?, ?)",
                (upload_id, "royalty-free music", original_name, str(destination.relative_to(ROOT)), size,
                 datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
        return {"id": upload_id, "filename": original_name, "size_bytes": size}
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


@app.post("/upload")
async def upload_videos(
    long_video: UploadFile = File(...),
    reference_video: UploadFile | None = File(default=None),
) -> dict[str, object]:
    """Phase 0: accept and persist source files; processing is not implemented yet."""
    primary = await save_upload(long_video, "long video")
    reference = await save_upload(reference_video, "reference video") if reference_video else None
    return {"message": "Upload saved. Use the Phase 2 clip finder to transcribe and score the long video.",
            "long_video": primary, "reference_video": reference}


@app.post("/clips/find")
async def analyze_long_video(long_video: UploadFile = File(...)) -> dict[str, object]:
    """Transcribe a 10-minute to 2-hour source locally and return top clip candidates."""
    saved = await save_upload(long_video, "clip finder source")
    video_path = UPLOAD_DIR / f"{saved['id']}{Path(saved['filename']).suffix.lower()}"
    try:
        duration = await run_in_threadpool(probe_duration_seconds, video_path)
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if duration < 600 or duration > 7200:
        raise HTTPException(status_code=422, detail="Source video must be between 10 minutes and 2 hours.")
    try:
        transcript = await run_in_threadpool(transcribe, video_path, "base")
        result = find_clips(transcript)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "video": {"id": saved["id"], "filename": saved["filename"],
                  "duration_seconds": round(duration, 3)},
        "transcription": transcript,
        "candidate_count": result["candidate_count"],
        "scoring_mode": result["scoring_mode"],
        "clips": result["clips"],
        "message": "Transcript and clip candidates are ready. Whisper ran locally; transcript wording is preserved from Whisper.",
    }


@app.post("/music/upload")
async def upload_licensed_music(music_file: UploadFile = File(...)) -> dict[str, object]:
    """Save a user-provided licensed music bed; the app does not bundle music."""
    return {"message": "Music saved. Use only music you have rights to use.",
            "music": await save_audio_upload(music_file)}


@app.post("/clips/render")
async def render_approved_clip(request: RenderRequest) -> dict[str, object]:
    """Render one user-approved Phase 2 candidate using an approved saved style."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        source_row = conn.execute(
            "SELECT stored_path FROM uploads WHERE id = ? AND role = ?",
            (request.source_id, "clip finder source"),
        ).fetchone()
    if not source_row:
        raise HTTPException(status_code=404, detail="Source upload was not found. Run the clip finder first.")
    source_path = ROOT / source_row[0]
    if not source_path.is_file():
        raise HTTPException(status_code=404, detail="Source video file is missing. Upload it again.")
    if request.end_seconds - request.start_seconds < 30 or request.end_seconds - request.start_seconds > 60:
        raise HTTPException(status_code=422, detail="Approved clips must be between 30 and 60 seconds.")
    safe_slug = re.sub(r"[^a-z0-9-]", "", request.style_slug.lower())
    profile_path = STYLE_DIR / f"{safe_slug}.json"
    if not safe_slug or not profile_path.is_file():
        raise HTTPException(status_code=404, detail="Choose a saved and approved Style Profile first.")
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Saved Style Profile JSON is invalid.") from exc
    music_path = None
    if request.music_upload_id:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            music_row = conn.execute(
                "SELECT stored_path FROM uploads WHERE id = ? AND role = ?",
                (request.music_upload_id, "royalty-free music"),
            ).fetchone()
        if not music_row:
            raise HTTPException(status_code=404, detail="Uploaded licensed music file was not found.")
        music_path = ROOT / music_row[0]
    output_dir = OUTPUT_DIR / request.source_id
    try:
        rendered = await run_in_threadpool(
            render_clip, source_path, output_dir, request.start_seconds, request.end_seconds,
            request.word_timestamps, profile, request.hook_line, music_path,
        )
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    rendered["style_name"] = profile.get("style_name", safe_slug)
    rendered["files"] = {
        key: f"/outputs/{rendered['id']}/{filename}"
        for key, filename in rendered.items()
        if key in {"mp4", "srt", "transcript", "title", "caption", "hashtags"}
    }
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute(
            "INSERT INTO rendered_clips VALUES (?, ?, ?, ?, ?)",
            (rendered["id"], request.source_id,
             str((output_dir / rendered["mp4"]).parent.relative_to(ROOT)),
             json.dumps(rendered, ensure_ascii=False), datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    return {"message": "Approved test clip rendered. Review it before generating more clips.", "clip": rendered}


@app.get("/outputs/{render_id}/{filename}")
def download_rendered_file(render_id: str, filename: str) -> FileResponse:
    """Serve only the six named assets created for a known render id."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        row = conn.execute("SELECT output_dir, files_json FROM rendered_clips WHERE id = ?", (render_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Rendered clip was not found.")
    try:
        metadata = json.loads(row[1])
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Rendered clip metadata is invalid.") from exc
    allowed_names = {metadata.get(key) for key in ("mp4", "srt", "transcript", "title", "caption", "hashtags")}
    if filename not in allowed_names:
        raise HTTPException(status_code=404, detail="This file is not part of the rendered clip.")
    path = (ROOT / row[0] / filename).resolve()
    if not path.is_relative_to(OUTPUT_DIR.resolve()) or not path.is_file():
        raise HTTPException(status_code=404, detail="Rendered file is missing.")
    media_type = "video/mp4" if path.suffix == ".mp4" else "text/plain; charset=utf-8"
    return FileResponse(path, media_type=media_type, filename=path.name)


@app.post("/styles/analyze")
async def analyze_style(
    reference_video: UploadFile = File(...),
    style_name: str = Form(default="Global Baatien Style"),
) -> dict[str, object]:
    """Analyze a user-provided local reference video and return an editable profile."""
    saved = await save_upload(reference_video, "style reference video")
    # save_upload names stored files with the generated upload id and original extension.
    candidates = list(UPLOAD_DIR.glob(f"{saved['id']}.*"))
    if not candidates:
        raise HTTPException(status_code=500, detail="Saved reference video could not be found.")
    video_path = candidates[0]
    try:
        profile = await run_in_threadpool(extract_style, video_path, style_name)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "profile": profile,
        "message": "Analysis is an estimate from rendered pixels. Exact source font and music-only volume cannot be recovered reliably.",
    }


@app.get("/styles")
def list_styles() -> list[dict[str, object]]:
    styles: list[dict[str, object]] = []
    for path in sorted(STYLE_DIR.glob("*.json")):
        try:
            styles.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return styles


@app.post("/styles/save")
def save_style(profile: dict[str, object]) -> dict[str, object]:
    try:
        validated = validate_profile(profile)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    slug = re.sub(r"[^a-z0-9]+", "-", validated["style_name"].strip().lower()).strip("-") or "style"
    destination = STYLE_DIR / f"{slug}.json"
    destination.write_text(json.dumps(validated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"saved": True, "path": f"styles/{destination.name}", "profile": validated}


@app.get("/styles/{slug}")
def get_style(slug: str) -> dict[str, object]:
    safe_slug = re.sub(r"[^a-z0-9-]", "", slug.lower())
    path = STYLE_DIR / f"{safe_slug}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Style profile was not found.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Style profile JSON is invalid.") from exc


def run() -> None:
    host = os.getenv("VCS_HOST", "127.0.0.1")
    port = int(os.getenv("VCS_PORT", "8000"))
    uvicorn.run("app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()
