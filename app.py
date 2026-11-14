"""Viral Clip Studio — Phase 0 upload intake and Phase 1 style profiles."""
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
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from style_extractor import PROFILE_FIELDS, extract_style, validate_profile

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
DB_PATH = ROOT / "viral_clip_studio.sqlite3"
TEMPLATE_PATH = ROOT / "templates" / "index.html"
STYLE_DIR = ROOT / "styles"
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


app = FastAPI(title="Viral Clip Studio", version="0.1.0-phase1", lifespan=lifespan)


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
        conn.commit()


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    if not TEMPLATE_PATH.is_file():
        raise HTTPException(status_code=500, detail="UI template is missing")
    return HTMLResponse(TEMPLATE_PATH.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "phase": 1, "ffmpeg": ffmpeg_status()})


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


@app.post("/upload")
async def upload_videos(
    long_video: UploadFile = File(...),
    reference_video: UploadFile | None = File(default=None),
) -> dict[str, object]:
    """Phase 0: accept and persist source files; processing is not implemented yet."""
    primary = await save_upload(long_video, "long video")
    reference = await save_upload(reference_video, "reference video") if reference_video else None
    return {"message": "Upload saved. Clip finding and editing are planned for later phases.",
            "long_video": primary, "reference_video": reference}


@app.post("/styles/analyze")
async def analyze_style(reference_video: UploadFile = File(...), style_name: str = "Global Baatien Style") -> dict[str, object]:
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
