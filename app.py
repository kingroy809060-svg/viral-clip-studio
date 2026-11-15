"""Viral Clip Studio — intake, style profiles, clip finder, and test-first batch editor."""
from __future__ import annotations

import os
import json
import re
import shutil
import sqlite3
import uuid
import copy
from contextlib import asynccontextmanager, closing
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
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


app = FastAPI(title="Viral Clip Studio", version="0.4.0-phase4", lifespan=lifespan)

VIDEO_TYPES = {"Podcast", "Talking Head", "Vlog", "Tutorial", "Motivational"}


class RenderRequest(BaseModel):
    source_id: str = Field(min_length=8, max_length=64)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    word_timestamps: list[dict[str, object]] = Field(default_factory=list)
    style_slug: str = Field(min_length=1, max_length=100)
    hook_line: str = Field(default="", max_length=500)
    music_upload_id: str | None = Field(default=None, max_length=64)
    video_type: str | None = Field(default=None, max_length=40)


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
        conn.execute("""CREATE TABLE IF NOT EXISTS batches (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""")
        conn.commit()


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    if not TEMPLATE_PATH.is_file():
        raise HTTPException(status_code=500, detail="UI template is missing")
    return HTMLResponse(TEMPLATE_PATH.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "phase": 4, "ffmpeg": ffmpeg_status()})


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
    """Phase 0 upload intake retained for direct/manual file checks."""
    primary = await save_upload(long_video, "long video")
    reference = await save_upload(reference_video, "reference video") if reference_video else None
    return {"message": "Upload saved. Use the Phase 4 batch workflow to analyze and edit the video.",
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
    if request.video_type:
        if request.video_type not in VIDEO_TYPES:
            raise HTTPException(status_code=422, detail="Video type is not supported.")
        profile = _profile_for_video_type(profile, request.video_type)
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
    rendered["video_type"] = request.video_type
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


def _profile_for_video_type(profile: dict[str, object], video_type: str) -> dict[str, object]:
    """Apply the Phase 4 edit preset while preserving the approved base style."""
    adapted = copy.deepcopy(profile)
    adapted["video_type"] = video_type
    if video_type == "Podcast":
        adapted["zoom_intensity_pct"] = 10
        adapted["speaker_tracking"] = True
    elif video_type == "Talking Head":
        adapted["zoom_intensity_pct"] = 15
    elif video_type == "Vlog":
        adapted["zoom_intensity_pct"] = 12
        adapted["preserve_scene_changes"] = True
    elif video_type == "Tutorial":
        adapted["caption_position"] = "top"
    elif video_type == "Motivational":
        adapted["zoom_intensity_pct"] = 0
        adapted["slow_zoom"] = True
    return adapted


def _detect_video_type(video_path: Path, transcript_text: str) -> tuple[str, float, str]:
    """Rule-based type estimate from transcript cues and sampled scene/face evidence."""
    text = transcript_text.lower()
    patterns = [
        ("Tutorial", ("tutorial", "step by step", "click on", "screen recording", "कैसे", "क्लिक करें", "पहला step")),
        ("Motivational", ("motivation", "never give up", "believe in yourself", "your dream", "हिम्मत", "मेहनत", "हार मत", "सपना")),
        ("Podcast", ("podcast", "interview", "our guest", "episode", "पॉडकास्ट", "इंटरव्यू")),
        ("Vlog", ("vlog", "day in my life", "travel with me", "let's go", "आज हम", "घूमने")),
    ]
    for video_type, cues in patterns:
        if any(cue in text for cue in cues):
            return video_type, 0.78, "transcript keyword cues"

    face_hits = scene_changes = samples = 0
    try:
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        previous = None
        duration = probe_duration_seconds(video_path)
        for fraction in (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95):
            capture.set(cv2.CAP_PROP_POS_MSEC, duration * fraction * 1000)
            ok, frame = capture.read()
            if not ok:
                continue
            samples += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if len(detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))):
                face_hits += 1
            thumb = cv2.resize(gray, (96, 54))
            if previous is not None and float(cv2.absdiff(thumb, previous).mean()) > 28:
                scene_changes += 1
            previous = thumb
        capture.release()
    except Exception:
        samples = 0
    if samples:
        face_ratio = face_hits / samples
        change_ratio = scene_changes / max(1, samples - 1)
        if face_ratio >= 0.72 and change_ratio < 0.45:
            return "Talking Head", 0.62, "stable sampled face framing"
        if change_ratio >= 0.45:
            return "Vlog", 0.58, "high sampled scene change rate"
        if face_ratio > 0:
            return "Podcast", 0.52, "sampled speaker face detected"
    return "Talking Head", 0.35, "default guess; review the detected type"


def _load_batch(batch_id: str) -> dict[str, object] | None:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        row = conn.execute("SELECT payload_json FROM batches WHERE id = ?", (batch_id,)).fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return None


def _save_batch(batch: dict[str, object]) -> None:
    batch["updated_at"] = datetime.now(timezone.utc).isoformat()
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute("UPDATE batches SET status = ?, payload_json = ?, updated_at = ? WHERE id = ?",
                     (batch["status"], json.dumps(batch, ensure_ascii=False), batch["updated_at"], batch["id"]))
        conn.commit()


def _candidate_from_ref(batch: dict[str, object], candidate_ref: str) -> tuple[dict[str, object], dict[str, object]]:
    match = re.fullmatch(r"(\d+):(\d+)", candidate_ref)
    if not match:
        raise HTTPException(status_code=422, detail="Candidate reference is invalid.")
    source_index, clip_index = map(int, match.groups())
    sources = batch.get("sources", [])
    if source_index >= len(sources) or clip_index >= len(sources[source_index].get("clips", [])):
        raise HTTPException(status_code=404, detail="Clip candidate was not found in this batch.")
    return sources[source_index], sources[source_index]["clips"][clip_index]


@app.post("/batches")
async def create_batch(
    background: list[UploadFile] = File(...),
    style_slug: str = Form(default="global-baatien-style"),
    video_type: str = Form(default="auto"),
    music_upload_id: str | None = Form(default=None),
) -> dict[str, object]:
    """Analyze one to four long videos and create a test-first batch."""
    if not 1 <= len(background) <= 4:
        raise HTTPException(status_code=422, detail="Upload between one and four long videos per batch.")
    safe_slug = re.sub(r"[^a-z0-9-]", "", style_slug.lower())
    style_path = STYLE_DIR / f"{safe_slug}.json"
    if not safe_slug or not style_path.is_file():
        raise HTTPException(status_code=404, detail="Choose a saved and approved Style Profile.")
    if video_type != "auto" and video_type not in VIDEO_TYPES:
        raise HTTPException(status_code=422, detail="Select Auto or one of the five supported video types.")
    if music_upload_id:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            music_row = conn.execute("SELECT 1 FROM uploads WHERE id = ? AND role = ?",
                                     (music_upload_id, "royalty-free music")).fetchone()
        if not music_row:
            raise HTTPException(status_code=404, detail="Uploaded licensed music file was not found.")

    batch_id = uuid.uuid4().hex[:16]
    now = datetime.now(timezone.utc).isoformat()
    batch: dict[str, object] = {
        "id": batch_id, "status": "analyzing", "style_slug": safe_slug,
        "style_name": json.loads(style_path.read_text(encoding="utf-8")).get("style_name", safe_slug),
        "requested_video_type": video_type, "sources": [], "test_candidate_ref": None,
        "music_upload_id": music_upload_id,
        "test_clip": None, "progress": {"completed": 0, "total": 0}, "results": [], "errors": [],
        "created_at": now, "updated_at": now,
    }
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute("INSERT INTO batches VALUES (?, ?, ?, ?, ?)",
                     (batch_id, "analyzing", json.dumps(batch, ensure_ascii=False), now, now))
        conn.commit()

    for file in background:
        try:
            saved = await save_upload(file, "clip finder source")
            source_path = UPLOAD_DIR / f"{saved['id']}{Path(saved['filename']).suffix.lower()}"
            duration = await run_in_threadpool(probe_duration_seconds, source_path)
            if duration < 600 or duration > 7200:
                raise ValueError(f"{saved['filename']}: video must be between 10 minutes and 2 hours.")
            transcript = await run_in_threadpool(transcribe, source_path, "base")
            picks = find_clips(transcript)
            if video_type == "auto":
                detected_type, confidence, type_reason = await run_in_threadpool(
                    _detect_video_type, source_path, str(transcript.get("text", "")))
            else:
                detected_type, confidence, type_reason = video_type, 1.0, "user selected"
            batch["sources"].append({
                "source_id": saved["id"], "filename": saved["filename"],
                "duration_seconds": round(duration, 3), "language": transcript.get("language", "unknown"),
                "video_type": detected_type, "type_confidence": confidence, "type_reason": type_reason,
                "candidate_count": picks["candidate_count"], "scoring_mode": picks["scoring_mode"],
                "clips": picks["clips"],
            })
        except (HTTPException, RuntimeError, ValueError, OSError) as exc:
            batch["errors"].append({"filename": file.filename or "video", "error": str(exc)})

    if not batch["sources"]:
        batch["status"] = "failed"
        _save_batch(batch)
        raise HTTPException(status_code=422, detail={"message": "No uploaded video could be analyzed.", "errors": batch["errors"]})
    batch["status"] = "analysis_ready"
    _save_batch(batch)
    return {"message": "Candidates are ready. Render one test clip and approve it before batch export.", "batch": batch}


@app.get("/batches/{batch_id}")
def get_batch(batch_id: str) -> dict[str, object]:
    batch = _load_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch was not found.")
    return batch


@app.post("/batches/{batch_id}/test")
async def render_batch_test(batch_id: str, candidate_ref: str = Form(...)) -> dict[str, object]:
    batch = _load_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch was not found.")
    if batch["status"] not in {"analysis_ready", "test_ready"}:
        raise HTTPException(status_code=409, detail="A test clip can only be rendered before batch approval.")
    source, candidate = _candidate_from_ref(batch, candidate_ref)
    request = RenderRequest(
        source_id=source["source_id"], start_seconds=candidate["start_seconds"],
        end_seconds=candidate["end_seconds"], word_timestamps=candidate["word_timestamps"],
        style_slug=batch["style_slug"], hook_line=candidate["hook_line"], video_type=source["video_type"],
        music_upload_id=batch.get("music_upload_id"),
    )
    rendered = await render_approved_clip(request)
    batch["status"] = "test_ready"
    batch["test_candidate_ref"] = candidate_ref
    batch["test_clip"] = rendered["clip"]
    _save_batch(batch)
    return {"message": "Test export is ready. Approve this exact test to unlock remaining clips.",
            "batch": batch, "test_clip": rendered["clip"]}


@app.post("/batches/{batch_id}/approve")
def approve_batch_test(batch_id: str) -> dict[str, object]:
    batch = _load_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch was not found.")
    if batch["status"] != "test_ready" or not batch.get("test_clip"):
        raise HTTPException(status_code=409, detail="Render and review one test clip before approving the batch.")
    batch["status"] = "approved"
    batch["approved_at"] = datetime.now(timezone.utc).isoformat()
    _save_batch(batch)
    return {"message": "Test approved. Remaining clips can now be generated.", "batch": batch}


async def _generate_remaining(batch_id: str) -> None:
    batch = _load_batch(batch_id)
    if not batch or batch["status"] != "generating":
        return
    remaining = [(source_index, clip_index) for source_index, source in enumerate(batch["sources"])
                 for clip_index, _ in enumerate(source["clips"])
                 if f"{source_index}:{clip_index}" != batch.get("test_candidate_ref")]
    progress = {"completed": 0, "total": len(remaining)}
    batch["progress"] = progress
    batch["results"] = []
    batch["errors"] = batch.get("errors", [])
    _save_batch(batch)
    for source_index, clip_index in remaining:
        source, candidate = _candidate_from_ref(batch, f"{source_index}:{clip_index}")
        try:
            rendered = await render_approved_clip(RenderRequest(
                source_id=source["source_id"], start_seconds=candidate["start_seconds"],
                end_seconds=candidate["end_seconds"], word_timestamps=candidate["word_timestamps"],
                style_slug=batch["style_slug"], hook_line=candidate["hook_line"], video_type=source["video_type"],
                music_upload_id=batch.get("music_upload_id"),
            ))
            batch["results"].append({"source": source["filename"], "candidate_ref": f"{source_index}:{clip_index}",
                                     "score": candidate["score"], "clip": rendered["clip"]})
        except Exception as exc:
            batch["errors"].append({"candidate_ref": f"{source_index}:{clip_index}", "error": str(exc)[:500]})
        progress["completed"] += 1
        batch["progress"] = progress
        _save_batch(batch)
    batch["status"] = "partial" if batch["errors"] else "complete"
    _save_batch(batch)


@app.post("/batches/{batch_id}/generate")
async def generate_batch(batch_id: str, background_tasks: BackgroundTasks) -> dict[str, object]:
    batch = _load_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch was not found.")
    if batch["status"] != "approved":
        raise HTTPException(status_code=409, detail="Approve the test clip before generating remaining clips.")
    batch["status"] = "generating"
    batch["progress"] = {"completed": 0, "total": sum(len(source["clips"]) for source in batch["sources"]) - 1}
    _save_batch(batch)
    background_tasks.add_task(_generate_remaining, batch_id)
    return {"message": "Remaining clips queued.", "batch": batch}


@app.get("/batches")
def list_batches() -> list[dict[str, object]]:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        rows = conn.execute("SELECT payload_json FROM batches ORDER BY created_at DESC LIMIT 50").fetchall()
    return [json.loads(row[0]) for row in rows]


@app.get("/clips/history")
def clip_history() -> list[dict[str, object]]:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        rows = conn.execute("""SELECT r.id, r.source_id, u.original_name, r.files_json, r.created_at
            FROM rendered_clips r LEFT JOIN uploads u ON u.id = r.source_id
            ORDER BY r.created_at DESC LIMIT 100""").fetchall()
    history = []
    for render_id, source_id, source_name, encoded, created_at in rows:
        try:
            item = json.loads(encoded)
            item.update({"source_id": source_id, "source_filename": source_name, "created_at": created_at})
            history.append(item)
        except json.JSONDecodeError:
            continue
    return history


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
