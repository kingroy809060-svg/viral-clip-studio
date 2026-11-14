"""Free, local heuristics for deriving an editable style profile from a video.

Raster video cannot reveal the original font file, and mixed audio cannot
reliably distinguish music from speech. Those values are explicitly estimates.
"""
from __future__ import annotations

import json
import re
import shutil
import statistics
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

PROFILE_FIELDS = (
    "style_name", "reference_duration_seconds", "caption_font_family",
    "caption_font_size_pct", "caption_font_color", "caption_position",
    "caption_background", "cut_speed_cuts_per_minute", "zoom_intensity_pct",
    "contrast", "saturation", "brightness", "music_volume_db", "hook_text", "end_cta",
)


def _run(args: list[str], timeout: float = 180) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(args, capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Could not run {Path(args[0]).name}: {exc}") from exc


def _probe(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe is required (install FFmpeg and ensure it is on PATH).")
    result = _run([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], 30)
    if result.returncode:
        raise RuntimeError("This file could not be read as a video. Check that it is a valid MP4 or MOV.")
    return json.loads(result.stdout)


def _frame_rgb(path: Path, at_seconds: float, width: int = 640, height: int = 360) -> np.ndarray | None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required (install FFmpeg and ensure it is on PATH).")
    result = _run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{max(0, at_seconds):.3f}",
        "-i", str(path), "-frames:v", "1", "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ], 30)
    expected = width * height * 3
    if result.returncode or len(result.stdout) != expected:
        return None
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(height, width, 3)


def _ocr_words(frame: np.ndarray) -> list[dict[str, Any]]:
    tesseract = shutil.which("tesseract")
    if not tesseract:
        return []
    height, width, _ = frame.shape
    ppm = f"P6\n{width} {height}\n255\n".encode() + frame.tobytes()
    try:
        result = subprocess.run([tesseract, "stdin", "stdout", "--psm", "11", "tsv"], input=ppm, capture_output=True, timeout=25, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows: list[dict[str, Any]] = []
    lines = result.stdout.decode("utf-8", "replace").splitlines()[1:]
    for line in lines:
        parts = line.split("\t", 11)
        if len(parts) != 12 or not parts[-1].strip():
            continue
        try:
            confidence = float(parts[10])
            x, y, w, h = map(int, parts[6:10])
        except ValueError:
            continue
        if confidence >= 20:
            rows.append({"text": parts[11].strip(), "confidence": confidence, "x": x, "y": y, "w": w, "h": h})
    return rows


def _caption_color(frame: np.ndarray, words: list[dict[str, Any]]) -> str:
    colors: list[tuple[int, int, int]] = []
    h, w, _ = frame.shape
    for word in words:
        x0, y0 = max(0, word["x"]), max(0, word["y"])
        x1, y1 = min(w, x0 + word["w"]), min(h, y0 + word["h"])
        pixels = frame[y0:y1, x0:x1].reshape(-1, 3)
        if not len(pixels):
            continue
        bright = pixels[(pixels.min(axis=1) > 175) & ((pixels.max(axis=1) - pixels.min(axis=1)) < 85)]
        yellow = pixels[(pixels[:, 0] > 155) & (pixels[:, 1] > 125) & (pixels[:, 2] < 150)]
        red = pixels[(pixels[:, 0] > 150) & (pixels[:, 0] > pixels[:, 1] * 1.35) & (pixels[:, 0] > pixels[:, 2] * 1.3)]
        green = pixels[(pixels[:, 1] > 145) & (pixels[:, 1] > pixels[:, 0] * 1.25) & (pixels[:, 1] > pixels[:, 2] * 1.2)]
        choice = max((bright, yellow, red, green), key=len)
        if len(choice):
            colors.append(tuple(int(v) for v in np.median(choice, axis=0)))
    if not colors:
        return "#FFFFFF (estimated; OCR found no clear caption pixels)"
    rgb = np.median(np.asarray(colors), axis=0).astype(int)
    return "#%02X%02X%02X" % tuple(rgb)


def _measure_frames(path: Path, duration: float) -> tuple[float, float, float]:
    ffmpeg = shutil.which("ffmpeg")
    count = max(1, min(72, int(duration)))
    fps = count / max(duration, 0.5)
    result = _run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path), "-an",
        "-vf", f"fps={fps:.8f},scale=160:90:force_original_aspect_ratio=increase,crop=160:90",
        "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ], max(180, min(1200, duration * 3)))
    frame_bytes = 160 * 90 * 3
    frame_count = len(result.stdout) // frame_bytes
    if result.returncode or frame_count == 0:
        raise RuntimeError("FFmpeg could not sample frames from this reference video.")
    frames = np.frombuffer(result.stdout[:frame_count * frame_bytes], dtype=np.uint8).reshape(frame_count, 90, 160, 3)
    luminance = (0.2126 * frames[..., 0] + 0.7152 * frames[..., 1] + 0.0722 * frames[..., 2])
    brightness = float(np.mean(luminance) / 255 * 100)
    contrast = float(np.mean(np.std(luminance, axis=(1, 2))) / 127.5 * 100)
    saturation = float(np.mean(np.max(frames, axis=-1).astype(float) - np.min(frames, axis=-1).astype(float)) / 255 * 100)
    return round(min(100, contrast), 1), round(min(100, saturation), 1), round(min(100, brightness), 1)


def _cut_speed(path: Path, duration: float) -> float:
    ffmpeg = shutil.which("ffmpeg")
    result = _run([
        ffmpeg, "-hide_banner", "-i", str(path), "-an", "-vf", "select='gt(scene,0.15)',showinfo",
        "-f", "null", "-",
    ], max(180, min(1200, duration * 3)))
    cuts = len(re.findall(rb"pts_time:", result.stderr))
    return round(cuts * 60 / duration, 1) if duration > 0 else 0.0


def _audio_level(path: Path, has_audio: bool, duration: float) -> float:
    if not has_audio:
        return -60.0
    ffmpeg = shutil.which("ffmpeg")
    result = _run([ffmpeg, "-hide_banner", "-i", str(path), "-vn", "-af", "volumedetect", "-f", "null", "-"], max(180, min(1200, duration * 3)))
    match = re.search(rb"mean_volume:\s*(-?[\d.]+)\s*dB", result.stderr)
    return round(float(match.group(1)), 1) if match else -60.0


def _zoom_estimate(path: Path, duration: float) -> float:
    """Estimate apparent scale change from center-registered frame samples.

    This is deliberately a visual estimate, not source-camera metadata. The
    center-scale correlation fallback keeps analysis available without OpenCV.
    """
    try:
        import cv2  # type: ignore
    except ImportError:
        cv2 = None
    sample_count = max(3, min(12, int(duration / 2)))
    times = np.linspace(max(0.1, duration * 0.1), max(0.2, duration * 0.9), sample_count)
    changes: list[float] = []
    previous = None
    for at in times:
        rgb = _frame_rgb(path, float(at), 320, 180)
        if rgb is None:
            continue
        gray = np.mean(rgb, axis=2).astype(np.uint8)
        if previous is not None:
            if cv2 is not None:
                points = cv2.goodFeaturesToTrack(previous, maxCorners=300, qualityLevel=0.01, minDistance=5)
                if points is not None and len(points) >= 8:
                    moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points, None, winSize=(21, 21), maxLevel=3)
                    if moved is not None and status is not None:
                        before, after = points[status.ravel() == 1], moved[status.ravel() == 1]
                        if len(before) >= 8:
                            matrix, inliers = cv2.estimateAffinePartial2D(before, after, method=cv2.RANSAC, ransacReprojThreshold=2.0)
                            inlier_count = int(inliers.sum()) if inliers is not None else 0
                            if matrix is not None and inlier_count >= 6:
                                scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
                                if 0.85 <= scale <= 1.15:
                                    changes.append(abs(scale - 1.0) * 100)
            else:
                # Coarse scale-correlation fallback when OpenCV has not been installed.
                h, w = gray.shape
                yy, xx = np.mgrid[0:h:2, 0:w:2]
                target = gray[::2, ::2].astype(np.float32)
                previous_float = previous.astype(np.float32)
                best_score, best_scale = -1.0, 1.0
                for scale in np.arange(0.94, 1.061, 0.01):
                    sx = np.rint((xx - (w - 1) / 2) / scale + (w - 1) / 2).astype(int)
                    sy = np.rint((yy - (h - 1) / 2) / scale + (h - 1) / 2).astype(int)
                    valid = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
                    source, current = previous_float[sy[valid], sx[valid]], target[valid]
                    if source.size < 100:
                        continue
                    source, current = source - source.mean(), current - current.mean()
                    denom = float(np.linalg.norm(source) * np.linalg.norm(current))
                    score = float(np.dot(source, current) / denom) if denom > 1e-6 else -1.0
                    if score > best_score:
                        best_score, best_scale = score, float(scale)
                if best_score >= 0.45:
                    changes.append(abs(best_scale - 1.0) * 100)
        previous = gray
    if not changes:
        return 0.0
    return round(float(np.clip(np.percentile(changes, 75), 0, 25)), 1)


def extract_style(video_path: str | Path, style_name: str = "Global Baatien Style") -> dict[str, Any]:
    """Analyze a local reference video and return exactly 15 editable fields."""
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Reference video not found: {path}")
    metadata = _probe(path)
    streams = metadata.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not video:
        raise RuntimeError("The uploaded reference file has no video stream.")
    duration = float(metadata.get("format", {}).get("duration") or video.get("duration") or 0)
    if duration <= 0:
        raise RuntimeError("Video duration could not be detected.")

    ocr_samples: list[tuple[float, np.ndarray, list[dict[str, Any]]]] = []
    for at in sorted({min(0.25, duration / 4), duration * 0.25, duration * 0.55, max(0.1, duration - 0.8)}):
        frame = _frame_rgb(path, at)
        if frame is not None:
            ocr_samples.append((at, frame, _ocr_words(frame)))
    detected = [(at, frame, words) for at, frame, words in ocr_samples if words]
    all_words = [word for _, _, words in detected for word in words]
    if detected:
        _, color_frame, color_words = max(detected, key=lambda sample: len(sample[2]))
        caption_color = _caption_color(color_frame, color_words)
        position_sample = max(detected, key=lambda sample: len(sample[2]))[2]
        center = statistics.median(word["y"] + word["h"] / 2 for word in position_sample) / 360
        position = "top" if center < 0.38 else "center" if center < 0.68 else "bottom"
        font_size_pct = round(statistics.median(word["h"] for word in all_words) / 360 * 100, 1)
        background = "dark/translucent (estimated)" if position == "bottom" else "none or unclear (estimated)"
        first_text = " ".join(word["text"] for word in min(detected, key=lambda x: x[0])[2])
        last_text = " ".join(word["text"] for word in max(detected, key=lambda x: x[0])[2])
        hook = first_text[:180]
        cta = last_text[:180] if len(detected) > 1 else ""
    else:
        caption_color, position, font_size_pct = "#FFFFFF (estimated; no OCR text detected)", "bottom", 4.5
        background, hook, cta = "none or unclear (estimated)", "", ""

    contrast, saturation, brightness = _measure_frames(path, duration)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    return {
        "style_name": style_name.strip() or "Global Baatien Style",
        "reference_duration_seconds": round(duration, 2),
        "caption_font_family": "Sans-serif (estimated; original font cannot be identified from pixels)",
        "caption_font_size_pct": font_size_pct,
        "caption_font_color": caption_color,
        "caption_position": position,
        "caption_background": background,
        "cut_speed_cuts_per_minute": _cut_speed(path, duration),
        "zoom_intensity_pct": _zoom_estimate(path, duration),
        "contrast": contrast,
        "saturation": saturation,
        "brightness": brightness,
        "music_volume_db": _audio_level(path, has_audio, duration),
        "hook_text": hook,
        "end_cta": cta,
    }


def validate_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Validate the editable 15-field profile before saving it."""
    if set(profile) != set(PROFILE_FIELDS):
        missing = sorted(set(PROFILE_FIELDS) - set(profile))
        extra = sorted(set(profile) - set(PROFILE_FIELDS))
        raise ValueError(f"Profile must have exactly 15 fields (missing={missing}, extra={extra}).")
    for key in ("style_name", "caption_font_family", "caption_font_color", "caption_position", "caption_background", "hook_text", "end_cta"):
        if not isinstance(profile[key], str):
            raise ValueError(f"{key} must be text.")
    for key in ("reference_duration_seconds", "caption_font_size_pct", "cut_speed_cuts_per_minute", "zoom_intensity_pct", "contrast", "saturation", "brightness", "music_volume_db"):
        try:
            profile[key] = float(profile[key])
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be numeric.") from None
    for key in ("caption_font_size_pct", "zoom_intensity_pct", "contrast", "saturation", "brightness"):
        if not 0 <= profile[key] <= 100:
            raise ValueError(f"{key} must be between 0 and 100.")
    if not profile["style_name"].strip():
        raise ValueError("Style name cannot be empty.")
    if profile["caption_position"] not in {"top", "center", "bottom"}:
        raise ValueError("Caption position must be top, center, or bottom.")
    return profile
