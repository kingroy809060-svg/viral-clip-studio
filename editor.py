"""Phase 3 vertical clip renderer powered by local FFmpeg and OpenCV."""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any


def _run(command: list[str], *, timeout: int = 900) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("FFmpeg timed out while rendering this clip.") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout)[-3000:]
        raise RuntimeError(f"FFmpeg failed to render this clip: {detail}")
    return result


def _probe(source: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe was not found on PATH. Install FFmpeg before rendering.")
    result = _run([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(source)], timeout=45)
    try:
        data = json.loads(result.stdout)
        video = next(stream for stream in data["streams"] if stream.get("codec_type") == "video")
        return {
            "width": int(video["width"]), "height": int(video["height"]),
            "duration": float(data.get("format", {}).get("duration", 0)),
            "has_audio": any(stream.get("codec_type") == "audio" for stream in data.get("streams", [])),
        }
    except (KeyError, StopIteration, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("Could not read the source video's audio/video streams.") from exc


def _face_center(source: Path, width: int, height: int, start: float, end: float) -> tuple[float, float]:
    """Estimate one stable crop centre from sampled frontal faces; fall back to frame centre."""
    try:
        import cv2

        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            return width / 2, height / 2
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        centers: list[tuple[float, float, int]] = []
        for moment in [start + (end - start) * fraction for fraction in (0.15, 0.4, 0.65, 0.9)]:
            capture.set(cv2.CAP_PROP_POS_MSEC, moment * 1000)
            ok, frame = capture.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
            if len(faces):
                x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
                centers.append((x + w / 2, y + h * 0.46, w * h))
        capture.release()
        if centers:
            weight = sum(item[2] for item in centers)
            return (sum(item[0] * item[2] for item in centers) / weight,
                    sum(item[1] * item[2] for item in centers) / weight)
    except Exception:
        pass
    return width / 2, height / 2


def _silence_intervals(source: Path, start: float, duration: float) -> list[tuple[float, float]]:
    """Find pauses long enough to remove, returning offsets relative to the clip."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg was not found on PATH. Install FFmpeg before rendering.")
    result = _run([
        ffmpeg, "-hide_banner", "-nostats", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
        "-i", str(source), "-vn", "-af", "silencedetect=noise=-35dB:d=0.45", "-f", "null", "-",
    ], timeout=120)
    log = result.stderr
    starts: list[float] = []
    intervals: list[tuple[float, float]] = []
    for line in log.splitlines():
        match_start = re.search(r"silence_start:\s*([\d.]+)", line)
        match_end = re.search(r"silence_end:\s*([\d.]+)", line)
        if match_start:
            starts.append(float(match_start.group(1)))
        if match_end and starts:
            silence_start = starts.pop(0)
            silence_end = min(duration, float(match_end.group(1)))
            # Leave short edit handles around each pause; discard only the central silent portion.
            left, right = silence_start + 0.08, silence_end - 0.08
            if right - left >= 0.25 and left > 0.04 and right < duration - 0.04:
                intervals.append((left, right))
    return intervals


def _keep_ranges(duration: float, cuts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    cursor = 0.0
    for left, right in cuts:
        if left > cursor:
            ranges.append((cursor, left))
        cursor = max(cursor, right)
    if duration > cursor:
        ranges.append((cursor, duration))
    return [(left, right) for left, right in ranges if right - left > 0.06]


def _remap_time(moment: float, cuts: list[tuple[float, float]]) -> float | None:
    removed = 0.0
    for left, right in cuts:
        if left <= moment < right:
            return None
        if right <= moment:
            removed += right - left
    return max(0.0, moment - removed)


def _ass_time(seconds: float) -> str:
    centiseconds = max(0, int(round(seconds * 100)))
    hours, centiseconds = divmod(centiseconds, 360000)
    minutes, centiseconds = divmod(centiseconds, 6000)
    whole_seconds, centiseconds = divmod(centiseconds, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"


def _srt_time(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def _ass_color(hex_color: str) -> str:
    match = re.fullmatch(r"#?([0-9a-fA-F]{6})", str(hex_color).strip())
    if not match:
        return "&H00E1E1F4"
    raw = match.group(1)
    return f"&H00{raw[4:6]}{raw[2:4]}{raw[0:2]}"


def _ass_escape(value: str) -> str:
    return value.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}").replace("\n", r"\N")


def _word_text(words: list[dict[str, Any]]) -> str:
    raw = [str(item.get("word", "")) for item in words]
    if any(item[:1].isspace() for item in raw[1:]):
        return "".join(raw).strip()
    return " ".join(item.strip() for item in raw if item.strip())


def _write_subtitles(
    ass_path: Path, srt_path: Path, words: list[dict[str, Any]], clip_start: float, clip_end: float,
    cuts: list[tuple[float, float]], profile: dict[str, Any], hook: str, output_duration: float,
) -> None:
    position = str(profile.get("caption_position", "bottom")).lower()
    alignment, margin_v = (8, 270) if position == "top" else ((5, 0) if position == "center" else (2, 150))
    font_size = max(24, min(96, round(1920 * float(profile.get("caption_font_size_pct", 3.6)) / 100)))
    font_name = str(profile.get("caption_font_family", "Arial"))
    if "estimated" in font_name.lower() or len(font_name) > 40 or not re.fullmatch(r"[\w -]+", font_name):
        font_name = "Arial"
    width_words: list[tuple[float, float, str]] = []
    for item in words:
        try:
            original_start, original_end = float(item["start"]), float(item["end"])
            if original_end <= clip_start or original_start >= clip_end:
                continue
            left, right = max(original_start, clip_start) - clip_start, min(original_end, clip_end) - clip_start
            mapped_left, mapped_right = _remap_time(left, cuts), _remap_time(right, cuts)
            if mapped_left is None or mapped_right is None or mapped_right <= mapped_left:
                continue
            token = str(item.get("word", "")).strip()
            if token:
                width_words.append((mapped_left, min(output_duration, mapped_right), token))
        except (KeyError, TypeError, ValueError):
            continue

    rows = ["[Script Info]", "ScriptType: v4.00+", "PlayResX: 1080", "PlayResY: 1920", "WrapStyle: 2", "ScaledBorderAndShadow: yes", "",
            "[V4+ Styles]", "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: Words,{font_name},{font_size},{_ass_color(str(profile.get('caption_font_color', '#F4E1C8')))},&H000000FF,&HAA000000,&H66000000,1,0,0,0,100,100,0,0,3,2,0,{alignment},60,60,{margin_v},1",
            f"Style: Hook,Arial,{min(96, max(40, font_size + 10))},&H00FFFFFF,&H000000FF,&H00000000,&H66000000,1,0,0,0,100,100,0,0,3,3,0,8,70,70,300,1", "",
            "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    events: list[tuple[float, str]] = []
    if hook.strip():
        events.append((0.0, f"Dialogue: 1,0:00:00.00,{_ass_time(min(2.0, output_duration))},Hook,,0,0,0,,{_ass_escape(hook.strip())}"))
    srt_entries: list[str] = []
    for index, (left, right, token) in enumerate(width_words, 1):
        finish = max(left + 0.08, right)
        escaped = _ass_escape(token)
        events.append((left, f"Dialogue: 0,{_ass_time(left)},{_ass_time(finish)},Words,,0,0,0,,{escaped}"))
        srt_entries.append(f"{index}\n{_srt_time(left)} --> {_srt_time(finish)}\n{token}\n")
    rows.extend(row for _, row in sorted(events, key=lambda item: item[0]))
    ass_path.write_text("\n".join(rows) + "\n", encoding="utf-8-sig")
    srt_path.write_text("\n".join(srt_entries), encoding="utf-8")


def render_clip(
    source_path: str | Path,
    output_dir: str | Path,
    start_seconds: float,
    end_seconds: float,
    word_timestamps: list[dict[str, Any]],
    style_profile: dict[str, Any],
    hook_line: str = "",
    music_path: str | Path | None = None,
    remove_silence: bool = True,
) -> dict[str, Any]:
    """Render one approved moment and its editable companion assets."""
    source, out = Path(source_path).resolve(), Path(output_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError("Source video is missing; upload it again before rendering.")
    start, end = float(start_seconds), float(end_seconds)
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValueError("Clip start/end time is invalid.")
    media = _probe(source)
    if end > media["duration"] + 0.05:
        raise ValueError("Clip end time is beyond the source video duration.")
    duration = end - start
    if duration < 1 or duration > 120:
        raise ValueError("A render must be between 1 and 120 seconds long.")
    music = Path(music_path).resolve() if music_path else None
    if music and not music.is_file():
        raise FileNotFoundError("Optional music file does not exist.")

    out.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex[:12]
    stem = f"viral_clip_{job_id}"
    mp4_path, srt_path = out / f"{stem}.mp4", out / f"{stem}.srt"
    ass_path = out / f"{stem}.ass"
    transcript_path = out / f"{stem}_transcript.txt"
    title_path, caption_path, hashtags_path = out / f"{stem}_title.txt", out / f"{stem}_caption.txt", out / f"{stem}_hashtags.txt"

    cuts = _silence_intervals(source, start, duration) if remove_silence and media["has_audio"] else []
    ranges = _keep_ranges(duration, cuts)
    if not ranges:
        ranges = [(0.0, duration)]
        cuts = []
    output_duration = sum(right - left for left, right in ranges)
    center_x, center_y = _face_center(source, media["width"], media["height"], start, end)
    source_w, source_h = media["width"], media["height"]
    zoom_pct = float(style_profile.get("zoom_intensity_pct", 0) or 0)
    zoom = 1 + (min(15.0, max(10.0, zoom_pct if zoom_pct > 0 else 12.0)) / 100.0)
    if source_w / source_h > 9 / 16:
        crop_h = min(source_h, source_w * 16 / 9) / zoom
        crop_w = crop_h * 9 / 16
    else:
        crop_w = min(source_w, source_h * 9 / 16) / zoom
        crop_h = crop_w * 16 / 9
    crop_x = max(0, min(source_w - crop_w, center_x - crop_w / 2))
    crop_y = max(0, min(source_h - crop_h, center_y - crop_h * 0.40))
    contrast = max(0.5, min(1.5, 0.75 + float(style_profile.get("contrast", 50)) / 200))
    saturation = max(0.0, min(2.0, float(style_profile.get("saturation", 50)) / 50))
    brightness = max(-0.1, min(0.1, (float(style_profile.get("brightness", 50)) - 50) / 500))
    _write_subtitles(ass_path, srt_path, word_timestamps, start, end, cuts, style_profile, hook_line, output_duration)

    def filt_path(path: Path) -> str:
        # FFmpeg filter syntax: escape backslashes, colons, quotes and commas in paths.
        return str(path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'").replace(",", "\\,")

    ass_filter = f"ass='{filt_path(ass_path)}'"
    chain: list[str] = []
    for index, (left, right) in enumerate(ranges):
        chain.append(
            f"[0:v]trim=start={start + left:.4f}:end={start + right:.4f},setpts=PTS-STARTPTS,"
            f"crop={crop_w:.3f}:{crop_h:.3f}:{crop_x:.3f}:{crop_y:.3f},scale=1080:1920:flags=lanczos,"
            f"eq=contrast={contrast:.4f}:brightness={brightness:.4f}:saturation={saturation:.4f},fps=30,setsar=1[{index}v]"
        )
        if media["has_audio"]:
            chain.append(f"[0:a]atrim=start={start + left:.4f}:end={start + right:.4f},asetpts=PTS-STARTPTS,volume=-3dB[{index}a]")
    if media["has_audio"]:
        concat_inputs = "".join(f"[{index}v][{index}a]" for index in range(len(ranges)))
        chain.append(f"{concat_inputs}concat=n={len(ranges)}:v=1:a=1[basev][basea]")
        chain.append(f"[basev]{ass_filter}[captioned]")
        if music:
            chain.append("[1:a]atrim=duration={:.4f},asetpts=PTS-STARTPTS,volume=-20dB[music]".format(output_duration))
            chain.append("[basea][music]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]")
        else:
            chain.append("[basea]anull[aout]")
    else:
        concat_inputs = "".join(f"[{index}v]" for index in range(len(ranges)))
        chain.append(f"{concat_inputs}concat=n={len(ranges)}:v=1:a=0[basev]")
        chain.append(f"[basev]{ass_filter}[captioned]")
        if music:
            chain.append("[1:a]atrim=duration={:.4f},asetpts=PTS-STARTPTS,volume=-20dB[aout]".format(output_duration))
    graph = out / f"{stem}_filter.txt"
    graph.write_text(";\n".join(chain) + "\n", encoding="utf-8")
    command = ["ffmpeg", "-hide_banner", "-y", "-i", str(source)]
    if music:
        command += ["-stream_loop", "-1", "-i", str(music)]
    command += ["-filter_complex_script", str(graph), "-map", "[captioned]"]
    if media["has_audio"] or music:
        command += ["-map", "[aout]"]
    command += ["-t", f"{output_duration:.4f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-r", "30"]
    if media["has_audio"] or music:
        command += ["-c:a", "aac", "-b:a", "192k"]
    command += ["-movflags", "+faststart", str(mp4_path)]
    try:
        _run(command, timeout=max(300, int(output_duration * 30)))
    finally:
        graph.unlink(missing_ok=True)
        ass_path.unlink(missing_ok=True)

    clip_words = [item for item in word_timestamps
                  if start <= float(item.get("start", -1)) < end
                  and _remap_time(float(item.get("start", -1)) - start, cuts) is not None]
    transcript = _word_text(clip_words)
    title = hook_line.strip() or (transcript.split(".", 1)[0][:90] if transcript else "New Reel")
    caption = transcript
    hashtags = "#Reels #InstagramReels #FacebookReels #ViralClip"
    transcript_path.write_text(transcript + "\n", encoding="utf-8")
    title_path.write_text(title + "\n", encoding="utf-8")
    caption_path.write_text(caption + "\n", encoding="utf-8")
    hashtags_path.write_text(hashtags + "\n", encoding="utf-8")
    return {
        "id": job_id, "mp4": mp4_path.name, "srt": srt_path.name,
        "transcript": transcript_path.name, "title": title_path.name,
        "caption": caption_path.name, "hashtags": hashtags_path.name,
        "duration_seconds": round(output_duration, 3), "width": 1080, "height": 1920,
        "fps": 30, "video_codec": "H.264", "audio_codec": "AAC" if media["has_audio"] or music else None,
        "removed_silence_seconds": round(sum(right - left for left, right in cuts), 3),
        "face_tracking": "OpenCV sampled frontal face" if (center_x, center_y) != (source_w / 2, source_h / 2) else "center crop fallback",
    }
