"""Transcript-based 30–60 second clip candidates with Gemini/rule scoring."""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

DEFAULT_MODEL = "gemini-3.8-flash"
WEIGHTS = {"hook": 30, "emotion": 30, "complete_idea": 20, "loop": 20}
HOOK_WORDS = {
    "why", "how", "what", "imagine", "secret", "truth", "mistake", "never", "shocking",
    "wait", "देखिए", "सोचिए", "क्यों", "कैसे", "क्या", "सच", "राज", "गलती", "कभी",
    "लेकिन", "फिर", "सुनो", "रुको", "असली",
}
EMOTION_WORDS = {
    "love", "fear", "angry", "shocked", "surprise", "cry", "happy", "sad", "danger",
    "secret", "impossible", "amazing", "heartbreaking", "प्यार", "डर", "गुस्सा", "हैरान",
    "खुशी", "दुख", "दर्द", "खतरा", "चौंक", "असंभव", "कमाल", "भावुक", "रोने",
}
OPEN_ENDINGS = {
    "but", "because", "until", "then", "however", "और", "लेकिन", "क्योंकि", "फिर", "तभी",
    "wait", "लेकिन फिर", "और फिर",
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w\u0900-\u097f']+", text.lower(), flags=re.UNICODE)


def _join_word_tokens(words: list[dict[str, Any]]) -> str:
    raw = [str(item.get("word", "")) for item in words]
    if any(item[:1].isspace() for item in raw[1:]):
        return "".join(raw).strip()
    return " ".join(item.strip() for item in raw if item.strip())


def _hook_line(text: str, max_words: int = 12) -> str:
    first = re.split(r"(?<=[.!?।])\s+", text.strip(), maxsplit=1)[0]
    words = first.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return first


def _rule_scores(text: str) -> dict[str, int]:
    tokens = _tokens(text)
    if not tokens:
        return {"hook": 0, "emotion": 0, "complete_idea": 0, "loop": 0}
    lowered = text.strip().lower()
    hook = 7 + min(8, len(tokens) // 12)
    if tokens[0] in HOOK_WORDS:
        hook += 8
    if "?" in text or "？" in text:
        hook += 5
    if any(word in HOOK_WORDS for word in tokens[:8]):
        hook += 3

    emotional_hits = sum(token in EMOTION_WORDS for token in tokens)
    emotion = min(23, emotional_hits * 5)
    emotion += min(5, text.count("!") * 2)
    if any(token in {"मैं", "मेरा", "मेरी", "मुझे", "my", "i", "we", "हम"} for token in tokens):
        emotion += 2

    sentence_ends = len(re.findall(r"[.!?।॥](?:\s|$)", text))
    word_count = len(tokens)
    complete = 5 + min(7, word_count // 18) + min(8, sentence_ends * 4)
    if 45 <= word_count <= 140:
        complete += 3
    if word_count < 20:
        complete = min(complete, 8)

    loop = 5
    if text.rstrip().endswith(("?", "؟", "…", "...")):
        loop += 7
    ending = " ".join(tokens[-3:])
    if ending in OPEN_ENDINGS or any(ending.endswith(item) for item in OPEN_ENDINGS):
        loop += 5
    if tokens[0] in tokens[-min(8, len(tokens)):]:
        loop += 3
    if any(phrase in lowered for phrase in ("part two", "अगले भाग", "आगे देखिए", "to be continued")):
        loop += 5

    return {
        "hook": min(WEIGHTS["hook"], max(0, hook)),
        "emotion": min(WEIGHTS["emotion"], max(0, emotion)),
        "complete_idea": min(WEIGHTS["complete_idea"], max(0, complete)),
        "loop": min(WEIGHTS["loop"], max(0, loop)),
    }


def _gemini_scores_batch(candidates: list[dict[str, Any]], api_key: str, model: str) -> dict[int, dict[str, int]]:
    compact = [{"index": item["index"], "text": item["transcript"][:1800]} for item in candidates]
    prompt = (
        "Score each verbatim transcript segment as a short vertical-video candidate. Do not rewrite, "
        "translate, or add words. Give integer component scores: hook 0-30, emotion 0-30, "
        "complete_idea 0-20, loop 0-20. Reward a compelling opening, clear emotional stakes, a "
        "complete thought, and an ending that encourages replay. Return only JSON in this exact shape: "
        '{"scores":[{"index":0,"hook":0,"emotion":0,"complete_idea":0,"loop":0}]}. '
        "Evaluate every input index once. Inputs: " + json.dumps(compact, ensure_ascii=False)
    )
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1},
    }).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        response_data = json.loads(response.read().decode("utf-8"))
    text = response_data["candidates"][0]["content"]["parts"][0]["text"]
    parsed = json.loads(text)
    output: dict[int, dict[str, int]] = {}
    for item in parsed.get("scores", []):
        index = int(item["index"])
        parts = {key: max(0, min(WEIGHTS[key], int(item[key]))) for key in WEIGHTS}
        output[index] = parts
    return output


def _apply_gemini(candidates: list[dict[str, Any]]) -> bool:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return False
    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    any_success = False
    # Keep requests small. A failed/rate-limited batch quietly retains rule scores.
    for offset in range(0, len(candidates), 12):
        batch = candidates[offset:offset + 12]
        try:
            scores = _gemini_scores_batch(batch, api_key, model)
        except (OSError, ValueError, KeyError, TypeError, urllib.error.URLError, urllib.error.HTTPError):
            continue
        for candidate in batch:
            parts = scores.get(candidate["index"])
            if parts is not None:
                candidate["score_parts"] = parts
                candidate["scoring_source"] = "gemini"
                any_success = True
    return any_success


def _candidate_windows(duration: float, min_duration: float, max_duration: float) -> list[tuple[float, float]]:
    window = min(45.0, max_duration, duration)
    if window < min_duration:
        return []
    if duration <= window:
        return [(0.0, duration)]
    step = 15.0
    last_start = duration - window
    starts = [float(i) for i in range(0, int(last_start) + 1, int(step))]
    if not starts or last_start - starts[-1] > 1.0:
        starts.append(last_start)
    return [(start, min(duration, start + window)) for start in starts]


def find_clips(
    transcription: dict[str, Any],
    min_duration: float = 30.0,
    max_duration: float = 60.0,
    limit: int = 8,
) -> dict[str, Any]:
    """Build sliding transcript windows, score all, and return up to `limit` spaced picks."""
    duration = float(transcription.get("duration_seconds") or 0.0)
    words = transcription.get("words") or []
    if not words:
        raise ValueError("Whisper found no timestamped speech. Check that the video has audible speech.")
    windows = _candidate_windows(duration, min_duration, max_duration)
    if not windows:
        raise ValueError("The source video is shorter than 30 seconds; no valid clip window can be made.")

    candidates: list[dict[str, Any]] = []
    for start, end in windows:
        selected = [item for item in words if start <= float(item.get("start", -1)) < end]
        text = _join_word_tokens(selected)
        if len(_tokens(text)) < 8:
            continue
        score_parts = _rule_scores(text)
        candidates.append({
            "index": len(candidates),
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "duration_seconds": round(end - start, 3),
            "hook_line": _hook_line(text),
            "transcript": text,
            "score_parts": score_parts,
            "score": sum(score_parts.values()),
            "scoring_source": "rule-based",
            "word_timestamps": selected,
        })

    if not candidates:
        raise ValueError("No 30–60 second window contained enough speech to score.")
    gemini_used = _apply_gemini(candidates)
    for item in candidates:
        item["score"] = sum(item["score_parts"].values())
        item.pop("index", None)
    candidates.sort(key=lambda item: (-item["score"], item["start_seconds"]))

    # Avoid returning many nearly identical moments from adjacent overlapping windows.
    selected_clips: list[dict[str, Any]] = []
    for candidate in candidates:
        overlaps = any(
            min(candidate["end_seconds"], chosen["end_seconds"])
            - max(candidate["start_seconds"], chosen["start_seconds"]) > 0.25 * min(
                candidate["duration_seconds"], chosen["duration_seconds"]
            )
            for chosen in selected_clips
        )
        if not overlaps:
            selected_clips.append(candidate)
        if len(selected_clips) >= max(1, min(8, int(limit))):
            break

    return {
        "candidate_count": len(candidates),
        "scoring_mode": "gemini+rule-based fallback" if gemini_used else "rule-based",
        "clips": selected_clips,
    }
