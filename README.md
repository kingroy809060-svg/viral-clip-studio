# Viral Clip Studio

Local-first video clipping app, built one verified phase at a time.

## Implemented now

### Phase 0 — setup

- FastAPI app launched with `python app.py`
- Startup FFmpeg check, also available at `/health`
- MP4/MOV upload intake, stored locally with SQLite metadata

### Phase 1 — reference style profile

- Upload a reference MP4/MOV and analyze duration, sampled-frame color metrics, scene-cut rate, OCR text, caption placement/size/color estimates, and mixed-audio level
- Review and edit all 15 profile fields in the UI
- Approve and save reusable JSON profiles in `styles/<style-name>.json`
- Default profile name: `Global Baatien Style`

The style analyzer needs the actual reel video file. A social profile URL by itself is not downloaded in Phase 1. A rasterized video cannot reveal the source font file, and a mixed audio track cannot reliably separate music from speech; those fields are estimates for review. OCR requires the optional system `tesseract` command. Zoom estimation tracks sampled image features with OpenCV, with a coarse pixel-scale fallback if OpenCV is absent; treat the result as an estimate.

### Phase 2 — local transcript and clip candidates

- Whisper `base` runs locally and returns a full transcript plus word-level timestamps; language detection is automatic for Hindi, Hinglish, English, and other Whisper-supported languages.
- 30–60 second transcript windows are scored on hook (30), emotion (30), complete idea (20), and loop potential (20). The UI shows the top non-overlapping clips, their exact Whisper text, score breakdown, and an editable full transcript.
- If `GEMINI_API_KEY` is set in `.env`, Gemini scores candidate batches. If the key is absent, the request fails, or the quota is reached, rule-based scores are used and the app continues.
- Phase 2 accepts source videos between 10 minutes and 3 hours. On first use, Whisper may need to download its model weights; transcription afterward runs locally.

### Phase 3 — single approved clip render

- Render one candidate at a time after choosing a saved Style Profile. The editor creates a 1080×1920, 30 fps H.264/CRF 18/yuv420p MP4; source audio is set to -3 dB.
- OpenCV samples frontal faces to choose a stable 9:16 crop center. When no face is detected, it uses a centered crop. The style zoom estimate is applied as a 10–15% punch (12% when the profile has no usable estimate), and profile contrast/saturation/brightness drive FFmpeg color grading.
- FFmpeg removes detected internal pauses longer than roughly half a second while preserving short handles. Word timestamps are shifted along with the removed pauses, then burned word-by-word into the video. The first two seconds can show the selected hook.
- Each render provides MP4, SRT, verbatim transcript, title, caption, and hashtags as separate downloads. No music is added by default; this avoids bundling unlicensed music. Source audio is a mixed track, so this phase cannot independently lower an embedded music bed while preserving speech.
- Phase 4 builds the enforced test-clip approval and batch workflow on top of this renderer.

### Phase 4 — end-to-end test-first batches

- The batch screen accepts one to four 10-minute–3-hour videos, an approved Style Profile (Global Baatien is selected by default), and an auto-detected or manually selected video type.
- It ranks candidates per source, then locks batch export behind a rendered test clip and a separate user approval. Only after approval can the remaining candidates render; status and progress are saved in SQLite.
- Saved Style Library and My Clips history are visible in the UI. Test and batch exports each keep their MP4, SRT, transcript, title, caption, and hashtags.
- Type detection uses transcript keywords and a few sampled frames. Podcast/Talking Head use OpenCV face-centering, Talking Head gets a stronger zoom, Tutorial moves captions to the top, and Motivational uses a gradual zoom. Vlog mode preserves scene changes and available source footage; it cannot invent or fetch additional B-roll.

### Phase 5 — download and share

- Download every clip's MP4, SRT, verbatim transcript, title, caption, and hashtags separately, or download the six deliverables together as a ZIP.
- Download a batch ZIP containing the reviewed test clip and every successfully generated clip. The bundle refreshes as the batch completes.
- Copy Caption + Hashtags copies the exact exported caption and hashtag text to the clipboard. Posting to Instagram/Facebook remains a manual step; this app does not request social account access or publish on your behalf.

## Run

```bash
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:8000>. Health check: <http://127.0.0.1:8000/health>.

For a public deployment, set `VCS_USERNAME` and a strong `VCS_PASSWORD` in the host's secret environment settings. All app pages and APIs require HTTP Basic Auth; `/health` remains public for deployment health checks. The app returns `503` for protected routes until a password is configured.

FFmpeg/ffprobe must be installed separately and available on `PATH`. Tesseract is optional for OCR. The app still starts if FFmpeg is missing, but video analysis reports that it is required.

## API

- `GET /health`
- `POST /upload` — accepts `long_video`, optional `reference_video`
- `POST /styles/analyze` — multipart fields `reference_video` and optional `style_name`
- `POST /styles/save` — JSON profile with exactly 15 fields
- `GET /styles` — list approved profiles
- `GET /styles/{slug}` — read one approved profile
- `POST /clips/find` — accepts `long_video` MP4/MOV and returns local Whisper transcript, word timestamps, and up to 8 scored moments
- `POST /music/upload` — accepts a user-provided licensed music bed (`MP3`, `WAV`, `M4A`, `AAC`, `OGG`, `FLAC`); no music is bundled
- `POST /clips/render` — JSON: `source_id`, `start_seconds`, `end_seconds`, `word_timestamps`, `style_slug`, and `hook_line`; renders one clip using an approved profile
- `GET /outputs/{render_id}/{filename}` — downloads one of the six files produced by that render
- `GET /downloads/render/{render_id}.zip` — downloads all six files for one rendered clip
- `GET /downloads/batch/{batch_id}.zip` — downloads the test clip and every completed batch clip with their companion files
- `POST /batches` — multipart `background` uploads (one to four videos), `style_slug`, `video_type` (`auto` or one of five types), and optional `music_upload_id`
- `GET /batches/{batch_id}` — batch analysis, test, approval, or generation status
- `POST /batches/{batch_id}/test` — multipart `candidate_ref` such as `0:0`; renders the single review clip
- `POST /batches/{batch_id}/approve` — approves the current test clip
- `POST /batches/{batch_id}/generate` — unlocks remaining candidates only after test approval
- `GET /batches` and `GET /clips/history` — saved batch and render history

## Local data

- Uploaded files: `uploads/`
- Rendered clips and companion files: `outputs/`
- Approved styles: `styles/`
- SQLite metadata: `viral_clip_studio.sqlite3`

## Not implemented yet

- Phase 6: live hosting

No paid API is used. Gemini is optional and falls back to local rule-based scoring.
