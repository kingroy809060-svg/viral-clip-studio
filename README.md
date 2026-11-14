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

## Run

```bash
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:8000>. Health check: <http://127.0.0.1:8000/health>.

FFmpeg/ffprobe must be installed separately and available on `PATH`. Tesseract is optional for OCR. The app still starts if FFmpeg is missing, but video analysis reports that it is required.

## API

- `GET /health`
- `POST /upload` — accepts `long_video`, optional `reference_video`
- `POST /styles/analyze` — multipart fields `reference_video` and optional `style_name`
- `POST /styles/save` — JSON profile with exactly 15 fields
- `GET /styles` — list approved profiles
- `GET /styles/{slug}` — read one approved profile

## Local data

- Uploaded files: `uploads/`
- Approved styles: `styles/`
- SQLite metadata: `viral_clip_studio.sqlite3`

## Not implemented yet

- Phase 2: local Whisper transcript and candidate clip scoring
- Phase 3: 9:16 clip rendering and editable text exports
- Phase 4: end-to-end workflow, history, and test-clip approval
- Phase 5: download/share conveniences
- Phase 6: GitHub publication and live hosting

No paid API is used by Phases 0–1.
