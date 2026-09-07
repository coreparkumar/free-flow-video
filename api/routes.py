import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

HLS_DIR = Path("storage/hls")
EVIDENCE_DIR = Path("storage/evidence")

HLS_DIR.mkdir(parents=True, exist_ok=True)
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Free Flow Video API", version="1.0.0")

app.mount("/hls", StaticFiles(directory=str(HLS_DIR)), name="hls")
app.mount("/evidence", StaticFiles(directory=str(EVIDENCE_DIR)), name="evidence")


@app.get("/")
async def root():
    return {
        "service": "free-flow-video",
        "status": "running",
        "endpoints": {
            "hls_playlist": "/hls/stream.m3u8",
            "hls_segments": "/hls/segment_*.ts",
            "evidence_clips": "/evidence/*.mp4",
            "health": "/health",
        },
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "hls_dir_exists": HLS_DIR.exists(), "evidence_dir_exists": EVIDENCE_DIR.exists()}


@app.get("/hls/stream.m3u8")
async def get_hls_playlist():
    playlist_path = HLS_DIR / "stream.m3u8"
    if not playlist_path.exists():
        raise HTTPException(status_code=404, detail="HLS playlist not yet generated")
    return FileResponse(playlist_path, media_type="application/vnd.apple.mpegurl")


@app.get("/evidence/list")
async def list_evidence_clips():
    clips = []
    for f in sorted(EVIDENCE_DIR.glob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True):
        stat = f.stat()
        clips.append(
            {
                "filename": f.name,
                "path": f"/evidence/{f.name}",
                "size_bytes": stat.st_size,
                "created": stat.st_mtime,
            }
        )
    return {"clips": clips}


@app.get("/evidence/{filename}")
async def get_evidence_clip(filename: str):
    filepath = EVIDENCE_DIR / filename
    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(status_code=404, detail="Evidence clip not found")
    return FileResponse(filepath, media_type="video/mp4")


def create_app() -> FastAPI:
    return app