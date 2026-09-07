# free-flow-video

A resilient, local-first video pipeline for ingesting unreliable RTSP streams, providing live HLS viewing, and generating evidence clips via local ML inference.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  RTSP Source │────▶│  Ingestion   │────▶│  Frame Buffer │
│  (Unreliable)│     │  (Resilient)  │     │  (Ring Buffer)│
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                 │
                    ┌────────────────────────────┼────────────────────────────┐
                    ▼                            ▼                            ▼
           ┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
           │   HLS Path      │           │   ML Path       │           │  Evidence Clip  │
           │ (Live Viewing)  │           │ (1-2 FPS Sample)│           │  (On Detection) │
           └────────┬────────┘           └────────┬────────┘           └────────┬────────┘
                    │                             │                            │
                    ▼                             ▼                            ▼
           ┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
           │ FFmpeg → .m3u8  │           │ YOLOv8n @ 640x640│           │  Last 5s + Next │
           │ + .ts segments  │           │ Person > 70%    │           │  5s → MP4       │
           └────────┬────────┘           └────────┬────────┘           └────────┬────────┘
                    │                             │                            │
                    ▼                             ▼                            ▼
           ┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
           │ FastAPI Static  │           │  Event Trigger  │           │  Local Storage  │
           │ File Serving    │           │  → Ring Buffer  │           │  /storage/evidence│
           └─────────────────┘           └─────────────────┘           └─────────────────┘
```

### Pipeline Split Strategy

The architecture deliberately separates concerns:

1. **HLS Path (Continuous)**: Full-frame-rate stream piped directly to FFmpeg for HLS chunking. Zero ML overhead. Served statically via FastAPI for minimal latency.

2. **ML Path (Sampled)**: Frames extracted at 1-2 FPS, downscaled to 640x640, fed to YOLOv8 nano on GPU. Ring buffer retains last ~10 seconds for evidence generation.

3. **Evidence Generation**: On detection (person > 70% confidence), the ring buffer provides 5 seconds pre-event + 5 seconds post-event frames, encoded to MP4.

## Resilience Features

- **Exponential Backoff**: RTSP reconnection with 1s → 2s → 4s → ... → 60s max delay
- **TCP Transport**: Forces RTSP over TCP for reliability over lossy Wi-Fi
- **Frame Ring Buffer**: Fixed-size `deque` prevents OOM on long uptimes
- **Async I/O**: `asyncio` for network ops; CPU-bound YOLO runs in executor
- **Graceful Degradation**: All stream reads have timeouts; errors logged not crashed

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU (RTX 4050 recommended) with CUDA drivers
- FFmpeg installed and in PATH
- RTSP source (IP Webcam app on Android, or `ffmpeg -re -i test.mp4 -f rtsp rtsp://localhost:8554/stream`)

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Environment variables:
- `RTSP_URL` - RTSP stream URL (default: `rtsp://localhost:8554/stream`)
- `HOST` - API bind host (default: `0.0.0.0`)
- `PORT` - API port (default: `8000`)

### Running

```bash
python main.py --rtsp-url "rtsp://192.168.1.100:8080/h264_ulaw.sdp"
```

### Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Service info |
| `GET /health` | Health check |
| `GET /hls/stream.m3u8` | HLS playlist (live) |
| `GET /hls/segment_*.ts` | HLS segments |
| `GET /evidence/list` | List evidence clips |
| `GET /evidence/{filename}` | Download evidence clip |

### Testing with Chaos

1. Start IP Webcam on Android (e.g., "IP Webcam" app → Start server)
2. Note the RTSP URL (typically `rtsp://<phone-ip>:8080/h264_ulaw.sdp`)
3. Run pipeline with that URL
4. Open `http://<server-ip>:8000/hls/stream.m3u8` in VLC or browser with hls.js
5. **Chaos test**: Toggle phone Wi-Fi on/off, move out of range, kill/restart IP Webcam app
6. Observe: Pipeline reconnects automatically; HLS continues; evidence clips generated on person detection

## Project Structure

```
free-flow-video/
├── api/
│   └── routes.py          # FastAPI endpoints for HLS & evidence
├── core/
│   ├── ingest.py          # RTSP connection, backoff, frame buffering
│   ├── ml_worker.py       # 1-2 FPS sampling + YOLO inference + evidence
│   └── hls_generator.py   # FFmpeg subprocess for HLS chunking
├── storage/
│   ├── evidence/          # Generated MP4 clips
│   └── hls/               # .m3u8 + .ts segments
├── main.py                # Entry point, wiring, lifecycle
├── requirements.txt
└── implementation_plan-v2.md
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi`, `uvicorn` | Async API server |
| `av` (PyAV) | RTSP demuxing, frame decoding |
| `ultralytics` | YOLOv8 nano inference |
| `opencv-python` | Frame resize, MP4 encoding |
| `numpy` | Array operations |

## Design Decisions

- **PyAV over OpenCV for ingest**: PyAV's `av.open` with `rtsp_transport=tcp` and timeout options provides better control over connection parameters than `cv2.VideoCapture`.
- **Separate FFmpeg process for HLS**: FFmpeg's native HLS muxer is battle-tested; avoids re-encoding in Python.
- **YOLOv8 nano (640x640)**: Runs ~30 FPS on RTX 4050; 1-2 FPS sampling leaves ample headroom.
- **Ring buffer via `deque(maxlen=N)`**: O(1) append/pop, fixed memory, automatic eviction.
- **No cloud deps**: Evidence stored locally; can add sync layer later if needed.

## License

MIT