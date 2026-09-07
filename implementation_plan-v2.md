# Implementation Plan: free-flow-video

## Overview
This document outlines the Minimum Viable Product (MVP) build for the Antrino Labs Video Engineer opportunity. The `free-flow-video` project is a resilient, local-first pipeline that ingests an unreliable RTSP stream, recovers from failures, splits the pipeline for live viewing (HLS) and machine learning inference, and auto-generates evidence clips.

## Tech Stack Required

### Hardware
* **Edge Processing Unit:** AMD Ryzen 7 7445HS processor, 16 GB RAM.
* **AI Acceleration:** NVIDIA GeForce RTX 4050 Laptop GPU (essential for running local vision models at scale).
* **Camera Source:** Android mobile device running an "IP Webcam" application to simulate network degradation.

### Software & Frameworks
* **Core Language:** Python 3.10+
* **API & Routing:** FastAPI (asyncio support for network bound tasks)
* **Ingestion & Video Processing:** `PyAV` (pythonic bindings for FFmpeg) or `OpenCV` alongside native `FFmpeg` subprocesses.
* **AI/ML:** `Ultralytics` (YOLOv8 nano/small) for fast, local object detection.
* **Streaming Protocol:** RTSP (input), HLS (output).
* **Version Control:** GitHub or Codeberg to manage source code and automate the deployment pipeline.

## Repository Structure
```text
free-flow-video/
├── api/
│   └── routes.py         # FastAPI endpoints for HLS delivery
├── core/
│   ├── ingest.py         # RTSP connection, backoff, and healing logic
│   └── ml_worker.py      # Frame sampling and YOLO inference utilizing the RTX 4050
├── storage/
│   ├── evidence/         # Output directory for generated .mp4 clips
│   └── hls/              # Temporary .m3u8 and .ts segments
├── .gitignore
├── requirements.txt
├── implementation_plan.md
└── README.md
```

## Essential Project Files
*   **`core/ingest.py`:** Implement the robust connection loop here, handling dropped packets, socket timeouts, and simulated camera reboots gracefully.
*   **`core/ml_worker.py`:** Isolate the vision logic. Ensure the system explicitly samples at 1-2 FPS to keep compute costs low before passing frames to the local GPU.
*   **`requirements.txt`:** Keep dependencies strictly limited to the essentials required for the pipeline, such as `fastapi`, `uvicorn`, `ultralytics`, and `av` (PyAV).

## Execution Plan

### Step 1: Simulate the Unreliable Edge
* **Action:** Install an IP Webcam app on the Android device to broadcast an RTSP stream over the local network. 
* **Validation:** Physically toggle the device's Wi-Fi connection to simulate dropped packets, camera reboots, and network hiccups. This acts as the chaos-testing source.

### Step 2: Build the Resilient Ingestion Loop
* **Action:** Create an asynchronous Python worker using FastAPI. 
* **Logic:** Implement a robust connection loop wrapping the RTSP ingest. Must include a `try/except` block to catch frame drops, `cv2.error`, or socket timeouts.
* **Resilience:** Apply an exponential backoff algorithm to reconnect automatically without restarting the service or causing memory leaks.

### Step 3: Implement the Pipeline Split (Live vs. ML)
* **Action:** Fork the ingested stream into two distinct paths:
  1. **Live HLS Path:** Pipe the continuous stream into FFmpeg to chunk into an HLS playlist (`.m3u8` / `.ts` segments) served statically via FastAPI for low-latency web viewing.
  2. **ML Sampling Path:** Extract frames at a controlled rate of 1-2 FPS. Downscale these frames to a lower resolution (e.g., 640x640) and hold them in a ring buffer to minimize memory overhead.

### Step 4: Integrate Local Vision Intelligence
* **Action:** Pass the 1-2 FPS sampled frames into the YOLOv8 model utilizing the RTX 4050 backend.
* **Logic:** Establish a confidence threshold (e.g., >70% for a "person"). When the condition is met, trigger a state change indicating an "Event".

### Step 5: Generate the Evidence Clip (Archiving)
* **Action:** Upon an "Event" trigger, extract the last 5 seconds of frames from the ring buffer and capture the subsequent 5 seconds of incoming frames.
* **Packaging:** Compile these frames into an MP4 file, tagged with a timestamp and event type (e.g., `event_person_timestamp.mp4`), demonstrating the ability to generate trustworthy, queryable records.

## High-Impact README Blueprint
The README is where architectural decisions are explicitly sold to the engineering team.
*   **The Problem:** Clearly define the unreliability of physical edge environments and the inherent instability of raw RTSP streams.
*   **The Architecture:** Detail the strategic pipeline split—separating the continuous, low-latency HLS conversion from the sampled, lower-resolution ML inference path.
*   **Chaos Testing:** Document the exact methodology used to simulate network degradation (e.g., actively toggling the Android camera source) and explain how the Python worker successfully recovered the stream without crashing.

---

## Rules of Engagement for AI Coding Tools
*When utilizing AI assistants (like OpenCode or other multi-agent frameworks) for vibe coding this POC, strictly enforce the following boundaries to ensure the output aligns with the architecture.*

### What to Do (Enforced Guidelines)
1. **Strictly Local-First:** All model inferences must run locally on the provided GPU hardware. No external API calls to OpenAI/Anthropic/Google for vision processing during this POC.
2. **Graceful Degradation:** All network calls and stream reads must have timeout parameters and explicit exception handling. Ensure the system logs errors rather than crashing.
3. **Resource Management:** Explicitly manage memory. If frames are being buffered, mandate the use of fixed-size ring buffers (e.g., `collections.deque(maxlen=N)`) to prevent out-of-memory (OOM) errors over long uptimes.
4. **Asynchronous I/O:** Enforce the use of `asyncio` for network operations and separate CPU-bound tasks (like YOLO inference) into different threads or processes to prevent blocking the ingestion loop.

### What NOT to Do (Hard Boundaries)
1. **NO "Happy Path" Assumptions:** Do not accept code that assumes the RTSP stream is always available. Reject basic `while True: ret, frame = cap.read()` loops unless wrapped in comprehensive failure recovery.
2. **NO Monolithic Architecture:** Do not dump all logic (ingestion, API, inference) into a single file. Separate concerns into discrete modules (`ingest.py`, `ml_worker.py`, `api.py`).
3. **NO Frame Bloat:** Do not feed every single 30 FPS frame into the ML model. The AI tool must respect the 1-2 FPS sampling constraint to keep compute costs low.
4. **NO Heavy ML Frameworks:** Do not introduce large, complex frameworks (like TensorFlow or full PyTorch ecosystems) if a lightweight library (like Ultralytics) achieves the goal.
5. **NO Cloud Storage Dependencies:** Do not write code that assumes S3 or Azure blob storage for the evidence clips. Clips must be saved locally to the file system first.
