import asyncio
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from core.ingest import FramePacket, RTSPIngestor

logger = logging.getLogger(__name__)


@dataclass
class DetectionEvent:
    event_type: str
    confidence: float
    timestamp: float
    frame: np.ndarray
    bbox: tuple[int, int, int, int]


class MLWorker:
    def __init__(
        self,
        ingestor: RTSPIngestor,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.7,
        target_classes: Optional[list[str]] = None,
        evidence_dir: str = "storage/evidence",
        pre_event_seconds: float = 5.0,
        post_event_seconds: float = 5.0,
        sample_fps: float = 2.0,
    ):
        self.ingestor = ingestor
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes or ["person"]
        self.evidence_dir = Path(evidence_dir)
        self.pre_event_seconds = pre_event_seconds
        self.post_event_seconds = post_event_seconds
        self.sample_fps = sample_fps

        self.evidence_dir.mkdir(parents=True, exist_ok=True)

        self._model = YOLO(model_path)
        self._model.to("cuda" if self._is_cuda_available() else "cpu")
        logger.info(f"YOLO model loaded on {'GPU' if self._is_cuda_available() else 'CPU'}")

        self._frame_buffer: deque[FramePacket] = deque(maxlen=int(sample_fps * (pre_event_seconds + post_event_seconds) + 10))
        self._running = False
        self._event_triggered = False
        self._post_event_frames_needed = 0
        self._current_event_frames: list[FramePacket] = []
        self._event_start_time: Optional[float] = None

    def _is_cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    async def start(self) -> None:
        self._running = True
        async for frame_packet in self._sample_frames():
            if not self._running:
                break
            await self._process_frame(frame_packet)

    async def stop(self) -> None:
        self._running = False

    async def _sample_frames(self):
        frame_interval = 1.0 / self.sample_fps
        last_sample_time = 0.0

        while self._running:
            current_time = time.time()
            if current_time - last_sample_time >= frame_interval:
                if self.ingestor._frame_buffer:
                    frame_packet = self.ingestor._frame_buffer[-1]
                    last_sample_time = current_time
                    yield frame_packet
            await asyncio.sleep(0.01)

    async def _process_frame(self, frame_packet: FramePacket) -> None:
        self._frame_buffer.append(frame_packet)

        if self._event_triggered:
            self._current_event_frames.append(frame_packet)
            self._post_event_frames_needed -= 1
            if self._post_event_frames_needed <= 0:
                await self._save_evidence_clip()
                self._reset_event_state()
            return

        resized_frame = cv2.resize(frame_packet.frame, (640, 640))
        results = self._model(resized_frame, verbose=False)[0]

        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = self._model.names[class_id]
            confidence = float(box.conf[0])

            if class_name in self.target_classes and confidence >= self.confidence_threshold:
                logger.info(f"Detection: {class_name} ({confidence:.2f}) at {frame_packet.timestamp}")
                await self._trigger_event(frame_packet, class_name, confidence, box.xyxy[0].tolist())
                break

    async def _trigger_event(
        self,
        frame_packet: FramePacket,
        event_type: str,
        confidence: float,
        bbox: list[float],
    ) -> None:
        self._event_triggered = True
        self._event_start_time = frame_packet.timestamp - self.pre_event_seconds
        self._post_event_frames_needed = int(self.sample_fps * self.post_event_seconds)

        pre_event_frames = self.ingestor.get_frames_since(self._event_start_time)
        self._current_event_frames = pre_event_frames + [frame_packet]

        logger.info(f"Event triggered: {event_type}, collecting {len(pre_event_frames)} pre-event frames + {self._post_event_frames_needed} post-event frames")

    async def _save_evidence_clip(self) -> None:
        if not self._current_event_frames:
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"event_{self._current_event_frames[0].frame_id}_{timestamp}.mp4"
        filepath = self.evidence_dir / filename

        height, width = self._current_event_frames[0].frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(filepath), fourcc, self.sample_fps, (width, height))

        for fp in self._current_event_frames:
            out.write(fp.frame)

        out.release()
        logger.info(f"Evidence clip saved: {filepath} ({len(self._current_event_frames)} frames)")

    def _reset_event_state(self) -> None:
        self._event_triggered = False
        self._current_event_frames = []
        self._event_start_time = None
        self._post_event_frames_needed = 0