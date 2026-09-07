import asyncio
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, Optional

import av
import cv2
import numpy as np
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).resolve().parent.parent / "config.env")


@dataclass
class FramePacket:
    frame: np.ndarray
    timestamp: float
    frame_id: int


class RTSPIngestor:
    def __init__(
        self,
        rtsp_url: Optional[str] = None,
        buffer_size: int = 150,
        reconnect_base_delay: float = 1.0,
        max_reconnect_delay: float = 60.0,
        connection_timeout: float = 10.0,
    ):
        self.rtsp_url = rtsp_url or os.getenv("rtsp_url", "rtsp://localhost:8554/stream")
        self.buffer_size = buffer_size
        self.reconnect_base_delay = reconnect_base_delay
        self.max_reconnect_delay = max_reconnect_delay
        self.connection_timeout = connection_timeout

        self._frame_buffer: deque[FramePacket] = deque(maxlen=buffer_size)
        self._frame_id = 0
        self._running = False
        self._container: Optional[av.container.InputContainer] = None
        self._stream: Optional[av.video.stream.VideoStream] = None

    async def start(self) -> None:
        self._running = True
        await self._connection_loop()

    async def stop(self) -> None:
        self._running = False
        if self._container:
            self._container.close()
            self._container = None

    def get_latest_frames(self, count: int) -> list[FramePacket]:
        return list(self._frame_buffer)[-count:]

    def get_frames_since(self, timestamp: float) -> list[FramePacket]:
        return [fp for fp in self._frame_buffer if fp.timestamp >= timestamp]

    async def _connection_loop(self) -> None:
        delay = self.reconnect_base_delay
        while self._running:
            try:
                await self._connect_and_read()
                delay = self.reconnect_base_delay
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"RTSP connection error: {e}. Reconnecting in {delay:.1f}s...")
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.max_reconnect_delay)

    async def _connect_and_read(self) -> None:
        logger.info(f"Connecting to RTSP stream: {self.rtsp_url}")

        loop = asyncio.get_event_loop()
        self._container = await loop.run_in_executor(
            None,
            lambda: av.open(
                self.rtsp_url,
                format="rtsp",
                options={
                    "rtsp_transport": "tcp",
                    "stimeout": str(int(self.connection_timeout * 1_000_000)),
                    "buffer_size": "1024000",
                },
            ),
        )

        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"
        logger.info("RTSP connection established")

        try:
            for packet in self._container.demux(self._stream):
                if not self._running:
                    break

                for frame in packet.decode():
                    if not self._running:
                        break

                    np_frame = frame.to_ndarray(format="bgr24")
                    timestamp = time.time()

                    self._frame_buffer.append(
                        FramePacket(frame=np_frame, timestamp=timestamp, frame_id=self._frame_id)
                    )
                    self._frame_id += 1

        except av.AVError as e:
            logger.warning(f"Stream ended or error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during stream read: {e}")
            raise
        finally:
            if self._container:
                self._container.close()
                self._container = None


class FrameGenerator:
    def __init__(self, ingestor: RTSPIngestor, target_fps: float = 2.0):
        self.ingestor = ingestor
        self.target_fps = target_fps
        self._frame_interval = 1.0 / target_fps
        self._last_yield_time = 0.0

    async def frames(self) -> AsyncGenerator[FramePacket, None]:
        while self.ingestor._running:
            current_time = time.time()
            if current_time - self._last_yield_time >= self._frame_interval:
                if self.ingestor._frame_buffer:
                    frame_packet = self.ingestor._frame_buffer[-1]
                    self._last_yield_time = current_time
                    yield frame_packet
            await asyncio.sleep(0.01)