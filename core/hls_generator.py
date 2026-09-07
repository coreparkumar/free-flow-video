import asyncio
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class HLSGenerator:
    def __init__(
        self,
        rtsp_url: str,
        output_dir: str = "storage/hls",
        segment_duration: int = 2,
        playlist_size: int = 10,
    ):
        self.rtsp_url = rtsp_url
        self.output_dir = Path(output_dir)
        self.segment_duration = segment_duration
        self.playlist_size = playlist_size
        self._process: Optional[subprocess.Popen] = None
        self._running = False

    async def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._running = True
        await self._run_ffmpeg()

    async def stop(self) -> None:
        self._running = False
        if self._process:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
            self._process = None

    async def _run_ffmpeg(self) -> None:
        playlist_path = self.output_dir / "stream.m3u8"
        segment_pattern = self.output_dir / "segment_%03d.ts"

        cmd = [
            "ffmpeg",
            "-y",
            "-rtsp_transport",
            "tcp",
            "-i",
            self.rtsp_url,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-tune",
            "zerolatency",
            "-g",
            str(self.segment_duration * 30),
            "-sc_threshold",
            "0",
            "-f",
            "hls",
            "-hls_time",
            str(self.segment_duration),
            "-hls_list_size",
            str(self.playlist_size),
            "-hls_flags",
            "delete_segments+append_list",
            "-hls_segment_filename",
            str(segment_pattern),
            str(playlist_path),
        ]

        logger.info(f"Starting FFmpeg HLS generation: {' '.join(cmd)}")

        while self._running:
            try:
                self._process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                stdout, stderr = await asyncio.get_event_loop().run_in_executor(
                    None, self._process.communicate
                )

                if not self._running:
                    break

                logger.warning(f"FFmpeg exited with code {self._process.returncode}")
                if stderr:
                    logger.error(f"FFmpeg stderr: {stderr.decode()[:500]}")

            except Exception as e:
                logger.error(f"FFmpeg error: {e}")

            if self._running:
                logger.info("Restarting FFmpeg in 5 seconds...")
                await asyncio.sleep(5)

    def is_running(self) -> bool:
        return self._running and self._process is not None and self._process.poll() is None