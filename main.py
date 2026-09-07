import argparse
import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from api.routes import create_app
from core.hls_generator import HLSGenerator
from core.ingest import RTSPIngestor
from core.ml_worker import MLWorker

load_dotenv(dotenv_path=Path(__file__).parent / "config.env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

RTSP_URL = os.getenv("RTSP_URL", "rtsp://localhost:8554/stream")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))


ingestor: RTSPIngestor | None = None
ml_worker: MLWorker | None = None
hls_generator: HLSGenerator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ingestor, ml_worker, hls_generator

    logger.info("Starting free-flow-video pipeline...")

    ingestor = RTSPIngestor(rtsp_url=RTSP_URL)
    ml_worker = MLWorker(ingestor=ingestor)
    hls_generator = HLSGenerator(rtsp_url=RTSP_URL)

    ingest_task = asyncio.create_task(ingestor.start())
    ml_task = asyncio.create_task(ml_worker.start())
    hls_task = asyncio.create_task(hls_generator.start())

    logger.info("All pipeline components started")

    yield

    logger.info("Shutting down pipeline...")
    ingest_task.cancel()
    ml_task.cancel()
    hls_task.cancel()

    await asyncio.gather(ingest_task, ml_task, hls_task, return_exceptions=True)

    if ingestor:
        await ingestor.stop()
    if ml_worker:
        await ml_worker.stop()
    if hls_generator:
        await hls_generator.stop()

    logger.info("Pipeline stopped")


app = create_app()
app.router.lifespan_context = lifespan


def main():
    global RTSP_URL
    parser = argparse.ArgumentParser(description="free-flow-video pipeline")
    parser.add_argument("--rtsp-url", default=RTSP_URL, help="RTSP stream URL")
    parser.add_argument("--host", default=HOST, help="API host")
    parser.add_argument("--port", type=int, default=PORT, help="API port")
    args = parser.parse_args()

    RTSP_URL = args.rtsp_url

    logger.info(f"RTSP URL: {RTSP_URL}")
    logger.info(f"API: http://{args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()