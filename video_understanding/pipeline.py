"""
pipeline.py
-----------
End-to-end Video Understanding Pipeline orchestrator.

Ties together extraction (S3), TwelveLabs indexing, embedding generation,
and result persistence (DynamoDB / S3).  Supports single-video and
batch-processing modes with progress tracking.

Usage
-----
Single video::

    from video_understanding.pipeline import VideoPipeline
    pipeline = VideoPipeline.from_env()
    result = pipeline.process_video("s3://my-bucket/videos/clip.mp4")

Batch::

    results = pipeline.process_batch(["s3://...", "s3://..."], max_workers=4)
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

import boto3

from video_understanding.twelvelabs_client import TwelveLabsClient, TwelveLabsAPIError

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class VideoRecord:
    """Immutable record representing a video through its pipeline lifecycle."""

    s3_uri: str
    index_id: str
    task_id: Optional[str] = None
    video_id: Optional[str] = None
    status: str = "pending"          # pending | indexing | indexed | embedding | embedded | failed
    embedding_s3_key: Optional[str] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())
    indexed_at: Optional[str] = None
    embedded_at: Optional[str] = None
    duration_sec: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class PipelineConfig:
    """Runtime configuration for the pipeline."""

    twelvelabs_api_key: str
    index_id: str
    s3_raw_bucket: str
    s3_embed_bucket: str
    dynamodb_table: str
    aws_region: str = "us-east-1"
    embedding_model: str = "Marengo-retrieval-2.7"
    presign_expiry_sec: int = 3600
    poll_interval_sec: int = 30
    indexing_timeout_sec: int = 1200
    embedding_timeout_sec: int = 600
    max_batch_workers: int = 4

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Build config from environment variables."""

        def _require(name: str) -> str:
            val = os.environ.get(name)
            if not val:
                raise EnvironmentError(f"Required environment variable '{name}' is not set.")
            return val

        return cls(
            twelvelabs_api_key=_require("TWELVELABS_API_KEY"),
            index_id=_require("TWELVELABS_INDEX_ID"),
            s3_raw_bucket=_require("S3_RAW_BUCKET"),
            s3_embed_bucket=_require("S3_EMBED_BUCKET"),
            dynamodb_table=_require("DYNAMODB_TABLE"),
            aws_region=os.environ.get("AWS_REGION", "us-east-1"),
            embedding_model=os.environ.get("EMBEDDING_MODEL", "Marengo-retrieval-2.7"),
            max_batch_workers=int(os.environ.get("MAX_BATCH_WORKERS", "4")),
        )


# ---------------------------------------------------------------------------
# Progress tracker
# ---------------------------------------------------------------------------


class BatchProgressTracker:
    """Simple progress tracker for batch runs."""

    def __init__(self, total: int) -> None:
        self.total = total
        self._done = 0
        self._failed = 0
        self._start = time.monotonic()

    def record_success(self) -> None:
        self._done += 1
        self._log()

    def record_failure(self) -> None:
        self._failed += 1
        self._done += 1
        self._log()

    def _log(self) -> None:
        elapsed = time.monotonic() - self._start
        pct = self._done / self.total * 100
        log.info(
            "Progress: %d/%d (%.1f%%) | failed=%d | elapsed=%.1fs",
            self._done,
            self.total,
            pct,
            self._failed,
            elapsed,
        )

    @property
    def success_count(self) -> int:
        return self._done - self._failed

    @property
    def failure_count(self) -> int:
        return self._failed


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class VideoPipeline:
    """
    Orchestrates the full video-understanding pipeline for one or many videos.

    Steps
    -----
    1. Generate presigned S3 URL for the video.
    2. Upload video to TwelveLabs index.
    3. Poll indexing task until ``ready``.
    4. Generate multimodal embedding.
    5. Store embedding vector as ``.npy`` in S3.
    6. Persist ``VideoRecord`` to DynamoDB.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._cfg = config
        self._tl = TwelveLabsClient(
            api_key=config.twelvelabs_api_key,
            max_retries=3,
        )
        self._s3 = boto3.client("s3", region_name=config.aws_region)
        self._dynamodb = boto3.resource("dynamodb", region_name=config.aws_region)
        self._table = self._dynamodb.Table(config.dynamodb_table)

    @classmethod
    def from_env(cls) -> "VideoPipeline":
        """Construct pipeline from environment variables."""
        return cls(PipelineConfig.from_env())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_video(self, s3_uri: str, extra_metadata: Optional[dict[str, Any]] = None) -> VideoRecord:
        """
        Run the full pipeline for a single video.

        Parameters
        ----------
        s3_uri : str
            S3 URI, e.g. ``s3://bucket/prefix/video.mp4``.
        extra_metadata : dict, optional
            Additional metadata to attach to the DynamoDB record.

        Returns
        -------
        VideoRecord
            Final state of the record after all pipeline stages.
        """
        record = VideoRecord(s3_uri=s3_uri, index_id=self._cfg.index_id, metadata=extra_metadata or {})
        log.info("Starting pipeline for: %s", s3_uri)

        try:
            record = self._stage_upload(record)
            record = self._stage_index(record)
            record = self._stage_embed(record)
            record = self._stage_store_embedding(record)
        except Exception as exc:
            record.status = "failed"
            record.error = str(exc)
            log.error("Pipeline failed for %s: %s", s3_uri, exc, exc_info=True)
        finally:
            self._persist_record(record)

        return record

    def process_batch(
        self,
        s3_uris: list[str],
        max_workers: Optional[int] = None,
    ) -> list[VideoRecord]:
        """
        Process a list of S3 video URIs in parallel.

        Parameters
        ----------
        s3_uris : list[str]
        max_workers : int, optional
            Overrides ``config.max_batch_workers``.

        Returns
        -------
        list[VideoRecord]
            One record per input URI in the same order.
        """
        workers = max_workers or self._cfg.max_batch_workers
        tracker = BatchProgressTracker(total=len(s3_uris))
        results: dict[str, VideoRecord] = {}

        log.info("Batch processing %d videos with %d workers.", len(s3_uris), workers)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_uri = {pool.submit(self.process_video, uri): uri for uri in s3_uris}
            for future in as_completed(future_to_uri):
                uri = future_to_uri[future]
                try:
                    record = future.result()
                    results[uri] = record
                    if record.status in ("embedded", "indexed"):
                        tracker.record_success()
                    else:
                        tracker.record_failure()
                except Exception as exc:
                    log.error("Unhandled exception for %s: %s", uri, exc)
                    results[uri] = VideoRecord(s3_uri=uri, index_id=self._cfg.index_id, status="failed", error=str(exc))
                    tracker.record_failure()

        log.info(
            "Batch complete: success=%d failed=%d total=%d",
            tracker.success_count,
            tracker.failure_count,
            len(s3_uris),
        )
        # Return in original order
        return [results[uri] for uri in s3_uris]

    def iter_process_batch(
        self,
        s3_uris: list[str],
        max_workers: Optional[int] = None,
    ) -> Iterator[VideoRecord]:
        """
        Generator variant of process_batch — yields records as they complete.
        Useful for streaming progress to a caller.
        """
        workers = max_workers or self._cfg.max_batch_workers
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_uri = {pool.submit(self.process_video, uri): uri for uri in s3_uris}
            for future in as_completed(future_to_uri):
                try:
                    yield future.result()
                except Exception as exc:
                    uri = future_to_uri[future]
                    log.error("Unhandled exception for %s: %s", uri, exc)
                    yield VideoRecord(
                        s3_uri=uri,
                        index_id=self._cfg.index_id,
                        status="failed",
                        error=str(exc),
                    )

    # ------------------------------------------------------------------
    # Pipeline stages (private)
    # ------------------------------------------------------------------

    def _stage_upload(self, record: VideoRecord) -> VideoRecord:
        """Presign S3 URL and submit to TwelveLabs."""
        bucket, key = self._parse_s3_uri(record.s3_uri)
        presigned_url = self._s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=self._cfg.presign_expiry_sec,
        )
        task = self._tl.upload_video(
            index_id=record.index_id,
            video_url=presigned_url,
            metadata={"s3_key": key, **record.metadata},
        )
        record.task_id = task["_id"]
        record.status = "indexing"
        log.info("Upload submitted: task_id=%s", record.task_id)
        return record

    def _stage_index(self, record: VideoRecord) -> VideoRecord:
        """Poll indexing task until terminal state."""
        assert record.task_id, "task_id must be set before polling"
        task = self._tl.wait_for_task(
            task_id=record.task_id,
            poll_interval=self._cfg.poll_interval_sec,
            timeout=self._cfg.indexing_timeout_sec,
        )
        if task["status"] == "failed":
            raise RuntimeError(f"TwelveLabs indexing task {record.task_id} failed: {task}")
        record.video_id = task.get("video_id")
        record.status = "indexed"
        record.indexed_at = datetime.now(tz=timezone.utc).isoformat()
        log.info("Indexed: video_id=%s", record.video_id)
        return record

    def _stage_embed(self, record: VideoRecord) -> VideoRecord:
        """Generate embedding for the indexed video."""
        assert record.video_id, "video_id must be set before embedding"
        embed_task = self._tl.generate_embedding(
            video_id=record.video_id,
            model=self._cfg.embedding_model,
        )
        record.metadata["embedding_vector"] = embed_task.get("embedding")
        record.metadata["embedding_dim"] = len(embed_task.get("embedding") or [])
        record.status = "embedding"
        log.info("Embedding generated: dim=%d", record.metadata["embedding_dim"])
        return record

    def _stage_store_embedding(self, record: VideoRecord) -> VideoRecord:
        """Serialize embedding to .npy and upload to S3."""
        import io
        import numpy as np

        vector = record.metadata.pop("embedding_vector", None)
        if not vector:
            log.warning("No embedding vector for video_id=%s — skipping S3 store.", record.video_id)
            return record

        arr = np.array(vector, dtype=np.float32)
        buf = io.BytesIO()
        np.save(buf, arr)
        buf.seek(0)

        s3_key = f"embeddings/{record.video_id}.npy"
        self._s3.put_object(
            Bucket=self._cfg.s3_embed_bucket,
            Key=s3_key,
            Body=buf.read(),
            ContentType="application/octet-stream",
            Metadata={
                "video_id": record.video_id or "",
                "embed_model": self._cfg.embedding_model,
                "vector_dim": str(arr.shape[0]),
            },
        )
        record.embedding_s3_key = s3_key
        record.status = "embedded"
        record.embedded_at = datetime.now(tz=timezone.utc).isoformat()
        log.info("Embedding stored: s3://%s/%s", self._cfg.s3_embed_bucket, s3_key)
        return record

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_record(self, record: VideoRecord) -> None:
        """Upsert a VideoRecord into DynamoDB."""
        item = record.to_dict()
        # Remove None values — DynamoDB doesn't accept them in put_item
        item = {k: v for k, v in item.items() if v is not None}
        try:
            self._table.put_item(Item=item)
            log.debug("Persisted record to DynamoDB: video_id=%s", record.video_id)
        except Exception as exc:
            log.error("Failed to persist record to DynamoDB: %s", exc)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
        """Parse ``s3://bucket/key`` into ``(bucket, key)``."""
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Not a valid S3 URI: {s3_uri}")
        path = s3_uri[5:]
        bucket, _, key = path.partition("/")
        return bucket, key
