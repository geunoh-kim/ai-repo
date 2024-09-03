"""
etl/transform.py
----------------
Transform module: normalises raw S3 video metadata into a validated,
canonical schema before loading into DynamoDB.

Uses pydantic v2 for schema enforcement, coercion, and detailed
validation error messages.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------


class VideoMetadata(BaseModel):
    """
    Canonical schema for a video record in the pipeline.

    All fields map directly to DynamoDB attributes.
    """

    # Primary identifiers
    video_id: str = Field(..., description="TwelveLabs video ID (tlv_...) or derived hash")
    s3_key: str = Field(..., description="S3 object key relative to the raw bucket")
    s3_uri: str = Field(..., description="Full s3:// URI")
    index_id: str = Field(..., description="TwelveLabs index ID")

    # Status
    status: str = Field(
        default="pending",
        description="Pipeline lifecycle state",
        pattern=r"^(pending|indexing|indexed|embedding|embedded|failed|skipped)$",
    )
    embedding_status: str = Field(
        default="pending",
        pattern=r"^(pending|embedded|embedding_failed|skipped)$",
    )

    # File attributes
    size_bytes: int = Field(..., ge=0)
    size_mb: float = Field(..., ge=0)
    content_type: str = Field(default="video/mp4")
    etag: str = Field(default="")

    # Temporal
    last_modified: datetime
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    indexed_at: Optional[datetime] = None
    embedded_at: Optional[datetime] = None

    # Video-level attributes (populated after indexing)
    duration_sec: Optional[float] = Field(default=None, ge=0)
    resolution: Optional[str] = None
    fps: Optional[float] = Field(default=None, ge=0)

    # Provenance
    source: str = Field(default="s3-ingest")
    tags: dict[str, str] = Field(default_factory=dict)
    embedding_s3: Optional[str] = None

    # ---------------------------------------------------------------------------
    # Validators
    # ---------------------------------------------------------------------------

    @field_validator("video_id")
    @classmethod
    def normalise_video_id(cls, v: str) -> str:
        return v.strip()

    @field_validator("s3_key")
    @classmethod
    def validate_s3_key(cls, v: str) -> str:
        if not v or v.startswith("/"):
            raise ValueError(f"s3_key must be a non-empty key without a leading slash; got: '{v}'")
        return v.strip()

    @field_validator("s3_uri")
    @classmethod
    def validate_s3_uri(cls, v: str) -> str:
        if not v.startswith("s3://"):
            raise ValueError(f"s3_uri must start with 's3://'; got: '{v}'")
        return v

    @field_validator("content_type")
    @classmethod
    def normalise_content_type(cls, v: str) -> str:
        return v.lower().strip()

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if not re.match(r"^\d+x\d+$", v):
            raise ValueError(f"resolution must be in WxH format, e.g. '1920x1080'; got: '{v}'")
        return v

    @model_validator(mode="after")
    def check_size_consistency(self) -> "VideoMetadata":
        expected_mb = round(self.size_bytes / 1_048_576, 2)
        if abs(expected_mb - self.size_mb) > 0.1:
            log.warning(
                "size_mb mismatch for %s: size_bytes implies %.2f MB but size_mb=%s",
                self.s3_key,
                expected_mb,
                self.size_mb,
            )
            # Auto-correct
            object.__setattr__(self, "size_mb", expected_mb)
        return self

    # ---------------------------------------------------------------------------
    # Serialisation helpers
    # ---------------------------------------------------------------------------

    def to_dynamodb_item(self) -> dict[str, Any]:
        """Return a DynamoDB-compatible dict (no None values, datetimes as ISO strings)."""
        raw = self.model_dump()
        return _clean_for_dynamodb(raw)

    model_config = {"validate_assignment": True}


# ---------------------------------------------------------------------------
# Transformer class
# ---------------------------------------------------------------------------


class VideoMetadataTransformer:
    """
    Transforms raw S3 / pipeline records into validated ``VideoMetadata`` objects.

    Typical flow::

        from etl.extract import S3VideoExtractor
        from etl.transform import VideoMetadataTransformer

        records = S3VideoExtractor("bucket", "prefix/").extract_all()
        transformer = VideoMetadataTransformer(index_id="idx_xyz", default_source="batch-2024")
        validated = transformer.transform_batch(r.to_dict() for r in records)
    """

    def __init__(
        self,
        index_id: str,
        default_source: str = "s3-ingest",
        derive_video_id: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        index_id : str
            TwelveLabs index ID to embed in every record.
        default_source : str
            Provenance label applied when the raw record has no ``source`` field.
        derive_video_id : bool
            If ``True`` and the raw record has no ``video_id``, derive a stable
            deterministic ID from the S3 URI using SHA-256.
        """
        self._index_id = index_id
        self._default_source = default_source
        self._derive_video_id = derive_video_id

    def transform(self, raw: dict[str, Any]) -> Optional[VideoMetadata]:
        """
        Transform a single raw record dict into a ``VideoMetadata``.

        Returns ``None`` and logs a warning if validation fails.
        """
        try:
            enriched = self._enrich(raw)
            return VideoMetadata(**enriched)
        except Exception as exc:
            log.warning("Validation failed for record %s: %s", raw.get("s3_key", "?"), exc)
            return None

    def transform_batch(
        self,
        raw_records: "Iterable[dict[str, Any]]",
    ) -> list[VideoMetadata]:
        """
        Transform many records; skips invalid ones.

        Returns a list of successfully validated ``VideoMetadata`` objects.
        """
        from typing import Iterable

        results: list[VideoMetadata] = []
        errors = 0
        for raw in raw_records:
            rec = self.transform(raw)
            if rec:
                results.append(rec)
            else:
                errors += 1

        log.info(
            "Transform complete: %d valid, %d invalid (skipped).",
            len(results),
            errors,
        )
        return results

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _enrich(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Fill in derived/default fields before validation."""
        enriched = dict(raw)

        # Ensure index_id
        enriched.setdefault("index_id", self._index_id)

        # Derive video_id if not present
        if not enriched.get("video_id") and self._derive_video_id:
            enriched["video_id"] = _derive_video_id(enriched.get("s3_uri") or enriched.get("s3_key", ""))

        # Default source
        enriched.setdefault("source", self._default_source)

        # Normalise size_mb
        if "size_mb" not in enriched and "size_bytes" in enriched:
            enriched["size_mb"] = round(int(enriched["size_bytes"]) / 1_048_576, 2)

        # Ensure datetimes are timezone-aware
        for field in ("last_modified", "created_at", "indexed_at", "embedded_at"):
            if isinstance(enriched.get(field), datetime) and enriched[field].tzinfo is None:
                enriched[field] = enriched[field].replace(tzinfo=timezone.utc)

        return enriched


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _derive_video_id(s3_uri: str) -> str:
    """Generate a deterministic video ID from an S3 URI."""
    digest = hashlib.sha256(s3_uri.encode()).hexdigest()[:16]
    return f"derived_{digest}"


def _clean_for_dynamodb(obj: Any) -> Any:
    """Recursively remove None values and convert datetimes to ISO strings."""
    if isinstance(obj, dict):
        return {k: _clean_for_dynamodb(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_clean_for_dynamodb(i) for i in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj
