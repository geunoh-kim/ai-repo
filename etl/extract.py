"""
etl/extract.py
--------------
Extract module: pulls video objects from AWS S3 for downstream processing.

Supports:
- Full extraction of all objects under a prefix
- Incremental extraction based on a watermark timestamp
- Metadata extraction (size, content-type, ETag, custom tags)
- Presigned URL generation for downstream upload steps
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

import boto3
from botocore.exceptions import ClientError

log = logging.getLogger(__name__)

# Supported video MIME types / extensions
_VIDEO_EXTENSIONS = frozenset(
    [".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v", ".ts", ".wmv"]
)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


class S3VideoRecord:
    """Represents a single video object in S3, enriched with metadata."""

    __slots__ = (
        "bucket",
        "key",
        "size_bytes",
        "last_modified",
        "etag",
        "content_type",
        "tags",
        "presigned_url",
        "s3_uri",
    )

    def __init__(
        self,
        bucket: str,
        key: str,
        size_bytes: int,
        last_modified: datetime,
        etag: str,
        content_type: str = "video/mp4",
        tags: Optional[dict[str, str]] = None,
        presigned_url: Optional[str] = None,
    ) -> None:
        self.bucket = bucket
        self.key = key
        self.size_bytes = size_bytes
        self.last_modified = last_modified
        self.etag = etag.strip('"')
        self.content_type = content_type
        self.tags = tags or {}
        self.presigned_url = presigned_url
        self.s3_uri = f"s3://{bucket}/{key}"

    @property
    def size_mb(self) -> float:
        return round(self.size_bytes / 1_048_576, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bucket": self.bucket,
            "s3_key": self.key,
            "s3_uri": self.s3_uri,
            "size_bytes": self.size_bytes,
            "size_mb": self.size_mb,
            "last_modified": self.last_modified.isoformat(),
            "etag": self.etag,
            "content_type": self.content_type,
            "tags": self.tags,
            "presigned_url": self.presigned_url,
        }

    def __repr__(self) -> str:
        return f"<S3VideoRecord {self.s3_uri} {self.size_mb} MB>"


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class S3VideoExtractor:
    """
    Extracts video objects from an S3 bucket prefix.

    Parameters
    ----------
    bucket : str
        S3 bucket name.
    prefix : str
        Object key prefix to scan (e.g. ``"raw/videos/"``).
    aws_region : str
        AWS region for the S3 client.
    presign_expiry : int
        Presigned URL TTL in seconds (default 3600).

    Example
    -------
    >>> extractor = S3VideoExtractor("my-bucket", "raw/videos/")
    >>> records = extractor.extract_new_since(watermark=datetime(2024, 1, 1, tzinfo=timezone.utc))
    >>> for r in records:
    ...     print(r.s3_uri, r.size_mb)
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        aws_region: str = "us-east-1",
        presign_expiry: int = 3600,
    ) -> None:
        self._bucket = bucket
        self._prefix = prefix
        self._presign_expiry = presign_expiry
        self._s3 = boto3.client("s3", region_name=aws_region)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def extract_all(self, fetch_tags: bool = False) -> list[S3VideoRecord]:
        """
        Return all video objects under the configured prefix.

        Parameters
        ----------
        fetch_tags : bool
            If ``True``, also retrieve object tags (adds one API call per object).

        Returns
        -------
        list[S3VideoRecord]
        """
        records = list(self._iter_objects(prefix=self._prefix, fetch_tags=fetch_tags))
        log.info("Extracted %d video objects from s3://%s/%s", len(records), self._bucket, self._prefix)
        return records

    def extract_new_since(
        self,
        watermark: datetime,
        fetch_tags: bool = False,
    ) -> list[S3VideoRecord]:
        """
        Return only video objects modified *after* the watermark datetime.

        Parameters
        ----------
        watermark : datetime
            Timezone-aware datetime. Objects with ``LastModified > watermark`` are returned.
        fetch_tags : bool

        Returns
        -------
        list[S3VideoRecord]
        """
        if watermark.tzinfo is None:
            watermark = watermark.replace(tzinfo=timezone.utc)

        records = [
            r for r in self._iter_objects(prefix=self._prefix, fetch_tags=fetch_tags)
            if r.last_modified > watermark
        ]
        log.info(
            "Incremental extract: %d new objects since %s from s3://%s/%s",
            len(records),
            watermark.isoformat(),
            self._bucket,
            self._prefix,
        )
        return records

    def extract_by_keys(self, keys: list[str], fetch_tags: bool = False) -> list[S3VideoRecord]:
        """
        Fetch metadata for an explicit list of S3 keys.

        Parameters
        ----------
        keys : list[str]
        fetch_tags : bool

        Returns
        -------
        list[S3VideoRecord]
        """
        records: list[S3VideoRecord] = []
        for key in keys:
            try:
                record = self._head_object(key, fetch_tags=fetch_tags)
                if record:
                    records.append(record)
            except ClientError as exc:
                log.error("Failed to head s3://%s/%s: %s", self._bucket, key, exc)
        log.info("Extracted %d / %d requested keys.", len(records), len(keys))
        return records

    def generate_presigned_urls(self, records: list[S3VideoRecord]) -> list[S3VideoRecord]:
        """
        Mutate each record in-place with a fresh presigned GET URL.

        Returns the same list for chaining.
        """
        for record in records:
            try:
                record.presigned_url = self._s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self._bucket, "Key": record.key},
                    ExpiresIn=self._presign_expiry,
                )
            except ClientError as exc:
                log.error("Failed to presign s3://%s/%s: %s", self._bucket, record.key, exc)
        return records

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _iter_objects(self, prefix: str, fetch_tags: bool) -> Iterator[S3VideoRecord]:
        """Paginate through S3 listing and yield ``S3VideoRecord`` objects."""
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                if not self._is_video(key):
                    continue
                tags = self._get_object_tags(key) if fetch_tags else {}
                content_type = self._guess_content_type(key)
                record = S3VideoRecord(
                    bucket=self._bucket,
                    key=key,
                    size_bytes=obj["Size"],
                    last_modified=obj["LastModified"],
                    etag=obj["ETag"],
                    content_type=content_type,
                    tags=tags,
                )
                yield record

    def _head_object(self, key: str, fetch_tags: bool) -> Optional[S3VideoRecord]:
        """Use HeadObject to get metadata for a single key."""
        try:
            resp = self._s3.head_object(Bucket=self._bucket, Key=key)
        except ClientError:
            return None

        tags = self._get_object_tags(key) if fetch_tags else {}
        return S3VideoRecord(
            bucket=self._bucket,
            key=key,
            size_bytes=resp["ContentLength"],
            last_modified=resp["LastModified"],
            etag=resp.get("ETag", ""),
            content_type=resp.get("ContentType", "video/mp4"),
            tags=tags,
        )

    def _get_object_tags(self, key: str) -> dict[str, str]:
        try:
            resp = self._s3.get_object_tagging(Bucket=self._bucket, Key=key)
            return {t["Key"]: t["Value"] for t in resp.get("TagSet", [])}
        except ClientError:
            return {}

    @staticmethod
    def _is_video(key: str) -> bool:
        return any(key.lower().endswith(ext) for ext in _VIDEO_EXTENSIONS)

    @staticmethod
    def _guess_content_type(key: str) -> str:
        ext = key.rsplit(".", 1)[-1].lower() if "." in key else ""
        _map = {
            "mp4": "video/mp4",
            "mov": "video/quicktime",
            "avi": "video/x-msvideo",
            "mkv": "video/x-matroska",
            "webm": "video/webm",
            "ts": "video/mp2t",
            "wmv": "video/x-ms-wmv",
        }
        return _map.get(ext, "application/octet-stream")
