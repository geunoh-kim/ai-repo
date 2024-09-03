"""
etl/load.py
-----------
Load module: persists validated ``VideoMetadata`` records to DynamoDB and S3.

Provides:
- Upsert (put_item) with conditional expressions to avoid stale overwrites
- Batch upsert using DynamoDB's batch_writer for throughput
- S3 manifest writer for downstream pipeline triggers
- Idempotent: re-running with the same records is safe
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import boto3
from boto3.dynamodb.conditions import Attr
from botocore.exceptions import ClientError

from etl.transform import VideoMetadata

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DynamoDB loader
# ---------------------------------------------------------------------------


class DynamoDBLoader:
    """
    Loads ``VideoMetadata`` records into a DynamoDB table with upsert semantics.

    The table is expected to have:
    - Partition key: ``video_id`` (String)

    Parameters
    ----------
    table_name : str
        DynamoDB table name.
    aws_region : str
        AWS region.
    allow_overwrite : bool
        If ``False``, a conditional expression prevents overwriting records
        already in a terminal state (``embedded``).

    Example
    -------
    >>> loader = DynamoDBLoader("video-metadata")
    >>> loader.upsert(record)
    >>> loader.upsert_batch(records)
    """

    def __init__(
        self,
        table_name: str,
        aws_region: str = "us-east-1",
        allow_overwrite: bool = True,
    ) -> None:
        self._table_name = table_name
        self._allow_overwrite = allow_overwrite
        self._dynamodb = boto3.resource("dynamodb", region_name=aws_region)
        self._table = self._dynamodb.Table(table_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert(self, record: VideoMetadata) -> bool:
        """
        Write a single ``VideoMetadata`` record to DynamoDB.

        Parameters
        ----------
        record : VideoMetadata

        Returns
        -------
        bool
            ``True`` on success, ``False`` if the write was skipped due to
            condition check failure.
        """
        item = record.to_dynamodb_item()
        kwargs: dict[str, Any] = {"Item": item}

        if not self._allow_overwrite:
            # Only write if the item doesn't already have status='embedded'
            kwargs["ConditionExpression"] = (
                Attr("video_id").not_exists() | ~Attr("status").eq("embedded")
            )

        try:
            self._table.put_item(**kwargs)
            log.debug("Upserted video_id=%s to %s", record.video_id, self._table_name)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                log.debug("Skipped video_id=%s (already in terminal state)", record.video_id)
                return False
            log.error("DynamoDB put_item failed for video_id=%s: %s", record.video_id, exc)
            raise

    def upsert_batch(self, records: list[VideoMetadata]) -> dict[str, int]:
        """
        Batch-write many records using DynamoDB's ``batch_writer``.

        DynamoDB batch_writer handles auto-retry of unprocessed items.

        Parameters
        ----------
        records : list[VideoMetadata]

        Returns
        -------
        dict
            ``{"success": N, "failed": M}`` summary.
        """
        success = 0
        failed = 0

        with self._table.batch_writer() as batch:
            for record in records:
                try:
                    batch.put_item(Item=record.to_dynamodb_item())
                    success += 1
                except Exception as exc:
                    log.error("Batch write failed for video_id=%s: %s", record.video_id, exc)
                    failed += 1

        log.info(
            "Batch upsert to '%s': success=%d failed=%d total=%d",
            self._table_name,
            success,
            failed,
            len(records),
        )
        return {"success": success, "failed": failed}

    def update_status(
        self,
        video_id: str,
        status: str,
        extra_fields: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Partially update the status and optional extra fields for a record.

        Parameters
        ----------
        video_id : str
        status : str
        extra_fields : dict, optional
            Additional attribute updates, e.g. ``{"indexed_at": "2024-..."}``.
        """
        now = datetime.now(tz=timezone.utc).isoformat()
        update_expr = "SET #st = :status, updated_at = :ua"
        expr_names = {"#st": "status"}
        expr_values: dict[str, Any] = {":status": status, ":ua": now}

        if extra_fields:
            for i, (k, v) in enumerate(extra_fields.items()):
                placeholder = f":extra{i}"
                update_expr += f", {k} = {placeholder}"
                expr_values[placeholder] = v

        self._table.update_item(
            Key={"video_id": video_id},
            UpdateExpression=update_expr,
            ExpressionAttributeNames=expr_names,
            ExpressionAttributeValues=expr_values,
        )
        log.debug("Updated status for video_id=%s → %s", video_id, status)

    def get_by_status(self, status: str, limit: int = 100) -> list[dict[str, Any]]:
        """
        Scan DynamoDB for records matching a given status.

        Note: For production use with large tables, add a GSI on ``status``.
        """
        from boto3.dynamodb.conditions import Attr

        items: list[dict[str, Any]] = []
        kwargs: dict[str, Any] = {
            "FilterExpression": Attr("status").eq(status),
            "Limit": limit,
        }
        while True:
            resp = self._table.scan(**kwargs)
            items.extend(resp.get("Items", []))
            last_key = resp.get("LastEvaluatedKey")
            if not last_key or len(items) >= limit:
                break
            kwargs["ExclusiveStartKey"] = last_key
        return items[:limit]


# ---------------------------------------------------------------------------
# S3 manifest writer
# ---------------------------------------------------------------------------


class S3ManifestWriter:
    """
    Writes a JSON manifest file to S3 listing videos ready for the next pipeline stage.

    Manifests are consumed by the Airflow S3KeySensor gate.

    Parameters
    ----------
    bucket : str
        S3 bucket for manifests.
    prefix : str
        Key prefix, e.g. ``"manifests/pending/"``.
    aws_region : str
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "manifests/pending/",
        aws_region: str = "us-east-1",
    ) -> None:
        self._bucket = bucket
        self._prefix = prefix.rstrip("/") + "/"
        self._s3 = boto3.client("s3", region_name=aws_region)

    def write(self, records: list[VideoMetadata], run_id: Optional[str] = None) -> str:
        """
        Serialise records to JSON and upload to S3.

        Parameters
        ----------
        records : list[VideoMetadata]
        run_id : str, optional
            Unique identifier for this manifest (defaults to UTC timestamp).

        Returns
        -------
        str
            S3 key of the written manifest file.
        """
        if not run_id:
            run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        key = f"{self._prefix}{run_id}.json"
        payload = [r.to_dynamodb_item() for r in records]
        body = json.dumps(payload, default=str, indent=2).encode()

        self._s3.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
        log.info(
            "Manifest written: s3://%s/%s (%d records, %d bytes)",
            self._bucket,
            key,
            len(records),
            len(body),
        )
        return key


# ---------------------------------------------------------------------------
# S3 embedding loader
# ---------------------------------------------------------------------------


class S3EmbeddingLoader:
    """
    Loads numpy embedding arrays to S3.

    Companion to the DynamoDBLoader: stores the actual vectors
    while DynamoDB stores the metadata / pointer.

    Parameters
    ----------
    bucket : str
        S3 bucket for embeddings.
    prefix : str
        Key prefix, e.g. ``"embeddings/"``.
    aws_region : str
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "embeddings/",
        aws_region: str = "us-east-1",
    ) -> None:
        self._bucket = bucket
        self._prefix = prefix.rstrip("/") + "/"
        self._s3 = boto3.client("s3", region_name=aws_region)

    def store(self, video_id: str, vector: list[float], model: str) -> str:
        """
        Serialise *vector* as a .npy file and upload to S3.

        Parameters
        ----------
        video_id : str
        vector : list[float]
        model : str
            Embedding model name for metadata annotation.

        Returns
        -------
        str
            S3 key of the stored .npy file.
        """
        import io
        import numpy as np

        arr = np.array(vector, dtype=np.float32)
        buf = io.BytesIO()
        np.save(buf, arr)
        buf.seek(0)

        key = f"{self._prefix}{video_id}.npy"
        self._s3.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=buf.read(),
            ContentType="application/octet-stream",
            Metadata={
                "video_id": video_id,
                "embed_model": model,
                "vector_dim": str(arr.shape[0]),
            },
        )
        log.info("Stored embedding: s3://%s/%s (dim=%d)", self._bucket, key, arr.shape[0])
        return key

    def load(self, video_id: str) -> "np.ndarray":
        """Download and deserialise a stored embedding vector."""
        import io
        import numpy as np

        key = f"{self._prefix}{video_id}.npy"
        try:
            obj = self._s3.get_object(Bucket=self._bucket, Key=key)
            buf = io.BytesIO(obj["Body"].read())
            return np.load(buf)
        except ClientError as exc:
            raise FileNotFoundError(
                f"Embedding not found at s3://{self._bucket}/{key}"
            ) from exc
