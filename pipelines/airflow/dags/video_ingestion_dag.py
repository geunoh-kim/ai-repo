"""
video_ingestion_dag.py
----------------------
Airflow DAG: Video Ingestion Pipeline

Pulls new video files from S3, uploads them to a TwelveLabs index,
monitors indexing status, and writes metadata to DynamoDB.

Schedule: every 2 hours
Owner:    ml-data-eng
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable
from airflow.operators.empty import EmptyOperator
from airflow.sensors.s3_key_sensor import S3KeySensor
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DAG defaults
# ---------------------------------------------------------------------------

DEFAULT_ARGS: dict[str, Any] = {
    "owner": "ml-data-eng",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=60),
}

S3_RAW_BUCKET: str = Variable.get("S3_RAW_BUCKET", default_var="raw-videos-bucket")
S3_MANIFEST_PREFIX: str = "manifests/pending/"
TWELVELABS_INDEX_ID: str = Variable.get("TWELVELABS_INDEX_ID", default_var="")
DYNAMODB_TABLE: str = Variable.get("DYNAMODB_TABLE", default_var="video-metadata")
MAX_BATCH_SIZE: int = 50  # videos per DAG run


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="video_ingestion_pipeline",
    description="Ingest new S3 videos into TwelveLabs index and record metadata in DynamoDB",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 */2 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["video", "twelvelabs", "ingestion", "s3"],
    doc_md=__doc__,
) as dag:

    # ------------------------------------------------------------------
    # 1. Gate: check that a manifest file has landed in S3
    # ------------------------------------------------------------------
    wait_for_manifest = S3KeySensor(
        task_id="wait_for_manifest",
        bucket_name=S3_RAW_BUCKET,
        bucket_key=f"{S3_MANIFEST_PREFIX}*.json",
        wildcard_match=True,
        aws_conn_id="aws_default",
        timeout=60 * 30,       # wait up to 30 min
        poke_interval=60,       # check every 60 s
        mode="reschedule",      # release worker slot while waiting
        soft_fail=False,
    )

    # ------------------------------------------------------------------
    # 2. Extract: list pending S3 video keys from the manifest
    # ------------------------------------------------------------------
    @task(task_id="extract_pending_videos")
    def extract_pending_videos() -> list[dict[str, Any]]:
        """Read the latest manifest from S3 and return a list of video records."""
        import boto3

        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")

        manifests = []
        for page in paginator.paginate(Bucket=S3_RAW_BUCKET, Prefix=S3_MANIFEST_PREFIX):
            manifests.extend(page.get("Contents", []))

        if not manifests:
            log.info("No manifest files found — skipping run.")
            return []

        # Pick the oldest manifest to process FIFO
        manifests.sort(key=lambda o: o["LastModified"])
        manifest_key = manifests[0]["Key"]
        log.info("Processing manifest: %s", manifest_key)

        obj = s3.get_object(Bucket=S3_RAW_BUCKET, Key=manifest_key)
        records: list[dict[str, Any]] = json.loads(obj["Body"].read())

        log.info("Manifest contains %d video records.", len(records))
        return records[:MAX_BATCH_SIZE]

    pending_videos = extract_pending_videos()

    # ------------------------------------------------------------------
    # 3. Upload & Index TaskGroup
    # ------------------------------------------------------------------
    with TaskGroup(group_id="upload_and_index") as upload_group:

        @task(task_id="upload_videos_to_twelvelabs")
        def upload_videos_to_twelvelabs(video_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """
            Upload each video URL (presigned S3) to TwelveLabs index.
            Returns a list of {video_id, task_id, s3_key} dicts.
            """
            import sys
            import os

            # Allow importing from repo root inside Airflow worker
            sys.path.insert(0, "/opt/airflow/dags/../..")
            from video_understanding.twelvelabs_client import TwelveLabsClient

            api_key = Variable.get("TWELVELABS_API_KEY")
            client = TwelveLabsClient(api_key=api_key)

            upload_results: list[dict[str, Any]] = []
            for record in video_records:
                s3_key: str = record["s3_key"]
                presigned_url: str = record["presigned_url"]
                try:
                    task_info = client.upload_video(
                        index_id=TWELVELABS_INDEX_ID,
                        video_url=presigned_url,
                        metadata={
                            "s3_key": s3_key,
                            "source": record.get("source", "airflow-ingest"),
                        },
                    )
                    upload_results.append(
                        {
                            "s3_key": s3_key,
                            "task_id": task_info["_id"],
                            "status": task_info["status"],
                        }
                    )
                    log.info("Submitted s3://%s — TL task_id=%s", s3_key, task_info["_id"])
                except Exception as exc:
                    log.error("Failed to upload %s: %s", s3_key, exc)
                    upload_results.append(
                        {"s3_key": s3_key, "task_id": None, "status": "upload_failed", "error": str(exc)}
                    )

            return upload_results

        @task(task_id="poll_indexing_status", retries=10, retry_delay=timedelta(minutes=2))
        def poll_indexing_status(upload_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """
            Poll TwelveLabs task status until all submitted tasks are ready or failed.
            Returns enriched records with final status and video_id.
            """
            import time
            import sys

            sys.path.insert(0, "/opt/airflow/dags/../..")
            from video_understanding.twelvelabs_client import TwelveLabsClient

            api_key = Variable.get("TWELVELABS_API_KEY")
            client = TwelveLabsClient(api_key=api_key)

            pending = [r for r in upload_results if r.get("task_id") and r["status"] not in ("ready", "failed", "upload_failed")]
            completed: list[dict[str, Any]] = [r for r in upload_results if r not in pending]

            timeout_sec = 60 * 20  # 20 minutes max polling
            interval_sec = 30
            elapsed = 0

            while pending and elapsed < timeout_sec:
                still_pending: list[dict[str, Any]] = []
                for record in pending:
                    task_status = client.get_task_status(record["task_id"])
                    if task_status["status"] in ("ready", "failed"):
                        record["status"] = task_status["status"]
                        record["video_id"] = task_status.get("video_id")
                        log.info(
                            "Task %s => status=%s video_id=%s",
                            record["task_id"],
                            record["status"],
                            record.get("video_id"),
                        )
                        completed.append(record)
                    else:
                        still_pending.append(record)
                pending = still_pending
                if pending:
                    time.sleep(interval_sec)
                    elapsed += interval_sec

            if pending:
                log.warning("%d tasks still pending after timeout — marking as timeout.", len(pending))
                for r in pending:
                    r["status"] = "timeout"
                completed.extend(pending)

            return completed

        uploaded = upload_videos_to_twelvelabs(pending_videos)
        indexed = poll_indexing_status(uploaded)

    # ------------------------------------------------------------------
    # 4. Write metadata to DynamoDB
    # ------------------------------------------------------------------
    @task(task_id="write_metadata_to_dynamodb")
    def write_metadata_to_dynamodb(indexed_records: list[dict[str, Any]]) -> None:
        """Upsert each video record into DynamoDB with final status."""
        import boto3
        from datetime import timezone

        dynamodb = boto3.resource("dynamodb", region_name=Variable.get("AWS_REGION", default_var="us-east-1"))
        table = dynamodb.Table(DYNAMODB_TABLE)

        now = datetime.now(tz=timezone.utc).isoformat()
        with table.batch_writer() as batch:
            for record in indexed_records:
                item = {
                    "video_id": record.get("video_id") or f"no-id-{record['s3_key']}",
                    "s3_key": record["s3_key"],
                    "index_id": TWELVELABS_INDEX_ID,
                    "status": record["status"],
                    "task_id": record.get("task_id"),
                    "indexed_at": now,
                    "embedding_status": "pending" if record["status"] == "ready" else "skipped",
                }
                batch.put_item(Item=item)
                log.info("DynamoDB upsert: video_id=%s status=%s", item["video_id"], item["status"])

        log.info("Wrote %d records to DynamoDB table '%s'.", len(indexed_records), DYNAMODB_TABLE)

    # ------------------------------------------------------------------
    # 5. Archive processed manifest
    # ------------------------------------------------------------------
    @task(task_id="archive_manifest")
    def archive_manifest() -> None:
        """Move processed manifest from pending/ to processed/ prefix."""
        import boto3

        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=S3_RAW_BUCKET, Prefix=S3_MANIFEST_PREFIX):
            for obj in page.get("Contents", []):
                src_key = obj["Key"]
                dst_key = src_key.replace("manifests/pending/", "manifests/processed/", 1)
                s3.copy_object(Bucket=S3_RAW_BUCKET, CopySource={"Bucket": S3_RAW_BUCKET, "Key": src_key}, Key=dst_key)
                s3.delete_object(Bucket=S3_RAW_BUCKET, Key=src_key)
                log.info("Archived manifest: %s → %s", src_key, dst_key)

    # ------------------------------------------------------------------
    # 6. Terminal tasks
    # ------------------------------------------------------------------
    success = EmptyOperator(task_id="pipeline_success")
    failure_alert = EmptyOperator(
        task_id="pipeline_failure_alert",
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    # ------------------------------------------------------------------
    # DAG wiring
    # ------------------------------------------------------------------
    metadata_written = write_metadata_to_dynamodb(indexed)
    manifest_archived = archive_manifest()

    (
        wait_for_manifest
        >> pending_videos
        >> upload_group
        >> metadata_written
        >> manifest_archived
        >> success
    )
    manifest_archived >> failure_alert
