"""
video_embedding_dag.py
----------------------
Airflow DAG: Video Embedding Pipeline

Queries DynamoDB for videos that have been indexed but not yet embedded,
calls the TwelveLabs Embed API to generate multimodal embeddings,
stores the embedding vectors as .npy files in S3, and updates DynamoDB.

Schedule: every 4 hours (offset from ingestion DAG)
Owner:    ml-data-eng
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DAG-level constants  (pulled from Airflow Variables for flexibility)
# ---------------------------------------------------------------------------

DEFAULT_ARGS: dict[str, Any] = {
    "owner": "ml-data-eng",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=60),
}

DYNAMODB_TABLE: str = Variable.get("DYNAMODB_TABLE", default_var="video-metadata")
S3_EMBED_BUCKET: str = Variable.get("S3_EMBED_BUCKET", default_var="video-embeddings-bucket")
EMBEDDING_MODEL: str = "Marengo-retrieval-2.7"
BATCH_SIZE: int = 20          # videos per Airflow task call
MAX_VIDEOS_PER_RUN: int = 200  # cap to avoid overloading the API


# ---------------------------------------------------------------------------
# Helper: DynamoDB scan for un-embedded videos
# ---------------------------------------------------------------------------

def _scan_unembedded_videos(limit: int = MAX_VIDEOS_PER_RUN) -> list[dict[str, Any]]:
    """Return DynamoDB records where status='ready' and embedding_status='pending'."""
    import boto3
    from boto3.dynamodb.conditions import Attr

    dynamodb = boto3.resource("dynamodb", region_name=Variable.get("AWS_REGION", default_var="us-east-1"))
    table = dynamodb.Table(DYNAMODB_TABLE)

    results: list[dict[str, Any]] = []
    scan_kwargs: dict[str, Any] = {
        "FilterExpression": Attr("status").eq("ready") & Attr("embedding_status").eq("pending"),
        "Limit": limit,
    }

    while True:
        response = table.scan(**scan_kwargs)
        results.extend(response.get("Items", []))
        last_key = response.get("LastEvaluatedKey")
        if not last_key or len(results) >= limit:
            break
        scan_kwargs["ExclusiveStartKey"] = last_key

    return results[:limit]


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="video_embedding_pipeline",
    description="Generate TwelveLabs multimodal embeddings for indexed videos and store in S3",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 */4 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["video", "twelvelabs", "embeddings", "s3", "dynamodb"],
    doc_md=__doc__,
) as dag:

    # ------------------------------------------------------------------
    # 1. Fetch videos that need embeddings
    # ------------------------------------------------------------------
    @task(task_id="fetch_pending_videos")
    def fetch_pending_videos() -> list[dict[str, Any]]:
        """Scan DynamoDB for videos ready for embedding."""
        records = _scan_unembedded_videos()
        log.info("Found %d videos pending embedding.", len(records))
        if not records:
            log.info("Nothing to do — all indexed videos are already embedded.")
        return records

    pending_records = fetch_pending_videos()

    # ------------------------------------------------------------------
    # 2. Batch and generate embeddings
    # ------------------------------------------------------------------
    with TaskGroup(group_id="generate_embeddings") as embed_group:

        @task(task_id="call_twelvelabs_embed_api")
        def call_twelvelabs_embed_api(video_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """
            For each video, call TwelveLabs Embed API and return
            a list of {video_id, embedding, embed_model, error?} dicts.
            """
            import sys

            sys.path.insert(0, "/opt/airflow/dags/../..")
            from video_understanding.twelvelabs_client import TwelveLabsClient

            api_key = Variable.get("TWELVELABS_API_KEY")
            client = TwelveLabsClient(api_key=api_key)

            results: list[dict[str, Any]] = []
            for record in video_records:
                video_id: str = record["video_id"]
                try:
                    embedding_data = client.generate_embedding(
                        video_id=video_id,
                        model=EMBEDDING_MODEL,
                    )
                    results.append(
                        {
                            "video_id": video_id,
                            "s3_key": record.get("s3_key", ""),
                            "embedding": embedding_data["embedding"],
                            "embed_model": EMBEDDING_MODEL,
                            "error": None,
                        }
                    )
                    log.info("Embedding generated for video_id=%s", video_id)
                except Exception as exc:
                    log.error("Embedding failed for video_id=%s: %s", video_id, exc)
                    results.append(
                        {
                            "video_id": video_id,
                            "s3_key": record.get("s3_key", ""),
                            "embedding": None,
                            "embed_model": EMBEDDING_MODEL,
                            "error": str(exc),
                        }
                    )
            return results

        @task(task_id="store_embeddings_to_s3")
        def store_embeddings_to_s3(embed_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """
            Serialize each embedding vector as a .npy file and upload to S3.
            Returns records enriched with embedding_s3 path.
            """
            import io
            import numpy as np
            import boto3

            s3 = boto3.client("s3")
            enriched: list[dict[str, Any]] = []

            for result in embed_results:
                if result["error"] or result["embedding"] is None:
                    result["embedding_s3"] = None
                    enriched.append(result)
                    continue

                video_id = result["video_id"]
                vector = np.array(result["embedding"], dtype=np.float32)
                buf = io.BytesIO()
                np.save(buf, vector)
                buf.seek(0)

                s3_key = f"embeddings/{video_id}.npy"
                s3.put_object(
                    Bucket=S3_EMBED_BUCKET,
                    Key=s3_key,
                    Body=buf.read(),
                    ContentType="application/octet-stream",
                    Metadata={
                        "video_id": video_id,
                        "embed_model": result["embed_model"],
                        "vector_dim": str(vector.shape[0]),
                    },
                )
                log.info("Stored embedding: s3://%s/%s (dim=%d)", S3_EMBED_BUCKET, s3_key, vector.shape[0])
                result["embedding_s3"] = s3_key
                enriched.append(result)

            return enriched

        embed_results = call_twelvelabs_embed_api(pending_records)
        stored_results = store_embeddings_to_s3(embed_results)

    # ------------------------------------------------------------------
    # 3. Update DynamoDB with embedding paths and status
    # ------------------------------------------------------------------
    @task(task_id="update_dynamodb_embedding_status")
    def update_dynamodb_embedding_status(stored_results: list[dict[str, Any]]) -> dict[str, int]:
        """
        Update each video's DynamoDB record with:
        - embedding_status: 'embedded' | 'embedding_failed'
        - embedding_s3: path to .npy file in S3
        - embedded_at: ISO timestamp
        """
        import boto3
        from datetime import timezone

        dynamodb = boto3.resource("dynamodb", region_name=Variable.get("AWS_REGION", default_var="us-east-1"))
        table = dynamodb.Table(DYNAMODB_TABLE)

        now = datetime.now(tz=timezone.utc).isoformat()
        success_count = 0
        failure_count = 0

        for result in stored_results:
            video_id = result["video_id"]
            if result.get("embedding_s3"):
                update_expr = "SET embedding_status = :es, embedding_s3 = :ep, embedded_at = :ea"
                expr_vals = {
                    ":es": "embedded",
                    ":ep": result["embedding_s3"],
                    ":ea": now,
                }
                success_count += 1
            else:
                update_expr = "SET embedding_status = :es, embed_error = :err, embedded_at = :ea"
                expr_vals = {
                    ":es": "embedding_failed",
                    ":err": result.get("error", "unknown"),
                    ":ea": now,
                }
                failure_count += 1

            table.update_item(
                Key={"video_id": video_id},
                UpdateExpression=update_expr,
                ExpressionAttributeValues=expr_vals,
            )
            log.info(
                "Updated DynamoDB: video_id=%s embedding_status=%s",
                video_id,
                "embedded" if result.get("embedding_s3") else "embedding_failed",
            )

        summary = {"success": success_count, "failed": failure_count, "total": len(stored_results)}
        log.info("Embedding run complete: %s", summary)
        return summary

    # ------------------------------------------------------------------
    # 4. Emit run metrics (stub — plug in your metrics backend)
    # ------------------------------------------------------------------
    @task(task_id="emit_run_metrics")
    def emit_run_metrics(summary: dict[str, int]) -> None:
        """Push embedding run metrics to CloudWatch / Prometheus / Datadog."""
        try:
            import boto3

            cw = boto3.client("cloudwatch", region_name=Variable.get("AWS_REGION", default_var="us-east-1"))
            cw.put_metric_data(
                Namespace="VideoUnderstandingPipeline",
                MetricData=[
                    {"MetricName": "EmbeddingsGenerated", "Value": summary["success"], "Unit": "Count"},
                    {"MetricName": "EmbeddingsFailed", "Value": summary["failed"], "Unit": "Count"},
                ],
            )
            log.info("Metrics emitted to CloudWatch: %s", summary)
        except Exception as exc:
            log.warning("Could not emit metrics (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Terminal tasks
    # ------------------------------------------------------------------
    done = EmptyOperator(task_id="pipeline_success")
    alert = EmptyOperator(
        task_id="pipeline_failure_alert",
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    # ------------------------------------------------------------------
    # DAG wiring
    # ------------------------------------------------------------------
    summary = update_dynamodb_embedding_status(stored_results)
    metrics = emit_run_metrics(summary)

    (
        pending_records
        >> embed_group
        >> summary
        >> metrics
        >> done
    )
    metrics >> alert
