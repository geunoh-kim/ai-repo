"""
glue_job.py
-----------
AWS Glue PySpark job for batch video metadata processing.

Reads raw video metadata from S3, normalizes and enriches it,
then writes the processed output back to S3 in Parquet format
for downstream analytics (Athena / Redshift Spectrum).

Usage (local test via pytest-spark or glue_local):
    python glue_job.py --JOB_NAME=video-metadata-etl \
                       --S3_INPUT_PATH=s3://raw-videos-bucket/metadata/ \
                       --S3_OUTPUT_PATH=s3://processed-bucket/metadata/

Deploy via AWS CLI:
    aws glue create-job --name video-metadata-etl \
        --role GlueExecutionRole \
        --command '{"Name":"glueetl","ScriptLocation":"s3://glue-scripts/glue_job.py","PythonVersion":"3"}' \
        --default-arguments '{"--job-language":"python","--TempDir":"s3://glue-tmp/"}'
"""

from __future__ import annotations

import sys
import logging
from datetime import datetime, timezone

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.transforms import DropNullFields
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Glue boilerplate
# ---------------------------------------------------------------------------

args = getResolvedOptions(
    sys.argv,
    [
        "JOB_NAME",
        "S3_INPUT_PATH",
        "S3_OUTPUT_PATH",
        "PARTITION_DATE",  # e.g. "2024-01-15"
    ],
)

sc = SparkContext()
glue_ctx = GlueContext(sc)
spark = glue_ctx.spark_session
job = Job(glue_ctx)
job.init(args["JOB_NAME"], args)

S3_INPUT_PATH: str = args["S3_INPUT_PATH"]
S3_OUTPUT_PATH: str = args["S3_OUTPUT_PATH"]
PARTITION_DATE: str = args.get("PARTITION_DATE", datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"))

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

RAW_SCHEMA = StructType(
    [
        StructField("video_id", StringType(), nullable=False),
        StructField("s3_key", StringType(), nullable=True),
        StructField("index_id", StringType(), nullable=True),
        StructField("status", StringType(), nullable=True),
        StructField("duration_sec", DoubleType(), nullable=True),
        StructField("file_size_bytes", LongType(), nullable=True),
        StructField("width", IntegerType(), nullable=True),
        StructField("height", IntegerType(), nullable=True),
        StructField("fps", DoubleType(), nullable=True),
        StructField("created_at", StringType(), nullable=True),
        StructField("indexed_at", StringType(), nullable=True),
        StructField("embedding_s3", StringType(), nullable=True),
        StructField("source", StringType(), nullable=True),
        StructField("tags", StringType(), nullable=True),  # JSON array as string
    ]
)

# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------

def extract(path: str) -> DataFrame:
    log.info("Reading raw metadata from: %s", path)
    return (
        spark.read
        .schema(RAW_SCHEMA)
        .option("multiLine", "true")
        .json(path)
    )

# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

def transform(df: DataFrame) -> DataFrame:
    log.info("Transforming metadata (%d rows)", df.count())

    df = (
        df
        # Drop rows with no video_id
        .filter(F.col("video_id").isNotNull() & (F.col("video_id") != ""))

        # Parse timestamps
        .withColumn("created_at_ts", F.to_timestamp("created_at"))
        .withColumn("indexed_at_ts", F.to_timestamp("indexed_at"))

        # Derived fields
        .withColumn("file_size_mb", F.round(F.col("file_size_bytes") / 1_048_576, 2))
        .withColumn("resolution", F.concat_ws("x", F.col("width").cast(StringType()), F.col("height").cast(StringType())))
        .withColumn("has_embedding", F.col("embedding_s3").isNotNull())
        .withColumn(
            "indexing_latency_sec",
            F.when(
                F.col("indexed_at_ts").isNotNull() & F.col("created_at_ts").isNotNull(),
                F.unix_timestamp("indexed_at_ts") - F.unix_timestamp("created_at_ts"),
            ).otherwise(None),
        )

        # Normalize status
        .withColumn(
            "status_normalized",
            F.when(F.col("status") == "ready", "indexed")
             .when(F.col("status").isin("failed", "timeout"), "failed")
             .otherwise("pending"),
        )

        # Add partition column
        .withColumn("dt", F.lit(PARTITION_DATE))

        # Drop raw timestamp strings and intermediate columns
        .drop("created_at", "indexed_at", "width", "height", "file_size_bytes")
    )

    # Drop fully null fields
    df = DropNullFields.apply(frame=glue_ctx.create_dynamic_frame.from_dataframe(df, glue_ctx)).toDF()

    log.info("Transform complete. Output schema: %s", df.schema.simpleString())
    return df

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load(df: DataFrame, path: str) -> None:
    output_path = f"{path.rstrip('/')}/dt={PARTITION_DATE}/"
    log.info("Writing %d rows to: %s", df.count(), output_path)
    (
        df.write
        .mode("overwrite")
        .parquet(output_path)
    )
    log.info("Write complete.")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raw_df = extract(S3_INPUT_PATH)
    processed_df = transform(raw_df)
    load(processed_df, S3_OUTPUT_PATH)
    job.commit()
    log.info("Glue job '%s' committed successfully.", args["JOB_NAME"])
