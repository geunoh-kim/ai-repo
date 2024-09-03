"""
infra/aws/s3_config.py
----------------------
AWS S3 bucket provisioning and configuration for the Video Understanding Pipeline.

Creates and configures:
- Raw video bucket     (versioning, lifecycle rules, event notifications)
- Embeddings bucket    (lifecycle rules for cost management)
- Manifests bucket     (used by Airflow S3KeySensor)

Run directly to provision all buckets::

    python -m infra.aws.s3_config --env prod --region us-east-1

Or import and call from infrastructure-as-code scripts.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any

import boto3
from botocore.exceptions import ClientError

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Bucket config data classes
# ---------------------------------------------------------------------------


@dataclass
class BucketSpec:
    name: str
    region: str
    versioning: bool = False
    lifecycle_rules: list[dict[str, Any]] = field(default_factory=list)
    notification_config: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    block_public_access: bool = True
    server_side_encryption: bool = True


def build_bucket_specs(env: str, region: str) -> list[BucketSpec]:
    """
    Return the full set of bucket specs for the given environment.

    Parameters
    ----------
    env : str
        Deployment environment, e.g. ``"dev" | "staging" | "prod"``.
    region : str
        AWS region.
    """
    common_tags = {
        "Project": "video-understanding-pipeline",
        "Environment": env,
        "ManagedBy": "infra/aws/s3_config.py",
        "Owner": "ml-data-eng",
    }

    raw_bucket = BucketSpec(
        name=f"video-raw-{env}",
        region=region,
        versioning=True,
        tags={**common_tags, "BucketRole": "raw-videos"},
        lifecycle_rules=[
            {
                "ID": "transition-old-raw-to-ia",
                "Status": "Enabled",
                "Filter": {"Prefix": "raw/"},
                "Transitions": [
                    {"Days": 30, "StorageClass": "STANDARD_IA"},
                    {"Days": 180, "StorageClass": "GLACIER"},
                ],
            },
            {
                "ID": "expire-processed-manifests",
                "Status": "Enabled",
                "Filter": {"Prefix": "manifests/processed/"},
                "Expiration": {"Days": 7},
            },
        ],
    )

    embed_bucket = BucketSpec(
        name=f"video-embeddings-{env}",
        region=region,
        versioning=False,
        tags={**common_tags, "BucketRole": "embeddings"},
        lifecycle_rules=[
            {
                "ID": "transition-embeddings-to-ia",
                "Status": "Enabled",
                "Filter": {"Prefix": "embeddings/"},
                "Transitions": [
                    {"Days": 90, "StorageClass": "STANDARD_IA"},
                ],
            },
        ],
    )

    manifest_bucket = BucketSpec(
        name=f"video-manifests-{env}",
        region=region,
        versioning=False,
        tags={**common_tags, "BucketRole": "manifests"},
        lifecycle_rules=[
            {
                "ID": "expire-pending-manifests",
                "Status": "Enabled",
                "Filter": {"Prefix": "manifests/pending/"},
                "Expiration": {"Days": 1},
            },
        ],
    )

    return [raw_bucket, embed_bucket, manifest_bucket]


# ---------------------------------------------------------------------------
# Provisioner
# ---------------------------------------------------------------------------


class S3Provisioner:
    """
    Creates and configures S3 buckets according to ``BucketSpec`` definitions.

    Parameters
    ----------
    aws_region : str
    dry_run : bool
        If ``True``, print planned actions without executing them.
    """

    def __init__(self, aws_region: str = "us-east-1", dry_run: bool = False) -> None:
        self._region = aws_region
        self._dry_run = dry_run
        self._s3 = boto3.client("s3", region_name=aws_region)

    def provision_all(self, specs: list[BucketSpec]) -> dict[str, bool]:
        """Provision all buckets. Returns {bucket_name: success} mapping."""
        results: dict[str, bool] = {}
        for spec in specs:
            try:
                self.provision(spec)
                results[spec.name] = True
            except Exception as exc:
                log.error("Failed to provision bucket '%s': %s", spec.name, exc)
                results[spec.name] = False
        return results

    def provision(self, spec: BucketSpec) -> None:
        """Create and configure a single bucket."""
        self._create_bucket(spec)
        if spec.block_public_access:
            self._set_public_access_block(spec.name)
        if spec.server_side_encryption:
            self._set_encryption(spec.name)
        if spec.versioning:
            self._enable_versioning(spec.name)
        if spec.lifecycle_rules:
            self._set_lifecycle(spec.name, spec.lifecycle_rules)
        if spec.tags:
            self._set_tags(spec.name, spec.tags)
        log.info("Bucket '%s' fully provisioned.", spec.name)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_bucket(self, spec: BucketSpec) -> None:
        if self._dry_run:
            log.info("[DRY RUN] Would create bucket: %s in %s", spec.name, spec.region)
            return
        try:
            kwargs: dict[str, Any] = {"Bucket": spec.name}
            if spec.region != "us-east-1":
                kwargs["CreateBucketConfiguration"] = {"LocationConstraint": spec.region}
            self._s3.create_bucket(**kwargs)
            log.info("Created bucket: %s", spec.name)
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
                log.info("Bucket already exists: %s (skipping creation)", spec.name)
            else:
                raise

    def _set_public_access_block(self, name: str) -> None:
        if self._dry_run:
            log.info("[DRY RUN] Would block public access on: %s", name)
            return
        self._s3.put_public_access_block(
            Bucket=name,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            },
        )
        log.info("Public access blocked: %s", name)

    def _set_encryption(self, name: str) -> None:
        if self._dry_run:
            log.info("[DRY RUN] Would enable SSE-S3 on: %s", name)
            return
        self._s3.put_bucket_encryption(
            Bucket=name,
            ServerSideEncryptionConfiguration={
                "Rules": [
                    {"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}
                ]
            },
        )
        log.info("SSE-S3 enabled: %s", name)

    def _enable_versioning(self, name: str) -> None:
        if self._dry_run:
            log.info("[DRY RUN] Would enable versioning on: %s", name)
            return
        self._s3.put_bucket_versioning(
            Bucket=name,
            VersioningConfiguration={"Status": "Enabled"},
        )
        log.info("Versioning enabled: %s", name)

    def _set_lifecycle(self, name: str, rules: list[dict[str, Any]]) -> None:
        if self._dry_run:
            log.info("[DRY RUN] Would set %d lifecycle rules on: %s", len(rules), name)
            return
        self._s3.put_bucket_lifecycle_configuration(
            Bucket=name,
            LifecycleConfiguration={"Rules": rules},
        )
        log.info("Lifecycle rules set (%d): %s", len(rules), name)

    def _set_tags(self, name: str, tags: dict[str, str]) -> None:
        if self._dry_run:
            log.info("[DRY RUN] Would tag bucket: %s with %s", name, tags)
            return
        self._s3.put_bucket_tagging(
            Bucket=name,
            Tagging={"TagSet": [{"Key": k, "Value": v} for k, v in tags.items()]},
        )
        log.info("Tags applied: %s", name)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Provision S3 buckets for the video pipeline.")
    parser.add_argument("--env", default="dev", choices=["dev", "staging", "prod"])
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    specs = build_bucket_specs(env=args.env, region=args.region)
    provisioner = S3Provisioner(aws_region=args.region, dry_run=args.dry_run)
    results = provisioner.provision_all(specs)

    failed = [name for name, ok in results.items() if not ok]
    if failed:
        log.error("Failed to provision: %s", failed)
        sys.exit(1)
    log.info("All buckets provisioned successfully.")
