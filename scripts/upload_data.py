#!/usr/bin/env python3
"""
upload_data.py — Upload the challenge dataset to S3
====================================================
Run this ONCE from your local machine (or any machine with the data).

Usage:
    python scripts/upload_data.py --bucket my-er-bucket --data-dir ./data
"""

import argparse
import os
from pathlib import Path
import boto3
from botocore.exceptions import ClientError


def upload_to_s3(bucket: str, prefix: str, local_dir: Path) -> None:
    s3 = boto3.client("s3")
    files = list(local_dir.rglob("*"))
    data_files = [f for f in files if f.is_file()]

    if not data_files:
        print(f"No files found in {local_dir}")
        return

    print(f"Uploading {len(data_files)} files to s3://{bucket}/{prefix}/")
    for fp in data_files:
        key = f"{prefix}/{fp.relative_to(local_dir).as_posix()}"
        try:
            s3.upload_file(str(fp), bucket, key)
            print(f"  ✓  {fp.name} → s3://{bucket}/{key}")
        except ClientError as e:
            print(f"  ✗  {fp.name} — ERROR: {e}")


def create_bucket_if_needed(bucket: str, region: str) -> None:
    s3 = boto3.client("s3", region_name=region)
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"Bucket s3://{bucket} already exists.")
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print(f"Creating bucket s3://{bucket} in {region} …")
            if region == "us-east-1":
                s3.create_bucket(Bucket=bucket)
            else:
                s3.create_bucket(
                    Bucket=bucket,
                    CreateBucketConfiguration={"LocationConstraint": region},
                )
            # Block all public access
            s3.put_public_access_block(
                Bucket=bucket,
                PublicAccessBlockConfiguration={
                    "BlockPublicAcls": True,
                    "IgnorePublicAcls": True,
                    "BlockPublicPolicy": True,
                    "RestrictPublicBuckets": True,
                },
            )
            print(f"Bucket created with all public access blocked.")
        else:
            raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket",   required=True, help="S3 bucket name")
    ap.add_argument("--data-dir", default="data", help="Local data directory")
    ap.add_argument("--prefix",   default="entity-resolution/data",
                    help="S3 key prefix")
    ap.add_argument("--region",   default="us-east-1")
    ap.add_argument("--create-bucket", action="store_true",
                    help="Create the S3 bucket if it doesn't exist")
    args = ap.parse_args()

    if args.create_bucket:
        create_bucket_if_needed(args.bucket, args.region)

    upload_to_s3(args.bucket, args.prefix, Path(args.data_dir))
    print("\nDone! Set ER_S3_BUCKET in your pipeline config or as a GitHub secret.")
