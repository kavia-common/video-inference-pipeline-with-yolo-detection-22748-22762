from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import boto3


@dataclass(frozen=True)
class S3Uri:
    """Parsed S3 URI."""
    bucket: str
    key: str


def parse_s3_uri(uri: str) -> S3Uri:
    """Parse s3://bucket/key URIs."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Invalid s3 uri: {uri}")
    return S3Uri(bucket=bucket, key=key)


def _s3_client():
    """
    Create a boto3 S3 client.

    Notes:
    - Credentials/region are expected to be provided via standard AWS env vars/roles.
    """
    return boto3.client("s3")


# PUBLIC_INTERFACE
def download_from_s3(s3_uri: str, local_path: Path) -> Path:
    """
    Download an S3 object to a local path.

    Parameters
    ----------
    s3_uri:
        s3://bucket/key
    local_path:
        Target file path.

    Returns
    -------
    Path
        The downloaded file path.
    """
    s3 = _s3_client()
    parsed = parse_s3_uri(s3_uri)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(parsed.bucket, parsed.key, str(local_path))
    return local_path


# PUBLIC_INTERFACE
def upload_to_s3(local_path: Path, s3_prefix_uri: str, object_name: Optional[str] = None) -> str:
    """
    Upload a local file to an S3 prefix.

    Parameters
    ----------
    local_path:
        Local file to upload.
    s3_prefix_uri:
        s3://bucket/prefix (prefix may be empty)
    object_name:
        Optional object name override. If None, uses local file name.

    Returns
    -------
    str
        Full s3://bucket/key URI of uploaded object.
    """
    if not local_path.exists():
        raise FileNotFoundError(str(local_path))

    prefix = parse_s3_uri(s3_prefix_uri + ("" if s3_prefix_uri.endswith("/") else "/") + "_")  # hack for parsing
    bucket = prefix.bucket
    base_prefix = prefix.key[:-1]  # remove "_" sentinel

    name = object_name or local_path.name
    key = f"{base_prefix}{name}" if base_prefix else name

    s3 = _s3_client()
    s3.upload_file(str(local_path), bucket, key)

    return f"s3://{bucket}/{key}"
