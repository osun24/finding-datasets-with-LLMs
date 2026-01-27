"""
Utility script to fetch replication batch results for specific OpenAI runs.

The script scans all batches created on October 22, 2025 and downloads the
outputs for the replicates generated via `csv-to-openai.py` using the batch
request files named `batch_requests_{model}-chemo-v3-{seed}.jsonl` for:

- Models: `o4-mini`, `gpt-5-mini`
- Seeds: 43, 44, 45, 46

Output files are saved alongside existing batch outputs using the
`batch-output-{model}-chemo-v3-{seed}.jsonl` naming convention. Any batch-level
error files are stored as `batch-errors-{model}-chemo-v3-{seed}.jsonl`.
"""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv

TARGET_DATE = datetime(2025, 10, 26).date()
TARGET_MODELS = ["o4-mini", "gpt-5-mini"]
TARGET_SEEDS = [47, 48, 49, 50, 51]
REQUEST_NAME_TEMPLATE = "batch_requests_{model}-chemo-v3-{seed}.jsonl"
OUTPUT_NAME_TEMPLATE = "batch-output-{model}-chemo-v3-{seed}.jsonl"
ERROR_NAME_TEMPLATE = "batch-errors-{model}-chemo-v3-{seed}.jsonl"


def get_openai_client():
    """
    Instantiate the OpenAI client while remaining compatible with both the
    post-1.0 (`OpenAI`) and pre-1.0 (`openai.Client`) SDK interfaces.
    """
    api_key = os.getenv("OPENAI")
    if not api_key:
        raise RuntimeError("OPENAI environment variable is not set.")

    try:
        from openai import OpenAI

        return OpenAI(api_key=api_key)
    except ImportError:
        import openai

        return openai.Client(api_key=api_key)


def resolve_timestamp(batch) -> Optional[datetime]:
    """Parse a batch object's creation timestamp into a timezone-aware datetime."""
    timestamp = getattr(batch, "created_at", None)
    if timestamp is None:
        timestamp = getattr(batch, "created", None)
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def download_file(client, file_id: str, destination: Path) -> None:
    """
    Download a file associated with a batch. Supports both the `files.content`
    streaming helper (new SDK) and the legacy `files.retrieve().bytes` pattern.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        content = client.files.content(file_id)
        # `content` is a `StreamingResponse` in the new SDK.
        with open(destination, "wb") as handle:
            handle.write(content.read())
        return
    except AttributeError:
        pass  # Fall back to older interface below.

    file_obj = client.files.retrieve(file_id)
    data = getattr(file_obj, "bytes", None)
    if data is None and hasattr(file_obj, "content"):
        data = file_obj.content

    if data is None:
        raise RuntimeError(f"Unable to download file {file_id}: unsupported SDK response.")

    with open(destination, "wb") as handle:
        handle.write(data)


def fetch_all_batches(client) -> List:
    """
    Fetch the complete set of batches available to the workspace, handling
    pagination automatically when supported.
    """
    batches = []
    params: Dict[str, str] = {}

    response = client.batches.list(limit=100, **params)
    batches.extend(getattr(response, "data", []))

    while getattr(response, "has_more", False):
        last_item = response.data[-1]
        after = getattr(last_item, "id", None)
        if not after:
            break
        response = client.batches.list(limit=100, after=after)
        batches.extend(getattr(response, "data", []))

    return batches


def match_target_batches(client, batches: Iterable) -> Dict[str, Dict]:
    """
    Identify batches that match the target date and request filenames. Returns a
    dictionary keyed by the request filename with metadata for downloading.
    """
    targets = {
        REQUEST_NAME_TEMPLATE.format(model=model, seed=seed): {"model": model, "seed": seed}
        for model in TARGET_MODELS
        for seed in TARGET_SEEDS
    }

    matches: Dict[str, Dict] = {}

    for batch in batches:
        created_dt = resolve_timestamp(batch)
        if not created_dt or created_dt.date() != TARGET_DATE:
            continue

        input_file_id = getattr(batch, "input_file_id", None)
        if not input_file_id:
            continue

        file_meta = client.files.retrieve(input_file_id)
        filename = getattr(file_meta, "filename", None)
        if not filename or filename not in targets:
            continue

        matches[filename] = {
            "batch": batch,
            "model": targets[filename]["model"],
            "seed": targets[filename]["seed"],
            "input_file_id": input_file_id,
            "output_file_id": getattr(batch, "output_file_id", None),
            "error_file_id": getattr(batch, "error_file_id", None),
            "status": getattr(batch, "status", "unknown"),
            "created_at": created_dt,
        }

    return matches


def main():
    load_dotenv()
    client = get_openai_client()

    print(f"🔍 Searching for replication batches from {TARGET_DATE.isoformat()}...")

    batches = fetch_all_batches(client)
    print(f"🧾 Retrieved {len(batches)} total batches.")

    matches = match_target_batches(client, batches)
    if not matches:
        print("⚠️ No matching replication batches found.")
        return

    for request_name, info in matches.items():
        model = info["model"]
        seed = info["seed"]
        status = info["status"]
        created_at = info["created_at"].isoformat()

        print(
            f"\n➡️  Batch for {model} seed {seed} (request: {request_name})"
            f"\n    Status: {status}"
            f"\n    Created: {created_at}"
            f"\n    Batch ID: {getattr(info['batch'], 'id', 'unknown')}"
        )

        output_file_id = info["output_file_id"]
        if output_file_id:
            destination = Path(OUTPUT_NAME_TEMPLATE.format(model=model, seed=seed))
            print(f"    ⬇️  Downloading output to {destination}...")
            download_file(client, output_file_id, destination)
        else:
            print("    ⚠️  No output file available yet.")

        error_file_id = info["error_file_id"]
        if error_file_id:
            destination = Path(ERROR_NAME_TEMPLATE.format(model=model, seed=seed))
            print(f"    ⚠️  Downloading error log to {destination}...")
            download_file(client, error_file_id, destination)

    missing = [
        REQUEST_NAME_TEMPLATE.format(model=model, seed=seed)
        for model in TARGET_MODELS
        for seed in TARGET_SEEDS
        if REQUEST_NAME_TEMPLATE.format(model=model, seed=seed) not in matches
    ]

    if missing:
        print("\n❗ The following replication batches were not found on the target date:")
        for name in missing:
            print(f"   - {name}")


if __name__ == "__main__":
    main()
