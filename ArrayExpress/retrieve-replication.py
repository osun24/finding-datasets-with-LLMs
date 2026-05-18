"""
Utility script to fetch ArrayExpress replication batch results.

The script scans batches created after May 17, 2026 at 12:00 AM PDT and
downloads outputs for input files whose filenames contain `arrayexpress` and
end in a numeric seed suffix, such as:

    batch_requests_gpt-5-mini-arrayexpress-v1-43.jsonl
    batch_requests_o4-mini-arrayexpress-v1-51.jsonl

Output filenames are derived from the input request filename by replacing
`batch_requests_` with `out-`. Error filenames use `batch-errors-`.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional
from zoneinfo import ZoneInfo

from dotenv import load_dotenv


SCRIPT_DIR = Path(__file__).resolve().parent

TARGET_CREATED_AFTER = datetime(
    2026,
    5,
    17,
    0,
    0,
    tzinfo=ZoneInfo("America/Los_Angeles"),
)
INPUT_FILENAME_KEYWORD = "arrayexpress"
SEED_SUFFIX_PATTERN = re.compile(r"-\d+\.jsonl$")


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
    streaming helper and the legacy `files.retrieve().bytes` pattern.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        content = client.files.content(file_id)
        with open(destination, "wb") as handle:
            handle.write(content.read())
        return
    except AttributeError:
        pass

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

    response = client.batches.list(limit=100)
    batches.extend(getattr(response, "data", []))

    while getattr(response, "has_more", False):
        last_item = response.data[-1]
        after = getattr(last_item, "id", None)
        if not after:
            break
        response = client.batches.list(limit=100, after=after)
        batches.extend(getattr(response, "data", []))

    return batches


def output_path_for_request(filename: str, prefix: str) -> Path:
    if filename.startswith("batch_requests_"):
        output_name = filename.replace("batch_requests_", prefix, 1)
    else:
        output_name = f"{prefix}{filename}"
    return SCRIPT_DIR / output_name


def has_seed_suffix(filename: str) -> bool:
    return bool(SEED_SUFFIX_PATTERN.search(filename))


def match_target_batches(client, batches: Iterable) -> List[dict]:
    """
    Identify batches that were created after the cutoff and whose input file
    names contain the ArrayExpress keyword.
    """
    matches = []

    for batch in batches:
        created_dt = resolve_timestamp(batch)
        if not created_dt or created_dt <= TARGET_CREATED_AFTER:
            continue

        input_file_id = getattr(batch, "input_file_id", None)
        if not input_file_id:
            continue

        file_meta = client.files.retrieve(input_file_id)
        filename = getattr(file_meta, "filename", None)
        if (
            not filename
            or INPUT_FILENAME_KEYWORD not in filename.lower()
            or not has_seed_suffix(filename)
        ):
            continue

        matches.append(
            {
                "batch": batch,
                "request_name": filename,
                "input_file_id": input_file_id,
                "output_file_id": getattr(batch, "output_file_id", None),
                "error_file_id": getattr(batch, "error_file_id", None),
                "status": getattr(batch, "status", "unknown"),
                "created_at": created_dt,
            }
        )

    return sorted(matches, key=lambda item: (item["created_at"], item["request_name"]))


def main() -> None:
    load_dotenv(SCRIPT_DIR / ".env")
    load_dotenv()
    client = get_openai_client()

    print(
        "Searching for replication batches created after "
        f"{TARGET_CREATED_AFTER.isoformat()} with "
        f"'{INPUT_FILENAME_KEYWORD}' in the input filename..."
    )

    batches = fetch_all_batches(client)
    print(f"Retrieved {len(batches)} total batches.")

    matches = match_target_batches(client, batches)
    if not matches:
        print("No matching replication batches found.")
        return

    for info in matches:
        request_name = info["request_name"]
        status = info["status"]
        created_at = info["created_at"].isoformat()

        print(
            f"\nBatch request: {request_name}"
            f"\n    Status: {status}"
            f"\n    Created: {created_at}"
            f"\n    Batch ID: {getattr(info['batch'], 'id', 'unknown')}"
        )

        output_file_id = info["output_file_id"]
        if output_file_id:
            destination = output_path_for_request(request_name, "out-")
            print(f"    Downloading output to {destination}...")
            download_file(client, output_file_id, destination)
        else:
            print("    No output file available yet.")

        error_file_id = info["error_file_id"]
        if error_file_id:
            destination = output_path_for_request(request_name, "batch-errors-")
            print(f"    Downloading error log to {destination}...")
            download_file(client, error_file_id, destination)


if __name__ == "__main__":
    main()
