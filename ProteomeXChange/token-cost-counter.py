"""
Count input/output tokens and costs for the runs listed in model_performance_summary.csv.
Input tokens are pulled from each JSONL response (same loop pattern used in csv-to-openai.py);
output tokens are recomputed with tiktoken for accuracy, then both are priced with the
provided per-1M-token rates (no cached input discount is applied).
"""

import csv
import json
from pathlib import Path
from typing import Dict, Tuple

import tiktoken

# Pricing in USD per 1M tokens (no cached-input discount applied).
PRICING_PER_MILLION: Dict[str, Dict[str, float]] = {
    "gpt-5.1": {"input": 0.625, "output": 5.0},
    "gpt-5": {"input": 0.625, "output": 5.0},
    "gpt-5-mini": {"input": 0.125, "output": 1.0},
    "gpt-5-nano": {"input": 0.025, "output": 0.2},
    "gpt-5-pro": {"input": 7.5, "output": 60.0},
    "gpt-4.1": {"input": 1.0, "output": 4.0},
    "gpt-4.1-mini": {"input": 0.2, "output": 0.8},
    "gpt-4.1-nano": {"input": 0.05, "output": 0.2},
    "gpt-4o": {"input": 1.25, "output": 5.0},
    "gpt-4o-2024-05-13": {"input": 2.5, "output": 7.5},
    "gpt-4o-mini": {"input": 0.075, "output": 0.3},
    "o1": {"input": 7.5, "output": 30.0},
    "o1-pro": {"input": 75.0, "output": 300.0},
    "o3-pro": {"input": 10.0, "output": 40.0},
    "o3": {"input": 1.0, "output": 4.0},
    "o3-deep-research": {"input": 5.0, "output": 20.0},
    "o4-mini": {"input": 0.55, "output": 2.2},
}


def safe_encoding(model_name: str):
    """Return a tiktoken encoding for the given model, falling back to cl100k_base."""
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def count_message_tokens(messages, encoding) -> int:
    """Count tokens for a list of messages (role/content) like csv-to-openai.py."""
    total = 0
    for message in messages or []:
        total += len(encoding.encode(message.get("role", "")))
        total += len(encoding.encode(message.get("content", "")))
    return total


def summarize_tokens(output_path: Path, default_model: str) -> Tuple[int, int]:
    """
    Return total (input_tokens, output_tokens) for a batch JSONL response file.
    Input tokens are summed from usage.prompt_tokens when present; if missing,
    we fall back to counting message tokens in the request body.
    Output tokens are recomputed with tiktoken using the response model.
    """
    total_prompt_tokens = 0
    total_completion_tokens = 0
    encodings: Dict[str, object] = {}

    def get_encoding(name: str):
        key = name or default_model
        if key not in encodings:
            encodings[key] = safe_encoding(key)
        return encodings[key]

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            response = record.get("response")
            if not response or response.get("error") or response.get("status_code") != 200:
                continue

            body = response.get("body", {})
            usage = body.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens")

            if prompt_tokens is None:
                messages = body.get("messages", [])
                prompt_tokens = count_message_tokens(messages, get_encoding(body.get("model")))

            total_prompt_tokens += prompt_tokens

            choices = body.get("choices") or []
            if not choices:
                continue

            message = choices[0].get("message", {})
            content = message.get("content", "")
            encoding = get_encoding(body.get("model"))
            total_completion_tokens += len(encoding.encode(content))

    return total_prompt_tokens, total_completion_tokens


def normalize_model_name(raw: str) -> str:
    """Normalize model names from the summary CSV to pricing keys."""
    return raw.strip().lower()


def clean_cell(value) -> str:
    """Convert None/blank cells to empty strings."""
    if value is None:
        return ""
    return str(value).strip()


def main():
    summary_path = Path("model_performance_summary.csv")
    output_path = Path("token_costs_edoc.csv")

    if not summary_path.exists():
        raise FileNotFoundError(f"Cannot find summary CSV at {summary_path}")

    with summary_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    results = []

    for row in rows:
        model = clean_cell(row.get("Model"))
        version = clean_cell(row.get("Version"))
        trial = clean_cell(row.get("Trial"))
        file_name = clean_cell(row.get("File"))

        if not file_name:
            print(f"Skipping {model} {version} {trial}: missing file name.")
            continue

        file_path = Path(file_name)
        if not file_path.exists():
            alt_path = summary_path.parent / file_name
            if alt_path.exists():
                file_path = alt_path
            else:
                print(f"Skipping {model} {version} {trial}: file {file_name} not found.")
                continue

        model_key = normalize_model_name(model)
        prompt_tokens, completion_tokens = summarize_tokens(file_path, model_key)

        pricing = PRICING_PER_MILLION.get(model_key)
        if not pricing:
            print(f"No pricing found for model '{model}'. Costs set to 0.")
            input_cost = output_cost = 0.0
        else:
            input_cost = prompt_tokens / 1_000_000 * pricing["input"]
            output_cost = completion_tokens / 1_000_000 * pricing["output"]

        results.append(
            {
                "model": model,
                "version": version,
                "trial": trial,
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "input_cost": round(input_cost, 4),
                "output_cost": round(output_cost, 4),
            }
        )

    fieldnames = ["model", "version", "trial", "input_tokens", "output_tokens", "input_cost", "output_cost"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {output_path} with {len(results)} rows.")


if __name__ == "__main__":
    main()
