"""
Count input/output tokens and costs for runs in model_performance_summary.csv.

The GEO summary now includes prompt-set, test-set, and combined rows. Split rows
are costed from their own output JSONL file; combined rows sum the prompt/test
files listed as combined:<prompt_file>+<test_file>.
"""

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    import tiktoken
except ImportError:
    tiktoken = None

SCRIPT_DIR = Path(__file__).resolve().parent

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
    if tiktoken is None:
        raise RuntimeError("tiktoken is required when a JSONL record lacks usage tokens.")
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def count_message_tokens(messages, encoding) -> int:
    """Count tokens for a list of chat messages, matching csv-to-openai.py."""
    total = 0
    for message in messages or []:
        total += len(encoding.encode(message.get("role", "")))
        total += len(encoding.encode(message.get("content", "")))
    return total


def normalize_model_name(raw: str) -> str:
    """Normalize model names from the summary CSV to pricing keys."""
    return clean_cell(raw).lower()


def clean_cell(value) -> str:
    """Convert None/blank cells to empty strings."""
    if value is None:
        return ""
    return str(value).strip()


def resolve_file_path(file_name: str, base_dir: Path) -> Path:
    path = Path(file_name)
    if not path.is_absolute():
        path = base_dir / path
    return path


def expand_file_spec(file_spec: str, base_dir: Path) -> List[Path]:
    """Return one or more JSONL paths from a File column value."""
    spec = clean_cell(file_spec)
    if spec.startswith("combined:"):
        return [
            resolve_file_path(part, base_dir)
            for part in spec.removeprefix("combined:").split("+")
            if part.strip()
        ]
    return [resolve_file_path(spec, base_dir)] if spec else []


def iter_successful_bodies(output_paths: Iterable[Path]):
    for output_path in output_paths:
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                response = record.get("response")
                if not response or response.get("error") or response.get("status_code") != 200:
                    continue
                yield response.get("body", {})


def summarize_tokens(output_paths: Iterable[Path], default_model: str) -> Tuple[int, int]:
    """
    Return total (input_tokens, output_tokens) for one or more batch outputs.
    API usage counts are preferred so output cost includes reasoning tokens.
    """
    total_prompt_tokens = 0
    total_completion_tokens = 0
    encodings: Dict[str, object] = {}

    def get_encoding(name: str):
        key = name or default_model
        if key not in encodings:
            encodings[key] = safe_encoding(key)
        return encodings[key]

    for body in iter_successful_bodies(output_paths):
        usage = body.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

        if prompt_tokens is None:
            messages = body.get("messages", [])
            prompt_tokens = count_message_tokens(messages, get_encoding(body.get("model")))

        if completion_tokens is None:
            choices = body.get("choices") or []
            content = ""
            if choices:
                content = choices[0].get("message", {}).get("content", "")
            completion_tokens = len(get_encoding(body.get("model")).encode(content))

        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens

    return total_prompt_tokens, total_completion_tokens


def main():
    summary_path = SCRIPT_DIR / "model_performance_summary.csv"
    output_path = SCRIPT_DIR / "token_costs.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Cannot find summary CSV at {summary_path}")

    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    results = []

    for row in rows:
        model = clean_cell(row.get("Model"))
        version = clean_cell(row.get("Version"))
        trial = clean_cell(row.get("Trial"))
        split = clean_cell(row.get("Split"))
        file_name = clean_cell(row.get("File"))

        if model == "ensemble-or" or file_name == "ensemble" or split == "combined-prompt-test":
            continue

        file_paths = expand_file_spec(file_name, SCRIPT_DIR)
        missing_paths = [path for path in file_paths if not path.exists()]
        if not file_paths or missing_paths:
            missing = ", ".join(str(path) for path in missing_paths) or file_name
            print(f"Skipping {model} {version} {trial} {split}: file not found: {missing}")
            continue

        model_key = normalize_model_name(model)
        prompt_tokens, completion_tokens = summarize_tokens(file_paths, model_key)

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
                "split": split,
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "input_cost": round(input_cost, 4),
                "output_cost": round(output_cost, 4),
                "total_cost": round(input_cost + output_cost, 4),
            }
        )

    fieldnames = [
        "model",
        "version",
        "trial",
        "split",
        "input_tokens",
        "output_tokens",
        "input_cost",
        "output_cost",
        "total_cost",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {output_path} with {len(results)} rows.")


if __name__ == "__main__":
    main()
