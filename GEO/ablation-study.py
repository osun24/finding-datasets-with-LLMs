import argparse
import csv
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROMPT_VERSION = "v2"
DEFAULT_SPLIT = "test-set"
DEFAULT_MODELS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-5-mini", "o4-mini"]
REPLICATION_MODELS = {"gpt-5-mini", "o4-mini"}
REPLICATION_SEEDS = [43, 44, 45, 46, 47, 48, 49, 50, 51]
BASE_SEED = 42
TOKEN_LIMIT = 128000

REDUCED_INPUT_CONFIGS = ["title_only", "title_description"]
FULL_INPUT_CONFIG = "title_description_clinical"
ALL_INPUT_CONFIGS = REDUCED_INPUT_CONFIGS + [FULL_INPUT_CONFIG]

TEST_POSITIVE_FILES = [
    "samples_table_GSE102287.csv",
    "samples_table_GSE14814.csv",
    "samples_table_GSE19188.csv",
    "samples_table_GSE29013.csv",
    "samples_table_GSE42127.csv",
    "samples_table_GSE42425.csv",
    "samples_table_GSE47115.csv",
    "samples_table_GSE50081.csv",
]

MODEL_ALIASES = sorted(
    [
        ("gpt-5-mini", "GPT-5-mini"),
        ("gpt-4o-mini", "GPT-4o-mini"),
        ("o4-mini", "o4-mini"),
        ("gpt-4.1-mini", "GPT-4.1-mini"),
    ],
    key=lambda item: len(item[0]),
    reverse=True,
)

GEO_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "q1_survival_data",
        "q2_stage_i_or_act",
        "q3_inclusion_justification",
        "q4_include_in_meta_analysis",
    ],
    "properties": {
        "q1_survival_data": {
            "type": "object",
            "additionalProperties": False,
            "required": ["answer", "evidence", "justification"],
            "properties": {
                "answer": {"type": "string", "enum": ["Yes", "No", "Unclear"]},
                "evidence": {"type": "string"},
                "justification": {"type": "string"},
            },
        },
        "q2_stage_i_or_act": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "answer",
                "stage_i_evidence",
                "act_evidence",
                "justification",
            ],
            "properties": {
                "answer": {"type": "string", "enum": ["Yes", "No", "Unclear"]},
                "stage_i_evidence": {"type": "string"},
                "act_evidence": {"type": "string"},
                "justification": {"type": "string"},
            },
        },
        "q3_inclusion_justification": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "meets_survival_criterion",
                "meets_stage_or_act_criterion",
                "justification",
            ],
            "properties": {
                "meets_survival_criterion": {
                    "type": "string",
                    "enum": ["Yes", "No", "Unclear"],
                },
                "meets_stage_or_act_criterion": {
                    "type": "string",
                    "enum": ["Yes", "No", "Unclear"],
                },
                "justification": {"type": "string"},
            },
        },
        "q4_include_in_meta_analysis": {
            "type": "object",
            "additionalProperties": False,
            "required": ["answer"],
            "properties": {
                "answer": {"type": "string", "enum": ["Yes", "No"]},
            },
        },
    },
}

GEO_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "geo_dataset_screening",
        "description": "Structured screening decision for GEO NSCLC meta-analysis datasets.",
        "schema": GEO_OUTPUT_SCHEMA,
        "strict": True,
    },
}


def build_prompt(dataset_text):
    system_message = (
        "You are an oncology expert evaluating clinical datasets based on "
        "inclusion/exclusion criteria. Respond concisely and clearly, returning "
        "answers in the given structure."
    )
    user_message = (
        "A clinical non-small cell lung cancer (NSCLC) dataset, its title, and "
        "description are included below.\n\n"
        "Inclusion Criteria:\n"
        "1. Presence of patient-specific variables for survival time and status "
        "(e.g., overall survival status and overall survival survival time; or "
        "progression-free survival status and progression-free survival time).\n"
        "2. At least one of the following: total-study sample composition of at "
        "least 50% stage I patients; or explicit adjuvant chemotherapy annotation.\n\n"
        "Let's think step by step and answer the following questions, matching the "
        "JSON schema keys:\n"
        "\t1.\tq1_survival_data: Does the dataset include patient-specific "
        "**variables** for survival status and survival time (e.g., OS, RFS, PFS)? "
        "If so, specify the variables.\n"
        "\t2.\tq2_stage_i_or_act: Does the dataset, as a whole, include a majority "
        "of patients (>50%) with Stage I NSCLC or explicit adjuvant chemotherapy "
        "(ACT) annotation? If so, specify.\n"
        "\t3.\tq3_inclusion_justification: Based on the criteria, justify whether "
        "the dataset should be considered for inclusion.\n"
        "\t4.\tq4_include_in_meta_analysis: Should this dataset be considered for "
        'inclusion in the meta-analysis? Answer "Yes" or "No".\n\n'
        f"{dataset_text}"
    )
    return system_message, user_message


def count_tokens(messages, model="gpt-4o-mini"):
    try:
        import tiktoken

        encoding = tiktoken.encoding_for_model(model)
        return sum(
            len(encoding.encode(message["role"]))
            + len(encoding.encode(message["content"]))
            for message in messages
        )
    except Exception:
        return sum(len(message["content"]) for message in messages) // 4


def read_comment_metadata(csv_path):
    title = ""
    summary = ""
    comments = []

    with csv_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped.startswith("#"):
                continue
            comments.append(stripped)
            if stripped.startswith("# TITLE:"):
                title = stripped
            elif stripped.startswith("# SUMMARY:"):
                summary = stripped

    return title, summary, comments


def build_input_text(csv_path, input_config):
    title, summary, comments = read_comment_metadata(csv_path)

    if input_config == "title_only":
        if not title:
            raise ValueError(f"{csv_path.name} does not contain a # TITLE comment.")
        return title

    if input_config == "title_description":
        parts = [part for part in [title, summary] if part]
        if not parts:
            raise ValueError(f"{csv_path.name} does not contain title/summary comments.")
        return "\n".join(parts)

    if input_config == FULL_INPUT_CONFIG:
        if pd is None:
            raise RuntimeError("pandas is required to regenerate the full-metadata input.")
        df = pd.read_csv(csv_path, comment="#", low_memory=False)
        csv_data = json.dumps(df.to_csv(index=False))
        if comments:
            return "\n".join(comments) + "\n\n" + csv_data
        return csv_data

    raise ValueError(f"Unknown input configuration: {input_config}")


def body_for_model(model, messages, seed):
    body = {
        "model": model,
        "messages": messages,
        "response_format": GEO_RESPONSE_FORMAT,
        "temperature": 0.0,
    }

    if model == "o4-mini":
        return {
            "model": model,
            "messages": messages,
            "response_format": GEO_RESPONSE_FORMAT,
            "temperature": 1,
            "seed": seed,
        }

    if model in {"gpt-5-mini", "gpt-5"}:
        return {
            "model": model,
            "messages": messages,
            "response_format": GEO_RESPONSE_FORMAT,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
            "seed": seed,
        }

    return body


def config_slug(input_config):
    return input_config.replace("_", "-")


def output_name_for_batch(batch_file):
    name = batch_file.name
    if not name.startswith("batch_requests_"):
        return f"out-{batch_file.stem}.jsonl"
    return "out-" + name[len("batch_requests_") :]


def create_batch_file(input_dir, output_path, model, input_config, seed=BASE_SEED):
    tasks = []
    total_tokens = 0

    for csv_path in sorted(input_dir.glob("*.csv")):
        dataset_text = build_input_text(csv_path, input_config)
        system_message, user_message = build_prompt(dataset_text)
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        task_tokens = count_tokens(messages, model=model)
        if task_tokens > TOKEN_LIMIT:
            print(f"{csv_path.name}: {task_tokens} tokens. Skipping token-limit overflow.")
            continue

        tasks.append(
            {
                "custom_id": csv_path.name,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body_for_model(model, messages, seed),
            }
        )
        total_tokens += task_tokens

    with output_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task) + "\n")

    return len(tasks), total_tokens


def create_batches(args):
    input_configs = args.configs
    if FULL_INPUT_CONFIG in input_configs and not args.allow_full_rerun:
        raise SystemExit(
            f"Refusing to create {FULL_INPUT_CONFIG} batches because that condition "
            "has already been evaluated. Pass --allow-full-rerun only if you truly "
            "intend to regenerate it."
        )

    split_dir = BASE_DIR / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    written = []
    for model in args.models:
        seeds = [BASE_SEED]
        if args.replicate_reasoning_models and model in REPLICATION_MODELS:
            seeds.extend(REPLICATION_SEEDS)

        for input_config in input_configs:
            for seed in seeds:
                seed_suffix = "" if seed == BASE_SEED else f"-{seed}"
                output_path = (
                    BASE_DIR
                    / f"batch_requests_{model}-chemo-may13-{PROMPT_VERSION}-"
                    f"{args.split}-ablation-{config_slug(input_config)}{seed_suffix}.jsonl"
                )
                task_count, total_tokens = create_batch_file(
                    split_dir,
                    output_path,
                    model=model,
                    input_config=input_config,
                    seed=seed,
                )
                written.append(output_path)
                print(
                    f"Wrote {output_path.name}: {task_count} tasks, "
                    f"~{total_tokens} prompt tokens"
                )

    return written


def get_openai_client():
    try:
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv()
        api_key = os.getenv("OPENAI") or os.getenv("OPENAI_API_KEY")
        return OpenAI(api_key=api_key)
    except ImportError as exc:
        raise RuntimeError("openai and python-dotenv are required for API calls.") from exc


def submit_batches(args):
    client = get_openai_client()
    batch_files = [Path(path) for path in args.batch_files]
    if not batch_files:
        batch_files = sorted(BASE_DIR.glob("batch_requests_*-ablation-*.jsonl"))

    manifest_entries = []
    for batch_file in batch_files:
        with batch_file.open("rb") as handle:
            uploaded = client.files.create(file=handle, purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        output_path = BASE_DIR / output_name_for_batch(batch_file)
        entry = {
            "batch_file": str(batch_file),
            "output_file": str(output_path),
            "input_file_id": uploaded.id,
            "batch_id": batch.id,
            "created_at": datetime.now().isoformat(),
        }
        manifest_entries.append(entry)
        print(f"Submitted {batch_file.name}: {batch.id}")

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = BASE_DIR / manifest_path
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump({"batches": manifest_entries}, handle, indent=2)
    print(f"Wrote manifest: {manifest_path}")


def retrieve_batches(args):
    client = get_openai_client()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = BASE_DIR / manifest_path

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    pending = {entry["batch_id"]: entry for entry in manifest.get("batches", [])}
    while pending:
        for batch_id, entry in list(pending.items()):
            batch = client.batches.retrieve(batch_id)
            print(f"{batch_id}: {batch.status}")
            if batch.status == "completed":
                if batch.output_file_id:
                    content = client.files.content(batch.output_file_id).read()
                    output_path = Path(entry["output_file"])
                    output_path.write_bytes(content)
                    print(f"Downloaded {output_path.name}")
                pending.pop(batch_id)
            elif batch.status in {"failed", "cancelled", "expired"}:
                if batch.error_file_id:
                    content = client.files.content(batch.error_file_id).read()
                    error_path = Path(entry["output_file"]).with_suffix(".errors.jsonl")
                    error_path.write_bytes(content)
                    print(f"Downloaded {error_path.name}")
                pending.pop(batch_id)

        if not args.wait or not pending:
            break
        time.sleep(args.poll_seconds)


def answer_is_yes(value):
    return str(value).strip().lower() == "yes"


def parse_q4_decision(content):
    parsed = json.loads(content)
    q4 = parsed.get("q4_include_in_meta_analysis", {})
    if isinstance(q4, dict):
        return answer_is_yes(q4.get("answer"))
    return answer_is_yes(q4)


def parse_batch_decisions(file_path):
    decisions = {}
    errors = []

    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            data = json.loads(line)
            custom_id = data.get("custom_id", "")
            try:
                body = ((data.get("response") or {}).get("body") or {})
                choices = body.get("choices") or []
                message = choices[0].get("message", {}) if choices else {}
                content = message.get("content") or ""
                decisions[custom_id] = parse_q4_decision(content)
            except Exception as exc:
                decisions[custom_id] = False
                errors.append(
                    {
                        "file": str(file_path),
                        "line": line_number,
                        "custom_id": custom_id,
                        "error": str(exc),
                    }
                )

    return decisions, errors


def calculate_metrics(decisions, positive_files, total_files_evaluated):
    positive_files = set(positive_files)
    predicted_positive = {name for name, include in decisions.items() if include}

    true_positive = len(predicted_positive & positive_files)
    false_positive = len(predicted_positive - positive_files)
    false_negative = len(positive_files - predicted_positive)
    true_negative = total_files_evaluated - true_positive - false_positive - false_negative

    sensitivity = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0
    )
    specificity = (
        true_negative / (true_negative + false_positive)
        if true_negative + false_positive
        else 0
    )
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0
    )
    accuracy = (
        (true_positive + true_negative) / total_files_evaluated
        if total_files_evaluated
        else 0
    )
    f1_score = (
        2 * precision * sensitivity / (precision + sensitivity)
        if precision + sensitivity
        else 0
    )

    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "total_files_evaluated": total_files_evaluated,
        "sensitivity": round(sensitivity, 3),
        "specificity": round(specificity, 3),
        "precision": round(precision, 3),
        "accuracy": round(accuracy, 3),
        "f1_score": round(f1_score, 3),
    }


def canonical_model_from_name(filename):
    for pattern, canonical in MODEL_ALIASES:
        if pattern in filename:
            return canonical
    return "Unknown"


def trial_from_name(filename):
    match = re.search(r"-([0-9]+)\.jsonl$", filename)
    if match:
        return f"seed{match.group(1)}"
    return "base"


def config_from_name(filename):
    if "-ablation-title-only" in filename:
        return "title_only"
    if "-ablation-title-description" in filename:
        return "title_description"
    return FULL_INPUT_CONFIG


def discover_output_files():
    full_outputs = [
        path
        for path in BASE_DIR.glob(f"out-*-chemo-may13-{PROMPT_VERSION}-test-set*.jsonl")
        if "-ablation-" not in path.name and path.stat().st_size > 0
    ]
    ablation_outputs = [
        path
        for path in BASE_DIR.glob(
            f"out-*-chemo-may13-{PROMPT_VERSION}-test-set-ablation-*.jsonl"
        )
        if path.stat().st_size > 0
    ]
    return sorted(full_outputs + ablation_outputs)


def median(values):
    values = sorted(values)
    if not values:
        return None
    midpoint = len(values) // 2
    if len(values) % 2:
        return values[midpoint]
    return (values[midpoint - 1] + values[midpoint]) / 2


def evaluate_outputs(args):
    output_files = [Path(path) for path in args.output_files] or discover_output_files()
    positive_files = TEST_POSITIVE_FILES
    total_files = len(list((BASE_DIR / args.split).glob("*.csv")))
    results = []
    errors = []

    for output_file in output_files:
        decisions, parse_errors = parse_batch_decisions(output_file)
        errors.extend(parse_errors)
        metrics = calculate_metrics(decisions, positive_files, total_files)
        results.append(
            {
                "model": canonical_model_from_name(output_file.name),
                "version": PROMPT_VERSION,
                "split": args.split,
                "input_config": config_from_name(output_file.name),
                "trial": trial_from_name(output_file.name),
                "file_path": str(output_file.relative_to(BASE_DIR)),
                "metrics": metrics,
            }
        )

    grouped = defaultdict(list)
    for result in results:
        grouped[(result["model"], result["input_config"])].append(result)

    baseline_by_model = {}
    for (model, input_config), group in grouped.items():
        if input_config != FULL_INPUT_CONFIG:
            continue
        baseline_by_model[model] = {
            metric: median([item["metrics"][metric] for item in group])
            for metric in group[0]["metrics"]
        }

    summary_rows = []
    for (model, input_config), group in sorted(grouped.items()):
        row = {
            "model": model,
            "input_config": input_config,
            "trials": len(group),
        }
        metric_names = list(group[0]["metrics"])
        for metric in metric_names:
            values = [item["metrics"][metric] for item in group]
            row[f"{metric}_median"] = median(values)
            row[f"{metric}_min"] = min(values)
            row[f"{metric}_max"] = max(values)

        baseline = baseline_by_model.get(model)
        if baseline and input_config != FULL_INPUT_CONFIG:
            for metric in ["sensitivity", "specificity", "precision", "accuracy", "f1_score"]:
                row[f"delta_{metric}_vs_full"] = round(
                    row[f"{metric}_median"] - baseline[metric], 3
                )
        summary_rows.append(row)

    results_path = BASE_DIR / args.results_json
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "prompt_version": PROMPT_VERSION,
                    "split": args.split,
                    "positive_files": positive_files,
                    "full_input_note": (
                        "title_description_clinical rows are parsed from existing "
                        "test-set outputs; no full-input batch is generated by default."
                    ),
                },
                "results": results,
                "parse_errors": errors,
            },
            handle,
            indent=2,
        )

    summary_path = BASE_DIR / args.summary_csv
    if summary_rows:
        fieldnames = sorted({key for row in summary_rows for key in row})
        front = ["model", "input_config", "trials"]
        fieldnames = front + [name for name in fieldnames if name not in front]
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"Evaluated {len(results)} output files.")
    print(f"Wrote {results_path.name}")
    print(f"Wrote {summary_path.name}")
    if errors:
        print(f"Parse errors: {len(errors)}; see {results_path.name}.")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Create and evaluate GEO input-configuration ablations for the final "
            "v2 screening prompt."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create-batches")
    create_parser.add_argument("--split", default=DEFAULT_SPLIT)
    create_parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    create_parser.add_argument("--configs", nargs="+", default=REDUCED_INPUT_CONFIGS)
    create_parser.add_argument("--allow-full-rerun", action="store_true")
    create_parser.add_argument("--replicate-reasoning-models", action="store_true")
    create_parser.set_defaults(func=create_batches)

    submit_parser = subparsers.add_parser("submit-batches")
    submit_parser.add_argument("batch_files", nargs="*")
    submit_parser.add_argument("--manifest", default="ablation_batch_manifest.json")
    submit_parser.set_defaults(func=submit_batches)

    retrieve_parser = subparsers.add_parser("retrieve-batches")
    retrieve_parser.add_argument("--manifest", default="ablation_batch_manifest.json")
    retrieve_parser.add_argument("--wait", action="store_true")
    retrieve_parser.add_argument("--poll-seconds", type=int, default=60)
    retrieve_parser.set_defaults(func=retrieve_batches)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("output_files", nargs="*")
    evaluate_parser.add_argument("--split", default=DEFAULT_SPLIT)
    evaluate_parser.add_argument("--results-json", default="ablation_performance_results.json")
    evaluate_parser.add_argument("--summary-csv", default="ablation_performance_summary.csv")
    evaluate_parser.set_defaults(func=evaluate_outputs)

    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
