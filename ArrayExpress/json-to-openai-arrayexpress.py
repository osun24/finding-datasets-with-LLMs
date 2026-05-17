#!/usr/bin/env python3
"""Create OpenAI batch requests for ArrayExpress neurodegeneration screening."""

import json
import os
import time
from pathlib import Path
import openai
import tiktoken
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONTEXT_GLOB = "*-screened/context_text/*.txt"
PROMPT_VERSION = "v1"

load_dotenv(SCRIPT_DIR / ".env")
client = openai.Client(api_key=os.getenv("OPENAI"))

SCHEMA_CONFIRMATION_PLACEHOLDER = "{ARRAYEXPRESS_CONTEXT_TEXT}"

YES_NO_UNCLEAR = {"type": "string", "enum": ["Yes", "No", "Unclear"]}

SOURCE_LIST_TARGETS = {
    "alzheimer-screened": "AD",
    "lbd-screened": "LBD",
    "als-ftd-screened": "ALS-FTD",
}

# Diseases for which familial cases are an exclusion criterion. ALS-FTD is
# omitted because familial ALS-FTD datasets are eligible for inclusion.
# “excluded familial AD and LBD datasets to increase data homogeneity” — Noori et al. 2021
# Schema name slugs (lowercase, no special chars) for the response_format `name` field.
RESPONSE_FORMAT_NAME_SLUG = {
    "AD": "ad",
    "LBD": "lbd",
    "ALS-FTD": "als_ftd",
}


# Ordered yes/no question slugs per disease, matching the build_prompt_* templates.
# The trailing inclusion_justification and include_dataset questions are appended
# automatically by make_output_schema.
DISEASE_YES_NO_QUESTIONS = {
    "AD": [
        "human_microarray_expression_profiling",
        "target_disease",
        "human_brain_tissue",
        "non_clinical",
        "case_control_design",
        "irrelevant_brain_region",
        "intermediate_phenotype",
        "familial_alzheimer_disease",
        "incompatible_technology",
    ],
    "LBD": [
        "human_microarray_expression_profiling",
        "target_disease",
        "human_brain_tissue",
        "non_clinical",
        "case_control_design",
        "irrelevant_brain_region",
        "familial_disease",
        "incompatible_technology",
    ],
    "ALS-FTD": [
        "human_microarray_expression_profiling",
        "target_disease",
        "human_brain_tissue",
        "non_clinical",
        "case_control_design",
        "irrelevant_brain_region",
        "incompatible_technology",
    ],
}


def _yes_no_qa_block() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["answer", "evidence", "justification"],
        "properties": {
            "answer": YES_NO_UNCLEAR,
            "evidence": {"type": "string"},
            "justification": {"type": "string"},
        },
    }


def make_output_schema(target_disease: str) -> dict:
    """Return a JSON schema tailored to a single target disease.

    Each yes/no criterion is its own top-level question (q1..qN), followed by
    a free-text inclusion_justification and a final Yes/No include_dataset
    decision. The question set differs per disease to match the prompt.
    """
    yes_no_slugs = DISEASE_YES_NO_QUESTIONS[target_disease]

    properties: dict[str, dict] = {}
    required: list[str] = []
    for index, slug in enumerate(yes_no_slugs, start=1):
        key = f"q{index}_{slug}"
        properties[key] = _yes_no_qa_block()
        required.append(key)

    justification_key = f"q{len(yes_no_slugs) + 1}_inclusion_justification"
    properties[justification_key] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["justification"],
        "properties": {
            "justification": {"type": "string"},
        },
    }
    required.append(justification_key)

    include_key = f"q{len(yes_no_slugs) + 2}_include_dataset"
    properties[include_key] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["answer"],
        "properties": {
            "answer": {"type": "string", "enum": ["Yes", "No"]},
        },
    }
    required.append(include_key)

    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


def make_response_format(target_disease: str) -> dict:
    slug = RESPONSE_FORMAT_NAME_SLUG[target_disease]
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"arrayexpress_{slug}_dataset_screening",
            "description": (
                f"Structured screening decision for ArrayExpress microarray datasets "
                f"in a {target_disease} screen."
            ),
            "schema": make_output_schema(target_disease),
            "strict": True,
        },
    }


def target_from_context_path(context_path: Path) -> str:
    source_list = context_path.parent.parent.name
    target = SOURCE_LIST_TARGETS.get(source_list)
    if target is None:
        raise ValueError(
            f"Cannot determine target disease from source list '{source_list}' "
            f"(context path: {context_path})"
        )
    return target


def build_prompt_ad(context_text: str) -> tuple[str, str]:
    system_message = (
        "You are a neurodegeneration transcriptomics expert evaluating clinical datasets, based on inclusion/exclusion criteria. Respond concisely and clearly, returning answers in the given structure."
    )
    user_message = (
        "An ArrayExpress/BioStudies dataset's context is included below.\n\n"
        
        "Inclusion Criteria:\n"
        "1. The dataset must be about human microarray expression profiling.\n"
        "2. The dataset must pertain to Alzheimer's disease (AD).\n"
        "3. The dataset must involve samples from CNS regions relevant to Alzheimer's disease (AD).\n\n"
        
        "Exclusion Criteria:\n"
        "4. If the dataset involves patient-derived in vitro cell lines or disease models, exclude it.\n"
        "5. If the study design is not case/control, exclude it.\n"
        "6. If the brain regions sampled are not significantly affected by AD neurodegeneration (e.g., cerebellum), exclude it.\n"
        "7. If the dataset only involves intermediate AD phenotypes, exclude it. Only disease AD samples classified as Braak V/VI (corresponding to a neocortical NFT stage) should be included.\n"
        "8. Familial AD datasets are excluded. Sporadic AD datasets are included.\n"
        "9. Technologies incompatible with microarray expression profiling, such as RNA-seq or proteomics, are excluded.\n\n"
        
        "Let's think step by step and answer the following questions, matching the JSON schema keys:\n"
        "\t1.\tq1_human_microarray_expression_profiling: Is the dataset about human microarray expression profiling? If so, please justify.\n"
        "\t2.\tq2_target_disease: Does the dataset pertain to Alzheimer's disease (AD)? If so, please justify.\n"
        "\t3.\tq3_human_brain_tissue: Does the dataset involve samples from human brain/CNS tissue? If so, please justify.\n"
        "\t4.\tq4_non_clinical: Does the dataset involve patient-derived in vitro cell lines or disease models? If so, please justify.\n"
        "\t5.\tq5_case_control_design: Does the dataset have a case/control study design? If so, please justify.\n"
        "\t6.\tq6_irrelevant_brain_region: Does the dataset involve brain regions not significantly affected by AD neurodegeneration (e.g., cerebellum)? If so, please justify.\n"
        "\t7.\tq7_intermediate_phenotype: Does the dataset only involve intermediate AD phenotypes (e.g., Braak III/IV)? If so, please justify.\n"
        "\t8.\tq8_familial_alzheimer_disease: Does the dataset only involve familial AD cases? If so, please justify.\n"
        "\t9.\tq9_incompatible_technology: Does the dataset involve technologies incompatible with microarray expression profiling, such as RNA-seq or proteomics? If so, please justify.\n\n"
        "\t10.\tq10_inclusion_justification: Based on the inclusion/exclusion criteria, justify whether the dataset should be considered for inclusion.\n"
        "\t11.\tq11_include_dataset: Should the dataset be considered for inclusion? (Answer \"Yes\" or \"No\")\n\n"
        
        f"{context_text}"
    )
    return system_message, user_message


def build_prompt_lbd(context_text: str) -> tuple[str, str]:
    system_message = (
        "You are a neurodegeneration transcriptomics expert evaluating clinical datasets, based on inclusion/exclusion criteria. Respond concisely and clearly, returning answers in the given structure."
    )
    user_message = (
        "An ArrayExpress/BioStudies dataset's context is included below.\n\n"
        
        "Inclusion Criteria:\n"
        "1. The dataset must be about human microarray expression profiling.\n"
        "2. The dataset must pertain to Lewy body diseases (LBD).\n"
        "3. The dataset must involve samples from CNS regions relevant to Lewy body diseases (LBD).\n\n"
        
        "Exclusion Criteria:\n"
        "4. If the dataset involves patient-derived in vitro cell lines or disease models, exclude it.\n"
        "5. If the study design is not case/control, exclude it.\n"
        "6. If the brain regions sampled are not significantly affected by LBD neurodegeneration, exclude it.\n"
        "7. Familial LBD datasets are excluded. Sporadic LBD datasets are included.\n"
        "8. Technologies incompatible with microarray expression profiling, such as RNA-seq or proteomics, are excluded.\n\n"
        
        "Let's think step by step and answer the following questions, matching the JSON schema keys:\n"
        "\t1.\tq1_human_microarray_expression_profiling: Is the dataset about human microarray expression profiling? If so, please justify.\n"
        "\t2.\tq2_target_disease: Does the dataset pertain to Lewy body diseases (LBD)? If so, please justify.\n"
        "\t3.\tq3_human_brain_tissue: Does the dataset involve samples from human brain/CNS tissue? If so, please justify.\n"
        "\t4.\tq4_non_clinical: Does the dataset involve patient-derived in vitro cell lines or disease models? If so, please justify.\n"
        "\t5.\tq5_case_control_design: Does the dataset have a case/control study design? If so, please justify.\n"
        "\t6.\tq6_irrelevant_brain_region: Does the dataset involve brain regions not significantly affected by LBD neurodegeneration? If so, please justify.\n"
        "\t7.\tq7_familial_disease: Does the dataset only involve familial LBD cases? If so, please justify.\n"
        "\t8.\tq8_incompatible_technology: Does the dataset involve technologies incompatible with microarray expression profiling, such as RNA-seq or proteomics? If so, please justify.\n\n"
        "\t9.\tq9_inclusion_justification: Based on the inclusion/exclusion criteria, justify whether the dataset should be considered for inclusion.\n"
        "\t10.\tq10_include_dataset: Should the dataset be considered for inclusion? (Answer \"Yes\" or \"No\")\n\n"
        
        f"{context_text}"
    )
    
    return system_message, user_message


def build_prompt_als_ftd(context_text: str) -> tuple[str, str]:
    system_message = (
        "You are a neurodegeneration transcriptomics expert evaluating clinical datasets, based on inclusion/exclusion criteria. Respond concisely and clearly, returning answers in the given structure."
    )
    user_message = (
        "An ArrayExpress/BioStudies dataset's context is included below.\n\n"
        
        "Inclusion Criteria:\n"
        "1. The dataset must be about human microarray expression profiling.\n"
        "2. The dataset must pertain to amyotrophic lateral sclerosis-frontotemporal dementia (ALS-FTD).\n"
        "3. The dataset must involve samples from CNS regions relevant to ALS-FTD.\n\n"
        
        "Exclusion Criteria:\n"
        "4. If the dataset involves patient-derived in vitro cell lines or disease models, exclude it.\n"
        "5. If the study design is not case/control, exclude it.\n"
        "6. If the brain regions sampled are not significantly affected by ALS-FTD neurodegeneration, exclude it.\n"
        "7. Technologies incompatible with microarray expression profiling, such as RNA-seq or proteomics, are excluded.\n\n"
        
        "Let's think step by step and answer the following questions, matching the JSON schema keys:\n"
        "\t1.\tq1_human_microarray_expression_profiling: Is the dataset about human microarray expression profiling? If so, please justify.\n"
        "\t2.\tq2_target_disease: Does the dataset pertain to amyotrophic lateral sclerosis-frontotemporal dementia (ALS-FTD)? If so, please justify.\n"
        "\t3.\tq3_human_brain_tissue: Does the dataset involve samples from human brain/CNS tissue? If so, please justify.\n"
        "\t4.\tq4_non_clinical: Does the dataset involve patient-derived in vitro cell lines or disease models? If so, please justify.\n"
        "\t5.\tq5_case_control_design: Does the dataset have a case/control study design? If so, please justify.\n"
        "\t6.\tq6_irrelevant_brain_region: Does the dataset involve brain regions not significantly affected by ALS-FTD neurodegeneration? If so, please justify.\n"
        "\t7.\tq7_incompatible_technology: Does the dataset involve technologies incompatible with microarray expression profiling, such as RNA-seq or proteomics? If so, please justify.\n\n"
        "\t8.\tq8_inclusion_justification: Based on the inclusion/exclusion criteria, justify whether the dataset should be considered for inclusion.\n"
        "\t9.\tq9_include_dataset: Should the dataset be considered for inclusion? (Answer \"Yes\" or \"No\")\n\n"
        
        f"{context_text}"
    )
    return system_message, user_message


PROMPT_BUILDERS = {
    "AD": build_prompt_ad,
    "LBD": build_prompt_lbd,
    "ALS-FTD": build_prompt_als_ftd,
}


def build_prompt(context_text: str, target_disease: str) -> tuple[str, str]:
    builder = PROMPT_BUILDERS.get(target_disease)
    if builder is None:
        raise ValueError(f"Unsupported target disease: {target_disease}")
    return builder(context_text)


def verify_schema_matches_prompt() -> None:
    print("\nReview the disease-specific prompt templates and JSON schema before creating batches.")
    print("=" * 80)
    print(f"Prompt version: {PROMPT_VERSION}")
    for target_disease in ("AD", "LBD", "ALS-FTD"):
        system_message, user_message = build_prompt(
            SCHEMA_CONFIRMATION_PLACEHOLDER, target_disease
        )
        print("\n" + "-" * 80)
        print(f"Prompt for target disease: {target_disease}")
        print("-" * 80)
        print("\nSystem message:\n")
        print(system_message)
        print("\nUser prompt template:\n")
        print(user_message)
    print("\n" + "=" * 80)
    for target_disease in ("AD", "LBD", "ALS-FTD"):
        print(f"\nJSON response_format for {target_disease}:\n")
        print(json.dumps(make_response_format(target_disease), indent=2))
    print("=" * 80)

    proceed = input(
        "Do these JSON schemas comply with the prompts being run? Type 'yes' to continue: "
    ).strip().lower()
    if proceed != "yes":
        raise SystemExit("Batch job creation aborted for schema/prompt review.")


def count_tokens(messages: list[dict[str, str]], model: str) -> int:
    encoding = tiktoken.encoding_for_model(model)
    total_tokens = 0
    for message in messages:
        total_tokens += len(encoding.encode(message["role"]))
        total_tokens += len(encoding.encode(message["content"]))
    return total_tokens


def build_body(
    model: str,
    messages: list[dict[str, str]],
    response_format: dict,
    reasoning: bool,
    seed: int,
) -> dict[str, object]:
    body: dict[str, object] = {
        "model": model,
        "messages": messages,
        "response_format": response_format,
        "temperature": 1 if reasoning else 0.0,
    }

    if model == "o4-mini":
        body["temperature"] = 1
        body["seed"] = seed

    if model in {"gpt-5-mini", "gpt-5"}:
        body = {
            "model": model,
            "messages": messages,
            "response_format": response_format,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
            "seed": seed,
        }

    return body


def create_batch_file(
    root: Path,
    input_glob: str,
    batch_file_path: Path,
    model: str,
    reasoning: bool,
    seed: int,
    verbose_prompts: bool,
) -> int:
    tasks = []
    total_tokens = 0

    for context_path in sorted(path for path in root.glob(input_glob) if path.is_file()):
        context_text = context_path.read_text(encoding="utf-8")
        target_disease = target_from_context_path(context_path)
        source_list = context_path.parent.parent.name
        accession = context_path.stem
        custom_id = f"{source_list}_{accession}"

        system_message, user_message = build_prompt(context_text, target_disease)
        response_format = make_response_format(target_disease)
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        if verbose_prompts:
            print(f"System message for {custom_id}:\n{system_message}\n")
            print(f"User message for {custom_id}:\n{user_message}\n")

        task_tokens = count_tokens(messages, model=model)
        if task_tokens > 128000:
            print(f"{custom_id}: {task_tokens} tokens. Skipping because it exceeds the token limit.")
            continue

        tasks.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": build_body(model, messages, response_format, reasoning, seed),
            }
        )
        total_tokens += task_tokens
        print(f"{custom_id}: {task_tokens} tokens")

    print(f"\nTotal estimated tokens for all requests: {total_tokens}")

    with batch_file_path.open("w", encoding="utf-8") as batch_file:
        for task in tasks:
            batch_file.write(json.dumps(task) + "\n")

    print(f"Wrote {len(tasks)} tasks to {batch_file_path}")
    return total_tokens


def upload_batch_file(batch_file_path: Path) -> str:
    uploaded_file = client.files.create(file=batch_file_path.open("rb"), purpose="batch")
    return uploaded_file.id


def create_batch_job(file_id: str) -> str:
    response = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    return response.id


def download_file(file_id: str, save_path: Path) -> None:
    content = client.files.content(file_id)
    save_path.write_bytes(content.read())


def monitor_batch(batch_id: str, output_prefix: Path) -> None:
    while True:
        response = client.batches.retrieve(batch_id)
        status = response.status
        output_file_id = response.output_file_id
        error_file_id = response.error_file_id
        print(f"Current status: {status}")
        print(f"Errors: {response.errors}")

        if status == "completed":
            if output_file_id:
                download_file(output_file_id, output_prefix.with_suffix(".output.jsonl"))
                print("Batch processing completed. Results saved.")
            if error_file_id:
                download_file(error_file_id, output_prefix.with_suffix(".errors.jsonl"))
                print("Some errors occurred. Details saved.")
            break
        if status in {"failed", "cancelled", "expired"}:
            if error_file_id:
                download_file(error_file_id, output_prefix.with_suffix(".errors.jsonl"))
                print(f"Some errors occurred. Details saved to {output_prefix.with_suffix('.errors.jsonl')}.")
            print(f"Batch job {status}.")
            break
        time.sleep(60)


def main() -> int:
    # gpt-5-mini; o4-mini; gpt-4.1-mini; gpt-4o-mini
    model = "o4-mini"
    verbose_prompts = False
    replication = True

    reasoning = model == "o4-mini"
    base_name = f"{model}-arrayexpress-neurodegeneration-{PROMPT_VERSION}"

    verify_schema_matches_prompt()

    if replication:
        for seed in [43, 44, 45, 46, 47, 48, 49, 50, 51]:
            name = f"{base_name}-{seed}"
            output_path = SCRIPT_DIR / f"batch_requests_{name}.jsonl"

            total_tokens = create_batch_file(
                root=SCRIPT_DIR,
                input_glob=DEFAULT_CONTEXT_GLOB,
                batch_file_path=output_path,
                model=model,
                reasoning=reasoning,
                seed=seed,
                verbose_prompts=verbose_prompts,
            )

            print(f"Model: {model}, Name: {name}, Seed: {seed}, {total_tokens} tokens.")

            file_id = upload_batch_file(output_path)
            print(f"Batch file uploaded. File ID: {file_id}")
            batch_id = create_batch_job(file_id)
            print(f"Batch job ID: {batch_id}")
    else:
        output_path = SCRIPT_DIR / f"batch_requests_{base_name}.jsonl"

        total_tokens = create_batch_file(
            root=SCRIPT_DIR,
            input_glob=DEFAULT_CONTEXT_GLOB,
            batch_file_path=output_path,
            model=model,
            reasoning=reasoning,
            seed=42,
            verbose_prompts=verbose_prompts,
        )

        proceed = input(
            f"Total tokens required: {total_tokens}. Do you want to upload and create a batch job? "
            "(yes/no): "
        ).strip().lower()
        if proceed != "yes":
            print("Batch job creation aborted.")
            return 0

        file_id = upload_batch_file(output_path)
        print(f"Batch file uploaded. File ID: {file_id}")
        batch_id = create_batch_job(file_id)
        print(f"Batch job ID: {batch_id}")

    print("Batch job created. Monitoring status...")
    monitor_batch(batch_id, output_path.with_suffix(""))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
