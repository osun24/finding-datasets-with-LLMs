import os
import json
import openai
import tiktoken
import time
from pathlib import Path
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent

load_dotenv(SCRIPT_DIR / ".env")
key = os.getenv('OPENAI')

client = openai.Client(api_key=key)

SCHEMA_CONFIRMATION_PLACEHOLDER = "{TITLE_AND_DESCRIPTION}"

# Variables have `evidence` fields on top of the `justification` field
PROTEOMEXCHANGE_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "q1_ovarian_cancer",
        "q2_healthy_controls_or_biomarker_discovery",
        "q2a_healthy_control_composition",
        "q3_proteomics",
        "q4_non_clinical",
        "q5_inclusion_justification",
        "q6_include_dataset",
    ],
    "properties": {
        "q1_ovarian_cancer": {
            "type": "object",
            "additionalProperties": False,
            "required": ["answer", "justification"],
            "properties": {
                "answer": {"type": "string", "enum": ["Yes", "No", "Unclear"]},
                "justification": {"type": "string"},
            },
        },
        "q2_healthy_controls_or_biomarker_discovery": {
            "type": "object",
            "additionalProperties": False,
            "required": ["answer", "evidence", "justification"],
            "properties": {
                "answer": {"type": "string", "enum": ["Yes", "No", "Unclear"]},
                "evidence": {"type": "string"},
                "justification": {"type": "string"},
            },
        },
        "q2a_healthy_control_composition": {
            "type": "object",
            "additionalProperties": False,
            "required": ["answer", "evidence", "justification"],
            "properties": {
                "answer": {
                    "type": "string",
                    "enum": ["Yes", "No", "Unclear", "Not applicable"],
                },
                "evidence": {"type": "string"},
                "justification": {"type": "string"},
            },
        },
        "q3_proteomics": {
            "type": "object",
            "additionalProperties": False,
            "required": ["answer", "justification"],
            "properties": {
                "answer": {"type": "string", "enum": ["Yes", "No", "Unclear"]},
                "justification": {"type": "string"},
            },
        },
        "q4_non_clinical": {
            "type": "object",
            "additionalProperties": False,
            "required": ["answer", "evidence", "justification"],
            "properties": {
                "answer": {"type": "string", "enum": ["Yes", "No", "Unclear"]},
                "evidence": {"type": "string"},
                "justification": {"type": "string"},
            },
        },
        "q5_inclusion_justification": {
            "type": "object",
            "additionalProperties": False,
            "required": ["meets_ovarian_cancer_criterion", "meets_healthy_controls_criterion", "meets_proteomics_criterion", "justification"],
            "properties": {
                "meets_ovarian_cancer_criterion": {"type": "string", "enum": ["Yes", "No", "Unclear"]},
                "meets_healthy_controls_criterion": {"type": "string", "enum": ["Yes", "No", "Unclear"]},
                "meets_proteomics_criterion": {"type": "string", "enum": ["Yes", "No", "Unclear"]},
                "justification": {"type": "string"},
            },
        },
        "q6_include_dataset": {
            "type": "object",
            "additionalProperties": False,
            "required": ["answer"],
            "properties": {
                "answer": {"type": "string", "enum": ["Yes", "No"]},
            },
        },
    },
}


PROTEOMEXCHANGE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "proteomexchange_dataset_screening",
        "description": "Structured screening decision for ProteomeXchange ovarian cancer proteomics datasets.",
        "schema": PROTEOMEXCHANGE_OUTPUT_SCHEMA,
        "strict": True,
    },
}

PROMPT_VERSION = "v2"
def build_prompt(study_data):
    """system_message = (
        "You are an oncology expert evaluating clinical datasets based on inclusion/exclusion criteria. Respond concisely and clearly, returning answers in the given structure."
    )
    
    user_message = (
        "A dataset's title and description are included below.\n\nInclusion Criteria: \n"
        "1. The dataset must be about ovarian cancer.\n"
        "2. The dataset must contain healthy controls.\n"
        "\t2a. If explicit sample compositions are provided, the dataset must contain at least 20% healthy controls.\n"
        "3. The dataset must be on proteomics.\n\n"
        "Let's think step by step and answer the following questions, matching the JSON schema keys:\n"
        "\t1.\tq1_ovarian_cancer: Is the dataset about ovarian cancer? If so, justify.\n"
        "\t2.\tq2_healthy_controls: Does the dataset contain healthy controls? If so, please explain.\n"
        "\t2a.\tq2a_healthy_control_composition: If the explicit sample compositions are provided, does the dataset contain at least 20% healthy controls?\n If so, please explain.\n"
        "\t3.\tq3_proteomics: Does it include proteomics? If so, please explain.\n"
        "\t4.\tq4_inclusion_justification: Based on the inclusion/exclusion criteria, justify whether the dataset should be considered for inclusion.\n"
        "\t5.\tq5_include_dataset: Should the dataset be considered for inclusion? (Answer \"Yes\" or \"No\")\n\n"
        f"{study_data}"
    )"""
    
    system_message = (
        "You are an oncology expert evaluating clinical datasets based on inclusion/exclusion criteria. Respond concisely and clearly, returning answers in the given structure."
    )
    
    user_message = (
        "A dataset's title and description are included below.\n\nInclusion Criteria: \n"
        "1. The dataset must be about ovarian cancer.\n"
        "2. The dataset must contain healthy patients as controls OR involve diagnostic biomarker discovery for ovarian cancer.\n"
        "\t2a. If explicit sample compositions are provided, the dataset must contain at least 20% healthy patients as controls. If explicit sample compositions are not provided but healthy patient controls are present, include the dataset for additional human review.\n"
        "3. The dataset must be on proteomics.\n\n"
        "Exclusion Criteria:\n"
        "4. Non-clinical datasets (e.g., xenografts, cell lines, animal models, and ex vivo studies) are excluded.\n\n"
        "Let's think step by step and answer the following questions, matching the JSON schema keys:\n"
        "\t1.\tq1_ovarian_cancer: Is the dataset about ovarian cancer? If so, justify.\n"
        "\t2.\tq2_healthy_controls_or_biomarker_discovery: Does the dataset contain healthy patients as controls or involve diagnostic biomarker discovery for ovarian cancer? If so, please explain.\n"
        "\t2a.\tq2a_healthy_control_composition: If the explicit sample compositions are provided, does the dataset contain at least 20% healthy patients as controls?\n If so, please explain.\n"
        "\t3.\tq3_proteomics: Does it include proteomics? If so, please explain.\n"
        "\t4.\tq4_non_clinical: Does the dataset include non-clinical samples (e.g., xenografts, cell lines, animal models, ex vivo studies, and in vitro studies)? If so, please explain.\n"
        "\t5.\tq5_inclusion_justification: Based on the inclusion/exclusion criteria, justify whether the dataset should be considered for inclusion.\n"
        "\t6.\tq6_include_dataset: Should the dataset be considered for inclusion? (Answer \"Yes\" or \"No\")\n\n"
        f"{study_data}"
    )

    return system_message, user_message


def verify_schema_matches_prompt():
    system_message, user_message = build_prompt(SCHEMA_CONFIRMATION_PLACEHOLDER)
    print("\nReview the prompt template and JSON schema before creating batches.")
    print("=" * 80)
    print(f"Prompt version: {PROMPT_VERSION}")
    print("\nSystem message:\n")
    print(system_message)
    print("\nUser prompt template:\n")
    print(user_message)
    print("\nJSON response_format:\n")
    print(json.dumps(PROTEOMEXCHANGE_RESPONSE_FORMAT, indent=2))
    print("=" * 80)

    proceed = input(
        "Does this JSON schema comply with the prompt being run? Type 'yes' to continue: "
    ).strip().lower()
    if proceed != "yes":
        raise SystemExit("Batch job creation aborted for schema/prompt review.")


def count_tokens(messages, model="gpt-4o-mini"):
    #Estimate the number of tokens used by the messages.
    encoding = tiktoken.encoding_for_model(model)
    total_tokens = 0
    for message in messages:
        total_tokens += len(encoding.encode(message["role"])) + len(encoding.encode(message["content"]))
    return total_tokens

def create_batch_file(input_directory, batch_file_path, model="gpt-4.1-mini", reasoning=False, seed = 42):
    tasks = []
    total_tokens = 0

    # Get all .txt files from the proteomexchange directory
    txt_files = [f for f in os.listdir(input_directory) if f.endswith('.txt')]
    
    for txt_file in txt_files:
        file_path = os.path.join(input_directory, txt_file)
        study_id = txt_file.replace('.txt', '')  # Use filename without extension as study ID
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Extract title and description from the file
            title = "No title"
            description = "No description"
            
            for line in lines:
                line = line.strip()
                if line.startswith("Title:"):
                    title = line[6:].strip()  # Remove "Title:" prefix
                elif line.startswith("Description:"):
                    description = line[12:].strip()  # Remove "Description:" prefix
            
            # Skip studies with no meaningful data
            if title == "No title available" and description == "No description available":
                continue
                
        except Exception as e:
            print(f"Error reading {txt_file}: {e}")
            continue
            
        study_data = f"Title: {title}\nDescription: {description}"
        
        system_message, user_message = build_prompt(study_data)
        
        print(f"System message for {study_id}:\n{system_message}\n")
        print(f"User message for {study_id}:\n{user_message}\n")

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        body = {
            "model": model,
            "messages": messages,
            "response_format": PROTEOMEXCHANGE_RESPONSE_FORMAT,
            "temperature": (1 if reasoning else 0.0),
        }

        task = {
            "custom_id": study_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }

        if model == "o4-mini":
            body = {
                "model": model,
                "messages": messages,
                "response_format": PROTEOMEXCHANGE_RESPONSE_FORMAT,
                "temperature": 1,
                "seed": seed,
            }
            task = {
                "custom_id": study_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }

        if model == "gpt-5-mini" or model == "gpt-5":
            body = {
                "model": model,
                "messages": messages,
                "response_format": PROTEOMEXCHANGE_RESPONSE_FORMAT,
                "top_p": 1, # top_p not supported for o4-mini nor gpt-5-mini other than default?
                "frequency_penalty": 0, # default
                "presence_penalty": 0, # default
                "n": 1,  # default
                "seed": seed,
            }
            task = {
                "custom_id": study_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            
        task_tokens = count_tokens(messages)

        if task_tokens > 128000:
            print(f"{study_id}: {task_tokens} tokens. Skipping because it exceeds the token limit.")
            continue
        tasks.append(task)
        total_tokens += task_tokens
        print(f"{study_id}: {task_tokens} tokens")

    print(f"\nTotal estimated tokens for all requests: {total_tokens}")

    with open(batch_file_path, 'w') as batch_file:
        for task in tasks:
            batch_file.write(json.dumps(task) + '\n')

    return total_tokens

def upload_batch_file(batch_file_path):
    file = client.files.create(file=open(batch_file_path, 'rb'), purpose="batch")
    return file.id

def create_batch_job(file_id):
    response = client.batches.create(input_file_id = file_id, endpoint="/v1/chat/completions", completion_window="24h")
    return response.id

def download_file(file_id, save_path):
    file = client.files.retrieve(file_id)
    with open(save_path, 'wb') as f:
        f.write(file.bytes)

def main():
    use_prompt_set = False
    split_name = "prompt-set" if use_prompt_set else "test-set"
    input_directory = SCRIPT_DIR / split_name  # Directory containing .txt files

    # gpt-5-mini; o4-mini; gpt-4.1-mini; gpt-4o-mini
    model = "o4-mini"  # Specify the model to use
    name = f"proteom-may13-{PROMPT_VERSION}-{split_name}"  # Specify the name for the batch job
    batch_file_path = SCRIPT_DIR / f"batch_requests_{model}-{name}.jsonl"

    # Count tokens and create batch file
    reasoning = True
    if model == "o4-mini":
        reasoning = True

    verify_schema_matches_prompt()
    
    replication = True
    if replication:
        for seed in [43, 44, 45, 46, 47, 48, 49, 50, 51]: #47, 48, 49, 50, 51 43, 44, 45, 46
            name = f"proteom-may13-{PROMPT_VERSION}-{split_name}-{seed}"  # Specify the name for the batch job
            batch_file_path = SCRIPT_DIR / f"batch_requests_{model}-{name}.jsonl"
            
            total_tokens = create_batch_file(input_directory, batch_file_path, model=model, reasoning=reasoning, seed=seed)

            print(f"Model: {model}, Name: {name}, Seed: {seed}, {total_tokens} tokens.")
            # Confirm before proceeding
            """proceed = input(f"Total tokens required: {total_tokens}. Do you want to proceed? (yes/no): ").strip().lower()
            if proceed != "yes":
                print("Batch job creation aborted.")
                return"""

            # Upload batch file and create batch job
            file_id = upload_batch_file(batch_file_path)
            print(f"Batch file uploaded. File ID: {file_id}")
            batch_id = create_batch_job(file_id)
            print(f"Batch job ID: {batch_id}")
    else:
        # Count tokens and create batch file
        total_tokens = create_batch_file(
            input_directory,
            batch_file_path,
            model=model,
            reasoning=reasoning,
        )

        # Confirm before proceeding
        proceed = input(f"Total tokens required: {total_tokens}. Do you want to proceed? (yes/no): ").strip().lower()
        if proceed != "yes":
            print("Batch job creation aborted.")
            return

        # Upload batch file and create batch job
        file_id = upload_batch_file(batch_file_path)
        print(f"Batch file uploaded. File ID: {file_id}")
        batch_id = create_batch_job(file_id)
        print(f"Batch job ID: {batch_id}")

    print("Batch job created. Monitoring status...")

    while True:
        response = client.batches.retrieve(batch_id)
        status = response.status
        output_file_id = response.output_file_id
        error_file_id = response.error_file_id
        print(f"Current status: {status}")
        print(f"Errors: {response.errors}")

        if status == "completed":
            if output_file_id:
                download_file(output_file_id, SCRIPT_DIR / f"batch-output-{model}-{name}.jsonl")
                print("Batch processing completed. Results saved.")
            if error_file_id:
                download_file(error_file_id, SCRIPT_DIR / f"batch-errors-{model}-{name}.jsonl")
                print("Some errors occurred. Details saved.")
            break
        elif status in {"failed", "cancelled", "expired"}:
            print(error_file_id)
            if error_file_id:
                download_file(error_file_id, SCRIPT_DIR / f"batch-errors-{model}-{name}.jsonl")
                print(f"Some errors occurred. Details saved to 'batch-errors-{model}-{name}.jsonl'.")
            print(f"Batch job {status}.")
            break
        time.sleep(60)  # Wait for 1 minute before checking again

if __name__ == "__main__":
    main()
