import os
import pandas as pd
import json
import openai
import tiktoken
import time
from dotenv import load_dotenv

load_dotenv()
key = os.getenv('OPENAI')

client = openai.Client(api_key=key, )

print(client.batches.list())

SCHEMA_CONFIRMATION_PLACEHOLDER = "{CSV_DATASET_CONTENT}"

# Variables have `evidence` fields on top of the `justification` field
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

PROMPT_VERSION = "v2"
def build_prompt(csv_text):
    # v1 - new, following JSON schema and template from Figure 1
    """system_message = (
        "You are an oncology expert evaluating clinical datasets based on inclusion/exclusion criteria. Respond concisely and clearly, returning answers in the given structure."
    )
    user_message = (
        "A clinical non-small cell lung cancer (NSCLC) dataset, its title, and description are included below.\n\n"
        
        "Inclusion Criteria:\n"
        "1. Presence of patient-specific survival time and status (e.g., overall survival or progression-free survival).\n"
        "2. At least one of the following: sample composition of at least 50% stage I patients; or explicit adjuvant chemotherapy annotation.\n\n"
        
        "Let's think step by step and answer the following questions, matching the JSON schema keys:\n"
        "\t1.\tq1_survival_data: Does the dataset include survival status and time data (e.g., OS, RFS, PFS)? If so, specify.\n"
        "\t2.\tq2_stage_i_or_act: Does it include a majority of patients (>50%) with Stage I NSCLC or explicit adjuvant chemotherapy (ACT) annotation? If so, specify.\n"
        "\t3.\tq3_inclusion_justification: Based on the criteria, justify whether the dataset should be considered for inclusion.\n"
        "\t4.\tq4_include_in_meta_analysis: Should this dataset be considered for inclusion in the meta-analysis? Answer \"Yes\" or \"No\".\n\n"
        f"{csv_text}"
    )"""
    
    #v2 — see prompt-versions.md for details on changes
    system_message = (
        "You are an oncology expert evaluating clinical datasets based on inclusion/exclusion criteria. Respond concisely and clearly, returning answers in the given structure."
    )
    user_message = (
        "A clinical non-small cell lung cancer (NSCLC) dataset, its title, and description are included below.\n\n"
        
        "Inclusion Criteria:\n"
        "1. Presence of patient-specific variables for survival time and status (e.g., overall survival status and overall survival survival time; or progression-free survival status and progression-free survival time).\n"
        "2. At least one of the following: total-study sample composition of at least 50% stage I patients; or explicit adjuvant chemotherapy annotation.\n\n"
        
        "Let's think step by step and answer the following questions, matching the JSON schema keys:\n"
        "\t1.\tq1_survival_data: Does the dataset include patient-specific **variables** for survival status and survival time (e.g., OS, RFS, PFS)? If so, specify the variables.\n"
        "\t2.\tq2_stage_i_or_act: Does the dataset, as a whole, include a majority of patients (>50%) with Stage I NSCLC or explicit adjuvant chemotherapy (ACT) annotation? If so, specify.\n"
        "\t3.\tq3_inclusion_justification: Based on the criteria, justify whether the dataset should be considered for inclusion.\n"
        "\t4.\tq4_include_in_meta_analysis: Should this dataset be considered for inclusion in the meta-analysis? Answer \"Yes\" or \"No\".\n\n"
        f"{csv_text}"
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
    print(json.dumps(GEO_RESPONSE_FORMAT, indent=2))
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

def create_batch_file(input_directory, batch_file_path, model="gpt-4.1-mini", reasoning =False, seed = 42):
    tasks = []
    total_tokens = 0

    for filename in os.listdir(input_directory):
        if filename.endswith(".csv"):
            csv_path = os.path.join(input_directory, filename)
            
            # Read the original file to extract comments
            with open(csv_path, 'r', encoding='utf-8') as f:
                original_lines = f.readlines()
            
            # Extract comment lines (lines starting with #)
            comment_lines = [line.strip() for line in original_lines if line.strip().startswith('#')]
            
            # Read CSV data (ignoring comments)
            df = pd.read_csv(csv_path, comment='#', low_memory=False)
            subset = df

            # Clean and escape .CSV content
            csv_data = json.dumps(subset.to_csv(index=False))
            
            # Add comments back to the top of csv_text
            if comment_lines:
                comments_text = '\n'.join(comment_lines)
                csv_text = f"{comments_text}\n\n{csv_data}"
            else:
                print("No comments found in the file!")
                csv_text = csv_data
            
            system_message, user_message = build_prompt(csv_text)
            
            print(f"System message for {filename}:\n{system_message}\n")
            print(f"User message for {filename}:\n{user_message}\n")


            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]

            body = {
                "model": model,
                "messages": messages,
                "response_format": GEO_RESPONSE_FORMAT,
                "temperature": (1 if reasoning else 0.0),
            }

            task = {
                "custom_id": filename,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            
            if model == "o4-mini":
                body = {
                    "model": model,
                    "messages": messages,
                    "response_format": GEO_RESPONSE_FORMAT,
                    "temperature": 1,
                    "seed": seed,
                }
                task = {
                    "custom_id": filename,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }

            if model == "gpt-5-mini" or model == "gpt-5":
                body = {
                    "model": model,
                    "messages": messages,
                    "response_format": GEO_RESPONSE_FORMAT,
                    "top_p": 1, # top_p not supported for o4-mini nor gpt-5-mini other than default?
                    "frequency_penalty": 0, # default
                    "presence_penalty": 0, # default
                    "n": 1,  # default
                    "seed": seed,
                }
                task = {
                    "custom_id": filename,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
            
            task_tokens = count_tokens(messages)

            if task_tokens > 128000:
                print(f"{filename}: {task_tokens} tokens. Skipping because it exceeds the token limit.")
                continue
            tasks.append(task)
            total_tokens += task_tokens
            print(f"{filename}: {task_tokens} tokens")

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
    input_directory = split_name

    # gpt-5-mini; o4-mini; gpt-4.1-mini; gpt-4o-mini
    model = "o4-mini"  # Specify the model to use
    name = f"chemo-may13-{PROMPT_VERSION}-{split_name}"  # Specify the name for the batch job
    batch_file_path = f"batch_requests_{model}-{name}.jsonl"

    verify_schema_matches_prompt()

    replication = True
    if replication:
        for seed in [43, 44, 45, 46, 47, 48, 49, 50, 51]: #47, 48, 49, 50, 51 #43, 44, 45, 46
            name = f"chemo-may13-{PROMPT_VERSION}-{split_name}-{seed}"  # Specify the name for the batch job
            batch_file_path = f"batch_requests_{model}-{name}.jsonl"
            
            total_tokens = create_batch_file(input_directory, batch_file_path, model=model, seed=seed)

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
        total_tokens = create_batch_file(input_directory, batch_file_path, model=model, reasoning = False)

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
                download_file(output_file_id, f"batch-output-{model}-{name}.jsonl")
                print("Batch processing completed. Results saved.")
            if error_file_id:
                download_file(error_file_id, f"batch-errors-{model}-{name}.jsonl")
                print("Some errors occurred. Details saved.")
            break
        elif status in {"failed", "cancelled", "expired"}:
            print(error_file_id)
            if error_file_id:
                download_file(error_file_id, f"batch-errors-{model}-{name}.jsonl")
                print(f"Some errors occurred. Details saved to 'batch-errors-{model}-{name}.jsonl'.")
            print(f"Batch job {status}.")
            break
        time.sleep(60)  # Wait for 1 minute before checking again

if __name__ == "__main__":
    main()
