import os
import json
import openai
import tiktoken
import time
from dotenv import load_dotenv

load_dotenv()
key = os.getenv('OPENAI')

client = openai.Client(api_key=key)

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
        
        """# v1
        system_message = (
            "You are an oncology expert evaluating clinical datasets based on inclusion/exclusion criteria. Respond concisely and clearly, returning answers in the exact same numbered list format as the questions. Do not merge different question responses. Do not repeat the questions in your response."
        )
        user_message = (
            "A dataset's title and description are included below. Inclusion Criteria: \n"
            "1. The dataset must be about ovarian cancer.\n"
            "2. The dataset must contain healthy controls.\n"
            "\t2a. If explicit sample compositions are provided, the dataset must contain at least 20% healthy controls.\n"
            "3. The dataset must be on proteomics.\n"
            "Let's think step by step and answer the following:\n"
            "\t1.\tIs the dataset about ovarian cancer? If so, justify.\n"
            "\t2.\tDoes the dataset contain healthy controls? If so, please explain.\n"
            "\t2a.\tIf the explicit sample compositions are provided, does the dataset contain at least 20% healthy controls?\n If so, please explain.\n"
            "\t3.\tDoes it include proteomics? If so, please explain.\n"
            "\t4.\tUsing the inclusion criteria, please explain whether the dataset should be included."
            "\t5. Based on the inclusion/exclusion criteria, justify whether the dataset should be considered for inclusion.\n"
            "\t6. Should the dataset be considered for inclusion? (Answer \"Yes\" or \"No.\")"
            f"{study_data}"
        ) """
        
        # v2
        """system_message = (
            "You are an oncology expert evaluating clinical datasets based on inclusion/exclusion criteria. Respond concisely and clearly, returning answers in the exact same numbered list format as the questions. Do not merge different question responses. Do not repeat the questions in your response."
        )
        user_message = (
            "A dataset's title and description are included below. Inclusion Criteria: \n"
            "1. The dataset must be about ovarian cancer.\n"
            "2. The dataset must contain healthy controls.\n"
            "\t2a. If explicit sample compositions are provided, the dataset must contain at least 20% healthy controls.\n"
            "\t2b. If the dataset is about diagnostic assays, it must include healthy controls as well.\n"
            "3. The dataset must be on proteomics.\n"
            "Let's think step by step and answer the following:\n"
            "\t1.\tIs the dataset about ovarian cancer? If so, justify.\n"
            "\t2.\tDoes the dataset contain healthy controls? If so, please explain.\n"
            "\t2a.\tIf the explicit sample compositions are provided, does the dataset contain at least 20% healthy controls?\n If so, please explain.\n"
            "\t3.\tDoes it include proteomics? If so, please explain.\n"
            "\t4.\tUsing the inclusion criteria, please explain whether the dataset should be included."
            "\t5. Based on the inclusion/exclusion criteria, justify whether the dataset should be considered for inclusion.\n"
            "\t6. Should the dataset be considered for inclusion? (Answer \"Yes\" or \"No.\")"
            f"{study_data}"
        ) """
        
        # v3
        system_message = (
            "You are an oncology expert evaluating clinical datasets based on inclusion/exclusion criteria. Respond concisely and clearly, returning answers in the exact same numbered list format as the questions. Do not merge different question responses. Do not repeat the questions in your response."
        )
        user_message = (
            "A dataset's title and description are included below. Inclusion Criteria: \n"
            "1. The dataset must be about ovarian cancer.\n"
            "2. The dataset must contain healthy controls.\n"
            "\t2a. If explicit sample compositions are provided, the dataset must have at least 20% healthy controls for inclusion.\n"
            "\t2b. If the dataset is about diagnostic assays and biomarkers, consider it as having healthy controls.\n"
            "3. The dataset must be on proteomics.\n"
            
            "Exclusion Criteria: \n"
            "4. Exclusively non-clinical datasets, such as in vitro and  animal studies, are excluded."

            "Let's think step by step and answer the following:\n"
            "\t1.\tIs the dataset about ovarian cancer? Justify.\n"
            "\t2.\tDoes the dataset contain healthy controls? If so, Justify.\n"
            "\t2a.\tIf the explicit sample compositions are provided, does the dataset contain at least 20% healthy controls?\n Justify.\n"
            "\t2b.\tIs the dataset about diagnostic biomarkers for diagnostic assays? Justify. \n"
            "\t3.\tDoes the dataset include proteomics? Justify.\n"
            "\t4.\tIs the dataset non-clinical? Justify."
            "\t5. Based on the inclusion/exclusion criteria, justify whether the dataset should be considered for inclusion.\n"
            "\t6. Should the dataset be considered for inclusion? (Answer \"Yes\" or \"No.\")"
            f"{study_data}"
        )
        
        print(f"System message for {study_id}:\n{system_message}\n")
        print(f"User message for {study_id}:\n{user_message}\n")

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        task = {
            "custom_id": study_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": messages,
                "temperature": (1 if reasoning else 0.0),  # Set temperature to 0 for non-reasoning tasks
                "seed": 42
            }
        }

        if model == "gpt-5-mini" or model == "gpt-5":
           task = {
            "custom_id": study_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": messages,
                "top_p": 1, # top_p not supported for o4-mini nor gpt-5-mini other than default?
                "frequency_penalty": 0, # default
                "presence_penalty": 0, # default
                "n": 1,  # default
                "seed": seed
            }
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
    input_directory = "proteomexchange_7_24_25"  # Directory containing .txt files
    model = "gpt-5-mini"  # Specify the model to use
    name = "edoc-proteom-paper-v3"  # Specify the name for the batch job
    batch_file_path = f"batch_requests_{model}-{name}.jsonl"

    # Count tokens and create batch file
    reasoning = True
    if model == "gpt-5-mini":
        reasoning = True
    
    replication = True
    if replication:
        for seed in [47, 48, 49, 50, 51]: #47, 48, 49, 50, 51 43, 44, 45, 46
            name = f"edoc-proteom-v3-{seed}"  # Specify the name for the batch job
            batch_file_path = f"batch_requests_{model}-{name}.jsonl"
            
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