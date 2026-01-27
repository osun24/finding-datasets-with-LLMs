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
            
            # SYSTEM for EARLY STAGE
            """You are an expert in analyzing clinical datasets for inclusion in a meta-analysis on recurrence after surgical resection in early-stage (< Stage IIA) NSCLC (non-small cell lung cancer). Evaluate the dataset description and data based on the following criteria:

            1. **Stage Focus:**
            - Majority of patients are diagnosed with Stage IA or IB NSCLC.
            - If stage information is missing from data fields, infer from the dataset description.

            2. **Recurrence Data:**
            - Includes recurrence status, recurrence-free survival, or time to recurrence.

            3. **Data Clarity:**
            - Parameter names and definitions are clear and standardized."""
            
            # USER for EARLY STAGE
            """The dataset description and data are provided below. Please evaluate and respond to the following:

        1. **Inclusion Decision:** Yes or No.

        2. **Justification:** 1-2 sentences indicating which criteria are met or unmet, including stage distribution if applicable.

        3. **Relevant Parameters:** If Yes, list the parameter names that satisfy the inclusion criteria.

                ---
                {csv_text}"""
                
            # SYSTEM for adj chemo
            """You are an expert in analyzing clinical datasets for inclusion in a meta-analysis "
                "on adjuvant chemotherapy (ACT) in NSCLC (non-small cell lung cancer). Your task "
                "is to determine whether the provided dataset includes relevant adjuvant chemotherapy "
                "(ACT) data for inclusion in the meta-dataset."""
                
            # USER for adj chemo
            """"The dataset below contains clinical and treatment information for NSCLC patients. "
                "You are provided with the first 10 patients and their covariates. Your task is to answer the following:\n"
                "\t1.\tShould this dataset be considered for inclusion in the meta-dataset? (Answer \"Yes\" or \"No.\")\n"
                "\t2.\tBriefly justify your answer in 1 sentence.\n"
                "\t3.\tIf your answer is \"Yes\", provide the parameter name.\n\n"
                f"{csv_text}"""
            
            # v1
            """system_message = (
                "You are an expert in analyzing clinical datasets for inclusion in a meta-analysis building ML survival models"
                "for adjuvant chemotherapy (ACT) treatment recommendation in NSCLC (non-small cell lung cancer). Your task "
                "is to determine whether the provided dataset includes survival status and time data and either of the following: adjuvant chemotherapy "
                "(ACT) status data OR includes a majority of patients (>50%) with Stage I NSCLC for inclusion in the meta-analysis."
            )
            user_message = (
                "The dataset below contains clinical information for NSCLC patients. Let's think step by step and answer the following:\n"
                "\t1.\tDoes the dataset include survival status and time data (e.g., OS, RFS, PFS)? If so, please specify.\n"
                "\t2.\tDoes it include a majority of patients (>50%) with Stage I NSCLC? If so, please specify.\n"
                "\t3.\tDoes it include adjuvant chemotherapy (ACT) status data? If so, please specify.\n"
                "\t4.\tBriefly justify if this dataset should be considered for inclusion in the meta-analysis in 1 sentence.\n"
                "\t5.\tShould this dataset be considered for inclusion in the meta-analysis? (Answer \"Yes\" or \"No.\")\n"
                f"{csv_text}"
            ) """
            
            # v2
            """system_message = (
                "You are an oncology expert evaluating clinical datasets based on inclusion/exclusion criteria. Respond concisely and clearly, returning answers in the same numbered list format as the questions."
            )
            user_message = (
                "A clinical non-small cell lung cancer (NSCLC) dataset, its title, and description are included below. Inclusion Criteria: \n"
                "1. The dataset must include patients' survival status and time variables (e.g., OS, RFS, PFS).\n And one or both of the following:\n"
                "2a. The dataset must include a majority of patients (>50%) with Stage I NSCLC.\n"
                "2b. The dataset must include adjuvant chemotherapy (ACT) status data.\n"
                "Let's think step by step and answer the following:\n"
                "\t1.\tDoes the dataset include survival status variable and survival time variable (e.g., OS, RFS, PFS)? If so, please specify the variables.\n"
                "\t2.\tDoes it include a majority of patients (>50%) with Stage I NSCLC? If so, please explain.\n"
                "\t3.\tDoes it include adjuvant chemotherapy (ACT) status data? If so, please explain.\n"
                "\t4.\tBriefly justify if this dataset should be considered for inclusion in the meta-analysis in 1 sentence.\n"
                "\t5.\tShould this dataset be considered for inclusion in the meta-analysis? (Answer \"Yes\" or \"No.\")\n"
                f"{csv_text}"
            )"""
            
            # v3
            system_message = (
                "You are an oncology expert evaluating clinical datasets based on inclusion/exclusion criteria. Respond concisely and clearly, returning answers in the same numbered list format as the questions."
            )
            user_message = (
                "A clinical non-small cell lung cancer (NSCLC) dataset, its title, and description are included below. Inclusion Criteria: \n"
                "1. The dataset must include explicit, patient-specific survival status and time variables (e.g., OS, RFS, PFS). Mere mention of survival in title or description is inadequate to meet this criterion.\n And one **OR** both of the following:\n"
                "2a. The dataset must include a majority of patients (>50%) with Stage I NSCLC.\n"
                "2b. The dataset must include adjuvant chemotherapy (ACT) status data.\n"
                "Let's think step by step and answer the following:\n"
                "\t1.\tDoes the dataset include survival status variable and survival time variable (e.g., OS, RFS, PFS)? If so, please specify the variables.\n"
                "\t2.\tDoes it include a majority of patients (>50%) with Stage I NSCLC? If so, please explain.\n"
                "\t3.\tDoes it include adjuvant chemotherapy (ACT) status data? If so, please explain.\n"
                "\t4.\tBriefly justify if this dataset should be considered for inclusion in the meta-analysis in 1 sentence.\n"
                "\t5.\tShould this dataset be considered for inclusion in the meta-analysis? (Answer \"Yes\" or \"No.\")\n"
                f"{csv_text}"
            )
            
            print(f"System message for {filename}:\n{system_message}\n")
            print(f"User message for {filename}:\n{user_message}\n")


            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]

            task = {
                "custom_id": filename,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": messages,
                    "temperature": (1 if reasoning else 0.0),  # Set temperature to 0 for non-reasoning tasks
                }
            }
            
            if model == "o4-mini":
                task = {
                "custom_id": filename,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": messages,
                    "temperature": 1,  # Set temperature to 0 for non-reasoning tasks
                    "seed": seed
                }
            }

            if model == "gpt-5-mini" or model == "gpt-5":
                task = {
                    "custom_id": filename,
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
    input_directory = "case-1"
    # gpt-5-mini; o4-mini; gpt-4.1-mini
    model = "gpt-5-mini"  # Specify the model to use
    name = "chemo-JSM-v3"  # Specify the name for the batch job
    batch_file_path = f"batch_requests_{model}-{name}.jsonl"

    replication = True
    if replication:
        for seed in [47, 48, 49, 50, 51]: #47, 48, 49, 50, 51 #43, 44, 45, 46
            name = f"chemo-v3-{seed}"  # Specify the name for the batch job
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