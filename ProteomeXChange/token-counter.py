import os
import tiktoken

def count_tokens(messages, model="gpt-4o-mini"):
    """Estimate the number of tokens used by the messages for the given model."""
    encoding = tiktoken.encoding_for_model(model)
    total_tokens = 0
    for message in messages:
        total_tokens += len(encoding.encode(message["role"])) + len(encoding.encode(message["content"]))
    return total_tokens

def build_messages_for_version(study_data: str, version: str):
    """Return (system_message, user_message) tuple for the given prompt version."""
    if version == "v1":
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
            "\t6. Should the dataset be considered for inclusion? (Answer \"Yes\" or \"No.\")\n"
            f"{study_data}"
        )
        return system_message, user_message

    if version == "v2":
        system_message = (
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
            "\t6. Should the dataset be considered for inclusion? (Answer \"Yes\" or \"No.\")\n"
            f"{study_data}"
        )
        return system_message, user_message

    # default to v3
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
        "\n"
        "Exclusion Criteria: \n"
        "4. Exclusively non-clinical datasets, such as in vitro and  animal studies, are excluded."
        "\n"
        "Let's think step by step and answer the following:\n"
        "\t1.\tIs the dataset about ovarian cancer? Justify.\n"
        "\t2.\tDoes the dataset contain healthy controls? If so, Justify.\n"
        "\t2a.\tIf the explicit sample compositions are provided, does the dataset contain at least 20% healthy controls?\n Justify.\n"
        "\t2b.\tIs the dataset about diagnostic biomarkers for diagnostic assays? Justify. \n"
        "\t3.\tDoes the dataset include proteomics? Justify.\n"
        "\t4.\tIs the dataset non-clinical? Justify.\n"
        "\t5. Based on the inclusion/exclusion criteria, justify whether the dataset should be considered for inclusion.\n"
        "\t6. Should the dataset be considered for inclusion? (Answer \"Yes\" or \"No.\")\n"
        f"{study_data}"
    )
    return system_message, user_message


def calculate_tokens_by_version(input_directory, versions=("v1", "v2", "v3"), tokenizer_model="gpt-4o-mini"):
    """Calculate total tokens across all studies for each prompt version.

    Returns:
      totals: dict mapping version -> total token count across all studies
      per_study: dict mapping version -> list of (study_id, token_count)
    """
    totals = {v: 0 for v in versions}
    per_study = {v: [] for v in versions}

    # Get all .txt files from the input directory
    txt_files = [f for f in os.listdir(input_directory) if f.endswith('.txt')]

    for txt_file in sorted(txt_files):
        file_path = os.path.join(input_directory, txt_file)
        study_id = txt_file.replace('.txt', '')

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Extract title and description from the file
            title = "No title"
            description = "No description"

            for line in lines:
                line = line.strip()
                if line.startswith("Title:"):
                    title = line[6:].strip()
                elif line.startswith("Description:"):
                    description = line[12:].strip()

            # Skip studies with no meaningful data
            if title == "No title available" and description == "No description available":
                continue

        except Exception as e:
            print(f"Error reading {txt_file}: {e}")
            continue

        study_data = f"Title: {title}\nDescription: {description}"

        for version in versions:
            system_message, user_message = build_messages_for_version(study_data, version)
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
            task_tokens = count_tokens(messages, model=tokenizer_model)
            # Practical safeguard for extreme outliers
            if task_tokens > 128_000:
                print(f"{study_id} ({version}): {task_tokens} tokens. Skipping (exceeds token limit).")
                continue
            totals[version] += task_tokens
            per_study[version].append((study_id, task_tokens))

    return totals, per_study


def main():
    input_directory = "proteomexchange_7_24_25"  # Directory containing .txt files
    # Tokenizer model used for counting tokens. Use a tiktoken-supported name.
    tokenizer_model = "gpt-4o-mini"  # falls back to cl100k_base if unavailable

    versions = ("v1", "v2", "v3")
    totals, per_study = calculate_tokens_by_version(
        input_directory=input_directory,
        versions=versions,
        tokenizer_model=tokenizer_model,
    )

    print("\nTotal estimated tokens needed per prompt version (summed over all studies):")
    for v in versions:
        print(f"  - {v}: {totals[v]} tokens")

    # Optional: show a few sample studies per version (comment out if noisy)
    for v in versions:
        if per_study[v]:
            print(f"\nFirst 5 studies for {v} (study_id: tokens):")
            for sid, tk in per_study[v][:5]:
                print(f"  {sid}: {tk}")

if __name__ == "__main__":
    main()