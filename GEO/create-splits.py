from pathlib import Path
import random
import shutil


CASE_DIR = Path(__file__).parent / "case-1"
PROMPT_DIR = Path(__file__).parent / "prompt-set"
TEST_DIR = Path(__file__).parent / "test-set"
RANDOM_SEED = 42
PROMPT_POSITIVE_COUNT = 4

POSITIVE_FILES = [
    "samples_table_GSE29013.csv",
    "samples_table_GSE14814.csv",
    "samples_table_GSE39279.csv",
    "samples_table_GSE50081.csv",
    "samples_table_GSE42127.csv",
    "samples_table_GSE37745.csv",
    "samples_table_GSE47115.csv",
    "samples_table_GSE16025.csv",
    "samples_table_GSE101929.csv",
    "samples_table_GSE19188.csv",
    "samples_table_GSE42425.csv",
    "samples_table_GSE102287.csv",
]


def copy_split(files, output_dir):
    output_dir.mkdir(exist_ok=True)

    for old_file in output_dir.glob("samples_table_*.csv"):
        old_file.unlink()

    for file_path in files:
        shutil.copy2(file_path, output_dir / file_path.name)


def main():
    rng = random.Random(RANDOM_SEED)

    all_files = sorted(CASE_DIR.glob("samples_table_*.csv"))
    positive_names = set(POSITIVE_FILES)
    all_names = {file_path.name for file_path in all_files}

    missing_positives = sorted(positive_names - all_names)
    if missing_positives:
        raise FileNotFoundError(
            "Positive files missing from case-1: " + ", ".join(missing_positives)
        )

    positives = [CASE_DIR / file_name for file_name in POSITIVE_FILES]
    negatives = [file_path for file_path in all_files if file_path.name not in positive_names]

    rng.shuffle(positives)
    rng.shuffle(negatives)

    prompt_positive = positives[:PROMPT_POSITIVE_COUNT]
    test_positive = positives[PROMPT_POSITIVE_COUNT:]

    prompt_ratio = len(prompt_positive) / len(positives)
    prompt_negative_count = round(len(negatives) * prompt_ratio)
    prompt_negative = negatives[:prompt_negative_count]
    test_negative = negatives[prompt_negative_count:]

    prompt_files = sorted(prompt_positive + prompt_negative)
    test_files = sorted(test_positive + test_negative)

    copy_split(prompt_files, PROMPT_DIR)
    copy_split(test_files, TEST_DIR)

    print(f"Prompt set: {len(prompt_positive)} positive, {len(prompt_negative)} negative")
    print(f"Test set: {len(test_positive)} positive, {len(test_negative)} negative")


if __name__ == "__main__":
    main()
