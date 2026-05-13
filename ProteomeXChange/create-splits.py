from pathlib import Path
import random
import shutil


CASE_DIR = Path(__file__).parent / "proteomexchange_7_24_25"
PROMPT_DIR = Path(__file__).parent / "prompt-set"
TEST_DIR = Path(__file__).parent / "test-set"
RANDOM_SEED = 42
PROMPT_POSITIVE_COUNT = 7

POSITIVE_FILES = [
   "PXD003794",
    "PXD009382",
    "PXD009655",
    "PXD011541",
    "PXD013827",
    "PXD014352",
    "PXD014474",
    "PXD015303",
    "PXD018417",
    "PXD020557",
    "PXD023508",
    "PXD023912",
    "PXD025864",
    "PXD030390",
    "PXD032299",
    "PXD033108",
    "PXD033169",
    "PXD036726",
    "PXD040959",
    "PXD041751",
    "PXD047188",
    "PXD056996"
]


def copy_split(files, output_dir):
    output_dir.mkdir(exist_ok=True)

    for old_file in output_dir.glob("PXD*.txt"):
        old_file.unlink()

    for file_path in files:
        shutil.copy2(file_path, output_dir / file_path.name)


def main():
    rng = random.Random(RANDOM_SEED)

    all_files = sorted(CASE_DIR.glob("PXD*.txt"))
    positive_names = set(POSITIVE_FILES)
    all_names = {file_path.stem for file_path in all_files}

    missing_positives = sorted(positive_names - all_names)
    if missing_positives:
        raise FileNotFoundError(
            "Positive files missing from proteomexchange_7_24_25: " + ", ".join(missing_positives)
        )

    positives = [CASE_DIR / f"{file_name}.txt" for file_name in POSITIVE_FILES]
    negatives = [file_path for file_path in all_files if file_path.stem not in positive_names]

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
