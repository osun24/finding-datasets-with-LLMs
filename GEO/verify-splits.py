from pathlib import Path

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
    "samples_table_GSE102287.csv"
]

PROMPT_DIR = Path(__file__).parent / "prompt-set"
TEST_DIR = Path(__file__).parent / "test-set"

def verify_splits():
    positive_set = set(POSITIVE_FILES)
    
    # Get files from prompt and test sets
    prompt_files = {f.name for f in PROMPT_DIR.glob("samples_table_*.csv")}
    test_files = {f.name for f in TEST_DIR.glob("samples_table_*.csv")}
    
    # Count how many positive files are in each set
    prompt_positive = positive_set & prompt_files
    test_positive = positive_set & test_files
    
    # Check for missing or misplaced files
    missing = positive_set - (prompt_files | test_files)
    
    print("=== Split Verification ===\n")
    print(f"Total positive files defined: {len(POSITIVE_FILES)}")
    print(f"Prompt set: {len(prompt_positive)} positive files found")
    print(f"Test set: {len(test_positive)} positive files found")
    print(f"Total found: {len(prompt_positive) + len(test_positive)}")
    
    if missing:
        print(f"\n⚠️  Missing positive files: {sorted(missing)}")
    else:
        print("\n✓ All positive files accounted for")
    
    # List which files are where
    print(f"\nPrompt set positive files: {sorted(prompt_positive)}")
    print(f"\nTest set positive files: {sorted(test_positive)}")

if __name__ == "__main__":
    verify_splits()
