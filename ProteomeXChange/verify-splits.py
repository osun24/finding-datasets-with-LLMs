from pathlib import Path

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

PROMPT_DIR = Path(__file__).parent / "prompt-set"
TEST_DIR = Path(__file__).parent / "test-set"

def verify_splits():
    positive_set = set(POSITIVE_FILES)
    
    # Get files from prompt and test sets
    prompt_files = {f.stem for f in PROMPT_DIR.glob("PXD*.txt")}
    test_files = {f.stem for f in TEST_DIR.glob("PXD*.txt")}
    
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
