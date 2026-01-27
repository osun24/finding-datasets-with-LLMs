import json
import os
import pandas as pd
import re
from datetime import datetime
from typing import Optional
from glob import glob
from pathlib import Path
from typing import List

# --- MATCH ---
actual = [
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

BASE_FILE_PATHS = [
    "out-gpt-4.1-mini-edoc-proteom-paper-v1.jsonl",
    "out-gpt-4.1-mini-edoc-proteom-paper-v2.jsonl",
    "out-gpt-4.1-mini-edoc-proteom-paper-v3.jsonl",
    "out-gpt-4o-mini-edoc-proteom-paper-v1.jsonl",
    "out-gpt-4o-mini-edoc-proteom-paper-v2.jsonl",
    "out-gpt-4o-mini-edoc-proteom-paper-v3.jsonl",
    "out-gpt-5-mini-edoc-proteom-paper-v1-seed42.jsonl",
    "out-gpt-5-mini-edoc-proteom-paper-v2-seed42.jsonl",
    "out-gpt-5-mini-edoc-proteom-paper-v3-seed42.jsonl",
    "out-o4-mini-edoc-proteom-paper-v1-seed42.jsonl",
    "out-o4-mini-edoc-proteom-paper-v2-seed42.jsonl",
    "out-o4-mini-edoc-proteom-paper-v3-seed42.jsonl",
    "batch-output-gpt-5-mini-edoc-proteom-paper-v1-seed42.jsonl",
    "batch-output-gpt-5-mini-edoc-proteom-paper-v2-seed42.jsonl",
    "batch-output-gpt-5-mini-edoc-proteom-paper-v3-seed42.jsonl",
    "batch-output-o4-mini-edoc-proteom-paper-v1-seed42.jsonl",
    "batch-output-o4-mini-edoc-proteom-paper-v2-seed42.jsonl",
    "batch-output-o4-mini-edoc-proteom-paper-v3-seed42.jsonl",
]

REPLICATION_PATTERNS = [
    "out-gpt-5-mini-edoc-proteom-v3-*.jsonl",
    "out-o4-mini-edoc-proteom-v3-*.jsonl",
    
]


def discover_file_paths() -> List[str]:
    """Gather all batch output files, including replication seeds."""
    discovered = set()
    for path in BASE_FILE_PATHS:
        if Path(path).exists():
            discovered.add(path)
    for pattern in REPLICATION_PATTERNS:
        for path in glob(pattern):
            if Path(path).is_file():
                discovered.add(path)
    return sorted(discovered)


file_paths = discover_file_paths()

MODEL_ALIASES = sorted(
    [
        ("gpt-5-mini", "GPT-5-mini"),
        ("gpt-4o-mini", "GPT-4o-mini"),
        ("4o-mini", "GPT-4o-mini"),
        ("o4-mini", "o4-mini"),
        ("gpt-4.1-mini", "GPT-4.1-mini"),
        ("4.1mini", "GPT-4.1-mini"),
        ("gpt-4.1-nano", "GPT-4.1-nano"),
        ("4.1nano", "GPT-4.1-nano"),
    ],
    key=lambda item: len(item[0]),
    reverse=True,
)

if not file_paths:
    print("⚠️ No batch output files found. Update the path patterns or download batch results first.")

# Initialize results storage
def load_or_create_results():
    """Load existing results or create new structure"""
    results_file = "model_performance_results.json"
    try:
        with open(results_file, 'r') as f:
            existing_results = json.load(f)
        print(f"📂 Loaded existing results from {results_file}")
        return existing_results
    except FileNotFoundError:
        print(f"📝 Creating new results file: {results_file}")
        return {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "description": "AI model performance evaluation for clinical dataset screening",
                "total_actual_studies": 0,
                "evaluation_criteria": "NSCLC clinical dataset inclusion/exclusion",
                "last_updated": datetime.now().isoformat()
            },
            "results": []
        }

all_results = load_or_create_results()

def parse_batch_output(file_path, csv_output):
    rows = []
    for line in open(file_path, 'r'):
        data = json.loads(line)
        content = data['response']['body']['choices'][0]['message']['content']
        model = data['response']['body']['model']
        content = content.replace("""""Yes"" or ""No.""""", '').replace("(Answer ""Yes"" or ""No."")", '').replace("""(Answer "Yes" or "No.")""", '').replace("Yes or No", "").replace("'Yes' or 'No'", "").replace("Yes/No", "")
        answers = {}
        
        # Handle multiple formatting patterns
        sections = re.split(r'\n(?=#{1,3}\s*\d+\.)', content.strip())  # ### 1.
        if len(sections) < 2:
            sections = re.split(r'\n(?=\*\*\d+\.)', content.strip())    # **1.
        if len(sections) < 2:
            sections = re.split(r'\n(?=\d+\.)', content.strip())        # 1.
        if len(sections) < 2:
            sections = re.split(r'\n(?=\d+\))', content.strip())        # 1)
        if len(sections) < 2:
            sections = re.split(r'(?i)\n(?=Step\s+\d+:)', content.strip())  # Step 1:

        for section in sections:
            patterns = [
                r'#{1,3}\s*(\d+)\.\s*(.+)',          # ### 1.
                r'\*\*(\d+)\.\s*(.+)',               # **1.
                r'^(\d+)\.\s*(.+)',                  # 1.
                r'^(\d+)\)\s*(.+)',                  # 1)
                r'(?i)^Step\s+(\d+):\s*(.+)',        # Step 1:
            ]
            for pattern in patterns:
                match = re.match(pattern, section.strip(), re.DOTALL)
                if match:
                    num, ans = match.groups()
                    cleaned_answer = re.sub(r'\*\*([^*]+)\*\*', r'\1', ans)  # remove bold
                    cleaned_answer = re.sub(r'---+', '', cleaned_answer)     # remove separators
                    cleaned_answer = re.sub(r'\s+', ' ', cleaned_answer.strip())
                    
                    cleaned_answer = cleaned_answer.replace("""""Yes"" or ""No.""""", '').replace("(Answer ""Yes"" or ""No."")", '')

                    cleaned_answer = cleaned_answer.replace("""Final summary: - Survival data: No - Majority Stage I: Yes (>50% patients appear Stage I) - ACT status: No - Inclusion: No, because survival and ACT data are essential for building survival ML models for ACT treatment recommendations.""", "")
                    answers[f"Q{num}"] = cleaned_answer
                    break
        
        cid = data.get('custom_id', '')
        row = {"custom_id": cid, "error": data.get('error', ''), "model": model}
        row.update(answers)
        try:
            row['in_actual'] = cid in actual
        except NameError:
            row['in_actual'] = False
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_output, index=False)
    print(f"Parsed batch output saved to {csv_output}")
    return df

def match_files(user_input, actual):
    matched = []
    false_positives = []  # User input but not in actual
    false_negatives = []  # In actual but not in user input
    
    for item in user_input:
        if item in actual:
            matched.append(item)
        else:
            false_positives.append(item)
    
    for item in actual:
        if item not in user_input:
            false_negatives.append(item)
    
    return matched, false_positives, false_negatives  

# calculate sensitivity, specificity, and accuracy
def calculate_metrics(matched, false_positives, false_negatives, total_files_evaluated):
    true_positive = len(matched)
    false_positive = len(false_positives)
    false_negative = len(false_negatives)
    
    # True Negatives: files that were correctly identified as NOT meeting criteria
    # This is total files evaluated minus all the positive cases (TP + FP + FN)
    true_negative = total_files_evaluated - (true_positive + false_positive + false_negative)
    
    print(f"True Positive: {true_positive}")
    print(f"False Positive: {false_positive}")
    print(f"False Negative: {false_negative}")
    print(f"True Negative: {true_negative}")
    print(f"Total files evaluated: {total_files_evaluated}")
    
    sensitivity = true_positive / (true_positive + false_negative) 
    specificity = true_negative / (true_negative + false_positive)
    
    # Correct accuracy calculation: (TP + TN) / (TP + FP + FN + TN)
    accuracy = (true_positive +true_negative) / total_files_evaluated
    
    # Precision and F1 score
    precision = true_positive / (true_positive + false_positive) 
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    
    print(f"Sensitivity (Recall): {sensitivity:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1 Score: {f1_score:.3f}")
    
    # Return all metrics as a dictionary for JSON storage
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "total_files_evaluated": total_files_evaluated,
        "sensitivity": round(sensitivity, 3),
        "specificity": round(specificity, 3),
        "precision": round(precision, 3),
        "accuracy": round(accuracy, 3),
        "f1_score": round(f1_score, 3)
    }

def extract_model_metadata(file_path):
    """Extract model, prompt version, and trial identifier from a batch output file."""
    filename = Path(file_path).name

    model = "Unknown"
    for pattern, canonical in MODEL_ALIASES:
        if pattern in filename:
            model = canonical
            break

    version_match = re.search(r'v(\d+)', filename)
    version = f"v{version_match.group(1)}" if version_match else "v1"

    seed_match = re.search(r'seed(\d+)', filename)
    if seed_match:
        trial = f"seed{seed_match.group(1)}"
    else:
        alt_seed_match = re.search(r'-v\d+-([0-9]+)(?=\.jsonl$)', filename)
        trial = f"seed{alt_seed_match.group(1)}" if alt_seed_match else None

    return {
        "model": model,
        "version": version,
        "trial": trial,
    }

def save_results_to_json(all_results, filename="model_performance_results.json"):
    """Save all results to JSON file"""
    with open(filename, 'w') as f:
        json.dump(all_results, indent=2, fp=f)
    print(f"\n📊 Results saved to {filename}")
    return filename

def show_detailed_responses(df, false_positives, false_negatives, first_col):
    """Show detailed AI responses for False Positives and False Negatives"""
    
    if false_positives:
        print("\n" + "="*60)
        print("FALSE POSITIVES - Detailed Responses:")
        print("="*60)
        for fp_file in false_positives:
            fp_row = df[df[first_col] == fp_file]
            if not fp_row.empty:
                print(f"\n📁 {fp_file}")
                print("-" * 50)
                # Show all Q columns
                for col in df.columns:
                    if col.startswith('Q') and not fp_row[col].isna().iloc[0]:
                        q_num = col[1:]  # Remove 'Q' prefix
                        response = fp_row[col].iloc[0]
                        print(f"Q{q_num}: {response}")
                print()
    
    if false_negatives:
        print("\n" + "="*60)
        print("FALSE NEGATIVES - Detailed Responses:")
        print("="*60)
        for fn_file in false_negatives:
            fn_row = df[df[first_col] == fn_file]
            if not fn_row.empty:
                print(f"\n📁 {fn_file}")
                print("-" * 50)
                # Show all Q columns
                for col in df.columns:
                    if col.startswith('Q') and not fn_row[col].isna().iloc[0]:
                        q_num = col[1:]  # Remove 'Q' prefix
                        response = fn_row[col].iloc[0]
                        print(f"Q{q_num}: {response}")
                print()
            else:
                print(f"\n📁 {fn_file} - NO AI RESPONSE FOUND (not in evaluated dataset)")
                print("-" * 50)
    
# Set the total actual studies count
all_results["metadata"]["total_actual_studies"] = len(actual)

for file_path in file_paths:
    csv_output = file_path.replace('.jsonl', '.csv')
    metadata = extract_model_metadata(file_path)
    model = metadata["model"]
    version = metadata["version"]
    trial = metadata["trial"]

    label = f"{model} {version}".strip()
    if trial:
        label = f"{label} [{trial}]"

    print(f"Processing {file_path} → {label}...")
    df = parse_batch_output(file_path, csv_output)

    # Determine the last question column (columns named like 'Q1', 'Q2', ...).
    # We can't assume it's the final DataFrame column because we add fields
    # like `in_actual` which may come after the Q columns.
    q_cols = [c for c in df.columns if c.startswith('Q')]
    if q_cols:
        # Sort question columns by number if possible, fallback to lexical order
        def _qnum(col):
            try:
                return int(col[1:])
            except Exception:
                return 0
        q_cols_sorted = sorted(q_cols, key=_qnum)
        last_col = q_cols_sorted[-1]
    else:
        # No Q-columns found; fall back to last column in DataFrame
        last_col = df.columns[-1]

    # Sort by the last question column in ascending order
    df_sorted = df.sort_values(by=last_col, ascending=True)

    # Filter rows where the last column contains "yes" (case-insensitive)
    filtered = df_sorted[df_sorted[last_col].str.contains("yes", case=False, na=False)]
    
    print(f"Identified studies from {last_col}: {len(filtered)}")
    
    # print identified studies by printing first column in filtered
    first_col = df.columns[0]
    identified_studies = filtered[first_col].tolist()
    
    # Run through matching process
    user_input = identified_studies
    if not user_input:
        print("No identified studies found.")
        continue
    
    matched, false_positives, false_negatives = match_files(user_input, actual)
    
    if matched:
        print("> Matched files:")
        for item in matched:
            print("  " + item)
    else:
        print("No files matched.")
            
    if false_positives:
        print("\n> False Positives (FP):")
        for item in false_positives:
            print(f"  {item}")
    
    if false_negatives:
        print("\n> False Negatives (FN):")
        for item in false_negatives:
            print(f"  {item}")
    
    # Calculate and display metrics
    total_files_evaluated = len(df)  # Total number of files/studies evaluated by the AI
    metrics = calculate_metrics(matched, false_positives, false_negatives, total_files_evaluated)
    
    # Store results for this model/version
    result_entry = {
        "model": model,
        "version": version,
        "trial": trial,
        "file_path": file_path,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "details": {
            "identified_studies": identified_studies,
            "matched_studies": matched,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "total_studies_in_dataset": len(actual),
            "trial": trial,
        }
    }
    
    # Check if this exact model/version combination already exists
    existing_index = None
    for i, existing_result in enumerate(all_results["results"]):
        if (
            existing_result.get("model") == model
            and existing_result.get("version") == version
            and existing_result.get("file_path") == file_path
            and (existing_result.get("trial") or "") == (trial or "")
        ):
            existing_index = i
            break
    
    if existing_index is not None:
        print(f"🔄 Updating existing entry for {model} {version}")
        all_results["results"][existing_index] = result_entry
    else:
        print(f"➕ Adding new entry for {model} {version}")
        all_results["results"].append(result_entry)
    
    # Update metadata
    all_results["metadata"]["last_updated"] = datetime.now().isoformat()
    
    # Show detailed responses for FP and FN
    show_detailed_responses(df, false_positives, false_negatives, first_col)

    print("\n" + "="*40 + "\n")

# Save all results to JSON
save_results_to_json(all_results)

# Also create a simplified DataFrame for quick analysis
summary_data = []
for result in all_results["results"]:
    row = {
        "Model": result.get("model", "Unknown"),
        "Version": result.get("version", ""),
        "Trial": result.get("trial", ""),
        "File": result.get("file_path", ""),
        **result["metrics"]  # Unpack all metrics
    }
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)

if not summary_df.empty:
    sort_cols = [col for col in ["Model", "Version", "Trial"] if col in summary_df.columns]
    summary_df = summary_df.sort_values(by=sort_cols).reset_index(drop=True)

summary_csv = "model_performance_summary.csv"
summary_df.to_csv(summary_csv, index=False)
print(f"📈 Summary table saved to {summary_csv}")

# Print final summary
print("\n" + "="*60)
print("FINAL SUMMARY:")
print("="*60)
for result in all_results["results"]:
    metrics = result["metrics"]
    print(f"{result['model']} {result['version']}: "
          f"Acc={metrics['accuracy']:.3f}, "
          f"Sen={metrics['sensitivity']:.3f}, "
          f"Spe={metrics['specificity']:.3f}, "
          f"F1={metrics['f1_score']:.3f}")
