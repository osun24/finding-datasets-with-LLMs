import os
import pandas as pd
from Bio import Entrez
import time
import re

# Set your email for NCBI Entrez
Entrez.email = ""  # Replace with your email

def fetch_geo_summary(gse_id):
    """Fetch summary information for a GSE ID from NCBI"""
    try:
        # Search for the GSE ID in the GDS database
        handle = Entrez.esearch(db="gds", term=f"{gse_id}[Accession]", retmax=1)
        search_results = Entrez.read(handle)
        handle.close()
        
        if not search_results["IdList"]:
            print(f"No results found for {gse_id}")
            return None, None
            
        # Get summary for the first result
        geo_id = search_results["IdList"][0]
        handle = Entrez.esummary(db="gds", id=geo_id, retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        
        if records:
            record = records[0]
            title = record.get('title', '')
            summary = record.get('summary', '')
            return title, summary
            
    except Exception as e:
        print(f"Error fetching data for {gse_id}: {e}")
        return None, None
    
    return None, None

def fetch_geo_detailed_info(gse_id):
    """Fetch detailed information from GEO using efetch"""
    try:
        # Search for the GSE ID
        handle = Entrez.esearch(db="gds", term=f"{gse_id}[Accession]", retmax=1)
        search_results = Entrez.read(handle)
        handle.close()
        
        if not search_results["IdList"]:
            return None, None
            
        # Fetch detailed record
        geo_id = search_results["IdList"][0]
        handle = Entrez.efetch(db="gds", id=geo_id, rettype="xml")
        metadata = handle.read()
        handle.close()
        
        # Parse title and summary from XML metadata
        title_match = re.search(r'<title>(.*?)</title>', metadata, re.DOTALL)
        summary_match = re.search(r'<summary>(.*?)</summary>', metadata, re.DOTALL)
        
        title = title_match.group(1).strip() if title_match else ""
        summary = summary_match.group(1).strip() if summary_match else ""
        
        # Clean up the text
        title = re.sub(r'\s+', ' ', title)
        summary = re.sub(r'\s+', ' ', summary)
        
        return title, summary
        
    except Exception as e:
        print(f"Error fetching detailed data for {gse_id}: {e}")
        return None, None

def extract_gse_id(filename):
    """Extract GSE ID from filename"""
    # Pattern: samples_table_GSE12345.csv
    match = re.search(r'GSE\d+', filename)
    return match.group(0) if match else None

def replace_first_row_with_metadata(csv_path, title, summary):
    """Replace the first row of a CSV file with title and summary"""
    try:
        # Read the CSV file
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            print(f"Empty file: {csv_path}")
            return False
        
        # Create new first row with title and summary
        new_first_row = f"# TITLE: {title}\n# SUMMARY: {summary}\n"
        
        # Remove the old first row (if it starts with #) and add new metadata
        if lines[0].startswith('#'):
            lines[0] = new_first_row
        else:
            lines.insert(0, new_first_row)
        
        # Write back to file
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print(f"Updated {csv_path}")
        return True
        
    except Exception as e:
        print(f"Error updating {csv_path}: {e}")
        return False

def main():
    input_directory = "outputs-1"
    
    if not os.path.exists(input_directory):
        print(f"Directory {input_directory} does not exist")
        return
    
    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(input_directory) if f.endswith('.csv')]
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    processed = 0
    failed = 0
    
    for filename in csv_files:
        print(f"\nProcessing {filename}...")
        
        # Extract GSE ID from filename
        gse_id = extract_gse_id(filename)
        if not gse_id:
            print(f"Could not extract GSE ID from {filename}")
            failed += 1
            continue
        
        print(f"GSE ID: {gse_id}")
        
        # Fetch title and summary from NCBI
        title, summary = fetch_geo_summary(gse_id)
        
        # If summary method fails, try detailed fetch
        if not title or not summary:
            print(f"Trying detailed fetch for {gse_id}")
            title, summary = fetch_geo_detailed_info(gse_id)
        
        if not title and not summary:
            print(f"Could not fetch metadata for {gse_id}")
            failed += 1
            continue
        
        # Use placeholder if one is missing
        if not title:
            title = f"No title available for {gse_id}"
        if not summary:
            summary = f"No summary available for {gse_id}"
        
        print(f"Title: {title[:100]}...")
        print(f"Summary: {summary[:100]}...")
        
        # Replace first row in CSV file
        csv_path = os.path.join(input_directory, filename)
        if replace_first_row_with_metadata(csv_path, title, summary):
            processed += 1
        else:
            failed += 1
        
        # Add delay to avoid rate limiting
        time.sleep(0.5)
    
    print(f"\nProcessing complete:")
    print(f"Successfully processed: {processed}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    main()
