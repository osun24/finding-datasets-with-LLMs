from Bio import Entrez
import pandas as pd
import time
import re

Entrez.email = ""  # Replace with your email

# USE SUMMARY
"""def fetch_geo_summary(geo_id):
    handle = Entrez.esummary(db="gds", id=geo_id, retmode="xml")
    records = Entrez.read(handle)
    handle.close()
    return records[0] if records else None"""

def save_metadata_to_excel(metadata_list, filename="geo_metadata.xlsx"):
    metadata_df = pd.DataFrame(metadata_list)
    try:
        existing_df = pd.read_excel(filename, engine='openpyxl')
        metadata_df = pd.concat([existing_df, metadata_df], ignore_index=True)
    except FileNotFoundError:
        pass
    metadata_df.to_excel(filename, index=False, engine='openpyxl')
    print(f"Metadata saved to {filename}")

def search_geo(term, retmax=5):
    handle = Entrez.esearch(db="gds", term=term, retmax=retmax)
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]

def fetch_geo_metadata(geo_id, retries=3, delay=0.3):
    for attempt in range(retries):
        try:
            time.sleep(delay)
            handle = Entrez.efetch(db="gds", id=geo_id, rettype="xml")
            metadata = handle.read()
            handle.close()
            return metadata
        except Exception as e:
            print(f"Error fetching metadata for GEO ID {geo_id} (Attempt {attempt+1}/{retries}): {e}")
            time.sleep(2 ** attempt)
    print(f"Failed to fetch metadata for GEO ID {geo_id} after {retries} retries.")
    return None

def fetch_geo_summary(geo_id):
    handle = Entrez.esummary(db="gds", id=geo_id, retmode="xml")
    records = Entrez.read(handle)
    handle.close()
    return records[0] if records else None

def extract_description(metadata):
    if pd.isnull(metadata):
        return None
    split_point = metadata.find("Organism:")
    if split_point != -1:
        return metadata[:split_point].strip()
    return metadata.strip()

def extract_field(metadata, field_name):
    if pd.isnull(metadata):
        return None
    field_marker = f"{field_name}:"
    if field_marker in metadata:
        try:
            field_start = metadata.index(field_marker) + len(field_marker)
            remaining_metadata = metadata[field_start:]
            next_field_index = re.search(r"\b\w+:", remaining_metadata)
            if next_field_index:
                next_field_start = next_field_index.start() + field_start
                return metadata[field_start:next_field_start].strip()
            return metadata[field_start:].strip()
        except ValueError:
            return None
    return None

def extract_samples(metadata):
    if pd.isnull(metadata):
        return None
    match = re.search(r'(\d+)\s+Samples', metadata)
    if match:
        return int(match.group(1))
    return None

def clean_platforms(metadata):
    if pd.isnull(metadata):
        return None
    return re.sub(r'\s+\d+\s+Samples', '', metadata).strip()

def process_final_excel(input_file, output_file):
    data = pd.read_excel(input_file)
    print(data.head())
    print(data['Metadata'])
    

    fields = {
        'Organism': 'Organism',
        'Type': 'Type',
        'Platform(s)': ['Platform', 'Platforms'],
        'FTP download': 'FTP download',
        'SeriesAccession': ['Series Accession', 'Accession'],
        'ID': 'ID'
    }

    data['Description'] = data['Metadata'].apply(extract_description)

    for column_name, field_name in fields.items():
        if isinstance(field_name, list):
            data[column_name] = data['Metadata'].apply(lambda x: extract_field(x, field_name[0]) or extract_field(x, field_name[1]))
        else:
            data[column_name] = data['Metadata'].apply(lambda x: extract_field(x, field_name))

    data['Samples'] = data['Platform(s)'].apply(extract_samples)
    data['Platform(s)'] = data['Platform(s)'].apply(clean_platforms)

    data.to_excel(output_file, index=False)
    print(f"Processed data saved to {output_file}")

def main():
    # need to run a match to check those that errored
    search_term = """((NSCLC OR ""non-small cell lung cancer"") AND Homo sapiens[Organism] AND ""gse""[Filter] AND 
(""expression profiling by array""[DataSet Type] OR 
""expression profiling by genome tiling array""[DataSet Type] OR 
""genome binding/occupancy profiling by array""[DataSet Type] OR 
""genome binding/occupancy profiling by genome tiling array""[DataSet Type] OR 
""genome variation profiling by array""[DataSet Type] OR 
""genome variation profiling by genome tiling array""[DataSet Type] OR 
""methylation profiling by array""[DataSet Type] OR 
""methylation profiling by genome tiling array""[DataSet Type] OR 
""non coding RNA profiling by array""[DataSet Type] OR 
""non coding RNA profiling by genome tiling array""[DataSet Type]))"""
    geo_ids = search_geo(search_term, retmax=5)
    print(f"Number of GEO datasets found: {len(geo_ids)}")

    if not geo_ids:
        print("No GEO datasets found.")
        return

    metadata_list = []
    excel_file = "adj-chemo_geo_wide_metadata.xlsx"
    final_file = "adj-chemo_metadata_split_fixed_series_accession.xlsx"

    for idx, geo_id in enumerate(geo_ids):
        metadata = fetch_geo_metadata(geo_id)
        summary = fetch_geo_summary(geo_id)
        print(summary)
        print(summary.keys())
        if metadata:
            print(f"Metadata fetched for GEO ID {geo_id}")
            metadata_list.append({"GEO ID": geo_id, "Metadata": metadata})
        
        if (idx + 1) % 10 == 0 or idx == len(geo_ids) - 1:
            save_metadata_to_excel(metadata_list, filename=excel_file)
            metadata_list = []

        if (idx + 1) % 100 == 0:
            print("Pausing to prevent rate-limiting...")
            time.sleep(5)

    # Process the final excel file into a formatted version
    process_final_excel(excel_file, final_file)

if __name__ == "__main__":
    main()