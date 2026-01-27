import json
import requests
from bs4 import BeautifulSoup
import time
import os, random


class ProteomeXchangeDatasetExtractor:
    """
    A class to extract ProteomeXchange dataset information and fetch detailed metadata.
    """
    
    def __init__(self, json_file_path, output_directory="dataset_details"):
        """
        Initialize the extractor with the JSON file path and output directory.
        
        Args:
            json_file_path (str): Path to the proteomexchange_datasets.json file
            output_directory (str): Directory to save the dataset text files
        """
        self.json_file_path = json_file_path
        self.output_directory = output_directory
        self.base_url = "https://proteomecentral.proteomexchange.org/cgi/GetDataset"
        self.datasets = []
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_directory, exist_ok=True)
    
    def load_datasets(self):
        """
        Load datasets from the JSON file.
        """
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.datasets = data.get('datasets', [])
        print(f"Loaded {len(self.datasets)} datasets from {self.json_file_path}")
    
    def extract_dataset_id(self, dataset):
        """
        Extract the dataset ID from a dataset entry.
        
        Args:
            dataset (list): A dataset entry from the JSON
            
        Returns:
            str: The dataset ID (e.g., 'PXD051234')
        """
        if isinstance(dataset, list) and len(dataset) > 0:
            return dataset[0]
        return None
    
    def fetch_dataset_details(self, dataset_id):
        """
        Fetch dataset title and description from ProteomeXchange website.
        
        Args:
            dataset_id (str): The dataset ID (e.g., 'PXD051234')
            
        Returns:
            tuple: (title, description) or (None, None) if failed
        """
        try:
            # Construct the URL
            url = f"{self.base_url}?ID={dataset_id}&test=no"
            
            # Make the request
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the dataset summary table
            summary_table = soup.find('table', class_='dataset-summary')
            
            title = None
            description = None
            
            if summary_table:
                rows = summary_table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        header = cells[0].get_text(strip=True)
                        content = cells[1].get_text(strip=True)
                        
                        if header == 'Title':
                            title = content
                        elif header == 'Description':
                            description = content
            
            return title, description
            
        except Exception as e:
            print(f"Error fetching details for {dataset_id}: {str(e)}")
            return None, None
    
    def save_dataset_to_file(self, dataset_id, title, description):
        """
        Save dataset information to a text file.
        
        Args:
            dataset_id (str): The dataset ID
            title (str): The dataset title
            description (str): The dataset description
        """
        filename = f"{dataset_id}.txt"
        filepath = os.path.join(self.output_directory, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Title: {title or 'No title available'}\n")
            f.write(f"Description: {description or 'No description available'}\n")
        
        print(f"Saved details for {dataset_id} to {filename}")
    
    def process_datasets(self, delay_seconds=1):
        """
        Process all datasets: fetch details and save to files.
        
        Args:
            delay_seconds (float): Delay between requests to be respectful to the server
        """
        if not self.datasets:
            print("No datasets loaded. Please run load_datasets() first.")
            return
        
        total_datasets = len(self.datasets)
        successful = 0
        failed = 0
        
        for i, dataset in enumerate(self.datasets, 1):
            dataset_id = self.extract_dataset_id(dataset)
            
            if not dataset_id:
                print(f"Skipping dataset {i}/{total_datasets}: No valid ID found")
                failed += 1
                continue
            
            print(f"Processing {i}/{total_datasets}: {dataset_id}")
            
            # Fetch details
            title, description = self.fetch_dataset_details(dataset_id)
            
            if title is not None or description is not None:
                self.save_dataset_to_file(dataset_id, title, description)
                successful += 1
            else:
                print(f"Failed to fetch details for {dataset_id}")
                failed += 1
            
            # Add delay to be respectful to the server
            if i < total_datasets:
                time.sleep(delay_seconds + random.uniform(-0.2, 0.2))
        
        print(f"\nProcessing complete!")
        print(f"Successfully processed: {successful}")
        print(f"Failed: {failed}")
        print(f"Output directory: {self.output_directory}")
    
    def run(self, delay_seconds=1):
        """
        Main method to run the entire extraction process.
        
        Args:
            delay_seconds (float): Delay between requests
        """
        print("Starting ProteomeXchange dataset extraction...")
        self.load_datasets()
        self.process_datasets(delay_seconds)


def main():
    """
    Main function to run the dataset extractor.
    """
    # Initialize the extractor
    extractor = ProteomeXchangeDatasetExtractor(
        json_file_path="proteomexchange_datasets.json",
        output_directory="proteomexchange_7_24_25"  # Specify your output directory
    )
    
    # Run the extraction process
    extractor.run(delay_seconds=3)


if __name__ == "__main__":
    main()
