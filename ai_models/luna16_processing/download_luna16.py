import os
import requests
import zipfile
import tarfile
import pandas as pd
from tqdm import tqdm
import urllib.request
import subprocess

class Luna16Downloader:
    def __init__(self, base_path="datasets/luna16"):
        self.base_path = base_path
        self.raw_data_path = os.path.join(base_path, "raw")
        self.processed_path = os.path.join(base_path, "processed")
        
        # Create directories
        os.makedirs(self.raw_data_path, exist_ok=True)
        os.makedirs(self.processed_path, exist_ok=True)
        
        # LUNA16 dataset URLs (subset - actual dataset requires registration)
        self.dataset_urls = {
            "annotations": "https://zenodo.org/record/3723295/files/annotations.csv",
            "candidates": "https://zenodo.org/record/3723295/files/candidate_annotations.csv",
            "sample_data": "https://www.kaggle.com/datasets/kmader/siim-medical-images/download"
        }
    
    def download_file(self, url, filename):
        """Download file with progress bar"""
        filepath = os.path.join(self.raw_data_path, filename)
        
        if os.path.exists(filepath):
            print(f"File already exists: {filename}")
            return filepath
            
        print(f"Downloading {filename}...")
        
        try:
            # For small files like CSVs
            if url.endswith('.csv'):
                df = pd.read_csv(url)
                df.to_csv(filepath, index=False)
            else:
                # For larger files
                urllib.request.urlretrieve(url, filepath)
                
            print(f"Downloaded: {filename}")
            return filepath
            
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            return None
    
    def extract_archive(self, filepath, extract_to):
        """Extract zip or tar files"""
        if filepath.endswith('.zip'):
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif filepath.endswith('.tar.gz') or filepath.endswith('.tgz'):
            with tarfile.open(filepath, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
        print(f"Extracted: {os.path.basename(filepath)}")
    
    def download_luna16(self):
        """Main method to download LUNA16 dataset"""
        print("Starting LUNA16 dataset download...")
        
        # Download annotations
        annotations_path = self.download_file(
            self.dataset_urls["annotations"], 
            "annotations.csv"
        )
        
        # Download candidates
        candidates_path = self.download_file(
            self.dataset_urls["candidates"], 
            "candidates.csv"
        )
        
        print("LUNA16 dataset download completed!")
        return annotations_path, candidates_path

def main():
    downloader = Luna16Downloader()
    downloader.download_luna16()

if __name__ == "__main__":
    main()