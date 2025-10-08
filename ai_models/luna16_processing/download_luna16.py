"""
File: ai_models/luna16_processing/download_luna16.py
Enhanced LUNA16 dataset downloader with support for all subsets
"""
import os
import requests
import zipfile
import tarfile
import pandas as pd
from tqdm import tqdm
import urllib.request
import gzip
import shutil
from pathlib import Path
import time

class Luna16Downloader:
    """
    Enhanced LUNA16 dataset downloader with support for all subsets
    """
    def __init__(self, base_path="datasets/luna16"):
        self.base_path = Path(base_path)
        self.raw_data_path = self.base_path / "raw"
        self.processed_path = self.base_path / "processed"
        
        # Create directories
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        # LUNA16 dataset URLs - All 10 subsets
        self.base_url = "https://zenodo.org/record/4121926/files"
        self.subsets = [f"subset{i}.zip" for i in range(10)]
        
        # Annotation files
        self.annotations_url = "https://zenodo.org/record/4121926/files/annotations.csv"
        self.candidates_url = "https://zenodo.org/record/4121926/files/candidates.csv"
    
    def download_file_with_progress(self, url, filename):
        """Download file with progress bar"""
        filepath = self.raw_data_path / filename
        
        if filepath.exists():
            print(f"✓ File already exists: {filename}")
            return filepath
            
        print(f"\nDownloading {filename}...")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            
            with open(filepath, 'wb') as file, tqdm(
                total=total_size,
                unit='iB',
                unit_scale=True,
                desc=filename
            ) as pbar:
                for data in response.iter_content(block_size):
                    size = file.write(data)
                    pbar.update(size)
            
            print(f"✓ Downloaded: {filename}")
            return filepath
            
        except Exception as e:
            print(f"✗ Error downloading {filename}: {e}")
            if filepath.exists():
                filepath.unlink()
            return None
    
    def extract_archive(self, filepath, extract_to=None):
        """Extract zip or tar.gz files"""
        if extract_to is None:
            extract_to = self.raw_data_path
        
        extract_to = Path(extract_to)
        extract_to.mkdir(parents=True, exist_ok=True)
        
        print(f"Extracting {filepath.name}...")
        
        try:
            if filepath.suffix == '.zip':
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif filepath.suffix == '.gz' or filepath.name.endswith('.tar.gz'):
                with tarfile.open(filepath, 'r:gz') as tar_ref:
                    tar_ref.extractall(extract_to)
            
            print(f"✓ Extracted: {filepath.name}")
            return True
            
        except Exception as e:
            print(f"✗ Error extracting {filepath.name}: {e}")
            return False
    
    def download_annotations(self):
        """Download annotation files"""
        print("\n=== Downloading Annotation Files ===")
        
        annotations_path = self.download_file_with_progress(
            self.annotations_url,
            "annotations.csv"
        )
        
        candidates_path = self.download_file_with_progress(
            self.candidates_url,
            "candidates.csv"
        )
        
        return annotations_path, candidates_path
    
    def download_subset(self, subset_num):
        """Download specific subset"""
        if subset_num < 0 or subset_num > 9:
            print(f"Invalid subset number: {subset_num}. Must be 0-9.")
            return None
        
        subset_filename = f"subset{subset_num}.zip"
        subset_url = f"{self.base_url}/{subset_filename}"
        
        filepath = self.download_file_with_progress(subset_url, subset_filename)
        
        if filepath and filepath.exists():
            # Extract to subset-specific directory
            extract_dir = self.raw_data_path / f"subset{subset_num}"
            success = self.extract_archive(filepath, extract_dir)
            
            if success:
                # Optionally remove zip file after extraction to save space
                # filepath.unlink()
                pass
            
            return extract_dir
        
        return None
    
    def download_subsets(self, subset_list=None):
        """Download multiple subsets"""
        if subset_list is None:
            subset_list = [0, 2]  # Default to subsets 0 and 2
        
        print(f"\n=== Downloading LUNA16 Subsets: {subset_list} ===")
        
        downloaded_subsets = []
        for subset_num in subset_list:
            extract_dir = self.download_subset(subset_num)
            if extract_dir:
                downloaded_subsets.append(extract_dir)
        
        return downloaded_subsets
    
    def download_all(self, subset_list=None):
        """Complete download pipeline"""
        print("\n" + "="*60)
        print("LUNA16 Dataset Download Manager")
        print("="*60)
        
        # Download annotations
        annotations_path, candidates_path = self.download_annotations()
        
        # Download subsets
        subset_dirs = self.download_subsets(subset_list)
        
        # Summary
        print("\n" + "="*60)
        print("Download Summary")
        print("="*60)
        print(f"Annotations: {'✓' if annotations_path else '✗'}")
        print(f"Candidates: {'✓' if candidates_path else '✗'}")
        print(f"Subsets downloaded: {len(subset_dirs)}")
        
        return {
            'annotations': annotations_path,
            'candidates': candidates_path,
            'subsets': subset_dirs
        }
    
    def get_dataset_info(self):
        """Get information about downloaded dataset"""
        info = {
            'total_subsets': 0,
            'total_scans': 0,
            'annotations_available': False,
            'candidates_available': False
        }
        
        # Check annotations
        ann_path = self.raw_data_path / "annotations.csv"
        cand_path = self.raw_data_path / "candidates.csv"
        
        info['annotations_available'] = ann_path.exists()
        info['candidates_available'] = cand_path.exists()
        
        # Count subsets and scans
        for i in range(10):
            subset_dir = self.raw_data_path / f"subset{i}"
            if subset_dir.exists():
                info['total_subsets'] += 1
                # Count .mhd files (scan files)
                mhd_files = list(subset_dir.glob("*.mhd"))
                info['total_scans'] += len(mhd_files)
        
        return info


def main():
    """Main execution function"""
    # Initialize downloader
    downloader = Luna16Downloader()
    
    # Download subsets 0 and 2 (you can modify this list)
    subset_list = [0, 2]  # Start with these two as they're downloading
    
    # Download everything
    results = downloader.download_all(subset_list=subset_list)
    
    # Display dataset info
    print("\n" + "="*60)
    print("Dataset Information")
    print("="*60)
    info = downloader.get_dataset_info()
    print(f"Total subsets: {info['total_subsets']}")
    print(f"Total scans: {info['total_scans']}")
    print(f"Annotations available: {info['annotations_available']}")
    print(f"Candidates available: {info['candidates_available']}")
    
    print("\n✓ Download complete! Ready for preprocessing.")
    print(f"Data location: {downloader.raw_data_path}")


if __name__ == "__main__":
    main()