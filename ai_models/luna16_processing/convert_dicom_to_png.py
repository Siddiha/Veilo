"""
File: ai_models/luna16_processing/convert_dicom_to_png.py
Convert LUNA16 .mhd/.raw files to PNG format for training
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import SimpleITK as sitk
from tqdm import tqdm
import cv2

class Luna16Converter:
    """
    Convert LUNA16 MHD/RAW files to PNG images
    """
    def __init__(self, input_path, output_path):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organized output
        self.normal_dir = self.output_path / "normal"
        self.nodule_dir = self.output_path / "nodules"
        self.normal_dir.mkdir(exist_ok=True)
        self.nodule_dir.mkdir(exist_ok=True)
        
        self.annotations = None
        self.candidates = None
    
    def load_annotations(self, annotations_path, candidates_path):
        """Load annotation CSV files"""
        try:
            self.annotations = pd.read_csv(annotations_path)
            self.candidates = pd.read_csv(candidates_path)
            print(f"✓ Loaded {len(self.annotations)} annotations")
            print(f"✓ Loaded {len(self.candidates)} candidates")
            return True
        except Exception as e:
            print(f"✗ Error loading annotations: {e}")
            return False
    
    def read_mhd_scan(self, filepath):
        """Read MHD file and return image array"""
        try:
            # Read the scan using SimpleITK
            scan = sitk.ReadImage(str(filepath))
            # Convert to numpy array
            scan_array = sitk.GetArrayFromImage(scan)
            
            # Get spacing information (important for nodule detection)
            spacing = np.array(scan.GetSpacing())[::-1]  # Reverse for numpy
            origin = np.array(scan.GetOrigin())[::-1]
            
            return scan_array, spacing, origin
        except Exception as e:
            print(f"✗ Error reading {filepath}: {e}")
            return None, None, None
    
    def normalize_scan(self, scan):
        """Normalize CT scan to standard HU window"""
        # Apply lung window: -1000 to 400 HU
        scan = np.clip(scan, -1000, 400)
        
        # Normalize to 0-255
        scan = scan.astype(np.float32)
        scan = (scan - scan.min()) / (scan.max() - scan.min() + 1e-8)
        scan = (scan * 255).astype(np.uint8)
        
        return scan
    
    def get_nodule_info(self, seriesuid):
        """Get nodule information for a scan"""
        if self.annotations is None:
            return []
        
        nodules = self.annotations[
            self.annotations['seriesuid'] == seriesuid
        ]
        
        nodule_list = []
        for _, row in nodules.iterrows():
            nodule_list.append({
                'coordX': row['coordX'],
                'coordY': row['coordY'],
                'coordZ': row['coordZ'],
                'diameter': row['diameter_mm']
            })
        
        return nodule_list
    
    def world_to_voxel(self, coord, origin, spacing):
        """Convert world coordinates to voxel coordinates"""
        voxel = np.abs((coord - origin) / spacing)
        return voxel.astype(int)
    
    def extract_slices_with_nodules(self, scan_array, spacing, origin, 
                                   nodules, seriesuid, num_slices=5):
        """Extract slices containing nodules"""
        saved_images = []
        
        for nodule in nodules:
            # Convert world coordinates to voxel
            world_coord = np.array([
                nodule['coordZ'],
                nodule['coordY'],
                nodule['coordX']
            ])
            
            voxel_coord = self.world_to_voxel(world_coord, origin, spacing)
            z, y, x = voxel_coord
            
            # Extract slices around the nodule
            for offset in range(-num_slices//2, num_slices//2 + 1):
                slice_idx = z + offset
                
                if 0 <= slice_idx < scan_array.shape[0]:
                    slice_img = scan_array[slice_idx]
                    slice_img = self.normalize_scan(slice_img)
                    
                    # Save with nodule information in filename
                    filename = f"{seriesuid}_slice{slice_idx:03d}_nodule.png"
                    filepath = self.nodule_dir / filename
                    
                    Image.fromarray(slice_img).save(filepath)
                    saved_images.append(filepath)
        
        return saved_images
    
    def extract_normal_slices(self, scan_array, seriesuid, 
                            num_slices=10, stride=5):
        """Extract normal slices (without nodules)"""
        saved_images = []
        
        # Sample slices uniformly across the scan
        total_slices = scan_array.shape[0]
        slice_indices = range(0, total_slices, stride)
        
        # Limit number of slices
        slice_indices = list(slice_indices)[:num_slices]
        
        for slice_idx in slice_indices:
            slice_img = scan_array[slice_idx]
            slice_img = self.normalize_scan(slice_img)
            
            filename = f"{seriesuid}_slice{slice_idx:03d}_normal.png"
            filepath = self.normal_dir / filename
            
            Image.fromarray(slice_img).save(filepath)
            saved_images.append(filepath)
        
        return saved_images
    
    def convert_single_scan(self, mhd_file, extract_nodules=True, 
                          extract_normals=True):
        """Convert single MHD file to PNG images"""
        # Read scan
        scan_array, spacing, origin = self.read_mhd_scan(mhd_file)
        
        if scan_array is None:
            return None
        
        # Get seriesuid from filename
        seriesuid = mhd_file.stem
        
        saved_images = []
        
        # Extract nodule slices
        if extract_nodules:
            nodules = self.get_nodule_info(seriesuid)
            if nodules:
                nodule_imgs = self.extract_slices_with_nodules(
                    scan_array, spacing, origin, nodules, seriesuid
                )
                saved_images.extend(nodule_imgs)
        
        # Extract normal slices
        if extract_normals:
            normal_imgs = self.extract_normal_slices(scan_array, seriesuid)
            saved_images.extend(normal_imgs)
        
        return saved_images
    
    def convert_batch(self, subset_dirs=None):
        """Convert all scans in specified subsets"""
        if subset_dirs is None:
            # Find all subset directories
            subset_dirs = list(self.input_path.glob("subset*"))
        
        print(f"\n=== Converting LUNA16 Scans to PNG ===")
        print(f"Input path: {self.input_path}")
        print(f"Output path: {self.output_path}")
        print(f"Subsets to process: {len(subset_dirs)}")
        
        total_images = 0
        total_scans = 0
        
        for subset_dir in subset_dirs:
            if not subset_dir.is_dir():
                continue
            
            print(f"\nProcessing {subset_dir.name}...")
            
            # Find all .mhd files
            mhd_files = list(subset_dir.glob("*.mhd"))
            print(f"Found {len(mhd_files)} scan files")
            
            for mhd_file in tqdm(mhd_files, desc=f"Converting {subset_dir.name}"):
                saved_imgs = self.convert_single_scan(mhd_file)
                
                if saved_imgs:
                    total_images += len(saved_imgs)
                    total_scans += 1
        
        print(f"\n✓ Conversion complete!")
        print(f"Total scans processed: {total_scans}")
        print(f"Total images extracted: {total_images}")
        print(f"  - Nodule images: {len(list(self.nodule_dir.glob('*.png')))}")
        print(f"  - Normal images: {len(list(self.normal_dir.glob('*.png')))}")
        
        return total_scans, total_images
    
    def create_dataset_summary(self):
        """Create summary CSV of converted dataset"""
        summary_data = []
        
        # Process nodule images
        for img_path in self.nodule_dir.glob("*.png"):
            summary_data.append({
                'filename': img_path.name,
                'path': str(img_path),
                'label': 'cancer',
                'class_id': 1
            })
        
        # Process normal images
        for img_path in self.normal_dir.glob("*.png"):
            summary_data.append({
                'filename': img_path.name,
                'path': str(img_path),
                'label': 'normal',
                'class_id': 0
            })
        
        df = pd.DataFrame(summary_data)
        summary_path = self.output_path / "dataset_summary.csv"
        df.to_csv(summary_path, index=False)
        
        print(f"\n✓ Dataset summary saved to: {summary_path}")
        print(f"Total images: {len(df)}")
        print(f"Class distribution:\n{df['label'].value_counts()}")
        
        return df


def main():
    """Main execution function"""
    # Setup paths
    input_path = Path("datasets/luna16/raw")
    output_path = Path("datasets/luna16/processed/images")
    annotations_path = Path("datasets/luna16/raw/annotations.csv")
    candidates_path = Path("datasets/luna16/raw/candidates.csv")
    
    # Create converter
    converter = Luna16Converter(input_path, output_path)
    
    # Load annotations
    if annotations_path.exists() and candidates_path.exists():
        converter.load_annotations(annotations_path, candidates_path)
    else:
        print("Warning: Annotation files not found. Only extracting sample slices.")
    
    # Convert all subsets
    converter.convert_batch()
    
    # Create summary
    converter.create_dataset_summary()
    
    print("\n✓ All conversions complete! Ready for preprocessing.")


if __name__ == "__main__":
    main()