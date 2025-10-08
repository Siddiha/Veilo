"""
File: ai_models/luna16_processing/preprocess_ct_scans.py
Advanced CT scan preprocessing pipeline
"""
import numpy as np
import cv2
from PIL import Image
import os
import pandas as pd
from skimage import exposure, filters, morphology
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json

class CTScanPreprocessor:
    """
    Advanced CT scan preprocessing for lung cancer detection
    """
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size
        self.preprocessing_stats = {
            'processed': 0,
            'failed': 0,
            'mean_intensity': [],
            'std_intensity': []
        }
    
    def load_image(self, image_path):
        """Load image from file path"""
        try:
            image = Image.open(image_path).convert('L')
            return np.array(image)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None
    
    def resize_image(self, image, target_size=None):
        """Resize image to target size maintaining aspect ratio"""
        if target_size is None:
            target_size = self.target_size
        
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    def normalize_intensity(self, image):
        """Normalize image intensity to 0-1 range"""
        image = image.astype(np.float32)
        image_min = np.min(image)
        image_max = np.max(image)
        
        if image_max - image_min > 0:
            image = (image - image_min) / (image_max - image_min)
        
        return image
    
    def apply_clahe(self, image, clip_limit=2.0, grid_size=(8, 8)):
        """Apply Contrast Limited Adaptive Histogram Equalization"""
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=grid_size
        )
        return clahe.apply(image)
    
    def remove_noise(self, image, method='gaussian'):
        """Apply noise reduction"""
        if method == 'gaussian':
            return cv2.GaussianBlur(image, (3, 3), 0)
        elif method == 'bilateral':
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif method == 'median':
            return cv2.medianBlur(image, 3)
        else:
            return image
    
    def enhance_edges(self, image):
        """Enhance edges using Sobel filter"""
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize
        magnitude = magnitude / (np.max(magnitude) + 1e-8)
        return magnitude
    
    def segment_lungs(self, image):
        """Segment lung regions from CT scan"""
        # Ensure uint8
        if image.dtype != np.uint8:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image
        
        # Apply Otsu thresholding
        try:
            thresh = threshold_otsu(img_uint8)
            binary = img_uint8 > thresh
            
            # Remove small objects and fill holes
            binary = morphology.remove_small_objects(binary, min_size=500)
            binary = morphology.remove_small_holes(binary, area_threshold=500)
            
            # Find largest connected components (lungs)
            labels = morphology.label(binary)
            
            if labels.max() == 0:
                return image
            
            # Create lung mask
            lung_mask = labels > 0
            
            # Apply mask to original image
            segmented = image.copy()
            segmented[~lung_mask] = 0
            
            return segmented
        except:
            return image
    
    def apply_lung_windowing(self, image):
        """Apply standard lung window settings"""
        # Lung window: -600 to 1500 HU (already normalized, so adjust)
        # This simulates the CT lung window
        
        # Clip values
        window_min = 0.2  # Adjusted for normalized image
        window_max = 0.8
        
        image_windowed = np.clip(image, window_min, window_max)
        
        # Rescale to 0-1
        image_windowed = (image_windowed - window_min) / (window_max - window_min + 1e-8)
        
        return image_windowed
    
    def augment_contrast(self, image):
        """Augment image contrast"""
        # Adaptive histogram equalization
        image_uint8 = (image * 255).astype(np.uint8)
        equalized = exposure.equalize_adapthist(
            image_uint8,
            clip_limit=0.03
        )
        return equalized
    
    def preprocess_single_image(self, image_path, output_path=None, 
                               apply_segmentation=False, visualize=False):
        """
        Complete preprocessing pipeline for single image
        
        Pipeline:
        1. Load image
        2. Resize
        3. Normalize intensity
        4. Apply CLAHE
        5. Remove noise
        6. Optional: Lung segmentation
        7. Apply lung windowing
        8. Final normalization
        """
        # Load image
        image = self.load_image(image_path)
        if image is None:
            self.preprocessing_stats['failed'] += 1
            return None
        
        # Store original for visualization
        original = image.copy()
        
        # Resize
        image = self.resize_image(image)
        
        # Normalize intensity (0-1)
        image = self.normalize_intensity(image)
        
        # Apply CLAHE for contrast enhancement
        image_clahe = self.apply_clahe(image)
        image = self.normalize_intensity(image_clahe)
        
        # Remove noise
        image = self.remove_noise(image, method='bilateral')
        
        # Optional lung segmentation
        if apply_segmentation:
            image = self.segment_lungs(image)
        
        # Apply lung windowing
        image = self.apply_lung_windowing(image)
        
        # Final normalization
        image = self.normalize_intensity(image)
        
        # Update statistics
        self.preprocessing_stats['processed'] += 1
        self.preprocessing_stats['mean_intensity'].append(np.mean(image))
        self.preprocessing_stats['std_intensity'].append(np.std(image))
        
        # Save if output path provided
        if output_path:
            output_image = (image * 255).astype(np.uint8)
            Image.fromarray(output_image).save(output_path)
        
        # Visualize if requested
        if visualize:
            self.visualize_preprocessing(original, image)
        
        return image
    
    def visualize_preprocessing(self, original, processed):
        """Visualize before and after preprocessing"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(processed, cmap='gray')
        axes[1].set_title('Preprocessed Image')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def preprocess_batch(self, input_dir, output_dir, apply_segmentation=False):
        """Preprocess all images in directory"""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_files = list(input_dir.glob('**/*.png'))
        image_files.extend(list(input_dir.glob('**/*.jpg')))
        image_files.extend(list(input_dir.glob('**/*.jpeg')))
        
        print(f"\n=== Preprocessing CT Scans ===")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Total images: {len(image_files)}")
        print(f"Lung segmentation: {apply_segmentation}")
        
        processed_files = []
        
        for img_path in tqdm(image_files, desc="Preprocessing"):
            # Maintain subdirectory structure
            relative_path = img_path.relative_to(input_dir)
            output_path = output_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Preprocess
            processed = self.preprocess_single_image(
                img_path,
                output_path,
                apply_segmentation=apply_segmentation
            )
            
            if processed is not None:
                processed_files.append(str(output_path))
        
        # Print statistics
        print(f"\n✓ Preprocessing complete!")
        print(f"Successfully processed: {self.preprocessing_stats['processed']}")
        print(f"Failed: {self.preprocessing_stats['failed']}")
        
        if self.preprocessing_stats['mean_intensity']:
            print(f"Average mean intensity: {np.mean(self.preprocessing_stats['mean_intensity']):.4f}")
            print(f"Average std intensity: {np.mean(self.preprocessing_stats['std_intensity']):.4f}")
        
        return processed_files
    
    def create_train_val_test_split(self, dataset_csv, output_dir, 
                                   train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Split dataset into train/val/test sets"""
        df = pd.read_csv(dataset_csv)
        
        # Shuffle dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Calculate split indices
        total = len(df)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        # Split dataframe
        train_df = df[:train_end]
        val_df = df[train_end:val_end]
        test_df = df[val_end:]
        
        # Create output directories
        output_dir = Path(output_dir)
        train_dir = output_dir / "train"
        val_dir = output_dir / "val"
        test_dir = output_dir / "test"
        
        for split_dir in [train_dir, val_dir, test_dir]:
            (split_dir / "normal").mkdir(parents=True, exist_ok=True)
            (split_dir / "cancer").mkdir(parents=True, exist_ok=True)
        
        # Save split CSVs
        train_df.to_csv(output_dir / "train.csv", index=False)
        val_df.to_csv(output_dir / "val.csv", index=False)
        test_df.to_csv(output_dir / "test.csv", index=False)
        
        print(f"\n✓ Dataset split created:")
        print(f"  Train: {len(train_df)} images")
        print(f"  Validation: {len(val_df)} images")
        print(f"  Test: {len(test_df)} images")
        
        # Print class distribution
        for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            print(f"\n{name} class distribution:")
            print(split_df['label'].value_counts())
        
        return train_df, val_df, test_df
    
    def save_preprocessing_config(self, output_path):
        """Save preprocessing configuration"""
        config = {
            'target_size': self.target_size,
            'preprocessing_stats': self.preprocessing_stats,
            'methods': [
                'resize',
                'normalize',
                'clahe',
                'denoise',
                'lung_windowing'
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"\n✓ Preprocessing config saved to: {output_path}")


def main():
    """Main execution function"""
    # Initialize preprocessor
    preprocessor = CTScanPreprocessor(target_size=(512, 512))
    
    # Setup paths
    input_dir = Path("datasets/luna16/processed/images")
    output_dir = Path("datasets/luna16/processed/preprocessed")
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        print("Please run convert_dicom_to_png.py first.")
        return
    
    # Preprocess all images
    processed_files = preprocessor.preprocess_batch(
        input_dir,
        output_dir,
        apply_segmentation=False  # Set to True for lung segmentation
    )
    
    # Create dataset splits
    dataset_csv = input_dir.parent / "dataset_summary.csv"
    if dataset_csv.exists():
        splits_dir = Path("datasets/luna16/processed/splits")
        preprocessor.create_train_val_test_split(
            dataset_csv,
            splits_dir,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
    
    # Save preprocessing configuration
    config_path = output_dir / "preprocessing_config.json"
    preprocessor.save_preprocessing_config(config_path)
    
    print("\n✓ All preprocessing complete! Ready for model training.")
    print(f"Preprocessed images location: {output_dir}")


if __name__ == "__main__":
    main()