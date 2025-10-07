import numpy as np
import cv2
from PIL import Image
import os
import pandas as pd
from skimage import exposure, filters
import matplotlib.pyplot as plt

class CTScanPreprocessor:
    def __init__(self):
        self.target_size = (512, 512)
    
    def load_image(self, image_path):
        """Load image from file path"""
        try:
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            return np.array(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def resize_image(self, image, target_size):
        """Resize image to target size"""
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    def normalize_intensity(self, image):
        """Normalize image intensity"""
        image = image.astype(np.float32)
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
        return image
    
    def apply_clahe(self, image, clip_limit=2.0, grid_size=(8,8)):
        """Apply Contrast Limited Adaptive Histogram Equalization"""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        return clahe.apply((image * 255).astype(np.uint8))
    
    def remove_noise(self, image):
        """Apply noise reduction"""
        # Gaussian blur for noise reduction
        denoised = cv2.GaussianBlur(image, (3, 3), 0)
        return denoised
    
    def enhance_edges(self, image):
        """Enhance edges using Sobel filter"""
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        return magnitude / np.max(magnitude)
    
    def preprocess_single_image(self, image_path, output_path=None):
        """Complete preprocessing pipeline for single image"""
        # Load image
        image = self.load_image(image_path)
        if image is None:
            return None
        
        # Resize
        image = self.resize_image(image, self.target_size)
        
        # Normalize intensity
        image = self.normalize_intensity(image)
        
        # Apply CLAHE for contrast enhancement
        image = self.apply_clahe(image)
        
        # Normalize again after CLAHE
        image = image.astype(np.float32) / 255.0
        
        # Remove noise
        image = self.remove_noise(image)
        
        # Save if output path provided
        if output_path:
            output_image = (image * 255).astype(np.uint8)
            Image.fromarray(output_image).save(output_path)
        
        return image
    
    def preprocess_batch(self, input_dir, output_dir):
        """Preprocess all images in directory"""
        os.makedirs(output_dir, exist_ok=True)
        processed_files = []
        
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"preprocessed_{filename}")
                
                processed_image = self.preprocess_single_image(input_path, output_path)
                if processed_image is not None:
                    processed_files.append(filename)
                    print(f"Processed: {filename}")
        
        print(f"Preprocessing complete. {len(processed_files)} files processed.")
        return processed_files

def main():
    preprocessor = CTScanPreprocessor()
    
    # Example usage
    input_dir = "datasets/luna16/processed/images"
    output_dir = "datasets/luna16/processed/preprocessed"
    
    preprocessor.preprocess_batch(input_dir, output_dir)

if __name__ == "__main__":
    main()