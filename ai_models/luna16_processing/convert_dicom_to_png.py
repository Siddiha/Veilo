import os
import pydicom
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

class DicomToPngConverter:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
    
    def read_dicom(self, filepath):
        """Read DICOM file and return image array"""
        try:
            dicom = pydicom.dcmread(filepath)
            image = dicom.pixel_array
            
            # Normalize to 0-255
            image = image.astype(np.float32)
            image = (image - image.min()) / (image.max() - image.min()) * 255.0
            image = image.astype(np.uint8)
            
            return image
        except Exception as e:
            print(f"Error reading DICOM {filepath}: {e}")
            return None
    
    def convert_single_file(self, dicom_path, png_path):
        """Convert single DICOM file to PNG"""
        image_array = self.read_dicom(dicom_path)
        if image_array is not None:
            image = Image.fromarray(image_array)
            image.save(png_path, 'PNG')
            return True
        return False
    
    def convert_batch(self, dicom_directory):
        """Convert all DICOM files in directory to PNG"""
        converted_files = []
        
        for filename in os.listdir(dicom_directory):
            if filename.lower().endswith('.dcm'):
                dicom_path = os.path.join(dicom_directory, filename)
                png_filename = filename.replace('.dcm', '.png').replace('.DCM', '.png')
                png_path = os.path.join(self.output_path, png_filename)
                
                if self.convert_single_file(dicom_path, png_path):
                    converted_files.append(png_filename)
                    print(f"Converted: {filename} -> {png_filename}")
        
        print(f"Conversion complete. {len(converted_files)} files converted.")
        return converted_files

def main():
    # Example usage
    converter = DicomToPngConverter(
        input_path="datasets/luna16/raw",
        output_path="datasets/luna16/processed/images"
    )
    converter.convert_batch("datasets/luna16/raw")

if __name__ == "__main__":
    main()