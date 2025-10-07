import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import json
from datetime import datetime

class LungCancerPredictor:
    def __init__(self, model_path):
        """
        Initialize the predictor with a trained model
        
        Args:
            model_path: Path to the saved model file
        """
        self.model = tf.keras.models.load_model(model_path)
        self.input_shape = self.model.input_shape[1:3]  # (height, width)
        self.class_names = ['Normal', 'Lung Cancer']
        
        print(f"Model loaded successfully. Input shape: {self.input_shape}")
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for prediction
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        try:
            # Load image
            if isinstance(image_path, str):
                image = Image.open(image_path).convert('L')  # Convert to grayscale
            else:
                # Assume it's already a PIL Image or numpy array
                image = image_path
            
            # Convert to numpy array if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Resize to model input size
            image = cv2.resize(image, self.input_shape)
            
            # Normalize pixel values
            image = image.astype(np.float32) / 255.0
            
            # Add batch and channel dimensions
            image = np.expand_dims(image, axis=0)  # Add batch dimension
            image = np.expand_dims(image, axis=-1)  # Add channel dimension
            
            return image
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def predict(self, image_path):
        """
        Make prediction on a single image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        processed_image = self.preprocess_image(image_path)
        
        if processed_image is None:
            return {
                'error': 'Failed to preprocess image',
                'confidence': 0.0,
                'class_name': 'Unknown',
                'class_index': -1
            }
        
        # Make prediction
        predictions = self.model.predict(processed_image)
        confidence = float(np.max(predictions[0]))
        class_index = int(np.argmax(predictions[0]))
        class_name = self.class_names[class_index]
        
        # Get probabilities for all classes
        probabilities = {
            self.class_names[i]: float(pred) 
            for i, pred in enumerate(predictions[0])
        }
        
        return {
            'class_index': class_index,
            'class_name': class_name,
            'confidence': confidence,
            'probabilities': probabilities,
            'timestamp': datetime.now().isoformat(),
            'is_cancer': class_index == 1
        }
    
    def predict_batch(self, image_paths):
        """
        Make predictions on multiple images
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of prediction results
        """
        results = []
        for image_path in image_paths:
            result = self.predict(image_path)
            result['image_path'] = image_path
            results.append(result)
        
        return results
    
    def analyze_confidence(self, confidence):
        """
        Analyze confidence level and provide interpretation
        
        Args:
            confidence: Prediction confidence score
            
        Returns:
            Confidence analysis dictionary
        """
        if confidence >= 0.9:
            level = "Very High"
            recommendation = "Strong evidence for diagnosis"
        elif confidence >= 0.7:
            level = "High"
            recommendation = "Good evidence for diagnosis"
        elif confidence >= 0.5:
            level = "Moderate"
            recommendation = "Suggest further examination"
        else:
            level = "Low"
            recommendation = "Requires expert review"
        
        return {
            'confidence_level': level,
            'recommendation': recommendation,
            'threshold_analysis': confidence >= 0.5
        }

def create_sample_test_images():
    """Create sample test images for demonstration"""
    os.makedirs('test_images', exist_ok=True)
    
    # Create some sample images (in real scenario, these would be real CT scans)
    for i in range(5):
        # Create random image that looks like CT scan
        img_array = np.random.rand(512, 512) * 255
        img = Image.fromarray(img_array.astype(np.uint8))
        img.save(f'test_images/sample_ct_{i}.png')
    
    print("Sample test images created in 'test_images' directory")

# Example usage and testing
if __name__ == "__main__":
    # Create sample images for testing
    create_sample_test_images()
    
    # Initialize predictor (replace with your actual model path)
    model_path = "trained_models/best_model.h5"  # Update this path
    
    if os.path.exists(model_path):
        predictor = LungCancerPredictor(model_path)
        
        # Test prediction
        test_image_path = "test_images/sample_ct_0.png"
        
        if os.path.exists(test_image_path):
            result = predictor.predict(test_image_path)
            
            print("\n" + "="*50)
            print("LUNG CANCER DETECTION RESULTS")
            print("="*50)
            print(f"Image: {test_image_path}")
            print(f"Prediction: {result['class_name']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Cancer Detected: {result['is_cancer']}")
            
            # Confidence analysis
            confidence_analysis = predictor.analyze_confidence(result['confidence'])
            print(f"Confidence Level: {confidence_analysis['confidence_level']}")
            print(f"Recommendation: {confidence_analysis['recommendation']}")
            
            print("\nClass Probabilities:")
            for class_name, prob in result['probabilities'].items():
                print(f"  {class_name}: {prob:.4f}")
                
        else:
            print(f"Test image not found: {test_image_path}")
            print("Please update the test_image_path or run create_sample_test_images() first")
    
    else:
        print(f"Model not found at: {model_path}")
        print("Please train a model first or update the model_path")
        
        # Create a simple demo model for testing
        print("\nCreating a demo model for testing...")
        from training.model_architectures import LungCancerModel, compile_model
        
        model_builder = LungCancerModel()
        demo_model = model_builder.create_simpler_model()
        demo_model = compile_model(demo_model)
        
        # Save demo model
        os.makedirs('trained_models', exist_ok=True)
        demo_model.save('trained_models/demo_model.h5')
        print("Demo model created at: trained_models/demo_model.h5")