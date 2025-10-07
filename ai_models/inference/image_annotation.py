import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ImageAnnotator:
    def __init__(self):
        self.colors = {
            'cancer': (255, 0, 0),      # Red
            'normal': (0, 255, 0),      # Green
            'suspicious': (255, 255, 0), # Yellow
            'annotation': (0, 255, 255), # Cyan
            'text': (255, 255, 255)     # White
        }
        
        self.font_path = None
        try:
            # Try to use Arial font
            self.font_path = "arial.ttf"
        except:
            print("Using default font - for better results install arial.ttf")
    
    def create_heatmap(self, image, prediction_probs, class_index):
        """
        Create heatmap overlay based on prediction confidence
        
        Args:
            image: Original image
            prediction_probs: Prediction probabilities
            class_index: Class index for heatmap
            
        Returns:
            Heatmap overlay image
        """
        # Create a simple heatmap (in practice, use Grad-CAM or similar)
        heatmap = np.zeros_like(image, dtype=np.float32)
        
        # Simple center-focused heatmap for demo
        # Replace this with actual activation mapping in production
        h, w = image.shape[:2]
        y, x = np.ogrid[0:h, 0:w]
        
        center_y, center_x = h // 2, w // 2
        sigma = min(h, w) // 4
        
        # Gaussian-like heatmap centered on image
        heatmap = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
        
        # Scale by prediction confidence
        confidence = prediction_probs[class_index]
        heatmap = heatmap * confidence
        
        return heatmap
    
    def annotate_image(self, image, prediction_result, confidence_metrics):
        """
        Annotate image with prediction results
        
        Args:
            image: Original image (numpy array or PIL Image)
            prediction_result: Prediction results dictionary
            confidence_metrics: Confidence metrics dictionary
            
        Returns:
            Annotated PIL Image
        """
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:  # Grayscale
                pil_image = Image.fromarray(image.astype(np.uint8))
            else:  # RGB
                pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Convert to RGB for annotation
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        draw = ImageDraw.Draw(pil_image)
        width, height = pil_image.size
        
        # Try to load font
        try:
            font_large = ImageFont.truetype(self.font_path, 24)
            font_medium = ImageFont.truetype(self.font_path, 18)
            font_small = ImageFont.truetype(self.font_path, 14)
        except:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Determine annotation color based on prediction
        is_cancer = prediction_result.get('is_cancer', False)
        confidence = confidence_metrics.get('max_confidence', 0.0)
        
        if is_cancer:
            border_color = self.colors['cancer']
            if confidence > 0.7:
                status_color = self.colors['cancer']
                status_text = "HIGH RISK - SUSPICIOUS LESION DETECTED"
            else:
                status_color = self.colors['suspicious']
                status_text = "SUSPICIOUS FINDING - FURTHER REVIEW NEEDED"
        else:
            border_color = self.colors['normal']
            if confidence > 0.7:
                status_color = self.colors['normal']
                status_text = "NORMAL - NO SUSPICIOUS FINDINGS"
            else:
                status_color = self.colors['suspicious']
                status_text = "UNCERTAIN - SUGGEST EXPERT REVIEW"
        
        # Add border
        border_width = 10
        draw.rectangle([0, 0, width, height], outline=border_color, width=border_width)
        
        # Add status bar at top
        status_bar_height = 40
        draw.rectangle([0, 0, width, status_bar_height], fill=status_color)
        
        # Add status text
        text_bbox = draw.textbbox((0, 0), status_text, font=font_medium)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (width - text_width) // 2
        draw.text((text_x, 10), status_text, fill=self.colors['text'], font=font_medium)
        
        # Add prediction information
        info_y = status_bar_height + 20
        
        # Prediction class and confidence
        class_name = prediction_result.get('class_name', 'Unknown')
        confidence_pct = confidence * 100
        
        draw.text((20, info_y), f"Diagnosis: {class_name}", 
                 fill=self.colors['text'], font=font_large)
        
        draw.text((20, info_y + 40), f"Confidence: {confidence_pct:.1f}%", 
                 fill=self.colors['text'], font=font_medium)
        
        # Confidence level
        conf_level = confidence_metrics.get('confidence_level', 'unknown').upper()
        draw.text((20, info_y + 70), f"Confidence Level: {conf_level}", 
                 fill=self.colors['text'], font=font_medium)
        
        # Add risk assessment if available
        risk_info = confidence_metrics.get('risk_assessment', {})
        if risk_info:
            risk_level = risk_info.get('risk_level', 'unknown').upper()
            draw.text((20, info_y + 100), f"Risk Level: {risk_level}", 
                     fill=self.colors['text'], font=font_medium)
        
        # Add probability breakdown
        probs = prediction_result.get('probabilities', {})
        prob_y = info_y + 140
        
        for i, (class_name, prob) in enumerate(probs.items()):
            color = self.colors['cancer'] if 'cancer' in class_name.lower() else self.colors['normal']
            draw.text((20, prob_y + i * 25), f"{class_name}: {prob:.3f}", 
                     fill=color, font=font_small)
        
        # Add timestamp
        timestamp = prediction_result.get('timestamp', '')
        if timestamp:
            draw.text((20, height - 30), f"Analyzed: {timestamp}", 
                     fill=self.colors['text'], font=font_small)
        
        # Add Veilo AI watermark
        watermark = "Veilo AI - Lung Cancer Detection"
        text_bbox = draw.textbbox((0, 0), watermark, font=font_small)
        text_width = text_bbox[2] - text_bbox[0]
        draw.text((width - text_width - 10, height - 30), watermark, 
                 fill=self.colors['text'], font=font_small)
        
        return pil_image
    
    def create_comparison_view(self, original_image, annotated_image, heatmap=None):
        """
        Create side-by-side comparison view
        
        Args:
            original_image: Original input image
            annotated_image: Annotated result image
            heatmap: Optional heatmap image
            
        Returns:
            Comparison image
        """
        images = [original_image, annotated_image]
        titles = ["Original CT Scan", "AI Analysis"]
        
        if heatmap is not None:
            images.append(heatmap)
            titles.append("Confidence Heatmap")
        
        # Create figure
        fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
        
        if len(images) == 1:
            axes = [axes]
        
        for ax, img, title in zip(axes, images, titles):
            if isinstance(img, Image.Image):
                img = np.array(img)
            
            if len(img.shape) == 2:  # Grayscale
                ax.imshow(img, cmap='gray')
            else:  # RGB
                ax.imshow(img)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def save_annotated_image(self, annotated_image, output_path, **kwargs):
        """
        Save annotated image to file
        
        Args:
            annotated_image: Annotated PIL Image
            output_path: Path to save the image
        """
        annotated_image.save(output_path, **kwargs)
        print(f"Annotated image saved to: {output_path}")

# Example usage
if __name__ == "__main__":
    # Create sample annotation
    annotator = ImageAnnotator()
    
    # Create sample image
    sample_image = Image.new('RGB', (512, 512), color='gray')
    
    # Sample prediction result
    sample_prediction = {
        'class_name': 'Lung Cancer',
        'class_index': 1,
        'is_cancer': True,
        'confidence': 0.85,
        'probabilities': {'Normal': 0.15, 'Lung Cancer': 0.85},
        'timestamp': '2024-01-15T10:30:00'
    }
    
    # Sample confidence metrics
    sample_confidence = {
        'max_confidence': 0.85,
        'confidence_level': 'high',
        'is_reliable': True,
        'risk_assessment': {
            'risk_level': 'high',
            'recommended_action': 'Immediate medical consultation recommended'
        }
    }
    
    # Create annotated image
    annotated = annotator.annotate_image(sample_image, sample_prediction, sample_confidence)
    
    # Save result
    annotator.save_annotated_image(annotated, 'sample_annotation.png')
    
    print("Sample annotation created and saved as 'sample_annotation.png'")