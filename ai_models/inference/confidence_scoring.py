import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple
import json

class ConfidenceScorer:
    def __init__(self, confidence_threshold=0.7, high_confidence_threshold=0.9):
        """
        Initialize confidence scorer
        
        Args:
            confidence_threshold: Minimum confidence for reliable prediction
            high_confidence_threshold: Threshold for high confidence predictions
        """
        self.confidence_threshold = confidence_threshold
        self.high_confidence_threshold = high_confidence_threshold
    
    def calculate_confidence_score(self, prediction_probs: np.ndarray) -> Dict:
        """
        Calculate comprehensive confidence scores
        
        Args:
            prediction_probs: Array of prediction probabilities
            
        Returns:
            Dictionary with confidence metrics
        """
        max_prob = np.max(prediction_probs)
        predicted_class = np.argmax(prediction_probs)
        
        # Calculate entropy-based uncertainty
        entropy = self._calculate_entropy(prediction_probs)
        
        # Calculate margin (difference between top two probabilities)
        sorted_probs = np.sort(prediction_probs)[::-1]
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
        
        # Confidence level categorization
        confidence_level = self._categorize_confidence(max_prob, margin)
        
        return {
            'max_confidence': float(max_prob),
            'predicted_class': int(predicted_class),
            'entropy': float(entropy),
            'margin': float(margin),
            'confidence_level': confidence_level,
            'is_reliable': max_prob >= self.confidence_threshold,
            'is_high_confidence': max_prob >= self.high_confidence_threshold
        }
    
    def _calculate_entropy(self, probabilities: np.ndarray) -> float:
        """Calculate entropy of probability distribution"""
        # Add small epsilon to avoid log(0)
        probabilities = probabilities + 1e-10
        probabilities = probabilities / np.sum(probabilities)
        return -np.sum(probabilities * np.log(probabilities))
    
    def _categorize_confidence(self, max_prob: float, margin: float) -> str:
        """Categorize confidence level"""
        if max_prob >= self.high_confidence_threshold and margin >= 0.3:
            return "very_high"
        elif max_prob >= self.confidence_threshold and margin >= 0.2:
            return "high"
        elif max_prob >= 0.5:
            return "moderate"
        else:
            return "low"
    
    def ensemble_confidence(self, predictions_list: List[np.ndarray]) -> Dict:
        """
        Calculate confidence from ensemble predictions
        
        Args:
            predictions_list: List of predictions from multiple models
            
        Returns:
            Ensemble confidence metrics
        """
        all_predictions = np.array(predictions_list)
        
        # Mean probabilities across ensemble
        mean_probs = np.mean(all_predictions, axis=0)
        
        # Standard deviation (uncertainty measure)
        std_probs = np.std(all_predictions, axis=0)
        
        # Agreement between models
        agreement = self._calculate_agreement(all_predictions)
        
        individual_scores = [self.calculate_confidence_score(pred) for pred in all_predictions]
        
        return {
            'mean_confidence': float(np.max(mean_probs)),
            'uncertainty': float(np.mean(std_probs)),
            'model_agreement': float(agreement),
            'individual_scores': individual_scores,
            'ensemble_reliable': agreement >= 0.8 and np.mean(std_probs) <= 0.2
        }
    
    def _calculate_agreement(self, predictions: np.ndarray) -> float:
        """Calculate agreement between multiple model predictions"""
        predicted_classes = np.argmax(predictions, axis=1)
        unique, counts = np.unique(predicted_classes, return_counts=True)
        return np.max(counts) / len(predicted_classes)
    
    def generate_confidence_report(self, 
                                 image_path: str, 
                                 prediction_result: Dict,
                                 confidence_metrics: Dict) -> Dict:
        """
        Generate comprehensive confidence report
        
        Args:
            image_path: Path to the analyzed image
            prediction_result: Original prediction results
            confidence_metrics: Calculated confidence metrics
            
        Returns:
            Comprehensive confidence report
        """
        report = {
            'image_info': {
                'path': image_path,
                'timestamp': prediction_result.get('timestamp', '')
            },
            'prediction': {
                'class_name': prediction_result.get('class_name', ''),
                'class_index': prediction_result.get('class_index', -1),
                'is_cancer': prediction_result.get('is_cancer', False)
            },
            'confidence_analysis': confidence_metrics,
            'risk_assessment': self._assess_risk(prediction_result, confidence_metrics),
            'recommendations': self._generate_recommendations(confidence_metrics)
        }
        
        return report
    
    def _assess_risk(self, prediction_result: Dict, confidence_metrics: Dict) -> Dict:
        """Assess risk based on prediction and confidence"""
        is_cancer = prediction_result.get('is_cancer', False)
        confidence = confidence_metrics.get('max_confidence', 0.0)
        confidence_level = confidence_metrics.get('confidence_level', 'low')
        
        if is_cancer:
            if confidence_level in ['very_high', 'high']:
                risk_level = "high"
                action = "Immediate medical consultation recommended"
            else:
                risk_level = "moderate"
                action = "Further diagnostic tests recommended"
        else:
            if confidence_level in ['very_high', 'high']:
                risk_level = "low"
                action = "Routine follow-up recommended"
            else:
                risk_level = "uncertain"
                action = "Additional screening recommended"
        
        return {
            'risk_level': risk_level,
            'recommended_action': action,
            'urgency': 'high' if risk_level == 'high' else 'medium' if risk_level == 'moderate' else 'low'
        }
    
    def _generate_recommendations(self, confidence_metrics: Dict) -> List[str]:
        """Generate recommendations based on confidence level"""
        recommendations = []
        confidence_level = confidence_metrics.get('confidence_level', 'low')
        is_reliable = confidence_metrics.get('is_reliable', False)
        
        if not is_reliable:
            recommendations.extend([
                "Low confidence prediction - manual review recommended",
                "Consider additional imaging or tests",
                "Consult with radiology specialist"
            ])
        else:
            if confidence_level == 'very_high':
                recommendations.append("High confidence prediction - suitable for clinical decision making")
            elif confidence_level == 'high':
                recommendations.append("Good confidence level - can be used with clinical correlation")
            else:
                recommendations.append("Moderate confidence - recommend correlation with patient history")
        
        return recommendations

# Example usage
if __name__ == "__main__":
    # Test confidence scoring
    scorer = ConfidenceScorer()
    
    # Sample predictions
    high_conf_pred = np.array([0.95, 0.05])  # High confidence
    low_conf_pred = np.array([0.55, 0.45])   # Low confidence
    uncertain_pred = np.array([0.51, 0.49])  # Very uncertain
    
    print("High Confidence Prediction:")
    print(scorer.calculate_confidence_score(high_conf_pred))
    
    print("\nLow Confidence Prediction:")
    print(scorer.calculate_confidence_score(low_conf_pred))
    
    print("\nUncertain Prediction:")
    print(scorer.calculate_confidence_score(uncertain_pred))
    
    # Test ensemble confidence
    ensemble_preds = [high_conf_pred, high_conf_pred, high_conf_pred]
    print("\nEnsemble Confidence:")
    print(scorer.ensemble_confidence(ensemble_preds))