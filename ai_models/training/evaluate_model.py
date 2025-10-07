import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
import pandas as pd
import os
import json

class ModelEvaluator:
    def __init__(self, model):
        self.model = model
    
    def predict(self, X_test):
        """Make predictions on test data"""
        return self.model.predict(X_test)
    
    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation"""
        # Make predictions
        y_pred_proba = self.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_true)
        
        # Precision, Recall, F1
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        results = {
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
            'roc_auc': roc_auc,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist()
            }
        }
        
        return results
    
    def plot_confusion_matrix(self, cm, class_names=['Normal', 'Cancer']):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        return plt
    
    def plot_roc_curve(self, fpr, tpr, roc_auc):
        """Plot ROC curve"""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        return plt
    
    def plot_predictions_samples(self, X_test, y_test, y_pred, num_samples=8):
        """Plot sample predictions with true vs predicted labels"""
        y_true_labels = np.argmax(y_test, axis=1)
        
        # Select random samples
        indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
        
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            image = X_test[idx].squeeze()
            true_label = y_true_labels[idx]
            pred_label = y_pred[idx]
            confidence = np.max(self.model.predict(X_test[idx:idx+1]))
            
            axes[i].imshow(image, cmap='gray')
            axes[i].set_title(f'True: {true_label}, Pred: {pred_label}\nConf: {confidence:.2f}')
            axes[i].axis('off')
            
            # Color code based on correctness
            if true_label == pred_label:
                axes[i].patch.set_edgecolor('green')
            else:
                axes[i].patch.set_edgecolor('red')
            axes[i].patch.set_linewidth(3)
        
        plt.tight_layout()
        return plt
    
    def save_evaluation_report(self, results, save_path='evaluation_report.json'):
        """Save evaluation results to JSON file"""
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, np.float32):
                json_results[key] = float(value)
            else:
                json_results[key] = value
        
        with open(save_path, 'w') as f:
            json.dump(json_results, f, indent=4)
        
        print(f"Evaluation report saved to: {save_path}")

def evaluate_saved_model(model_path, X_test, y_test):
    """Evaluate a saved model"""
    model = tf.keras.models.load_model(model_path)
    evaluator = ModelEvaluator(model)
    return evaluator.evaluate_model(X_test, y_test)

# Example usage
if __name__ == "__main__":
    # This would be used after training
    from train_cnn import LungCancerTrainer
    
    config = {
        'batch_size': 16,
        'epochs': 10,
        'learning_rate': 0.001,
        'validation_split': 0.2,
        'model_save_path': 'trained_models'
    }
    
    trainer = LungCancerTrainer(config)
    X_train, X_val, y_train, y_val = trainer.load_data()
    
    # Load trained model
    model = tf.keras.models.load_model('trained_models/best_model.h5')
    
    # Evaluate
    evaluator = ModelEvaluator(model)
    results = evaluator.evaluate_model(X_val, y_val)
    
    # Plot results
    cm = np.array(results['confusion_matrix'])
    evaluator.plot_confusion_matrix(cm)
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    evaluator.plot_roc_curve(
        results['roc_curve']['fpr'],
        results['roc_curve']['tpr'],
        results['roc_auc']
    )
    plt.savefig('roc_curve.png')
    plt.show()
    
    # Save report
    evaluator.save_evaluation_report(results)