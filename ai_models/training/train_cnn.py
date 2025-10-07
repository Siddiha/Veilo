import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import json
import datetime

from model_architectures import LungCancerModel, compile_model
from data_augmentation import CTScanDataGenerator
from evaluate_model import ModelEvaluator

class LungCancerTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.history = None
        
        # Create directories
        os.makedirs(config['model_save_path'], exist_ok=True)
        os.makedirs(config['log_dir'], exist_ok=True)
    
    def load_data(self):
        """Load and prepare training data"""
        print("Loading training data...")
        
        # This is a placeholder - replace with actual data loading
        # For now, we'll create dummy data for demonstration
        num_samples = 1000
        img_height, img_width = 512, 512
        
        # Create dummy data (replace with actual data loading)
        X_train = np.random.rand(num_samples, img_height, img_width, 1).astype(np.float32)
        y_train = np.random.randint(0, 2, size=(num_samples,))
        
        # Convert to categorical
        y_train = tf.keras.utils.to_categorical(y_train, 2)
        
        return train_test_split(
            X_train, y_train, 
            test_size=self.config['validation_split'],
            random_state=42,
            stratify=y_train
        )
    
    def calculate_class_weights(self, y_train):
        """Calculate class weights for imbalanced data"""
        y_classes = np.argmax(y_train, axis=1)
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_classes),
            y=y_classes
        )
        return dict(enumerate(class_weights))
    
    def create_callbacks(self):
        """Create training callbacks"""
        callbacks = [
            # Model checkpoint
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(
                    self.config['model_save_path'],
                    'best_model.h5'
                ),
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),
            
            # Early stopping
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard
            tf.keras.callbacks.TensorBoard(
                log_dir=self.config['log_dir'],
                histogram_freq=1
            )
        ]
        
        return callbacks
    
    def train_model(self):
        """Main training function"""
        print("Starting model training...")
        
        # Load data
        X_train, X_val, y_train, y_val = self.load_data()
        
        # Calculate class weights
        class_weights = self.calculate_class_weights(y_train)
        print(f"Class weights: {class_weights}")
        
        # Create model
        model_builder = LungCancerModel()
        self.model = model_builder.create_cnn_model()
        self.model = compile_model(self.model, self.config['learning_rate'])
        
        print("Model architecture:")
        self.model.summary()
        
        # Create data generators
        train_generator = CTScanDataGenerator(
            X_train, y_train,
            batch_size=self.config['batch_size'],
            augment=True,
            shuffle=True
        )
        
        val_generator = CTScanDataGenerator(
            X_val, y_val,
            batch_size=self.config['batch_size'],
            augment=False,
            shuffle=False
        )
        
        # Create callbacks
        callbacks = self.create_callbacks()
        
        # Train model
        self.history = self.model.fit(
            train_generator,
            epochs=self.config['epochs'],
            validation_data=val_generator,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Save final model
        final_model_path = os.path.join(
            self.config['model_save_path'],
            'final_model.h5'
        )
        self.model.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")
        
        return self.history
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Precision
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
            axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
        
        # Recall
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
            axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.config['model_save_path'], 'training_history.png')
        plt.savefig(plot_path)
        plt.show()

def main():
    # Training configuration
    config = {
        'batch_size': 16,
        'epochs': 50,
        'learning_rate': 0.001,
        'validation_split': 0.2,
        'early_stopping_patience': 15,
        'model_save_path': 'trained_models',
        'log_dir': 'training_logs'
    }
    
    # Create and run trainer
    trainer = LungCancerTrainer(config)
    history = trainer.train_model()
    
    # Plot training history
    trainer.plot_training_history()
    
    # Evaluate model
    evaluator = ModelEvaluator(trainer.model)
    X_train, X_val, y_train, y_val = trainer.load_data()
    evaluation_results = evaluator.evaluate_model(X_val, y_val)
    
    print("Training completed!")
    print(f"Validation Accuracy: {evaluation_results['accuracy']:.4f}")
    print(f"Validation Precision: {evaluation_results['precision']:.4f}")
    print(f"Validation Recall: {evaluation_results['recall']:.4f}")

if __name__ == "__main__":
    main()