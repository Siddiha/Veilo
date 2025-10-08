"""
File: ai_models/training/train_cnn.py
Enhanced CNN training pipeline for lung cancer detection
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import json
import datetime
from pathlib import Path
from PIL import Image

from model_architectures import LungCancerModel, compile_model
from data_augmentation import CTScanDataGenerator
from evaluate_model import ModelEvaluator

class LungCancerTrainer:
    """
    Complete training pipeline for lung cancer detection model
    """
    def __init__(self, config):
        self.config = config
        self.model = None
        self.history = None
        
        # Create directories
        self.model_save_path = Path(config['model_save_path'])
        self.log_dir = Path(config['log_dir'])
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Data paths
        self.data_path = Path(config.get('data_path', 'datasets/luna16/processed/preprocessed'))
    
    def load_data_from_csv(self, csv_path, data_dir):
        """Load data from CSV file"""
        df = pd.read_csv(csv_path)
        
        images = []
        labels = []
        
        print(f"Loading data from {csv_path}...")
        
        for _, row in df.iterrows():
            img_path = Path(row['path'])
            
            # Handle different path structures
            if not img_path.exists():
                img_path = data_dir / row['filename']
            
            if img_path.exists():
                try:
                    img = Image.open(img_path).convert('L')
                    img = np.array(img).astype(np.float32) / 255.0
                    
                    # Ensure correct shape
                    if img.shape != (512, 512):
                        img = tf.image.resize(img[..., np.newaxis], (512, 512)).numpy().squeeze()
                    
                    images.append(img)
                    labels.append(row['class_id'])
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        X = np.array(images)
        y = np.array(labels)
        
        # Add channel dimension
        if len(X.shape) == 3:
            X = X[..., np.newaxis]
        
        # Convert labels to categorical
        y = tf.keras.utils.to_categorical(y, 2)
        
        print(f"Loaded {len(X)} images")
        print(f"Shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        return X, y
    
    def load_data(self):
        """Load and prepare training data from LUNA16 processed dataset"""
        print("="*60)
        print("Loading LUNA16 Dataset")
        print("="*60)
        
        splits_dir = Path("datasets/luna16/processed/splits")
        
        # Check if splits exist
        train_csv = splits_dir / "train.csv"
        val_csv = splits_dir / "val.csv"
        
        if not train_csv.exists() or not val_csv.exists():
            print("Error: Dataset splits not found!")
            print("Please run preprocess_ct_scans.py first.")
            return None, None, None, None
        
        # Load data
        X_train, y_train = self.load_data_from_csv(train_csv, self.data_path)
        X_val, y_val = self.load_data_from_csv(val_csv, self.data_path)
        
        print(f"\n✓ Data loaded successfully")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Print class distribution
        print(f"\nTraining class distribution:")
        print(f"  Normal: {np.sum(np.argmax(y_train, axis=1) == 0)}")
        print(f"  Cancer: {np.sum(np.argmax(y_train, axis=1) == 1)}")
        
        return X_train, X_val, y_train, y_val
    
    def calculate_class_weights(self, y_train):
        """Calculate class weights for imbalanced data"""
        y_classes = np.argmax(y_train, axis=1)
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_classes),
            y=y_classes
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        print(f"\nClass weights: {class_weight_dict}")
        return class_weight_dict
    
    def create_callbacks(self):
        """Create training callbacks"""
        callbacks = [
            # Model checkpoint - save best model
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.model_save_path / 'best_model.h5'),
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
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard
            tf.keras.callbacks.TensorBoard(
                log_dir=str(self.log_dir),
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            ),
            
            # CSV Logger
            tf.keras.callbacks.CSVLogger(
                str(self.log_dir / 'training_log.csv'),
                separator=',',
                append=False
            )
        ]
        
        return callbacks
    
    def train_model(self):
        """Main training function"""
        print("\n" + "="*60)
        print("Starting Model Training")
        print("="*60)
        
        # Load data
        X_train, X_val, y_train, y_val = self.load_data()
        
        if X_train is None:
            print("Failed to load data. Exiting.")
            return None
        
        # Calculate class weights
        class_weights = self.calculate_class_weights(y_train)
        
        # Create model
        print("\n" + "="*60)
        print("Building Model")
        print("="*60)
        
        model_builder = LungCancerModel(
            input_shape=(512, 512, 1),
            num_classes=2
        )
        
        # Choose model architecture
        architecture = self.config.get('architecture', 'cnn')
        
        if architecture == 'cnn':
            self.model = model_builder.create_cnn_model()
        elif architecture == 'densenet':
            self.model = model_builder.create_densenet_model()
        elif architecture == 'resnet':
            self.model = model_builder.create_resnet_model()
        else:
            print(f"Unknown architecture: {architecture}. Using CNN.")
            self.model = model_builder.create_cnn_model()
        
        self.model = compile_model(self.model, self.config['learning_rate'])
        
        print(f"\nModel architecture: {architecture}")
        print(f"Total parameters: {self.model.count_params():,}")
        self.model.summary()
        
        # Create data generators
        print("\n" + "="*60)
        print("Creating Data Generators")
        print("="*60)
        
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
        print("\n" + "="*60)
        print("Training Started")
        print("="*60)
        print(f"Epochs: {self.config['epochs']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print(f"Steps per epoch: {len(train_generator)}")
        print(f"Validation steps: {len(val_generator)}")
        print("="*60 + "\n")
        
        self.history = self.model.fit(
            train_generator,
            epochs=self.config['epochs'],
            validation_data=val_generator,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Save final model
        final_model_path = self.model_save_path / 'final_model.h5'
        self.model.save(final_model_path)
        print(f"\n✓ Final model saved to: {final_model_path}")
        
        # Save training history
        history_path = self.model_save_path / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history.history, f, indent=4)
        print(f"✓ Training history saved to: {history_path}")
        
        return self.history
    
    def plot_training_history(self, save=True):
        """Plot and optionally save training history"""
        if self.history is None:
            print("No training history available.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Training', linewidth=2)
            axes[1, 0].plot(self.history.history['val_precision'], label='Validation', linewidth=2)
            axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Training', linewidth=2)
            axes[1, 1].plot(self.history.history['val_recall'], label='Validation', linewidth=2)
            axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plot_path = self.model_save_path / 'training_history.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✓ Training plots saved to: {plot_path}")
        
        plt.show()


def main():
    """Main execution function"""
    # Training configuration
    config = {
        'batch_size': 16,
        'epochs': 50,
        'learning_rate': 0.001,
        'early_stopping_patience': 15,
        'architecture': 'cnn',  # Options: 'cnn', 'densenet', 'resnet'
        'model_save_path': 'trained_models',
        'log_dir': 'training_logs',
        'data_path': 'datasets/luna16/processed/preprocessed'
    }
    
    print("="*60)
    print("Veilo AI - Lung Cancer Detection Model Training")
    print("="*60)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*60)
    
    # Create and run trainer
    trainer = LungCancerTrainer(config)
    history = trainer.train_model()
    
    if history is not None:
        # Plot training history
        trainer.plot_training_history(save=True)
        
        # Evaluate model
        print("\n" + "="*60)
        print("Evaluating Model")
        print("="*60)
        
        X_train, X_val, y_train, y_val = trainer.load_data()
        if X_val is not None:
            evaluator = ModelEvaluator(trainer.model)
            evaluation_results = evaluator.evaluate_model(X_val, y_val)
            
            print("\n" + "="*60)
            print("Final Results")
            print("="*60)
            print(f"Validation Accuracy: {evaluation_results['accuracy']:.4f}")
            print(f"Validation Precision: {evaluation_results['precision']:.4f}")
            print(f"Validation Recall: {evaluation_results['recall']:.4f}")
            print(f"Validation F1-Score: {evaluation_results['f1_score']:.4f}")
            print(f"ROC AUC: {evaluation_results['roc_auc']:.4f}")
            
            # Save evaluation results
            eval_path = trainer.model_save_path / 'evaluation_results.json'
            evaluator.save_evaluation_report(evaluation_results, eval_path)
        
        print("\n" + "="*60)
        print("✓ Training Complete!")
        print("="*60)
        print(f"Model saved to: {trainer.model_save_path}")
        print(f"Logs saved to: {trainer.log_dir}")


if __name__ == "__main__":
    main()