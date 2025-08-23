# backend/train_model.py - AI Model Training Script
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import os
import requests
import zipfile
from pathlib import Path
import shutil

class LungCancerModelTrainer:
    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size
        self.model = None
        self.history = None
        self.classes = ['NORMAL', 'PNEUMONIA']
        
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            'models',
            'dataset/train/NORMAL',
            'dataset/train/PNEUMONIA', 
            'dataset/test/NORMAL',
            'dataset/test/PNEUMONIA',
            'logs'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created directory: {directory}")
    
    def download_dataset_info(self):
        """Provide information about downloading the dataset"""
        print("\n🔗 DATASET DOWNLOAD INSTRUCTIONS:")
        print("=" * 60)
        print("1. Go to: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        print("2. Click 'Download' (requires Kaggle account)")
        print("3. Extract the downloaded zip file")
        print("4. Copy the contents to match this structure:")
        print("   dataset/")
        print("   ├── train/")
        print("   │   ├── NORMAL/     (1,341 images)")
        print("   │   └── PNEUMONIA/  (3,875 images)")
        print("   └── test/")
        print("       ├── NORMAL/     (234 images)")
        print("       └── PNEUMONIA/  (390 images)")
        print("\n📊 Dataset Info:")
        print("   - Total images: ~5,840")
        print("   - Image format: JPEG")
        print("   - Size: ~1.2GB")
        print("   - Classes: Normal vs Pneumonia")
        print("\n⚠️  This is a high-quality medical dataset used in research!")
    
    def check_dataset(self):
        """Check if dataset is properly structured"""
        required_paths = [
            'dataset/train/NORMAL',
            'dataset/train/PNEUMONIA',
            'dataset/test/NORMAL', 
            'dataset/test/PNEUMONIA'
        ]
        
        dataset_ready = True
        for path in required_paths:
            if not os.path.exists(path):
                print(f"❌ Missing: {path}")
                dataset_ready = False
            else:
                file_count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"✅ Found {file_count} images in {path}")
        
        return dataset_ready
    
    def create_model(self):
        """Create CNN architecture optimized for medical imaging"""
        model = models.Sequential([
            # Input layer with data augmentation
            layers.Input(shape=(*self.img_size, 3)),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(), 
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global pooling and dense layers
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(2, activation='softmax')  # 2 classes: Normal, Pneumonia
        ])
        
        # Compile with appropriate metrics for medical imaging
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        print("✅ Model architecture created successfully")
        return model
    
    def create_data_generators(self):
        """Create data generators with medical-appropriate augmentation"""
        # Training data augmentation (medical-safe transformations)
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,       # Small rotations only
            width_shift_range=0.1,   # Minimal shifts
            height_shift_range=0.1,
            horizontal_flip=True,    # OK for chest X-rays
            zoom_range=0.1,         # Slight zoom
            fill_mode='nearest',
            validation_split=0.2
        )
        
        # Test data - only rescaling
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            'dataset/train',
            target_size=self.img_size,
            batch_size=32,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        validation_generator = train_datagen.flow_from_directory(
            'dataset/train', 
            target_size=self.img_size,
            batch_size=32,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        test_generator = test_datagen.flow_from_directory(
            'dataset/test',
            target_size=self.img_size,
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"✅ Training samples: {train_generator.samples}")
        print(f"✅ Validation samples: {validation_generator.samples}")
        print(f"✅ Test samples: {test_generator.samples}")
        print(f"✅ Class indices: {train_generator.class_indices}")
        
        return train_generator, validation_generator, test_generator
    
    def train_model(self, train_gen, val_gen, epochs=50):
        """Train the model with medical-appropriate callbacks"""
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        # Callbacks for optimal training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=8,
                min_lr=0.0001,
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger('logs/training.log')
        ]
        
        print(f"\n🚀 Starting training for {epochs} epochs...")
        print("📊 Monitoring: Accuracy, Loss, Precision, Recall")
        
        # Calculate steps per epoch
        steps_per_epoch = train_gen.samples // train_gen.batch_size
        validation_steps = val_gen.samples // val_gen.batch_size
        
        # Train the model
        history = self.model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history
        print("✅ Training completed!")
        return history
    
    def evaluate_model(self, test_gen):
        """Comprehensive model evaluation"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        print("\n📊 EVALUATING MODEL...")
        
        # Basic evaluation
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(test_gen, verbose=1)
        
        # Calculate F1 score
        f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
        
        print(f"\n🎯 FINAL RESULTS:")
        print(f"   Test Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
        print(f"   Test Precision: {test_precision:.4f}")
        print(f"   Test Recall:    {test_recall:.4f}")
        print(f"   F1 Score:       {f1_score:.4f}")
        print(f"   Test Loss:      {test_loss:.4f}")
        
        # Medical interpretation
        if test_accuracy > 0.90:
            print("🌟 EXCELLENT - Research-grade performance!")
        elif test_accuracy > 0.85:
            print("✅ VERY GOOD - Clinical potential!")
        elif test_accuracy > 0.80:
            print("👍 GOOD - Promising results!")
        else:
            print("⚠️  NEEDS IMPROVEMENT - Consider more data/training")
        
        return test_accuracy, test_precision, test_recall, f1_score
    
    def plot_training_history(self):
        """Create comprehensive training plots"""
        if self.history is None:
            raise ValueError("No training history available.")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Lung Cancer Detection Model - Training History', fontsize=16)
        
        # Accuracy plot
        axes[0, 0].plot(self.history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[0, 1].plot(self.history.history['loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision plot
        axes[1, 0].plot(self.history.history['precision'], 'g-', label='Training Precision', linewidth=2)
        axes[1, 0].plot(self.history.history['val_precision'], 'orange', label='Validation Precision', linewidth=2)
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Recall plot
        axes[1, 1].plot(self.history.history['recall'], 'purple', label='Training Recall', linewidth=2)
        axes[1, 1].plot(self.history.history['val_recall'], 'brown', label='Validation Recall', linewidth=2)
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Training plots saved as 'training_history.png'")
    
    def save_final_model(self):
        """Save the final trained model"""
        if self.model is None:
            raise ValueError("No model to save.")
        
        model_path = 'models/lung_cancer_detector.h5'
        self.model.save(model_path)
        
        # Save model info
        model_info = {
            'model_path': model_path,
            'image_size': self.img_size,
            'classes': self.classes,
            'training_date': str(tf.timestamp()),
            'framework': 'TensorFlow/Keras'
        }
        
        import json
        with open('models/model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"✅ Final model saved: {model_path}")
        print(f"✅ Model info saved: models/model_info.json")
        
        # Model size
        model_size = os.path.getsize(model_path) / (1024*1024)
        print(f"📦 Model size: {model_size:.1f} MB")

def main():
    """Main training pipeline"""
    print("🏥 LUNG CANCER DETECTION AI TRAINING")
    print("=" * 50)
    
    # Initialize trainer
    trainer = LungCancerModelTrainer(img_size=(224, 224))
    
    # Setup
    print("\n1️⃣  Setting up directories...")
    trainer.setup_directories()
    
    # Dataset info
    print("\n2️⃣  Dataset information:")
    trainer.download_dataset_info()
    
    # Check dataset
    print("\n3️⃣  Checking dataset...")
    if not trainer.check_dataset():
        print("\n❌ Dataset not ready! Please download and extract the dataset first.")
        print("Run this script again after setting up the dataset.")
        return False
    
    # Create model
    print("\n4️⃣  Creating model architecture...")
    model = trainer.create_model()
    print(f"📊 Model parameters: {model.count_params():,}")
    
    # Create data generators  
    print("\n5️⃣  Preparing data generators...")
    train_gen, val_gen, test_gen = trainer.create_data_generators()
    
    # Train model
    print("\n6️⃣  Training model...")
    history = trainer.train_model(train_gen, val_gen, epochs=50)
    
    # Evaluate model
    print("\n7️⃣  Final evaluation...")
    trainer.evaluate_model(test_gen)
    
    # Plot results
    print("\n8️⃣  Creating training plots...")
    trainer.plot_training_history()
    
    # Save model
    print("\n9️⃣  Saving final model...")
    trainer.save_final_model()
    
    print("\n🎉 TRAINING COMPLETE!")
    print("✅ Your AI model is ready to use!")
    print("🚀 Start the Flask API with: python app.py")
    
    return True

if __name__ == "__main__":
    # Check TensorFlow setup
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # Run training
    success = main()
    
    if success:
        print("\n🎊 SUCCESS! Your lung cancer detection AI is trained and ready!")
    else:
        print("\n💡 TIP: Download the dataset first, then run this script again.")