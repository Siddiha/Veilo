import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121, ResNet50, VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

class LungCancerModel:
    def __init__(self, input_shape=(512, 512, 1), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def create_cnn_model(self):
        """Create a custom CNN model for lung cancer detection"""
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=self.input_shape,
                         kernel_regularizer=l2(0.01)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu',
                         kernel_regularizer=l2(0.01)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu',
                         kernel_regularizer=l2(0.01)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu',
                         kernel_regularizer=l2(0.01)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global Average Pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def create_densenet_model(self):
        """Create model using DenseNet121 pre-trained weights"""
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=(512, 512, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def create_resnet_model(self):
        """Create model using ResNet50 pre-trained weights"""
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(512, 512, 3)
        )
        
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model

def compile_model(model, learning_rate=0.001):
    """Compile the model with appropriate settings"""
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    return model

def create_ensemble_model(models_list, input_shape=(512, 512, 1)):
    """Create an ensemble of models"""
    model_input = tf.keras.Input(shape=input_shape)
    
    # Get predictions from each model
    outputs = []
    for i, model in enumerate(models_list):
        # Ensure each model has unique name
        model._name = f"model_{i}"
        outputs.append(model(model_input))
    
    # Average predictions
    averaged_outputs = layers.Average()(outputs)
    
    ensemble_model = tf.keras.Model(
        inputs=model_input, 
        outputs=averaged_outputs
    )
    
    return ensemble_model