import tensorflow as tf
import numpy as np
import albumentations as A
import cv2

class DataAugmentor:
    def __init__(self):
        self.augmentation_pipeline = A.Compose([
            # Geometric transformations
            A.Rotate(limit=15, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.1, 
                rotate_limit=10, 
                p=0.5
            ),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            
            # Brightness/contrast adjustments
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.5
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.3),
            
            # Morphological operations
            A.CoarseDropout(
                max_holes=8, 
                max_height=8, 
                max_width=8, 
                p=0.3
            ),
        ])
    
    def augment_image(self, image, mask=None):
        """Apply augmentation to single image"""
        if mask is not None:
            augmented = self.augmentation_pipeline(image=image, mask=mask)
            return augmented['image'], augmented['mask']
        else:
            augmented = self.augmentation_pipeline(image=image)
            return augmented['image'], None
    
    def create_tf_augmentation_layer(self):
        """Create TensorFlow data augmentation layer"""
        return tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.2),
        ])
    
    def augment_dataset(self, images, labels, augmentation_factor=2):
        """Augment entire dataset"""
        augmented_images = []
        augmented_labels = []
        
        for i in range(len(images)):
            image = images[i]
            label = labels[i]
            
            # Add original image
            augmented_images.append(image)
            augmented_labels.append(label)
            
            # Add augmented versions
            for _ in range(augmentation_factor):
                aug_image, _ = self.augment_image(image)
                augmented_images.append(aug_image)
                augmented_labels.append(label)
        
        return np.array(augmented_images), np.array(augmented_labels)

class CTScanDataGenerator(tf.keras.utils.Sequence):
    """Data generator for CT scan images with real-time augmentation"""
    
    def __init__(self, images, labels, batch_size=32, augment=False, shuffle=True):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.augmentor = DataAugmentor()
        self.indexes = np.arange(len(images))
        
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[
            index * self.batch_size:(index + 1) * self.batch_size
        ]
        
        batch_images = self.images[batch_indexes]
        batch_labels = self.labels[batch_indexes]
        
        if self.augment:
            augmented_images = []
            for image in batch_images:
                aug_image, _ = self.augmentor.augment_image(image)
                augmented_images.append(aug_image)
            batch_images = np.array(augmented_images)
        
        return batch_images, batch_labels
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)