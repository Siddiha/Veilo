"""
File: ai_models/setup.py
Setup script for Veilo AI models
"""
import subprocess
import sys
import os
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def check_python_version():
    """Check if Python version is compatible"""
    print_header("Checking Python Version")
    
    required_version = (3, 8)
    current_version = sys.version_info[:2]
    
    print(f"Current Python version: {sys.version}")
    
    if current_version >= required_version:
        print(f"✓ Python {current_version[0]}.{current_version[1]} is compatible")
        return True
    else:
        print(f"✗ Python {required_version[0]}.{required_version[1]}+ is required")
        print(f"  Current version: {current_version[0]}.{current_version[1]}")
        return False

def install_requirements():
    """Install required packages"""
    print_header("Installing Dependencies")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("✗ requirements.txt not found")
        return False
    
    print("Installing packages from requirements.txt...")
    print("This may take several minutes...\n")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("\n✓ All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("\n✗ Error installing dependencies")
        return False

def create_directory_structure():
    """Create necessary directories"""
    print_header("Creating Directory Structure")
    
    directories = [
        "datasets/luna16/raw",
        "datasets/luna16/processed/images",
        "datasets/luna16/processed/preprocessed",
        "datasets/luna16/processed/splits",
        "trained_models",
        "training_logs",
        "test_images",
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")
    
    print("\n✓ Directory structure created")
    return True

def test_imports():
    """Test if all required packages can be imported"""
    print_header("Testing Package Imports")
    
    packages = [
        ("tensorflow", "TensorFlow"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("sklearn", "scikit-learn"),
        ("matplotlib", "Matplotlib"),
        ("tqdm", "tqdm"),
    ]
    
    all_success = True
    
    for package, name in packages:
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - Failed to import")
            all_success = False
    
    if all_success:
        print("\n✓ All packages imported successfully")
    else:
        print("\n✗ Some packages failed to import")
    
    return all_success

def check_gpu():
    """Check if GPU is available for TensorFlow"""
    print_header("Checking GPU Availability")
    
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"✓ Found {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
            print("\n✓ GPU acceleration available")
        else:
            print("ℹ No GPU detected - Training will use CPU")
            print("  (This is fine but will be slower)")
        
        return True
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        return False

def verify_luna16_setup():
    """Verify LUNA16 dataset setup"""
    print_header("Verifying LUNA16 Dataset Setup")
    
    raw_path = Path("datasets/luna16/raw")
    
    if not raw_path.exists():
        print("ℹ LUNA16 dataset not downloaded yet")
        print("  Run: python run_complete_pipeline.py --step 1")
        return True
    
    # Check for annotation files
    annotations = raw_path / "annotations.csv"
    candidates = raw_path / "candidates.csv"
    
    if annotations.exists() and candidates.exists():
        print("✓ Annotation files found")
    else:
        print("ℹ Annotation files not found")
    
    # Check for subsets
    subsets = list(raw_path.glob("subset*"))
    if subsets:
        print(f"✓ Found {len(subsets)} subset(s) downloaded")
    else:
        print("ℹ No dataset subsets found")
        print("  Run: python run_complete_pipeline.py --step 1")
    
    return True

def create_example_config():
    """Create example configuration file"""
    print_header("Creating Example Configuration")
    
    config_content = """# Veilo AI Training Configuration
# Copy this to config.py and modify as needed

TRAINING_CONFIG = {
    # Model settings
    'architecture': 'cnn',  # Options: 'cnn', 'densenet', 'resnet'
    'input_shape': (512, 512, 1),
    'num_classes': 2,
    
    # Training parameters
    'batch_size': 16,
    'epochs': 50,
    'learning_rate': 0.001,
    'early_stopping_patience': 15,
    
    # Data paths
    'data_path': 'datasets/luna16/processed/preprocessed',
    'model_save_path': 'trained_models',
    'log_dir': 'training_logs',
    
    # Data split
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
}

# Dataset settings
DATASET_CONFIG = {
    'subsets': [0, 2],  # Which LUNA16 subsets to use
    'target_size': (512, 512),
    'apply_augmentation': True,
    'apply_segmentation': False,
}
"""
    
    config_file = Path("example_config.py")
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"✓ Created: {config_file}")
    print("  Copy and customize for your needs")
    
    return True

def print_next_steps():
    """Print next steps for user"""
    print_header("Setup Complete! Next Steps")
    
    print("1. Download LUNA16 Dataset:")
    print("   python run_complete_pipeline.py --step 1")
    print()
    print("2. Convert to PNG:")
    print("   python run_complete_pipeline.py --step 2")
    print()
    print("3. Preprocess Images:")
    print("   python run_complete_pipeline.py --step 3")
    print()
    print("4. Train Model:")
    print("   python run_complete_pipeline.py --step 4")
    print()
    print("Or run everything at once:")
    print("   python run_complete_pipeline.py")
    print()
    print("For help:")
    print("   python run_complete_pipeline.py --help")
    print()
    print("="*70)

def main():
    """Main setup function"""
    print("="*70)
    print("  Veilo AI - Model Setup Script")
    print("="*70)
    
    # Run all setup steps
    steps = [
        ("Python Version", check_python_version),
        ("Dependencies", install_requirements),
        ("Directory Structure", create_directory_structure),
        ("Package Imports", test_imports),
        ("GPU Check", check_gpu),
        ("LUNA16 Setup", verify_luna16_setup),
        ("Example Config", create_example_config),
    ]
    
    results = []
    
    for step_name, step_func in steps:
        try:
            result = step_func()
            results.append((step_name, result))
        except Exception as e:
            print(f"\n✗ Error in {step_name}: {e}")
            results.append((step_name, False))
    
    # Print summary
    print_header("Setup Summary")
    
    for step_name, result in results:
        status = "✓" if result else "✗"
        print(f"{status} {step_name}")
    
    all_success = all(result for _, result in results)
    
    if all_success:
        print("\n" + "="*70)
        print("  ✓ SETUP SUCCESSFUL!")
        print("="*70)
        print_next_steps()
    else:
        print("\n" + "="*70)
        print("  ⚠ SETUP COMPLETED WITH WARNINGS")
        print("="*70)
        print("\nSome steps had issues, but you can still proceed.")
        print("Check the messages above for details.")
        print_next_steps()

if __name__ == "__main__":
    main()