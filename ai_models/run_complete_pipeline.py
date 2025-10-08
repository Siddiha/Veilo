"""
File: ai_models/run_complete_pipeline.py
Master script to run the complete AI model pipeline
"""
import sys
import os
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from luna16_processing.download_luna16 import Luna16Downloader
from luna16_processing.convert_dicom_to_png import Luna16Converter
from luna16_processing.preprocess_ct_scans import CTScanPreprocessor
from training.train_cnn import LungCancerTrainer


class VeiloAIPipeline:
    """
    Complete pipeline for Veilo AI lung cancer detection system
    """
    def __init__(self):
        self.base_path = Path("datasets/luna16")
        self.results = {}
    
    def print_header(self, text):
        """Print formatted header"""
        print("\n" + "="*70)
        print(f"  {text}")
        print("="*70 + "\n")
    
    def step_1_download_data(self, subset_list=None):
        """Step 1: Download LUNA16 dataset"""
        self.print_header("STEP 1: Downloading LUNA16 Dataset")
        
        if subset_list is None:
            subset_list = [0, 2]  # Default subsets
        
        downloader = Luna16Downloader()
        results = downloader.download_all(subset_list=subset_list)
        
        self.results['download'] = results
        
        print("\n✓ Step 1 Complete: Data downloaded successfully")
        return results
    
    def step_2_convert_to_png(self):
        """Step 2: Convert DICOM/MHD files to PNG"""
        self.print_header("STEP 2: Converting Scans to PNG Format")
        
        input_path = self.base_path / "raw"
        output_path = self.base_path / "processed" / "images"
        annotations_path = self.base_path / "raw" / "annotations.csv"
        candidates_path = self.base_path / "raw" / "candidates.csv"
        
        # Check if input exists
        if not input_path.exists():
            print("Error: Raw data not found. Please run step 1 first.")
            return None
        
        converter = Luna16Converter(input_path, output_path)
        
        # Load annotations if available
        if annotations_path.exists() and candidates_path.exists():
            converter.load_annotations(annotations_path, candidates_path)
        
        # Convert all scans
        total_scans, total_images = converter.convert_batch()
        
        # Create dataset summary
        df = converter.create_dataset_summary()
        
        self.results['conversion'] = {
            'total_scans': total_scans,
            'total_images': total_images,
            'dataset_df': df
        }
        
        print("\n✓ Step 2 Complete: Scans converted to PNG")
        return total_scans, total_images
    
    def step_3_preprocess(self):
        """Step 3: Preprocess images"""
        self.print_header("STEP 3: Preprocessing Images")
        
        input_dir = self.base_path / "processed" / "images"
        output_dir = self.base_path / "processed" / "preprocessed"
        
        # Check if input exists
        if not input_dir.exists():
            print("Error: Converted images not found. Please run step 2 first.")
            return None
        
        preprocessor = CTScanPreprocessor(target_size=(512, 512))
        
        # Preprocess all images
        processed_files = preprocessor.preprocess_batch(
            input_dir,
            output_dir,
            apply_segmentation=False
        )
        
        # Create train/val/test splits
        dataset_csv = input_dir.parent / "dataset_summary.csv"
        if dataset_csv.exists():
            splits_dir = self.base_path / "processed" / "splits"
            train_df, val_df, test_df = preprocessor.create_train_val_test_split(
                dataset_csv,
                splits_dir,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15
            )
            
            self.results['preprocessing'] = {
                'processed_files': len(processed_files),
                'train_samples': len(train_df),
                'val_samples': len(val_df),
                'test_samples': len(test_df)
            }
        
        # Save preprocessing configuration
        config_path = output_dir / "preprocessing_config.json"
        preprocessor.save_preprocessing_config(config_path)
        
        print("\n✓ Step 3 Complete: Images preprocessed and split")
        return processed_files
    
    def step_4_train_model(self, config=None):
        """Step 4: Train the model"""
        self.print_header("STEP 4: Training Model")
        
        if config is None:
            config = {
                'batch_size': 16,
                'epochs': 50,
                'learning_rate': 0.001,
                'early_stopping_patience': 15,
                'architecture': 'cnn',
                'model_save_path': 'trained_models',
                'log_dir': 'training_logs',
                'data_path': 'datasets/luna16/processed/preprocessed'
            }
        
        # Check if preprocessed data exists
        data_path = Path(config['data_path'])
        splits_dir = Path("datasets/luna16/processed/splits")
        
        if not splits_dir.exists():
            print("Error: Preprocessed data not found. Please run step 3 first.")
            return None
        
        # Create trainer and train
        trainer = LungCancerTrainer(config)
        history = trainer.train_model()
        
        if history is not None:
            # Plot training history
            trainer.plot_training_history(save=True)
            
            # Evaluate model
            from training.evaluate_model import ModelEvaluator
            X_train, X_val, y_train, y_val = trainer.load_data()
            
            if X_val is not None:
                evaluator = ModelEvaluator(trainer.model)
                evaluation_results = evaluator.evaluate_model(X_val, y_val)
                
                self.results['training'] = {
                    'final_accuracy': evaluation_results['accuracy'],
                    'final_precision': evaluation_results['precision'],
                    'final_recall': evaluation_results['recall'],
                    'final_f1': evaluation_results['f1_score'],
                    'roc_auc': evaluation_results['roc_auc']
                }
                
                # Save evaluation results
                eval_path = Path(config['model_save_path']) / 'evaluation_results.json'
                evaluator.save_evaluation_report(evaluation_results, eval_path)
        
        print("\n✓ Step 4 Complete: Model training finished")
        return history
    
    def run_complete_pipeline(self, subset_list=None, skip_steps=None):
        """Run the complete pipeline"""
        if skip_steps is None:
            skip_steps = []
        
        self.print_header("Veilo AI - Complete Pipeline Execution")
        
        print("Pipeline Steps:")
        print("  1. Download LUNA16 Dataset")
        print("  2. Convert to PNG Format")
        print("  3. Preprocess Images")
        print("  4. Train Model")
        print()
        
        # Step 1: Download
        if 1 not in skip_steps:
            try:
                self.step_1_download_data(subset_list)
            except Exception as e:
                print(f"Error in Step 1: {e}")
                return False
        else:
            print("⊘ Step 1 skipped")
        
        # Step 2: Convert
        if 2 not in skip_steps:
            try:
                self.step_2_convert_to_png()
            except Exception as e:
                print(f"Error in Step 2: {e}")
                return False
        else:
            print("⊘ Step 2 skipped")
        
        # Step 3: Preprocess
        if 3 not in skip_steps:
            try:
                self.step_3_preprocess()
            except Exception as e:
                print(f"Error in Step 3: {e}")
                return False
        else:
            print("⊘ Step 3 skipped")
        
        # Step 4: Train
        if 4 not in skip_steps:
            try:
                self.step_4_train_model()
            except Exception as e:
                print(f"Error in Step 4: {e}")
                return False
        else:
            print("⊘ Step 4 skipped")
        
        # Print final summary
        self.print_summary()
        
        return True
    
    def print_summary(self):
        """Print pipeline execution summary"""
        self.print_header("Pipeline Execution Summary")
        
        if 'download' in self.results:
            print("✓ Data Download:")
            print(f"  - Subsets downloaded: {len(self.results['download'].get('subsets', []))}")
        
        if 'conversion' in self.results:
            print("\n✓ Conversion:")
            print(f"  - Total scans processed: {self.results['conversion']['total_scans']}")
            print(f"  - Total images extracted: {self.results['conversion']['total_images']}")
        
        if 'preprocessing' in self.results:
            print("\n✓ Preprocessing:")
            print(f"  - Preprocessed images: {self.results['preprocessing']['processed_files']}")
            print(f"  - Training samples: {self.results['preprocessing']['train_samples']}")
            print(f"  - Validation samples: {self.results['preprocessing']['val_samples']}")
            print(f"  - Test samples: {self.results['preprocessing']['test_samples']}")
        
        if 'training' in self.results:
            print("\n✓ Model Training:")
            print(f"  - Final Accuracy: {self.results['training']['final_accuracy']:.4f}")
            print(f"  - Final Precision: {self.results['training']['final_precision']:.4f}")
            print(f"  - Final Recall: {self.results['training']['final_recall']:.4f}")
            print(f"  - Final F1-Score: {self.results['training']['final_f1']:.4f}")
            print(f"  - ROC AUC: {self.results['training']['roc_auc']:.4f}")
        
        print("\n" + "="*70)
        print("  ✓ PIPELINE COMPLETE - Model Ready for Deployment!")
        print("="*70)


def main():
    """Main execution with command line arguments"""
    parser = argparse.ArgumentParser(
        description='Veilo AI - Lung Cancer Detection Pipeline'
    )
    
    parser.add_argument(
        '--subsets',
        type=int,
        nargs='+',
        default=[0, 2],
        help='LUNA16 subsets to download (default: 0 2)'
    )
    
    parser.add_argument(
        '--skip',
        type=int,
        nargs='+',
        default=[],
        help='Steps to skip (1=download, 2=convert, 3=preprocess, 4=train)'
    )
    
    parser.add_argument(
        '--step',
        type=int,
        choices=[1, 2, 3, 4],
        help='Run only specific step'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = VeiloAIPipeline()
    
    # Run specific step or complete pipeline
    if args.step:
        if args.step == 1:
            pipeline.step_1_download_data(args.subsets)
        elif args.step == 2:
            pipeline.step_2_convert_to_png()
        elif args.step == 3:
            pipeline.step_3_preprocess()
        elif args.step == 4:
            pipeline.step_4_train_model()
    else:
        # Run complete pipeline
        pipeline.run_complete_pipeline(
            subset_list=args.subsets,
            skip_steps=args.skip
        )


if __name__ == "__main__":
    # Check if running with arguments
    if len(sys.argv) > 1:
        main()
    else:
        # Default execution
        print("="*70)
        print("  Veilo AI - Lung Cancer Detection Pipeline")
        print("="*70)
        print("\nUsage examples:")
        print("  # Run complete pipeline with default subsets (0, 2):")
        print("  python run_complete_pipeline.py")
        print()
        print("  # Download specific subsets:")
        print("  python run_complete_pipeline.py --subsets 0 1 2")
        print()
        print("  # Skip already completed steps:")
        print("  python run_complete_pipeline.py --skip 1 2")
        print()
        print("  # Run only specific step:")
        print("  python run_complete_pipeline.py --step 3")
        print()
        print("="*70)
        print("\nRunning with default settings...")
        print("="*70)
        
        # Run with defaults
        pipeline = VeiloAIPipeline()
        pipeline.run_complete_pipeline(subset_list=[0, 2])