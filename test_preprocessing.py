import sys
import os
sys.path.append(os.path.dirname(__file__))

from app import get_preprocessing_summary

if __name__ == "__main__":
    print("Running preprocessing...")
    summary = get_preprocessing_summary()
    print("\nPreprocessing Summary:")
    print(f"Original rows: {summary['original_rows']}")
    print(f"Processed rows: {summary['processed_rows']}")
    print(f"Dropped rows: {summary['dropped_rows']}")
    print(f"New features: {summary['new_features']}")
    print(f"\nPreprocessed file saved to: {summary['preprocessed_path']}")
    
    # Verify file exists
    if os.path.exists(summary['preprocessed_path']):
        print(f"✓ File created successfully!")
        file_size = os.path.getsize(summary['preprocessed_path'])
        print(f"  File size: {file_size:,} bytes")
    else:
        print(f"✗ File was not created!")
