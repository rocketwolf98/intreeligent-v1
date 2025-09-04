#!/usr/bin/env python3
"""
Clean a CSV file to only include entries for images that still exist.
"""

import pandas as pd
from pathlib import Path
import os
import argparse

def clean_csv(csv_path, image_dir, image_extensions=None):
    """
    Remove entries from CSV where the corresponding image file no longer exists.
    
    Args:
        csv_path: Path to the CSV file
        image_dir: Path to the directory containing images
        image_extensions: List of image extensions to search for (default: ['.tif'])
    """
    if image_extensions is None:
        image_extensions = ['.tif']
    # Read the CSV file
    df = pd.read_csv(csv_path)
    print(f"Original CSV has {len(df)} entries")
    
    # Get list of existing image files
    image_dir_path = Path(image_dir)
    existing_files = set()
    
    for ext in image_extensions:
        pattern = f"*{ext}"
        for file_path in image_dir_path.glob(pattern):
            existing_files.add(file_path.name)
    
    print(f"Found {len(existing_files)} existing image files")
    
    # Filter DataFrame to only include rows where image_path exists
    df_cleaned = df[df['image_path'].isin(existing_files)]
    
    print(f"Cleaned CSV has {len(df_cleaned)} entries")
    print(f"Removed {len(df) - len(df_cleaned)} entries with missing images")
    
    # Save the cleaned CSV
    backup_path = csv_path.replace('.csv', '_backup.csv')
    print(f"Backing up original CSV to: {backup_path}")
    df.to_csv(backup_path, index=False)
    
    print(f"Saving cleaned CSV to: {csv_path}")
    df_cleaned.to_csv(csv_path, index=False)
    
    # Show some statistics
    if 'label' in df_cleaned.columns:
        label_counts = df_cleaned['label'].value_counts()
        print("\nLabel distribution in cleaned dataset:")
        for label, count in label_counts.items():
            print(f"  {label}: {count}")

def main():
    parser = argparse.ArgumentParser(description="Clean a CSV file to only include entries for images that still exist")
    parser.add_argument("csv_path", help="Path to the CSV file")
    parser.add_argument("image_dir", help="Path to the directory containing images")
    parser.add_argument("-e", "--extensions", nargs="+", default=[".tif"], 
                        help="Image file extensions to search for (default: .tif)")
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file '{args.csv_path}' does not exist")
        return
    
    if not os.path.exists(args.image_dir):
        print(f"Error: Image directory '{args.image_dir}' does not exist")
        return
    
    # Clean the CSV
    clean_csv(args.csv_path, args.image_dir, args.extensions)

if __name__ == "__main__":
    main()