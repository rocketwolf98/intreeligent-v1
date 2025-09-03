#!/usr/bin/env python3
"""
Script to delete a specified percentage of training images randomly.
"""

import os
import random
import argparse
from pathlib import Path

def delete_images_by_percentage(directory_path, percentage_to_delete, dry_run=True):
    """
    Delete specified percentage of image files randomly from the given directory.
    
    Args:
        directory_path: Path to the directory containing images
        percentage_to_delete: Percentage of files to delete (0-100)
        dry_run: If True, only print what would be deleted (default: True)
    """
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    # Get all image files
    image_files = []
    directory = Path(directory_path)
    
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    total_files = len(image_files)
    print(f"Found {total_files} image files")
    
    if total_files == 0:
        print("No image files found!")
        return
    
    # Calculate how many to delete
    files_to_delete = int(total_files * (percentage_to_delete / 100))
    files_to_keep = total_files - files_to_delete
    
    print(f"Will delete {files_to_delete} files ({percentage_to_delete}%)")
    print(f"Will keep {files_to_keep} files ({100 - percentage_to_delete}%)")
    
    # Randomly select files to delete
    random.shuffle(image_files)
    files_for_deletion = image_files[:files_to_delete]
    
    if dry_run:
        print("\n--- DRY RUN MODE ---")
        print("Files that would be deleted:")
        for i, file_path in enumerate(files_for_deletion[:10]):  # Show first 10
            print(f"  {file_path.name}")
        if len(files_for_deletion) > 10:
            print(f"  ... and {len(files_for_deletion) - 10} more files")
        print("\nTo actually delete files, run with dry_run=False")
    else:
        print("\n--- DELETING FILES ---")
        deleted_count = 0
        failed_count = 0
        
        for file_path in files_for_deletion:
            try:
                file_path.unlink()
                deleted_count += 1
                if deleted_count % 1000 == 0:
                    print(f"Deleted {deleted_count}/{files_to_delete} files...")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
                failed_count += 1
        
        print(f"\nCompleted!")
        print(f"Successfully deleted: {deleted_count}")
        print(f"Failed to delete: {failed_count}")
        print(f"Files remaining: {total_files - deleted_count}")

def main():
    parser = argparse.ArgumentParser(description="Delete a specified percentage of images randomly from a directory")
    parser.add_argument("directory", help="Path to the directory containing images")
    parser.add_argument("-p", "--percentage", type=float, default=95.0, 
                        help="Percentage of files to delete (default: 95.0)")
    parser.add_argument("--execute", action="store_true", 
                        help="Actually delete files (default is dry-run mode)")
    
    args = parser.parse_args()
    
    # Validate percentage
    if args.percentage < 0 or args.percentage > 100:
        print("Error: Percentage must be between 0 and 100")
        return
    
    # Validate directory
    if not os.path.exists(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist")
        return
    
    dry_run = not args.execute
    
    if dry_run:
        print("=== DRY RUN MODE ===")
        print("Use --execute flag to actually delete files")
    else:
        print("=== EXECUTION MODE ===")
    
    delete_images_by_percentage(args.directory, args.percentage, dry_run=dry_run)
    
    if dry_run:
        print("\nTo actually delete files, run with --execute flag")

if __name__ == "__main__":
    main()