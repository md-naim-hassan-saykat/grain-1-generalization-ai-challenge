#!/usr/bin/env python3
"""
Script to create a properly structured zip file for Codabench.
This script creates a zip with competition.yaml at the root level.
"""
import os
import zipfile
from pathlib import Path

def create_bundle():
    """Create a zip bundle with competition.yaml at the root."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    bundle_name = script_dir / "grain_classification_bundle.zip"
    
    # Files/directories to exclude
    exclude_patterns = [
        ".git",
        ".ipynb_checkpoints",
        ".DS_Store",
        "preparation.ipynb",
        "STATUS.md",
        "utilities",
        "data_split_summary.json",
        "__pycache__",
        "*.pyc"
    ]
    
    # Remove old zip if it exists
    if bundle_name.exists():
        print(f"Removing old bundle: {bundle_name}")
        bundle_name.unlink()
    
    print(f"Creating bundle: {bundle_name}")
    print(f"From directory: {script_dir}")
    
    # Create zip file
    with zipfile.ZipFile(bundle_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through all files in Competition_Bundle
        for root, dirs, files in os.walk(script_dir):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]
            
            # Process files
            for file in files:
                file_path = Path(root) / file
                
                # Skip excluded files
                if any(pattern in str(file_path) for pattern in exclude_patterns):
                    continue
                
                # Skip the bundle zip itself
                if file_path == bundle_name:
                    continue
                
                # Get relative path from Competition_Bundle directory
                arcname = file_path.relative_to(script_dir)
                
                # Add to zip
                zipf.write(file_path, arcname)
                if 'competition.yaml' in str(arcname) or 'ingestion' in str(arcname) or 'scoring' in str(arcname):
                    print(f"  Added: {arcname}")
    
    print(f"\n✓ Bundle created: {bundle_name}")
    print(f"✓ competition.yaml should be at the root of the zip")
    print(f"\nTo verify, check that competition.yaml is listed (not Competition_Bundle/competition.yaml)")
    
    # Verify competition.yaml is in the zip
    with zipfile.ZipFile(bundle_name, 'r') as zipf:
        files = zipf.namelist()
        if 'competition.yaml' in files:
            print("✓ VERIFIED: competition.yaml is at the root of the zip")
        else:
            print("⚠ WARNING: competition.yaml not found at root level!")
            print(f"  Files in zip: {[f for f in files if 'competition' in f]}")

if __name__ == "__main__":
    create_bundle()
