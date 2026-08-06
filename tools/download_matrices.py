"""
Download selected matrices from the SuiteSparse Matrix Collection.

Usage:
    python tools/download_matrices.py
"""

import os
import tarfile
from ssgetpy import search
from scipy.io import mmread

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

MATRIX_FOLDER = os.path.join(os.path.dirname(__file__), "..", "matrices")

os.makedirs(MATRIX_FOLDER, exist_ok=True)

MATRIX_NAMES = [
    "494_bus",
    "662_bus",
    "1138_bus",
    "west0479",
    "bcsstk13",
    "bcsstk14",
    "nos1",
    "nos2",
    "ash219",
    "arc130"
]

# ------------------------------------------------------------
# Download matrices
# ------------------------------------------------------------

def download_matrices():
    print("Downloading matrices into:", MATRIX_FOLDER)

    for name in MATRIX_NAMES:
        try:
            print(f"\nSearching for {name}...")
            result = search(name=name)

            if len(result) == 0:
                print(f"Matrix {name} not found.")
                continue

            print(f"Downloading {name}...")
            result.download(destpath=MATRIX_FOLDER)

        except Exception as e:
            print(f"Could not download {name}: {e}")

# ------------------------------------------------------------
# Extract .tar.gz files
# ------------------------------------------------------------

def extract_archives():
    print("\nExtracting archives...")

    for filename in os.listdir(MATRIX_FOLDER):
        if filename.endswith(".tar.gz"):
            filepath = os.path.join(MATRIX_FOLDER, filename)
            print(f"Extracting {filename}...")

            try:
                with tarfile.open(filepath, "r:gz") as tar:
                    tar.extractall(MATRIX_FOLDER)
            except Exception as e:
                print(f"Error extracting {filename}: {e}")

    print("Extraction complete.")

# ------------------------------------------------------------
# Show downloaded files
# ------------------------------------------------------------

def list_downloaded_files():
    print("\nFiles in matrices folder:")
    for root, dirs, files in os.walk(MATRIX_FOLDER):
        for file in files:
            print(os.path.join(root, file))

# ------------------------------------------------------------
# Test loading one matrix
# ------------------------------------------------------------

def test_load():
    test_path = os.path.join(MATRIX_FOLDER, "494_bus", "494_bus.mtx")

    if os.path.exists(test_path):
        A = mmread(test_path).tocsr()
        print("\nTest matrix loaded:")
        print("Shape:", A.shape)
        print("Nonzeros:", A.nnz)
    else:
        print("\nTest matrix 494_bus.mtx not found.")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    download_matrices()
    extract_archives()
    list_downloaded_files()
    test_load()
