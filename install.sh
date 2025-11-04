#!/bin/bash

# Active Learning for Molecular Docking - Installation Script
# Downloads qVina2.1 and prepares the environment

set -e

echo "=========================================="
echo "Active Learning for Molecular Docking"
echo "Installation Script"
echo "=========================================="
echo ""

# Create bin directory
echo "Creating bin directory..."
mkdir -p bin

# Download qVina2.1
echo "Downloading qVina2.1..."
if wget -q --show-progress "https://github.com/QVina/qvina/raw/refs/heads/master/bin/qvina2.1" -O bin/qvina2.1; then
    chmod +x bin/qvina2.1
    echo "✓ qVina2.1 installed successfully"
else
    echo "✗ Failed to download qVina2.1"
    echo "  Please download manually from: https://github.com/QVina/qvina"
    exit 1
fi

# Verify qVina2.1
echo ""
echo "Verifying qVina2.1 installation..."
if ./bin/qvina2.1 --version 2>&1 | grep -q "qvina"; then
    echo "✓ qVina2.1 is working correctly"
else
    echo "⚠ qVina2.1 may not be working correctly"
fi

echo ""
echo "Installation complete!"
echo ""
echo "Next steps:"
echo "  1. Activate conda environment: conda activate active-learning-docking"
echo "  2. Run example: python main.py -o results -l inputs/reference.pdb -p inputs/protein.pdb -d inputs/candidates.csv"
