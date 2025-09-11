#!/bin/bash

# Active Learning Molecular Docking - Example Run Script
# This script demonstrates how to run the framework with example data

set -e  # Exit on any error

echo "=========================================="
echo "Active Learning Molecular Docking Example"
echo "=========================================="

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "active-learning-docking" ]]; then
    echo "Warning: Conda environment 'active-learning-docking' is not activated"
    echo "Please run: conda activate active-learning-docking"
    echo ""
fi

# Test installation first
echo "1. Testing installation..."
python test_installation.py
if [ $? -ne 0 ]; then
    echo "Installation test failed. Please check your setup."
    exit 1
fi

echo ""
echo "2. Checking required files..."

# Check if example files exist
if [ ! -f "inputs/protein.pdb" ]; then
    echo "Warning: Example protein file not found at inputs/protein.pdb"
    echo "Please provide your own protein structure file"
fi

if [ ! -f "inputs/reference.pdb" ]; then
    echo "Warning: Example reference ligand not found at inputs/reference.pdb"
    echo "Please provide your own reference ligand file"
fi

if [ ! -f "inputs/candidates.csv" ]; then
    echo "Warning: Example compound database not found at inputs/candidates.csv"
    echo "Please provide your own compound database with 'Catalog ID' and 'smiles' columns"
fi

if [ ! -f "config.yml" ]; then
    echo "Creating example configuration file..."
    cat > config_example.yml << 'EOF'
# Example configuration for Active Learning Molecular Docking
input:
  protein_file: "inputs/protein.pdb"
  ligand_file: "inputs/reference.pdb"
  database_file: "inputs/candidates.csv"
  initial_csv: null

output:
  base_dir: "example_results"
  log_file: "active_learning.log"

active_learning:
  budget: 1000                    # Small budget for quick example
  batch_size: 5                   # Small batch size
  n_cycles: 3                     # Few cycles for demonstration
  conf_num: 1
  similarity_threshold: 0.2
  hbond_similarity_threshold: 0.3
  
  exploration:
    initial_ratio: 0.8
    final_ratio: 0.2
    decay_type: "linear"

molecular:
  filtering:
    enable_element_filter: true
    enable_deduplication: true
    supported_elements: ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
  
  fingerprint:
    type: "morgan"
    radius: 2
    n_bits: 1024
  
  rmsd:
    threshold: 2.0
    timeout: 60
  
  hbond:
    distance_cutoff: 3.5
    angle_cutoff: 120

docking:
  executable_path: "./bin/qvina2.1"
  box:
    size: [20, 20, 20]
    center: "auto"
  parameters:
    exhaustiveness: 8
    n_poses: 1
    energy_range: 3
  n_cpu: 4

model:
  type: "tabm"
  architecture:
    embed_dim: 64              # Smaller for quick example
    hidden_dims: [128]
    dropout: 0.2
    batch_norm: true
  
  training:
    learning_rate: 0.001
    weight_decay: 0.0
    epochs: 20                 # Fewer epochs for quick example
    batch_size: 64
    early_stopping:
      patience: 5
      min_delta: 0.001
  
  uncertainty:
    method: "mc_dropout"
    mc_samples: 20             # Fewer samples for speed
  
  hyperparameter_tuning:
    enable: false              # Disabled for quick example
    method: "random"
    n_trials: 50
    timeout: 1800

system:
  max_threads: 4
  memory:
    cleanup_interval: 2
    max_cycle_history: 1
  
  gpu:
    enable: true
    device_id: 0
  
  logging:
    level: "INFO"
    format: "%(asctime)s - %(levelname)s - %(message)s"
    console_output: true
  
  random_seed: 42

validation:
  required_csv_columns: ["Catalog ID", "smiles"]
  max_file_size_mb: 1000
  constraints:
    budget: [1, 1000000]
    batch_size: [1, 10000]
    n_cycles: [1, 100]
    similarity_threshold: [0.0, 1.0]
EOF
    echo "Created config_example.yml with example settings"
fi

echo ""
echo "3. Configuration options:"
echo "   a) Use command line arguments (requires all files)"
echo "   b) Use YAML configuration file"
echo "   c) Run quick validation test"
echo ""

read -p "Choose option (a/b/c) [b]: " choice
choice=${choice:-b}

case $choice in
    a)
        echo ""
        echo "Running with command line arguments..."
        echo "Command: python main.py --protein inputs/protein.pdb --ligand inputs/reference.pdb --database inputs/candidates.csv --output example_results"
        
        if [ -f "inputs/protein.pdb" ] && [ -f "inputs/reference.pdb" ] && [ -f "inputs/candidates.csv" ]; then
            python main.py \
                --protein inputs/protein.pdb \
                --ligand inputs/reference.pdb \
                --database inputs/candidates.csv \
                --output example_results
        else
            echo "Error: Required input files not found. Please provide:"
            echo "  - inputs/protein.pdb"
            echo "  - inputs/reference.pdb" 
            echo "  - inputs/candidates.csv"
            exit 1
        fi
        ;;
    b)
        echo ""
        echo "Running with YAML configuration..."
        
        if [ -f "config.yml" ]; then
            config_file="config.yml"
        elif [ -f "config_example.yml" ]; then
            config_file="config_example.yml"
        else
            echo "Error: No configuration file found (config.yml or config_example.yml)"
            exit 1
        fi
        
        echo "Using configuration file: $config_file"
        echo "Command: python main.py --config $config_file"
        
        python main.py --config "$config_file"
        ;;
    c)
        echo ""
        echo "Running configuration validation test..."
        python -c "
import os
import sys
import yaml

try:
    config_file = None
    if os.path.exists('config.yml'):
        config_file = 'config.yml'
    elif os.path.exists('config_example.yml'):
        config_file = 'config_example.yml'
    else:
        print('No configuration file found')
        sys.exit(1)
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f'✓ {config_file} is valid')
    print('Configuration validation completed successfully!')
except Exception as e:
    print(f'Configuration validation failed: {e}')
    sys.exit(1)
"
        ;;
    *)
        echo "Invalid option. Exiting."
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Example run completed!"
echo ""
echo "Next steps:"
echo "1. Check the results in the output directory"
echo "2. Review the log file for detailed information"
echo "3. Customize the configuration for your specific needs"
echo "4. Scale up the budget and cycles for production runs"
echo ""
echo "For help: python main.py --help"
echo "Documentation: See README.md for detailed usage"
echo "=========================================="