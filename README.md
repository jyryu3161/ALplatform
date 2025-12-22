# Active Learning for Molecular Docking

An active learning framework for efficient virtual screening using TabM neural networks with Monte Carlo dropout uncertainty estimation and qVina2.1 molecular docking.

## Dependencies

Install the conda environment as follows:

```bash
conda env create -f environment.yml
conda activate active-learning-docking
```

To download qVina2.1 and prepare the environment:
```bash
bash install.sh
```

**Note on GPU Support:**
The default installation uses CPU-only PyTorch. For GPU acceleration (recommended for large-scale screening):
```bash
conda env create -f environment-gpu.yml
conda activate active-learning-docking
```

## Input File Format

### Chemical Database (CSV or Directory)

**Option 1: Single CSV File**

The database CSV file must contain the following columns:

```csv
Catalog ID,smiles
ZINC000001,CCCc1ccc(O)cc1
ZINC000002,CC(C)Cc1ccc(C(C)C(=O)O)cc1
ZINC000003,CC1=CC=C(C=C1)C(=O)O
```

**Option 2: Directory with Multiple CSV Files**

You can also provide a directory path containing multiple CSV files. All CSV files in the directory will be automatically loaded and merged:

```
inputs/csvs/
├── library1.csv
├── library2.csv
└── library3.csv
```

Usage:
```bash
python main.py -d inputs/csvs/ -l ligand.pdb -p protein.pdb -o results
```

All files must have the same format (Catalog ID and smiles columns). Duplicate Catalog IDs across files will be automatically removed during the merge process.

### Example Files

Example input files are provided in the `inputs/` directory:
- `inputs/reference.pdb` - Example reference ligand
- `inputs/protein.pdb` - Example target protein
- `inputs/candidates.csv` - Example compound database

## How to run

Before running, make sure the conda environment is activated: `conda activate active-learning-docking`

```bash
python main.py -o OUTPUT_DIRECTORY -l LIGAND_PDB -p PROTEIN_PDB -d DATABASE_CSV
```

Example usage:

```bash
python main.py \
    -o ./results \
    -l inputs/reference.pdb \
    -p inputs/protein.pdb \
    -d inputs/candidates.csv \
    --batch_size 10 \
    --n_cycles 10
```

Example with hyperparameter auto-tuning:

```bash
python main.py \
    -o ./results \
    -l inputs/reference.pdb \
    -p inputs/protein.pdb \
    -d inputs/candidates.csv \
    --auto_tune \
    --epochs 150
```

### Using Configuration Files

Three example YAML configuration files are provided for different use cases:

**1. config_basic.yaml** - Basic configuration with essential parameters
```bash
python main.py --config config_basic.yaml
```

**2. config_full.yaml** - Complete configuration with all parameters and detailed comments
```bash
python main.py --config config_full.yaml
```

**3. config_autotune.yaml** - Configuration with automatic hyperparameter tuning enabled
```bash
python main.py --config config_autotune.yaml
```

**Override YAML settings with command-line arguments:**
```bash
# Command-line arguments take priority over YAML settings
python main.py --config config_basic.yaml --epochs 200 --batch_size 15
```

**Example: Using directory input with YAML**

You can set the database to a directory in your YAML file:
```yaml
# config_basic.yaml
output_dir: "results"
ligand: "inputs/reference.pdb"
protein: "inputs/protein.pdb"
database: "inputs/csvs/"  # Directory with multiple CSV files

batch_size: 10
n_cycles: 10
similarity: 0.2
```

Then run:
```bash
python main.py --config config_basic.yaml
```

## Parameters

### Required Arguments

- `-o OUTPUT_DIR` : Output directory for results
- `-l LIGAND_PDB` : Reference ligand PDB file for RMSD calculation
- `-p PROTEIN_PDB` : Target protein PDB file for docking
- `-d DATABASE` : Chemical database (CSV file or directory containing CSV files)
  - Single file: `inputs/candidates.csv`
  - Directory: `inputs/csvs/` (automatically merges all CSV files)

### Active Learning Parameters

- `-i INITIAL_CSV` : Initial training set CSV file (optional)
- `-s SIMILARITY` : Molecular similarity threshold (default: 0.2)
- `--hbond_similarity` : H-bond similarity threshold (default: 0.3)
- `--rmsd_threshold` : RMSD threshold for valid poses (default: 2.0)
- `--budget` : Total compound evaluation budget (default: 100,000)
- `--batch_size` : Compounds selected per cycle (default: 10)
- `--n_cycles` : Number of active learning cycles (default: 10)
- `--conf_num` : Conformers per molecule (default: 1)

### TabM Neural Network Parameters

- `--auto_tune` : Enable automatic hyperparameter tuning via Optuna
- `--tuning_method` : Tuning method: 'grid' or 'random' (default: 'random')
- `--learning_rate` : Learning rate for TabM (default: 0.001)
- `--dropout` : Dropout rate for regularization (default: 0.2)
- `--epochs` : Training epochs per cycle (default: 100)
- `--batch_size_nn` : Neural network batch size (default: 256)
- `--mc_dropout_samples` : Monte Carlo dropout samples for uncertainty (default: 50)

### Exploration Strategy

- `--initial_exploration_ratio` : Initial exploration ratio (default: 0.8)
- `--final_exploration_ratio` : Final exploration ratio (default: 0.2)

### System Settings

- `--max_threads` : Maximum CPU threads for parallel processing (default: 8)
- `--disable_element_filter` : Disable element filtering for docking compatibility
- `--disable_deduplication` : Disable SMILES-based molecular deduplication

## Results

The results are organized in the output directory as follows:

```
results/
├── final_results.csv                          # Top-ranked compounds with docking scores
├── active_learning.log                        # Detailed execution log
├── cycle_performance_summary.csv              # Performance metrics per cycle
├── prediction_analysis.csv                    # Model prediction analysis
├── cycle_1/                                   # First cycle results
│   ├── cycle_results.csv                      # Selected compounds for this cycle
│   ├── detailed_results_with_predictions.csv  # Predictions with uncertainty
│   └── detailed_results_before_filtering.csv  # Pre-filter results
├── cycle_2/                                   # Second cycle results
│   ├── cycle_results.csv
│   ├── detailed_results_with_predictions.csv
│   └── detailed_results_before_filtering.csv
├── ...
├── cycle_N/                                   # Final cycle results
└── pose/                                      # Molecular docking poses
    ├── pose_ZINC000001.pdbqt                  # Individual compound poses
    ├── pose_ZINC000002.pdbqt
    └── ...
```

**Key output files:**
- `final_results.csv`: Top-ranked compounds with columns: Catalog ID, smiles, docking_score, uncertainty, similarity, rmsd
- `active_learning.log`: Complete execution log with timestamps and performance metrics
- `cycle_performance_summary.csv`: R² scores, timing, and memory usage per cycle


## Contact

For more information, check the [CSB_Lab](https://www.csb-lab.net/)

---

*This framework combines active learning with molecular docking for efficient hit discovery in drug development.*
