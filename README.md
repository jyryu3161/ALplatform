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

### Chemical Database (CSV)

The database CSV file must contain the following columns:

```csv
Catalog ID,smiles
ZINC000001,CCCc1ccc(O)cc1
ZINC000002,CC(C)Cc1ccc(C(C)C(=O)O)cc1
ZINC000003,CC1=CC=C(C=C1)C(=O)O
```

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

Example with YAML configuration file:

```bash
# Create config.yml with your parameters
python main.py --config config.yml
```

**YAML Configuration Example:**

```yaml
# config.yml - Example configuration
output_dir: "results"
ligand: "inputs/reference.pdb"
protein: "inputs/protein.pdb"
database: "inputs/candidates.csv"

# Active Learning
similarity: 0.2
budget: 100000
batch_size: 10
n_cycles: 10

# TabM Neural Network
auto_tune: false
learning_rate: 0.001
dropout: 0.2
epochs: 100
mc_dropout_samples: 50

# System
max_threads: 8
```

## Parameters

### Required Arguments

- `-o OUTPUT_DIR` : Output directory for results
- `-l LIGAND_PDB` : Reference ligand PDB file for RMSD calculation
- `-p PROTEIN_PDB` : Target protein PDB file for docking
- `-d DATABASE_CSV` : Chemical database CSV file with 'Catalog ID' and 'smiles' columns

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

The result file is saved as `OUTPUT_DIRECTORY/final_results.csv` in CSV format. The report includes:
- `Catalog ID`: Unique compound identifier
- `smiles`: SMILES string of the compound
- `docking_score`: Predicted binding affinity
- `uncertainty`: Model uncertainty estimate
- `similarity`: Tanimoto similarity to reference ligand
- `rmsd`: RMSD to reference ligand pose

For sub-data:
- Active learning logs are saved as `OUTPUT_DIRECTORY/active_learning.log`
- Cycle-specific results are saved as `OUTPUT_DIRECTORY/cycle_*/`
- Model performance metrics are saved as `OUTPUT_DIRECTORY/cycle_performance_summary.csv`
- Molecular docking poses are saved as `OUTPUT_DIRECTORY/pose/pose_*.pdbqt`



## Contact

For more information, check the [CSB_Lab](https://www.csb-lab.net/)

---

*This framework combines active learning with molecular docking for efficient hit discovery in drug development.*
