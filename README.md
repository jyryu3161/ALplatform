# Active Learning for Molecular Docking

An active learning framework for efficient virtual screening using TabM neural networks with Monte Carlo dropout uncertainty estimation and qVina2.1 molecular docking.

## Using in Google Colaboratory

Active Learning for Molecular Docking is also available in [Google Colaboratory](https://colab.research.google.com/)
*(Coming soon)*

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

### Parameters

#### Required Arguments

- `-o OUTPUT_DIR` : Output directory for results
- `-l LIGAND_PDB` : Reference ligand PDB file for RMSD calculation
- `-p PROTEIN_PDB` : Target protein PDB file for docking
- `-d DATABASE_CSV` : Chemical database CSV file with 'Catalog ID' and 'smiles' columns

#### Active Learning Parameters

- `-i INITIAL_CSV` : Initial training set CSV file (optional)
- `-s SIMILARITY` : Molecular similarity threshold (default: 0.2)
- `--hbond_similarity` : H-bond similarity threshold (default: 0.3)
- `--rmsd_threshold` : RMSD threshold for valid poses (default: 2.0)
- `--budget` : Total compound evaluation budget (default: 100,000)
- `--batch_size` : Compounds selected per cycle (default: 10)
- `--n_cycles` : Number of active learning cycles (default: 10)
- `--conf_num` : Conformers per molecule (default: 1)

#### TabM Neural Network Parameters

- `--auto_tune` : Enable automatic hyperparameter tuning via Optuna
- `--tuning_method` : Tuning method: 'grid' or 'random' (default: 'random')
- `--learning_rate` : Learning rate for TabM (default: 0.001)
- `--dropout` : Dropout rate for regularization (default: 0.2)
- `--epochs` : Training epochs per cycle (default: 100)
- `--batch_size_nn` : Neural network batch size (default: 256)
- `--mc_dropout_samples` : Monte Carlo dropout samples for uncertainty (default: 50)

#### Exploration Strategy

- `--initial_exploration_ratio` : Initial exploration ratio (default: 0.8)
- `--final_exploration_ratio` : Final exploration ratio (default: 0.2)

#### System Settings

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

## YAML Configuration

Create a `config.yml` file for batch parameter settings:

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

Then run:
```bash
python main.py --config config.yml
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

## Architecture

The framework consists of modular components:

```
main.py                      # Entry point
Activedocking.py            # Active learning orchestrator
unified_model.py            # TabM neural network with MC dropout
docking_engine.py           # qVina2.1 integration
sample_acquisition.py       # Sample selection strategies
molecular_processing.py     # Molecular filtering and processing
data_loader.py             # Data loading and preprocessing
memory_manager.py          # Memory optimization
cycle_monitor.py           # Performance monitoring
result_saver.py            # Results management
```

Supporting modules:
- `RMSD.py` - MCS-based RMSD calculation
- `Hbond.py` - Hydrogen bond pattern analysis
- `molecule_filter.py` - Element filtering and validation
- `fingerprint.py` - Morgan fingerprint calculation
- `exceptions.py` - Custom exception handling

## Performance Notes

**Computational Requirements:**
- Recommended RAM: 4-8 GB for databases with 100,000 compounds
- CPU: Multi-core processor recommended (8+ cores)
- GPU: Optional, speeds up neural network training significantly

**Typical Performance:**
- Processing speed: ~1,000 compounds/hour (CPU) or ~5,000 compounds/hour (GPU)
- Model training: Minutes per cycle (GPU) or hours per cycle (CPU)
- Docking: Dominated by qVina2.1 performance (CPU-bound)

**qVina2.1 vs AutoDock Vina:**
qVina2.1 provides up to 20x faster docking with equivalent accuracy, making it ideal for high-throughput active learning workflows.

## Troubleshooting

### Installation Issues

If you encounter dependency conflicts:
```bash
# Clean conda cache
conda clean --all

# Retry installation
conda env remove -n active-learning-docking
conda env create -f environment.yml
```

### Missing qVina2.1

```bash
# Manual installation
wget https://github.com/QVina/qvina/raw/refs/heads/master/bin/qvina2.1
chmod +x qvina2.1
mkdir -p bin && mv qvina2.1 bin/
```

### Memory Issues

For memory-constrained systems:
```bash
python main.py --batch_size 5 --batch_size_nn 64
```

### Invalid SMILES

Check your database format:
```bash
# Validate SMILES strings
python -c "
import pandas as pd
from rdkit import Chem
df = pd.read_csv('inputs/candidates.csv')
invalid = [i for i, smi in enumerate(df['smiles']) if Chem.MolFromSmiles(smi) is None]
print(f'Invalid SMILES at rows: {invalid[:10]}')
"
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For more information, check the [GitHub repository](https://github.com/YOUR_USERNAME/ActiveLearning_only_tabm)

---

*This framework combines active learning with molecular docking for efficient hit discovery in drug development.*
