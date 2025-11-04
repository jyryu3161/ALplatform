# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

[Code Writing Rules - Always Follow]

1. Do NOT include temporary or experimental code, test blocks, debug prints, or placeholder comments.
2. Avoid writing duplicate logic if a function or method already exists—reuse or refactor instead.
3. Do NOT create unnecessary new files; modify or add to the existing file and structure as requested.
4. Use try-except blocks only when necessary, with specific exception classes, and in the smallest possible scope.
5. Do NOT add unnecessary external libraries. If absolutely required, include a comment explaining why.
6. Write functions and classes concisely, including a clear docstring for purpose, parameters, and return values.
7. Follow standard code style guidelines (PEP8 for Python) for readability and consistency.
8. The provided code must be complete, executable, and modify only the requested scope.
9. Avoid over-compressing complex logic into single lines; prioritize readability.
10. Before adding potentially unsafe or unintended behaviors, explain the reasoning first.
11. Always answer in Korean.

[Documentation Rules]
12. Do NOT create multiple separate .md files for documentation.
13. Organize all project information, setup instructions, usage examples, and documentation in a single comprehensive [README.md](http://readme.md/) file in English.
14. Structure the [README.md](http://readme.md/) with clear sections including: Project Overview, Installation, Usage, API Documentation, Examples, and Contributing Guidelines.
15. If code modifications or feature updates change the usage, always update the [README.md](http://readme.md/) to the latest version before finalizing.


## System Overview

This is an Active Learning framework for molecular docking that uses the TabM (Tabular Machine Learning) deep neural network model. The system iteratively selects promising molecular candidates from large chemical databases, performs molecular docking simulations, and trains a TabM model to predict docking scores with uncertainty estimation using Monte Carlo dropout.

## Main Entry Point

Run the system using `main.py`:

```bash
# Basic usage
python main.py -o results -l inputs/reference.pdb -p inputs/protein.pdb -d inputs/candidates.csv

# With hyperparameter auto-tuning (recommended)
python main.py -o results -l inputs/reference.pdb -p inputs/protein.pdb -d inputs/candidates.csv --auto_tune --epochs 150

# Custom active learning parameters
python main.py -o results -l inputs/reference.pdb -p inputs/protein.pdb -d inputs/candidates.csv --batch_size 20 --n_cycles 15 --similarity 0.3
```

Required inputs:
- `-l/--ligand`: Reference ligand PDB file for RMSD and similarity calculations
- `-p/--protein`: Target protein PDB file for docking
- `-d/--database`: Chemical database CSV with 'Catalog ID' and 'smiles' columns
- `-o/--output_dir`: Output directory for all results

## Core Architecture

### ActiveLearningWithConformers (`Activedocking.py`)
Main orchestrator class that:
- Manages active learning cycles with exploration-exploitation balance
- Handles molecular filtering, deduplication, and conformer generation
- Coordinates between TabM model training and molecular docking
- Uses qvina2.1 for AutoDock Vina-based molecular docking
- Calculates RMSD similarity using Maximum Common Substructure (MCS)
- Analyzes hydrogen bond interactions for additional molecular similarity

### UnifiedModel (`unified_model.py`)
TabMNetwork implementation:
- PyTorch neural network with embedding layer and batch normalization
- Uses Monte Carlo dropout for uncertainty estimation during inference
- Supports automatic hyperparameter tuning via Optuna
- GPU acceleration when CUDA is available
- Configurable architecture: embedding → 256-unit hidden layer → scalar output

### Supporting Modules
- `RMSD.py`: MCS-based RMSD calculation between reference and candidate molecules
- `Hbond.py`: Hydrogen bond pattern analysis for molecular similarity
- `molecule_filter.py`: Filters molecules with unsupported elements for docking compatibility
- `cycle_monitor.py`: Performance monitoring and memory management across AL cycles
- `result_saver.py`: Handles output file generation and cycle result persistence
- `fingerprint.py`: Morgan fingerprint calculation with error handling

## Key Parameters

### Active Learning
- `--budget`: Total compound evaluation budget (default: 100,000)
- `--batch_size`: Compounds selected per cycle (default: 10)  
- `--n_cycles`: Number of active learning cycles (default: 10)
- `--similarity`: Molecular similarity threshold for candidate selection (default: 0.2)
- `--hbond_similarity`: H-bond similarity threshold (default: 0.3)
- `--rmsd_threshold`: RMSD threshold for valid poses (default: 2.0)

### TabM Neural Network
- `--learning_rate`: Adam optimizer learning rate (default: 0.001)
- `--dropout`: Dropout rate for regularization (default: 0.2)
- `--epochs`: Training epochs per cycle (default: 100)
- `--batch_size_nn`: Neural network batch size (default: 256)
- `--mc_dropout_samples`: Monte Carlo samples for uncertainty estimation (default: 50)

### Hyperparameter Tuning
- `--auto_tune`: Enable Optuna-based hyperparameter optimization
- `--tuning_method`: 'random' or 'grid' search strategy

## Output Structure

Results are saved in the specified output directory:
- `final_results.csv`: Top-ranked compounds with docking scores
- `cycle_*/`: Individual cycle results and intermediate data
- `pose/`: Molecular docking pose files  
- `active_learning.log`: Detailed execution log
- `cycle_performance_summary.csv`: Performance metrics per cycle

## External Dependencies

The system requires qvina2.1 binary located in `bin/qvina2.1` for molecular docking operations. Key Python dependencies include PyTorch, RDKit, ODDT, Optuna, and scikit-learn.

## Development Notes

- The system uses extensive memory management and cleanup between cycles
- All molecular processing includes error handling for invalid SMILES structures
- Fingerprint calculations are cached to improve performance
- The exploration ratio decreases linearly from initial to final values across cycles
- Model predictions include both point estimates and uncertainty quantification