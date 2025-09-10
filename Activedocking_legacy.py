import os
import csv
import logging
import numpy as np
import pandas as pd
import optuna
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import __version__ as rdkit_version
from oddt import toolkit
from oddt.docking import autodock_vina
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, RegressorMixin
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gc
import warnings
import contextlib
warnings.filterwarnings('ignore')
import json

try:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    warnings.filterwarnings("ignore", message=".*GetValence.*")
    warnings.filterwarnings("ignore", message=".*MorganGenerator.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="rdkit")
except ImportError:
    pass

from RMSD import MCSRMSDCalculator
from Hbond import HBondCalculator
from molecule_filter import MoleculeFilter, MoleculeDeduplicator
from cycle_monitor import CyclePerformanceMonitor, log_memory_usage
from unified_model import UnifiedModel
from result_saver import ResultSaverMixin
from fingerprint import get_morgan_fingerprint_safe

class ActiveLearningWithConformers(ResultSaverMixin):
    def __init__(self, csv_file, protein_file, ligand_file, output_dir, 
                 similarity_threshold=0.2, budget=100000, batch_size=2500, n_cycles=10, 
                 conf_num=1, hbond_similarity_threshold=0.5, initial_csv=None, 
                 enable_element_filter=True, enable_deduplication=True,
                 # 모델 관련 파라미터
                 auto_tune=False, tuning_method='random', 
                 max_threads=8,
                 learning_rate=1e-3, dropout=0.2, epochs=100, model_batch_size=256,
                 mc_dropout_samples=50,
                 # Active Learning 파라미터
                 initial_exploration_ratio=0.8, final_exploration_ratio=0.2):
        super().__init__()
        self.csv_file = csv_file
        self.initial_csv = initial_csv
        self.output_dir = output_dir
        self.ligand_file = ligand_file
        self.budget = budget
        self.batch_size = batch_size
        self.n_cycles = n_cycles
        self.conf_num = conf_num
        self.similarity_threshold = similarity_threshold
        self.hbond_similarity_threshold = hbond_similarity_threshold
        self.enable_element_filter = enable_element_filter
        self.enable_deduplication = enable_deduplication
        self.initial_exploration_ratio = initial_exploration_ratio
        self.final_exploration_ratio = final_exploration_ratio
        self.exploration_ratio = initial_exploration_ratio
        self.docking_scores = {}
        self.interaction_data = {}
        self.structure_info = {}
        self.invalid_smiles = []
        self.initial_candidates = {}
        self.fingerprint_cache = {}
        self.fingerprint_computed = False
        self.predicted_scores = {}  # {cid: predicted_score}
        self.prediction_uncertainty = {}  # {cid: std_prediction}
        self.acquisition_scores = {}  # {cid: acquisition_score}
        self.cycle_predictions = {}  # {cycle: {cid: prediction_data}}
        self.memory_cleanup_interval = 5  # 5사이클마다 정리
        self.max_cycle_predictions_history = 1  
        self.pose_dir = os.path.join(output_dir, 'pose')
        os.makedirs(self.pose_dir, exist_ok=True)
        self.cycle_monitor = CyclePerformanceMonitor(output_dir)
        
        if self.enable_deduplication:
            self.deduplicator = MoleculeDeduplicator()
            logging.info("SMILES-based deduplication enabled")
        else:
            self.deduplicator = None
            
        if self.enable_element_filter:
            self.molecule_filter = MoleculeFilter()
            logging.info("Element filtering enabled")
        else:
            self.molecule_filter = None
        
        if not self.initialize_components(protein_file, ligand_file):
            raise ValueError("Failed to initialize components")
        
        if not hasattr(self, 'predicted_scores'):
            self.predicted_scores = {}
        if not hasattr(self, 'prediction_uncertainty'):
            self.prediction_uncertainty = {}
        if not hasattr(self, 'acquisition_scores'):
            self.acquisition_scores = {}
        if not hasattr(self, 'cycle_predictions'):
            self.cycle_predictions = {}
        
        self.model = UnifiedModel(
            auto_tune=auto_tune,
            tuning_method=tuning_method,
            max_threads=max_threads,
            learning_rate=learning_rate,
            dropout=dropout,
            epochs=epochs,
            batch_size=model_batch_size,
            mc_dropout_samples=mc_dropout_samples
        )
        
        logging.info(f"Model initialized: {self.model.model_type}")

    def cleanup_memory(self, force_cleanup=False):
        try:
            if len(self.cycle_predictions) > self.max_cycle_predictions_history:
                sorted_cycles = sorted(self.cycle_predictions.keys())
                cycles_to_remove = sorted_cycles[:-self.max_cycle_predictions_history]
                
                for cycle in cycles_to_remove:
                    del self.cycle_predictions[cycle]
                    logging.debug(f"Removed cycle {cycle} predictions from memory")
            
            if force_cleanup and hasattr(self, 'fingerprint_cache'):
                active_cids = set([self.cids[i] for i in self.training_indices + self.screening_indices])
                
                cache_keys_to_remove = []
                for cid in self.fingerprint_cache:
                    if cid not in active_cids:
                        cache_keys_to_remove.append(cid)
                
                for cid in cache_keys_to_remove:
                    del self.fingerprint_cache[cid]
                
                if cache_keys_to_remove:
                    logging.info(f"Cleaned {len(cache_keys_to_remove)} unused fingerprints from cache")
            
            gc.collect()
            
            log_memory_usage(f"after cleanup (cycle {getattr(self, 'current_cycle', 'unknown')})")
            
        except Exception as e:
            logging.warning(f"Memory cleanup failed: {str(e)}")

    def should_cleanup_memory(self, cycle_number):
        return cycle_number % self.memory_cleanup_interval == 0

    def deduplicate_by_smiles(self):
        if not self.enable_deduplication or self.deduplicator is None:
            logging.info("SMILES deduplication disabled")
            return 0
        
        original_count = len(self.structure_info)
        logging.info(f"Starting SMILES deduplication for {original_count:,} compounds...")
        
        self.structure_info, duplicate_count = self.deduplicator.deduplicate_dictionary(self.structure_info)
        
        removed_cids = set(self.deduplicator.get_removed_compounds())
        
        self.deduplicator.filter.cleanup_removed_compounds(
            removed_cids,
            self.docking_scores,
            self.interaction_data,
            getattr(self, 'fingerprint_cache', None),
            getattr(self, 'molecule_labels', None)
        )
        
        self.cids = list(self.structure_info.keys())
        self.training_indices = []
        self.screening_indices = list(range(len(self.cids)))
        
        final_count = len(self.structure_info)
        dedup_rate = (duplicate_count / original_count * 100) if original_count > 0 else 0
        
        logging.info(f"SMILES deduplication completed:")
        logging.info(f" Unique compounds: {final_count:,}")
        logging.info(f" Duplicates removed: {duplicate_count:,} ({dedup_rate:.1f}%)")
        logging.info(f" Original total: {original_count:,}")
        
        return duplicate_count
        
    def save_deduplication_report(self):
        if self.enable_deduplication and self.deduplicator is not None:
            self.deduplicator.save_report(self.output_dir)

    def filter_molecules_by_elements(self, molecule_dict):
        if not self.enable_element_filter or self.molecule_filter is None:
            return molecule_dict
        
        total_compounds = len(molecule_dict)
        logging.info(f"Applying element filtering for {total_compounds:,} compounds...")
        
        filtered_dict, _ = self.molecule_filter.filter_molecule_dict(molecule_dict)

        return filtered_dict

    def load_initial_candidates(self):
        if not self.initial_csv or not os.path.exists(self.initial_csv):
            logging.info("No initial CSV file provided or file doesn't exist")
            return {}
        
        try:
            df = pd.read_csv(self.initial_csv)
            if 'Catalog ID' not in df.columns or 'smiles' not in df.columns:
                logging.error("Initial CSV file must contain 'Catalog ID' and 'smiles' columns")
                return {}
            
            df = df.dropna(subset=['Catalog ID', 'smiles'])
            original_count = len(df)
            
            # 1단계: 중복 제거 (SMILES 기반) - 가장 효율적인 첫 번째 필터
            logging.info(f"Initial CSV: Starting with {original_count:,} compounds")
            
            initial_raw = {}
            for _, row in df.iterrows():
                cid = str(row['Catalog ID'])
                smiles = row['smiles']
                initial_raw[cid] = smiles
            
            # SMILES 기반 중복 제거 수행
            if self.enable_deduplication and self.deduplicator is not None:
                logging.info("Applying SMILES deduplication to initial CSV...")
                initial_dedup, duplicate_count = self.deduplicator.deduplicate_dictionary(initial_raw)
                dedup_rate = (duplicate_count / original_count * 100) if original_count > 0 else 0
                logging.info(f"Initial CSV deduplication: {original_count:,} → {len(initial_dedup):,} ({dedup_rate:.1f}% duplicates removed)")
            else:
                initial_dedup = initial_raw
                logging.info("SMILES deduplication disabled for initial CSV")
            
            # 2단계: 유사도 필터링 - 중복 제거 후 더 적은 수에 대해 핑거프린트 계산
            ref_mol = next(toolkit.readfile('pdb', self.ligand_file))
            ref_smiles = ref_mol.write('smi').strip()
            ref_rdkit_mol = Chem.MolFromSmiles(ref_smiles)
            ref_fp = get_morgan_fingerprint_safe(ref_rdkit_mol, 2, 1024)
            
            if ref_fp is None:
                logging.error("Failed to generate reference fingerprint")
                return {}
            
            initial_similar = {}
            fingerprint_calculated = 0
            fingerprint_failed = 0
            
            for cid, smiles in tqdm(initial_dedup.items(), desc="Similarity filtering (initial)"):
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    self.invalid_smiles.append(smiles)
                    fingerprint_failed += 1
                    continue
                
                mol_fp = get_morgan_fingerprint_safe(mol, 2, 1024)
                if mol_fp is None:
                    fingerprint_failed += 1
                    continue
                
                fingerprint_calculated += 1
                similarity = DataStructs.TanimotoSimilarity(ref_fp, mol_fp)
                
                if similarity >= self.similarity_threshold:
                    initial_similar[cid] = smiles
            
            logging.info(f"Initial CSV similarity filtering: {len(initial_dedup):,} → {len(initial_similar):,} compounds")
            logging.info(f"Fingerprints calculated: {fingerprint_calculated:,}, Failed: {fingerprint_failed:,}")
            
            # 3단계: 원소 필터링 - 가장 빠른 마지막 필터
            initial_candidates = self.filter_molecules_by_elements(initial_similar)
            final_count = len(initial_candidates)
            
            logging.info(f"Initial CSV processing completed:")
            logging.info(f"  Original: {original_count:,} → Final: {final_count:,} compounds")
            logging.info(f"  Overall efficiency: {(final_count/original_count*100):.1f}% retained")
            
            del df, initial_raw, initial_dedup, initial_similar, ref_fp
            gc.collect()
            
            return initial_candidates
            
        except Exception as e:
            logging.error(f"Error loading initial CSV: {str(e)}")
            return {}

    def initialize_components(self, protein_file, ligand_file):
        self.pdb_info = self.read_pdb_info(protein_file)
        if not self.pdb_info:
            logging.error("Failed to load PDB info")
            return False
        
        toolkit.backend = 'rdkit'
        self.protein = next(toolkit.readfile('pdb', protein_file))
        self.protein.protein = True
        self.ligand = next(toolkit.readfile('pdb', ligand_file))
        self.vina = autodock_vina(
            protein=self.protein,
            auto_ligand=self.ligand,
            size=(20, 20, 20),
            num_modes=3,
            exhaustiveness=8,
            energy_range=5,
            n_cpu=16,
            executable='./bin/qvina2.1'
        )

        ref_mol = Chem.MolFromPDBFile(ligand_file)
        self.mcs_calculator = MCSRMSDCalculator(ref_mol)
        self.hbond_calculator = HBondCalculator(protein_file, ligand_file)
        return True

    def read_pdb_info(self, pdb_file):
        atom_residue_info = {}
        if not os.path.exists(pdb_file):
            logging.error(f"PDB file not found: {pdb_file}")
            return {}

        with open(pdb_file, 'r') as fp:
            for line in fp:
                if not (line.startswith('ATOM') or line.startswith('HETATM')):
                    continue
                    
                atom_num = int(line[6:11].strip() or 0)
                residue_name = line[17:20].strip() or "UNK"
                chain_id = line[21:22].strip() or "X"
                residue_num = line[22:26].strip() or "0"
                
                if all([atom_num, residue_name, chain_id, residue_num]):
                    atom_residue_info[atom_num] = f'{residue_name}_{chain_id}_{residue_num}'

        return atom_residue_info

    def load_candidates(self):
        # 최적화된 필터링 순서로 Initial CSV 처리
        self.initial_candidates = self.load_initial_candidates()
        
        # 메인 라이브러리 처리 시작
        df = pd.read_csv(self.csv_file)
        if 'Catalog ID' not in df.columns or 'smiles' not in df.columns:
            logging.error("Main CSV file must contain 'Catalog ID' and 'smiles' columns")
            return False
        
        df = df.dropna(subset=['Catalog ID', 'smiles'])
        original_main_count = len(df)

        # 1단계: 메인 라이브러리 중복 제거 (SMILES 기반)
        logging.info(f"Main library: Starting with {original_main_count:,} compounds")
        
        main_raw = {}
        for _, row in df.iterrows():
            cid = str(row['Catalog ID'])
            smiles = row['smiles']
            main_raw[cid] = smiles
        
        # SMILES 기반 중복 제거 수행
        if self.enable_deduplication and self.deduplicator is not None:
            logging.info("Applying SMILES deduplication to main library...")
            main_dedup, main_duplicate_count = self.deduplicator.deduplicate_dictionary(main_raw)
            main_dedup_rate = (main_duplicate_count / original_main_count * 100) if original_main_count > 0 else 0
            logging.info(f"Main library deduplication: {original_main_count:,} → {len(main_dedup):,} ({main_dedup_rate:.1f}% duplicates removed)")
        else:
            main_dedup = main_raw
            logging.info("SMILES deduplication disabled for main library")

        # 2단계: 유사도 필터링 - 중복 제거 후 더 적은 수에 대해 핑거프린트 계산
        ref_mol = next(toolkit.readfile('pdb', self.ligand_file))
        ref_smiles = ref_mol.write('smi').strip()
        ref_rdkit_mol = Chem.MolFromSmiles(ref_smiles)
        ref_fp = get_morgan_fingerprint_safe(ref_rdkit_mol, 2, 1024)
        
        if ref_fp is None:
            logging.error("Failed to generate reference fingerprint")
            return False
        
        main_similar = {}
        main_fingerprint_calculated = 0
        main_fingerprint_failed = 0
        
        for cid, smiles in tqdm(main_dedup.items(), desc="Similarity filtering (main library)"):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                self.invalid_smiles.append(smiles)
                main_fingerprint_failed += 1
                continue

            mol_fp = get_morgan_fingerprint_safe(mol, 2, 1024)
            if mol_fp is None:
                main_fingerprint_failed += 1
                continue
                
            main_fingerprint_calculated += 1
            similarity = DataStructs.TanimotoSimilarity(ref_fp, mol_fp)

            if similarity >= self.similarity_threshold:
                main_similar[cid] = smiles
        
        logging.info(f"Main library similarity filtering: {len(main_dedup):,} → {len(main_similar):,} compounds")
        logging.info(f"Fingerprints calculated: {main_fingerprint_calculated:,}, Failed: {main_fingerprint_failed:,}")
        
        # 3단계: 원소 필터링 - 가장 빠른 마지막 필터
        main_library = self.filter_molecules_by_elements(main_similar)
        
        logging.info(f"Main library processing completed:")
        logging.info(f"  Original: {original_main_count:,} → Final: {len(main_library):,} compounds")
        logging.info(f"  Overall efficiency: {(len(main_library)/original_main_count*100):.1f}% retained")
        
        # 라이브러리 병합 및 최종 통계
        all_compounds = main_library.copy()
        
        overlap_count = 0
        for cid, smiles in self.initial_candidates.items():
            if cid in all_compounds:
                overlap_count += 1
            else:
                all_compounds[cid] = smiles
        
        logging.info(f"Combined library: {len(all_compounds)} compounds")
        logging.info(f"  - From main library: {len(main_library)}")
        logging.info(f"  - From initial CSV (unique): {len(self.initial_candidates) - overlap_count}")
        logging.info(f"  - Overlapping compounds: {overlap_count}")

        self.structure_info = all_compounds
        self.cids = list(self.structure_info.keys())
        self.training_indices = []
        self.screening_indices = list(range(len(self.cids)))
        
        # 전체 라이브러리 최종 통계 및 효율성 리포트
        initial_csv_count = 0
        if self.initial_csv and os.path.exists(self.initial_csv):
            try:
                initial_csv_count = len(pd.read_csv(self.initial_csv).dropna(subset=['Catalog ID', 'smiles']))
            except:
                initial_csv_count = 0
                
        total_initial = original_main_count + initial_csv_count
        final_total = len(all_compounds)
        
        logging.info(f"="*60)
        logging.info(f"OPTIMIZED FILTERING PIPELINE COMPLETED")
        logging.info(f"="*60)
        logging.info(f"Combined Library Final Statistics:")
        logging.info(f"  Total Original: {total_initial:,} compounds") 
        logging.info(f"  Final Combined: {final_total:,} compounds")
        logging.info(f"  Overall Efficiency: {(final_total/total_initial*100):.1f}% retained" if total_initial > 0 else "  Overall Efficiency: N/A")
        logging.info(f"  Processing Order: Deduplication → Similarity → Element Filtering")
        logging.info(f"="*60)
        
        del df, main_raw, main_dedup, main_similar, main_library, ref_fp
        gc.collect()
        
        return True

    def generate_conformers(self, cid, smiles):
        mol = None
        test_mol = None
        conformers = []
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logging.warning(f"Failed to create molecule from SMILES for {cid}")
                return None

            mol = Chem.AddHs(mol)
            
            test_mol = Chem.Mol(mol)
            embed_result = AllChem.EmbedMolecule(test_mol)
            if embed_result != 0:
                logging.warning(f"Failed to embed molecule for {cid}")
                return None
            
            conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=self.conf_num)
            
            if not conf_ids:
                logging.warning(f"Failed to generate conformers for {cid}")
                return [test_mol]
                
            mmff_results = AllChem.MMFFOptimizeMoleculeConfs(mol)
            
            for conf_id in conf_ids:
                conf_mol = Chem.Mol(mol, False, conf_id)
                
                if mmff_results and conf_id < len(mmff_results):
                    if mmff_results[conf_id][0] != 0:  
                        logging.debug(f"MMFF optimization failed for conformer {conf_id} of {cid}")
                
                conformers.append(conf_mol)
            
            if not conformers:
                logging.warning(f"No conformers generated for {cid}")
                return [test_mol]  
            
            logging.info(f"Generated {len(conformers)} conformers for {cid}")
            return conformers
            
        except Exception as e:
            logging.error(f"Error generating conformers for {cid}: {str(e)}")
            return None
            
        finally:
            try:
                if mol is not None:
                    del mol
                if test_mol is not None:
                    del test_mol
            except:
                pass  
            
            if hasattr(self, '_conformer_count'):
                self._conformer_count += 1
                if self._conformer_count % 100 == 0:
                    gc.collect()
            else:
                self._conformer_count = 1

    def get_docking_score(self, cid):
        if cid in self.docking_scores:
            return self.docking_scores[cid]
        
        self.interaction_data[cid] = {
            'mcs_size': 0,
            'mcs_similarity': 0.0,
            'rmsd': float('inf'),
            'hbond_similarity': 0.0
        }
        
        smiles = self.structure_info[cid]
        conformers = self.generate_conformers(cid, smiles)
            
        if not conformers:
            logging.debug(f"Failed to generate conformers for {cid}")
            return None
            
        valid_poses = []
            
        for conf_idx, conf in enumerate(conformers):
            mol_block = Chem.MolToMolBlock(conf)
            oddt_mol = toolkit.readstring('sdf', mol_block)
            
            results = None
            
            if oddt_mol is not None:
                try:
                    with open(os.devnull, 'w') as f, contextlib.redirect_stderr(f):
                        results = self.vina.dock(oddt_mol)
                except (AssertionError, Exception) as e:
                    logging.debug(f"Docking failed for {cid}: {str(e)}")
                    continue
                
            if not results:
                continue
                    
            for pose in results:
                try:
                    score = float(pose.data['vina_affinity'])
                    pose_mol = Chem.MolFromPDBBlock(pose.write('pdb'))
                        
                    if pose_mol is None:
                        continue
                            
                    mcs_size, mcs_similarity, rmsd = self.mcs_calculator.calculate_mcs_and_rmsd(pose_mol)
                    hbond_similarity = self.hbond_calculator.calculate_hbond_similarity(pose_mol)
                        
                    valid_poses.append((score, rmsd, pose, (mcs_size, mcs_similarity, rmsd, hbond_similarity), conf_idx))
                except Exception as e:
                    logging.debug(f"Pose processing failed for {cid}: {str(e)}")
                    continue
        
        if not valid_poses:
            logging.debug(f"No valid poses found for {cid}")
            return None
            
        try:
            min_rmsd = min(pose[1] for pose in valid_poses)
            rmsd_threshold = min_rmsd * 1.5
                
            good_rmsd_poses = [pose for pose in valid_poses if pose[1] <= rmsd_threshold]
            if not good_rmsd_poses:
                good_rmsd_poses = valid_poses
                
            best_pose = min(good_rmsd_poses, key=lambda x: x[0])
            best_score, best_rmsd, best_pose_obj, best_data, best_conf_idx = best_pose
                
            self.docking_scores[cid] = best_score
            self.interaction_data[cid] = {
                'mcs_size': best_data[0],
                'mcs_similarity': best_data[1],
                'rmsd': best_data[2],
                'hbond_similarity': best_data[3]
            }
                
            pose_path = os.path.join(self.pose_dir, f'pose_{cid}.pdbqt')
            best_pose_obj.write('pdbqt', pose_path, overwrite=True)
                
            logging.debug(f"Selected best pose for {cid} - Score: {best_score:.3f}, "
                        f"RMSD: {best_rmsd:.3f}, HBond Similarity: {best_data[3]:.3f},"
                        f"Conformer: {best_conf_idx}, Total poses: {len(valid_poses)}")
                
            return best_score
        except Exception as e:
            logging.debug(f"Final processing failed for {cid}: {str(e)}")
            return None

    def initialize(self, n_initial=None):
        if n_initial is None:
            n_initial = self.batch_size * 2
            
        initial_from_csv = []
        
        if self.initial_candidates:
            available_initial = []
            for cid in self.initial_candidates:
                if cid in self.structure_info: 
                    idx = self.cids.index(cid)
                    if idx in self.screening_indices:
                        available_initial.append(idx)
            
            n_from_initial = min(len(available_initial), n_initial)
            if n_from_initial > 0:
                initial_from_csv = np.random.choice(available_initial, n_from_initial, replace=False).tolist()
                logging.info(f"Selected {n_from_initial} compounds from initial CSV")
        
        remaining_needed = n_initial - len(initial_from_csv)
        remaining_indices = list(set(self.screening_indices) - set(initial_from_csv))
        
        additional_from_library = []
        if remaining_needed > 0 and remaining_indices:
            n_from_library = min(remaining_needed, len(remaining_indices))
            additional_from_library = np.random.choice(remaining_indices, n_from_library, replace=False).tolist()
            logging.info(f"Selected {n_from_library} additional compounds from main library")
        
        self.training_indices = initial_from_csv + additional_from_library
        self.screening_indices = list(set(self.screening_indices) - set(self.training_indices))
        
        total_initial = len(self.training_indices)
        logging.info(f"Total initial training set: {total_initial} compounds")

        if total_initial == 0:
            logging.error("No compounds selected for initial training")
            return False

        logging.info("Performing initial docking...")
        successful_docking = 0
        failed_docking = 0
        
        for idx in tqdm(self.training_indices, desc="Initial docking"):
            cid = self.cids[idx]
            score = self.get_docking_score(cid)
            
            if score is not None:
                successful_docking += 1
            else:
                failed_docking += 1
                
        valid_initial = sum(1 for cid in self.docking_scores 
                            if cid in self.interaction_data 
                            and self.interaction_data[cid]['rmsd'] <= 2.0 and
                            self.interaction_data[cid]['hbond_similarity'] >= self.hbond_similarity_threshold)
                            
        logging.info(f"Initial docking completed. Successful: {successful_docking}, Failed: {failed_docking}")
        logging.info(f"Found {valid_initial} valid poses (RMSD ≤ 2.0, HBond similarity ≥ {self.hbond_similarity_threshold})")
        
        if successful_docking < 2:
            logging.error(f"Insufficient successful docking results: {successful_docking}")
            return False
                        
        return True

    def precompute_all_fingerprints(self):
        if self.fingerprint_computed:
            logging.info("Fingerprints already computed, skipping...")
            return True
            
        total_compounds = len(self.cids)
        logging.info(f"Pre-computing ECFP fingerprints for {total_compounds:,} compounds...")
        
        successful_count = 0
        failed_count = 0
        
        batch_size = 1000
        
        with tqdm(range(0, len(self.cids), batch_size), 
                desc="Computing ECFP fingerprints", 
                unit="batches",
                ncols=100) as pbar:
            
            for start_idx in pbar:
                end_idx = min(start_idx + batch_size, len(self.cids))
                batch_cids = self.cids[start_idx:end_idx]
                
                batch_success = 0
                batch_failed = 0
                
                for cid in batch_cids:
                    try:
                        smiles = self.structure_info[cid]
                        mol = Chem.MolFromSmiles(smiles)
                        
                        if mol is None:
                            batch_failed += 1
                            continue
                        
                        fp = get_morgan_fingerprint_safe(mol, 2, 1024)
                        if fp is None:
                            batch_failed += 1
                            continue
                            
                        fp_list = list(fp)
                        self.fingerprint_cache[cid] = np.array(fp_list, dtype=np.float32)
                        batch_success += 1
                        
                    except Exception as e:
                        batch_failed += 1
                        logging.debug(f"Error computing fingerprint for {cid}: {str(e)}")
                        continue
                
                successful_count += batch_success
                failed_count += batch_failed
                
                gc.collect()
                
                if (successful_count + failed_count) > 0:
                    success_rate = successful_count / (successful_count + failed_count) * 100
                    pbar.set_postfix({
                        'Success': f'{successful_count:,}',
                        'Failed': f'{failed_count:,}', 
                        'Rate': f'{success_rate:.1f}%'
                    })
        
        self.fingerprint_computed = True
        
        success_rate = successful_count / (successful_count + failed_count) * 100 if (successful_count + failed_count) > 0 else 0
        
        logging.info(f"ECFP computation completed:")
        logging.info(f" Successful: {successful_count:,} ({success_rate:.1f}%)")
        logging.info(f" Failed: {failed_count:,}")
        logging.info(f" Total processed: {successful_count + failed_count:,}")
        
        return successful_count > 0
    
    def get_fingerprints(self, cid):
        if not self.fingerprint_computed:
            raise RuntimeError("Fingerprints not pre-computed. Call precompute_all_fingerprints() first.")
        
        if cid not in self.fingerprint_cache:
            logging.warning(f"Fingerprint not found for CID: {cid}")
            return None
        
        return self.fingerprint_cache[cid]

    def get_fingerprints_batch(self, cid_list):
        fingerprints = []
        valid_indices = []
        
        for i, cid in enumerate(cid_list):
            fp = self.get_fingerprints(cid)
            if fp is not None:
                fingerprints.append(fp)
                valid_indices.append(i)
        
        if fingerprints:
            return np.array(fingerprints, dtype=np.float32), valid_indices
        else:
            return np.array([]), valid_indices

    def train_model(self):
        log_memory_usage("before model training")
        
        # 유효한 훈련 데이터 준비
        valid_indices = [idx for idx in self.training_indices 
                        if self.cids[idx] in self.docking_scores 
                        and self.cids[idx] in self.interaction_data
                        and self.interaction_data[self.cids[idx]]['rmsd'] <= 2.0
                        and self.interaction_data[self.cids[idx]]['hbond_similarity'] >= self.hbond_similarity_threshold]
        
        if len(valid_indices) < 5:
            logging.warning(f"Insufficient valid samples for training: {len(valid_indices)}")
            return False
        
        cid_list = [self.cids[i] for i in valid_indices]
        X, valid_fp_indices = self.get_fingerprints_batch(cid_list)
        
        final_valid_indices = [valid_indices[i] for i in valid_fp_indices]
        y = np.array([self.docking_scores[self.cids[i]] for i in final_valid_indices], dtype=np.float32)
        
        if len(X) == 0 or len(y) == 0:
            logging.warning("Empty training data")
            return False
        
        logging.info(f"Training {self.model.model_type} with {len(y)} compounds")
        
        self.model.fit(X, y)
        
        metrics = self.model.evaluate_all_metrics(X, y)
        
        logging.info(f"Training Performance:")
        logging.info(f"  - R²: {metrics['r2']:.4f}")
        logging.info(f"  - MAE: {metrics['mae']:.4f}")
        logging.info(f"  - RMSE: {metrics['rmse']:.4f}")
        logging.info(f"  - Samples: {metrics['n_samples']}")

        if hasattr(self, 'current_cycle') and self.current_cycle % 5 == 0: 
            self.save_trained_model(self.current_cycle)
        
        del X, y, cid_list
        gc.collect()
        log_memory_usage("after model training")
        
        return True, metrics  # 성능 지표도 함께 반환

    def save_trained_model(self, cycle_number=None):
        if not self.model.is_fitted:
            logging.warning("Model is not trained yet, cannot save")
            return
        
        if cycle_number is not None:
            model_dir = os.path.join(self.output_dir, f'models', f'cycle_{cycle_number}')
        else:
            model_dir = os.path.join(self.output_dir, 'models', 'final')
        
        try:
            self.model.save_model(model_dir)
            
            additional_info = {
                'cycle': cycle_number,
                'total_training_samples': len(self.training_indices),
                'valid_training_samples': len([idx for idx in self.training_indices 
                                            if self.cids[idx] in self.docking_scores]),
                'similarity_threshold': self.similarity_threshold,
                'hbond_similarity_threshold': self.hbond_similarity_threshold,
                'exploration_ratio': getattr(self, 'exploration_ratio', 'N/A')
            }
            
            info_file = os.path.join(model_dir, 'training_info.json')
            with open(info_file, 'w') as f:
                json.dump(additional_info, f, indent=2)
            
            logging.info(f"Model saved for cycle {cycle_number if cycle_number else 'final'}")
            
        except Exception as e:
            logging.error(f"Failed to save model: {str(e)}")

    def save_best_model(self):
        best_model_dir = os.path.join(self.output_dir, 'models', 'best')
        
        try:
            self.model.save_model(best_model_dir)
            
            if hasattr(self.model, 'best_params_') and self.model.best_params_:
                best_info = {
                    'best_parameters': self.model.best_params_,
                    'model_type': self.model.model_type,
                    'auto_tuned': self.model.auto_tune,
                    'tuning_method': self.model.tuning_method,
                    'final_training_size': len(self.training_indices)
                }
                
                best_info_file = os.path.join(best_model_dir, 'best_model_info.json')
                with open(best_info_file, 'w') as f:
                    json.dump(best_info, f, indent=2)
            
            logging.info("Best model saved")
            
        except Exception as e:
            logging.error(f"Failed to save best model: {str(e)}")

    def acquire_samples(self):
        if not self.screening_indices:
            logging.warning("No screening samples available")
            return 0
        
        log_memory_usage("before sample acquisition")
        
        batch_size = 5000 
        all_scores = []

        cycle_predictions = {}

        for start_idx in range(0, len(self.screening_indices), batch_size):
            end_idx = min(start_idx + batch_size, len(self.screening_indices))
            batch_indices = self.screening_indices[start_idx:end_idx]
            
            batch_cids = [self.cids[i] for i in batch_indices]
            X_batch, valid_fp_indices = self.get_fingerprints_batch(batch_cids)
            
            if len(X_batch) == 0:
                continue
            
            mean_predictions, std_predictions = self.model.predict(X_batch)
            
            exploit_scores = np.zeros_like(mean_predictions)
            if len(mean_predictions) > 1 and np.max(mean_predictions) > np.min(mean_predictions):
                exploit_scores = (mean_predictions - np.min(mean_predictions)) / (np.max(mean_predictions) - np.min(mean_predictions))

            explore_scores = np.zeros_like(std_predictions)
            if len(std_predictions) > 1 and np.max(std_predictions) > np.min(std_predictions):
                explore_scores = (std_predictions - np.min(std_predictions)) / (np.max(std_predictions) - np.min(std_predictions))
            
            acquisition_scores = (1 - self.exploration_ratio) * exploit_scores + self.exploration_ratio * explore_scores
            
            batch_valid_indices = [batch_indices[i] for i in valid_fp_indices]

            for i, (idx, pred_score, std_score, acq_score) in enumerate(zip(
                batch_valid_indices, mean_predictions, std_predictions, acquisition_scores)):
                cid = self.cids[idx]
                
                self.predicted_scores[cid] = float(pred_score)
                self.prediction_uncertainty[cid] = float(std_score)
                self.acquisition_scores[cid] = float(acq_score)
                
                cycle_predictions[cid] = {
                    'predicted_score': float(pred_score),
                    'uncertainty': float(std_score),
                    'acquisition_score': float(acq_score),
                    'exploitation_score': float(exploit_scores[i]),
                    'exploration_score': float(explore_scores[i])
                }

            all_scores.extend(zip(batch_valid_indices, acquisition_scores))
            
            del X_batch, mean_predictions, std_predictions, acquisition_scores
            gc.collect()

        if hasattr(self, 'current_cycle'):
            self.cycle_predictions[self.current_cycle] = cycle_predictions

        if not all_scores:
            logging.warning("No valid fingerprints found for screening samples")
            return 0
        
        sorted_scores = sorted(all_scores, key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, _ in sorted_scores[:self.batch_size*2]]
        
        cycle_compounds = []
        candidate_results = []
        failed_docking = 0
        
        for idx in tqdm(selected_indices, desc="Evaluating new samples"):
            cid = self.cids[idx]
            cycle_compounds.append(cid)  
            score = self.get_docking_score(cid)
            
            if score is None:
                failed_docking += 1
                continue
                
            if cid in self.interaction_data:
                rmsd = self.interaction_data[cid]['rmsd']
                hbond_similarity = self.interaction_data[cid]['hbond_similarity']
                
                if rmsd <= 2.0 and hbond_similarity >= self.hbond_similarity_threshold:
                    candidate_results.append((idx, cid, score, rmsd, hbond_similarity))
        
        if hasattr(self, 'current_cycle'):
            if hasattr(self, 'cycle_predictions') and self.current_cycle in self.cycle_predictions:
                self.save_cycle_detailed_results_with_predictions(self.current_cycle, cycle_compounds)
            else:
                self.save_cycle_detailed_results(self.current_cycle, cycle_compounds)

        if failed_docking > 0:
            logging.info(f"Docking failed for {failed_docking} compounds in this cycle")

        if not candidate_results:
            logging.warning("No compounds passed quality filters (RMSD ≤ 2.0, HBond similarity ≥ threshold)")
            return 0
        
        candidate_results.sort(key=lambda x: x[2])

        n_select = min(self.batch_size, len(candidate_results))
        selected_samples = candidate_results[:n_select]

        valid_samples = [result[0] for result in selected_samples] 

        if valid_samples:
            self.training_indices.extend(valid_samples)
            self.screening_indices = list(set(self.screening_indices) - set(valid_samples))
            logging.info(f"  - Total candidates docked: {len(selected_indices)}")
            logging.info(f"  - Passed RMSD ≤ 2.0 and HBond similarity ≥ {self.hbond_similarity_threshold}: {len(candidate_results)}")
            logging.info(f"  - Selected top {n_select} based on docking scores")
        else:
            logging.warning("No samples passed RMSD filter in this cycle")

        del all_scores, selected_indices, candidate_results
        gc.collect()
        
        log_memory_usage("after sample acquisition")
        return len(valid_samples)
    
    def run(self):
        log_memory_usage("at startup")
        
        try:
            logging.info("Starting Active Learning...")
            
            if not self.load_candidates():
                logging.error("Failed to load candidates")
                return [], None
            
            # 중복제거는 이미 load_candidates()에서 각 라이브러리별로 처리되었음
            # 전역 중복제거 호출 제거로 성능 최적화 달성
            logging.info("Skipping global deduplication - already handled per library for optimization")
            
            # 중복제거 리포트는 각 라이브러리에서 저장됨
            if self.enable_deduplication and self.deduplicator is not None:
                self.save_deduplication_report()
            
            if not self.precompute_all_fingerprints():
                logging.error("Failed to compute fingerprints")
                return [], None
            
            logging.info("Starting initialization with deduplicated library...")
            if not self.initialize():
                logging.error("Initialization failed")
                return [], None
            
            initial_compounds = [self.cids[idx] for idx in self.training_indices if self.cids[idx] in self.docking_scores]
            self.current_cycle = 0
            self.save_cycle_detailed_results(0, initial_compounds)

            valid_initial = sum(1 for cid in self.docking_scores 
                                if cid in self.interaction_data 
                                and self.interaction_data[cid]['rmsd'] <= 2.0
                                and self.interaction_data[cid]['hbond_similarity'] >= self.hbond_similarity_threshold)
                                
            logging.info(f"Initialization complete. Valid initial samples: {valid_initial}")

            total_valid_compounds = valid_initial
            
            for cycle in range(self.n_cycles):
                cycle_number = cycle + 1
                self.current_cycle = cycle_number
                
                self.cycle_monitor.start_cycle(cycle_number)

                # Exploration ratio scheduling
                progress = cycle / max(1, self.n_cycles - 1)
                self.exploration_ratio = self.initial_exploration_ratio - (self.initial_exploration_ratio - self.final_exploration_ratio) * progress
                
                logging.info(f"\nStarting cycle {cycle_number}/{self.n_cycles} with exploration ratio: {self.exploration_ratio:.3f}")

                training_success, metrics = self.train_model()
                if not training_success:
                    logging.warning(f"Cycle {cycle_number}: Model training failed")
                    self.cycle_monitor.end_cycle(0, metrics=None)
                    continue

                logging.info("Selecting and evaluating new samples...")
                new_valid_samples = self.acquire_samples()
                    
                if new_valid_samples == 0:
                    logging.warning(f"Cycle {cycle_number}: No new valid samples found")
                    self.cycle_monitor.end_cycle(0, metrics=metrics)
                    continue
                        
                total_valid_compounds += new_valid_samples
                logging.info(f"Cycle {cycle_number}: Added {new_valid_samples} valid samples")
                logging.info(f"Total valid compounds so far: {total_valid_compounds}")

                self.save_cycle_results(cycle_number)
                
                compounds_processed = self.batch_size * 2
                self.cycle_monitor.end_cycle(compounds_processed, metrics=metrics)  # metrics 전달

                if self.should_cleanup_memory(cycle_number):
                    logging.info(f"Performing memory cleanup at cycle {cycle_number}")
                    self.cleanup_memory(force_cleanup=True)
                elif cycle_number % 2 == 0:  # 짝수 사이클마다 가벼운 정리
                    self.cleanup_memory(force_cleanup=False)

                if len(self.training_indices) >= self.budget:
                    logging.info("Reached budget limit. Stopping.")
                    break

            logging.info("\nActive Learning completed")

            self.save_trained_model()  
            self.save_best_model()     
            self.save_final_results()
            self.cycle_monitor.log_final_summary()
            self.save_prediction_analysis() 
            self.cycle_monitor.log_final_summary()

            filtered_results = []
            for cid, score in self.docking_scores.items():
                if (cid in self.interaction_data and 
                    self.interaction_data[cid]['rmsd'] <= 2.0 and
                    self.interaction_data[cid]['hbond_similarity'] >= self.hbond_similarity_threshold):
                    filtered_results.append((cid, score))

            return sorted(filtered_results, key=lambda x: x[1]), self.model
            
        except Exception as e:
            logging.error(f"Error during execution: {str(e)}")
            raise
        finally:
            gc.collect()
            log_memory_usage("at completion")