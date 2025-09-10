import logging
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import csv
import os

class MoleculeFilter:
    UNSUPPORTED_ELEMENTS = {
        'Li', 'Be', 'B', 'Si', 'Ru', 'Al', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Rh', 'Pd',
        'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
        'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W',
        'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr',
        'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
        'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh',
        'Fl', 'Mc', 'Lv', 'Ts', 'Og'
    }
    
    SUPPORTED_ELEMENTS = {
        'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'
    }
    
    def __init__(self):
        self.filtered_count = 0
        self.invalid_smiles_count = 0
        self.unsupported_elements_found = set()
        self.duplicate_groups = {}
        self.removed_duplicates = []
        
    def contains_unsupported_elements(self, mol):
        if mol is None:
            return True, set()
            
        found_unsupported = set()
        
        for atom in mol.GetAtoms():
            element = atom.GetSymbol()
            if element in self.UNSUPPORTED_ELEMENTS:
                found_unsupported.add(element)
                
        return len(found_unsupported) > 0, found_unsupported
    
    def filter_molecule_dict(self, molecule_dict):
        logging.info(f"Filtering {len(molecule_dict)} molecules from dictionary...")
        
        self.filtered_count = 0
        self.invalid_smiles_count = 0
        self.unsupported_elements_found = set()
        
        filtered_dict = {}
        
        for mol_id, smiles in tqdm(molecule_dict.items(), desc="Filtering molecules"):
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is None:
                self.invalid_smiles_count += 1
                continue
            
            has_unsupported, unsupported_elements = self.contains_unsupported_elements(mol)
            
            if has_unsupported:
                self.filtered_count += 1
                self.unsupported_elements_found.update(unsupported_elements)
                continue
            
            filtered_dict[mol_id] = smiles
        
        total_processed = len(molecule_dict)
        total_kept = len(filtered_dict)
        
        filter_stats = {
            'total_processed': total_processed,
            'total_kept': total_kept,
            'total_filtered': self.filtered_count,
            'invalid_smiles': self.invalid_smiles_count,
            'unsupported_elements_found': sorted(self.unsupported_elements_found),
            'filter_rate': (self.filtered_count / total_processed * 100) if total_processed > 0 else 0
        }
        
        logging.info(f"Dictionary filtering completed")
        logging.info(f"Total processed: {total_processed}")
        logging.info(f"Total kept: {total_kept}")
        logging.info(f"Total filtered out: {self.filtered_count}")
        logging.info(f"Filter rate: {filter_stats['filter_rate']:.2f}%")
        
        if self.unsupported_elements_found:
            logging.info(f"Unsupported elements found: {sorted(self.unsupported_elements_found)}")
        
        return filtered_dict, filter_stats
    
    def canonicalize_smiles(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        except Exception as e:
            logging.debug(f"Error canonicalizing SMILES '{smiles}': {e}")
            return None
    
    def deduplicate_by_smiles(self, structure_info):        
        self.duplicate_groups = {}
        self.removed_duplicates = []
        
        smiles_to_cids = {}
        invalid_smiles_count = 0
        
        for cid, smiles in structure_info.items():
            canonical_smiles = self.canonicalize_smiles(smiles)
            
            if canonical_smiles is None:
                invalid_smiles_count += 1
                logging.debug(f"Invalid SMILES for {cid}: {smiles}")
                continue
                
            if canonical_smiles not in smiles_to_cids:
                smiles_to_cids[canonical_smiles] = []
            smiles_to_cids[canonical_smiles].append(cid)
        
        new_structure_info = {}
        duplicate_count = 0
        
        for canonical_smiles, cid_list in smiles_to_cids.items():
            if len(cid_list) == 1:
                cid = cid_list[0]
                new_structure_info[cid] = structure_info[cid]
            else:
                kept_cid = cid_list[0]
                removed_cids = cid_list[1:]
                
                new_structure_info[kept_cid] = structure_info[kept_cid]
                self.duplicate_groups[kept_cid] = removed_cids
                self.removed_duplicates.extend(removed_cids)
                duplicate_count += len(removed_cids)
                
                logging.debug(f"Duplicate SMILES '{canonical_smiles}': kept {kept_cid}, removed {removed_cids}")
        
        original_count = len(structure_info)
        final_count = len(new_structure_info)
        '''
        logging.info(f"SMILES deduplication completed:")
        logging.info(f"  - Original compounds: {original_count}")
        logging.info(f"  - After deduplication: {final_count}")
        logging.info(f"  - Duplicates removed: {duplicate_count}")
        logging.info(f"  - Invalid SMILES: {invalid_smiles_count}")
        logging.info(f"  - Deduplication rate: {duplicate_count/original_count*100:.1f}%")
        '''
        return new_structure_info, duplicate_count
    
    def cleanup_removed_compounds(self, removed_cids, *data_dicts):
        removed_counts = []
        
        for data_dict in data_dicts:
            if data_dict is None:
                removed_counts.append(0)
                continue
                
            removed_count = 0
            for cid in list(data_dict.keys()):
                if cid in removed_cids:
                    del data_dict[cid]
                    removed_count += 1
            removed_counts.append(removed_count)
        
        return removed_counts
    
    def save_deduplication_report(self, output_dir):
        if not self.removed_duplicates:
            logging.info("No duplicates were removed, skipping deduplication report")
            return
            
        report_file = os.path.join(output_dir, 'deduplication_report.csv')
        
        with open(report_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Kept_CID', 'Removed_CID', 'Duplicate_Type'])
            
            for kept_cid, removed_cids in self.duplicate_groups.items():
                for removed_cid in removed_cids:
                    writer.writerow([kept_cid, removed_cid, 'SMILES_duplicate'])
        
        logging.info(f"Deduplication report saved to {report_file}")
        logging.info(f"Total duplicate groups: {len(self.duplicate_groups)}")
        logging.info(f"Total removed compounds: {len(self.removed_duplicates)}")


class MoleculeDeduplicator:    
    def __init__(self):
        self.filter = MoleculeFilter()
    
    def deduplicate_dictionary(self, structure_info):
        return self.filter.deduplicate_by_smiles(structure_info)
    
    def get_removed_compounds(self):
        return self.filter.removed_duplicates
    
    def save_report(self, output_dir):
        self.filter.save_deduplication_report(output_dir)