import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS

class MCSRMSDCalculator:
    def __init__(self, reference_mol):          
        self.reference_mol = reference_mol
        self.reference_coords = self.get_coords(reference_mol)
        
    def get_coords(self, mol):
        if mol is None or mol.GetNumConformers() == 0:
            return None
        conf = mol.GetConformer()
        return np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
            
    def calculate_mcs_and_rmsd(self, query_mol):
        mcs = rdFMCS.FindMCS(
            [self.reference_mol, query_mol],
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            matchValences=True,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            timeout=60
        )
        
        if mcs.numAtoms == 0:
            return 0, 0.0, float('inf')
                
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
        if mcs_mol is None:
            return 0, 0.0, float('inf')
            
        mcs_size = mcs.numAtoms
        max_atoms = max(self.reference_mol.GetNumAtoms(), query_mol.GetNumAtoms())
        mcs_similarity = mcs_size / max_atoms if max_atoms > 0 else 0.0
        
        ref_match = self.reference_mol.GetSubstructMatch(mcs_mol)
        query_match = query_mol.GetSubstructMatch(mcs_mol)
        
        if not ref_match or not query_match:
            return mcs_size, mcs_similarity, float('inf')
        
        ref_coords = np.array([self.reference_coords[i] for i in ref_match])
        query_coords = np.array([query_mol.GetConformer().GetAtomPosition(i) for i in query_match])
        
        rmsd = self.calculate_rmsd(ref_coords, query_coords)
        return mcs_size, mcs_similarity, rmsd
            
    def calculate_rmsd(self, coords1, coords2):
        if coords1 is None or coords2 is None or coords1.shape != coords2.shape:
            return float('inf')
        
        centroid1 = np.mean(coords1, axis=0)
        centroid2 = np.mean(coords2, axis=0)
        
        coords1_centered = coords1 - centroid1
        coords2_centered = coords2 - centroid2
        
        H = coords1_centered.T @ coords2_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        if np.linalg.det(R) < 0:
            Vt[-1,:] *= -1
            R = Vt.T @ U.T
        
        coords2_aligned = (R @ coords2_centered.T).T + centroid1
        
        squared_diff = np.sum((coords1 - coords2_aligned) ** 2, axis=1)
        return np.sqrt(np.mean(squared_diff))