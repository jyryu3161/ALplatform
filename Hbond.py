import numpy as np
from rdkit import Chem

class HBondCalculator:
    def __init__(self, protein_file, reference_file):
        self.protein = Chem.MolFromPDBFile(protein_file, sanitize=True, removeHs=False)
        self.reference = Chem.MolFromPDBFile(reference_file, sanitize=True, removeHs=False)
        
        if not self.protein or not self.reference:
            raise ValueError("Failed to load protein or reference molecule")
            
        self.ref_hbonds = self.find_hbonds(self.reference, self.protein)
        self.ref_hbond_count = len(self.ref_hbonds)

    def is_hbond_donor(self, atom):
        if atom.GetSymbol() in ['N', 'O']:
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == 'H':
                    return True
        return False

    def is_hbond_acceptor(self, atom):
        return atom.GetSymbol() in ['N', 'O'] and atom.GetExplicitValence() < atom.GetTotalValence()

    def get_distance(self, pos1, pos2):
        return np.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pos1, pos2)))

    def find_hbonds(self, ligand, protein, distance_cutoff=3.5, angle_cutoff=120):
        hbonds = []
        ligand_conf = ligand.GetConformer()
        protein_conf = protein.GetConformer()

        for l_atom in ligand.GetAtoms():
            if self.is_hbond_donor(l_atom):
                l_pos = ligand_conf.GetAtomPosition(l_atom.GetIdx())
                
                for p_atom in protein.GetAtoms():
                    if self.is_hbond_acceptor(p_atom):
                        p_pos = protein_conf.GetAtomPosition(p_atom.GetIdx())
                        distance = self.get_distance(l_pos, p_pos)
                        
                        if distance <= distance_cutoff:
                            for l_neighbor in l_atom.GetNeighbors():
                                if l_neighbor.GetSymbol() == 'H':
                                    h_pos = ligand_conf.GetAtomPosition(l_neighbor.GetIdx())
                                    angle = self.calculate_angle(h_pos, l_pos, p_pos)
                                    if angle >= angle_cutoff:
                                        hbonds.append((l_atom.GetIdx(), p_atom.GetIdx(), distance))

        for p_atom in protein.GetAtoms():
            if self.is_hbond_donor(p_atom):
                p_pos = protein_conf.GetAtomPosition(p_atom.GetIdx())
                
                for l_atom in ligand.GetAtoms():
                    if self.is_hbond_acceptor(l_atom):
                        l_pos = ligand_conf.GetAtomPosition(l_atom.GetIdx())
                        distance = self.get_distance(l_pos, p_pos)
                        
                        if distance <= distance_cutoff:
                            for p_neighbor in p_atom.GetNeighbors():
                                if p_neighbor.GetSymbol() == 'H':
                                    h_pos = protein_conf.GetAtomPosition(p_neighbor.GetIdx())
                                    angle = self.calculate_angle(h_pos, p_pos, l_pos)
                                    if angle >= angle_cutoff:
                                        hbonds.append((l_atom.GetIdx(), p_atom.GetIdx(), distance))

        return hbonds

    def calculate_angle(self, pos1, pos2, pos3):
        v1 = np.array(pos1) - np.array(pos2)
        v2 = np.array(pos3) - np.array(pos2)
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  
        return np.degrees(np.arccos(cos_angle))

    def calculate_hbond_similarity(self, pose_mol):
        if not pose_mol:
            return 0.0
            
        pose_hbonds = self.find_hbonds(pose_mol, self.protein)
        
        if self.ref_hbond_count == 0:
            return 1.0 if len(pose_hbonds) == 0 else 0.0
            
        similarity = len(pose_hbonds) / max(self.ref_hbond_count, len(pose_hbonds))
        return similarity