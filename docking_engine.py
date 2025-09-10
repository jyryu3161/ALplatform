"""
분자 도킹 엔진 모듈

AutoDock Vina를 기반으로 한 분자 도킹 시뮬레이션을 담당합니다.
SMILES로부터 형태를 생성하고 도킹 스코어를 계산합니다.
"""

import os
import logging
import contextlib
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from oddt import toolkit
from oddt.docking import autodock_vina

from RMSD import MCSRMSDCalculator
from Hbond import HBondCalculator
from exceptions import DockingError, MolecularProcessingError


class ConformerGenerator:
    """분자 형태(conformer) 생성 클래스"""
    
    def __init__(self, conf_num: int = 1):
        """
        Args:
            conf_num: 생성할 형태의 개수
        """
        self.conf_num = conf_num
        self._conformer_count = 0
    
    def generate_conformers(self, cid: str, smiles: str) -> Optional[List[Chem.Mol]]:
        """
        주어진 SMILES로부터 분자 형태를 생성합니다.
        
        Args:
            cid: 화합물 ID
            smiles: SMILES 문자열
            
        Returns:
            생성된 형태들의 리스트 또는 None
            
        Raises:
            MolecularProcessingError: 형태 생성 실패 시
        """
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
            self._cleanup_memory(mol, test_mol)
    
    def _cleanup_memory(self, mol: Optional[Chem.Mol], test_mol: Optional[Chem.Mol]):
        """메모리 정리"""
        try:
            if mol is not None:
                del mol
            if test_mol is not None:
                del test_mol
        except:
            pass  
        
        self._conformer_count += 1
        if self._conformer_count % 100 == 0:
            import gc
            gc.collect()


class DockingEngine:
    """
    분자 도킹 시뮬레이션 엔진
    
    AutoDock Vina를 사용하여 분자와 단백질 간의 결합 친화도를 예측합니다.
    """
    
    def __init__(self, 
                 protein_file: str,
                 ligand_file: str, 
                 conf_num: int = 1,
                 vina_executable: str = './bin/qvina2.1'):
        """
        Args:
            protein_file: 단백질 PDB 파일 경로
            ligand_file: 기준 리간드 PDB 파일 경로
            conf_num: 생성할 형태의 개수
            vina_executable: Vina 실행파일 경로
        """
        self.protein_file = protein_file
        self.ligand_file = ligand_file
        self.conf_num = conf_num
        self.vina_executable = vina_executable
        
        self.conformer_generator = ConformerGenerator(conf_num)
        self.docking_scores: Dict[str, float] = {}
        self.interaction_data: Dict[str, Dict[str, Any]] = {}
        
        # PDB 정보 읽기
        self.pdb_info = self._read_pdb_info(protein_file)
        
        # 도킹 환경 초기화
        self._initialize_docking_environment()
        
        # 계산기들 초기화
        self._initialize_calculators()
    
    def _read_pdb_info(self, pdb_file: str) -> Dict[int, str]:
        """PDB 파일에서 원자-잔기 정보를 읽습니다."""
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
    
    def _initialize_docking_environment(self):
        """도킹 환경을 초기화합니다."""
        try:
            toolkit.backend = 'rdkit'
            self.protein = next(toolkit.readfile('pdb', self.protein_file))
            self.protein.protein = True
            self.ligand = next(toolkit.readfile('pdb', self.ligand_file))
            
            self.vina = autodock_vina(
                protein=self.protein,
                auto_ligand=self.ligand,
                size=(20, 20, 20),
                num_modes=3,
                exhaustiveness=8,
                energy_range=5,
                n_cpu=16,
                executable=self.vina_executable
            )
            
            logging.info("Docking environment initialized successfully")
            
        except Exception as e:
            raise DockingError(f"Failed to initialize docking environment: {str(e)}")
    
    def _initialize_calculators(self):
        """RMSD 및 H-bond 계산기를 초기화합니다."""
        try:
            ref_mol = Chem.MolFromPDBFile(self.ligand_file)
            self.mcs_calculator = MCSRMSDCalculator(ref_mol)
            self.hbond_calculator = HBondCalculator(self.protein_file, self.ligand_file)
            
            logging.info("Molecular calculators initialized successfully")
            
        except Exception as e:
            raise DockingError(f"Failed to initialize calculators: {str(e)}")
    
    def get_docking_score(self, cid: str, smiles: str, pose_dir: str = None) -> Optional[float]:
        """
        분자의 도킹 스코어를 계산합니다.
        
        Args:
            cid: 화합물 ID
            smiles: SMILES 문자열
            pose_dir: 포즈 저장 디렉터리 (선택사항)
            
        Returns:
            최적 도킹 스코어 또는 None
        """
        if cid in self.docking_scores:
            return self.docking_scores[cid]
        
        # 기본값 설정
        self.interaction_data[cid] = {
            'mcs_size': 0,
            'mcs_similarity': 0.0,
            'rmsd': float('inf'),
            'hbond_similarity': 0.0
        }
        
        # 형태 생성
        conformers = self.conformer_generator.generate_conformers(cid, smiles)
        if not conformers:
            logging.debug(f"Failed to generate conformers for {cid}")
            return None
        
        # 도킹 수행
        valid_poses = self._dock_conformers(cid, conformers)
        if not valid_poses:
            logging.debug(f"No valid poses found for {cid}")
            return None
        
        # 최적 포즈 선택 및 저장
        best_score = self._select_best_pose(cid, valid_poses, pose_dir)
        return best_score
    
    def _dock_conformers(self, cid: str, conformers: List[Chem.Mol]) -> List[Tuple]:
        """형태들에 대해 도킹을 수행합니다."""
        valid_poses = []
        
        for conf_idx, conf in enumerate(conformers):
            mol_block = Chem.MolToMolBlock(conf)
            oddt_mol = toolkit.readstring('sdf', mol_block)
            
            if oddt_mol is None:
                continue
            
            # 도킹 수행
            results = self._perform_docking(oddt_mol, cid)
            if not results:
                continue
            
            # 결과 처리
            for pose in results:
                pose_data = self._process_pose(cid, pose, conf_idx)
                if pose_data:
                    valid_poses.append(pose_data)
        
        return valid_poses
    
    def _perform_docking(self, oddt_mol, cid: str):
        """실제 도킹을 수행합니다."""
        try:
            with open(os.devnull, 'w') as f, contextlib.redirect_stderr(f):
                results = self.vina.dock(oddt_mol)
            return results
        except (AssertionError, Exception) as e:
            logging.debug(f"Docking failed for {cid}: {str(e)}")
            return None
    
    def _process_pose(self, cid: str, pose, conf_idx: int) -> Optional[Tuple]:
        """포즈를 처리하고 점수를 계산합니다."""
        try:
            score = float(pose.data['vina_affinity'])
            pose_mol = Chem.MolFromPDBBlock(pose.write('pdb'))
            
            if pose_mol is None:
                return None
            
            # 상호작용 데이터 계산
            mcs_size, mcs_similarity, rmsd = self.mcs_calculator.calculate_mcs_and_rmsd(pose_mol)
            hbond_similarity = self.hbond_calculator.calculate_hbond_similarity(pose_mol)
            
            interaction_data = (mcs_size, mcs_similarity, rmsd, hbond_similarity)
            return (score, rmsd, pose, interaction_data, conf_idx)
            
        except Exception as e:
            logging.debug(f"Pose processing failed for {cid}: {str(e)}")
            return None
    
    def _select_best_pose(self, cid: str, valid_poses: List[Tuple], pose_dir: str = None) -> float:
        """최적 포즈를 선택합니다."""
        try:
            # RMSD 기반 필터링
            min_rmsd = min(pose[1] for pose in valid_poses)
            rmsd_threshold = min_rmsd * 1.5
            
            good_rmsd_poses = [pose for pose in valid_poses if pose[1] <= rmsd_threshold]
            if not good_rmsd_poses:
                good_rmsd_poses = valid_poses
            
            # 최적 스코어 선택
            best_pose = min(good_rmsd_poses, key=lambda x: x[0])
            best_score, best_rmsd, best_pose_obj, best_data, best_conf_idx = best_pose
            
            # 결과 저장
            self.docking_scores[cid] = best_score
            self.interaction_data[cid] = {
                'mcs_size': best_data[0],
                'mcs_similarity': best_data[1],
                'rmsd': best_data[2],
                'hbond_similarity': best_data[3]
            }
            
            # 포즈 파일 저장
            if pose_dir:
                self._save_pose(cid, best_pose_obj, pose_dir)
            
            logging.debug(f"Selected best pose for {cid} - Score: {best_score:.3f}, "
                        f"RMSD: {best_rmsd:.3f}, HBond Similarity: {best_data[3]:.3f}, "
                        f"Conformer: {best_conf_idx}, Total poses: {len(valid_poses)}")
            
            return best_score
            
        except Exception as e:
            logging.debug(f"Final processing failed for {cid}: {str(e)}")
            return None
    
    def _save_pose(self, cid: str, pose_obj, pose_dir: str):
        """포즈를 파일로 저장합니다."""
        try:
            os.makedirs(pose_dir, exist_ok=True)
            pose_path = os.path.join(pose_dir, f'pose_{cid}.pdbqt')
            pose_obj.write('pdbqt', pose_path, overwrite=True)
        except Exception as e:
            logging.warning(f"Failed to save pose for {cid}: {str(e)}")
    
    def batch_docking(self, compound_data: Dict[str, str], pose_dir: str = None) -> Dict[str, Optional[float]]:
        """
        여러 화합물에 대해 배치 도킹을 수행합니다.
        
        Args:
            compound_data: {cid: smiles} 딕셔너리
            pose_dir: 포즈 저장 디렉터리
            
        Returns:
            {cid: docking_score} 딕셔너리
        """
        results = {}
        
        for cid, smiles in tqdm(compound_data.items(), desc="Batch docking"):
            score = self.get_docking_score(cid, smiles, pose_dir)
            results[cid] = score
            
        return results
    
    def get_interaction_data(self, cid: str) -> Optional[Dict[str, Any]]:
        """화합물의 상호작용 데이터를 반환합니다."""
        return self.interaction_data.get(cid)
    
    def get_valid_compounds(self, rmsd_threshold: float = 2.0, 
                           hbond_threshold: float = 0.3) -> List[str]:
        """
        품질 기준을 만족하는 화합물 목록을 반환합니다.
        
        Args:
            rmsd_threshold: RMSD 임계값
            hbond_threshold: H-bond 유사도 임계값
            
        Returns:
            유효한 화합물 ID 리스트
        """
        valid_cids = []
        
        for cid, data in self.interaction_data.items():
            if (cid in self.docking_scores and
                data['rmsd'] <= rmsd_threshold and
                data['hbond_similarity'] >= hbond_threshold):
                valid_cids.append(cid)
        
        return valid_cids
    
    def get_statistics(self) -> Dict[str, Any]:
        """도킹 결과 통계를 반환합니다."""
        if not self.docking_scores:
            return {'total_docked': 0}
        
        scores = list(self.docking_scores.values())
        rmsds = [data['rmsd'] for data in self.interaction_data.values() if data['rmsd'] != float('inf')]
        hbond_sims = [data['hbond_similarity'] for data in self.interaction_data.values()]
        
        return {
            'total_docked': len(self.docking_scores),
            'score_mean': np.mean(scores),
            'score_std': np.std(scores),
            'score_min': np.min(scores),
            'score_max': np.max(scores),
            'rmsd_mean': np.mean(rmsds) if rmsds else float('inf'),
            'rmsd_std': np.std(rmsds) if rmsds else 0,
            'hbond_sim_mean': np.mean(hbond_sims) if hbond_sims else 0,
            'hbond_sim_std': np.std(hbond_sims) if hbond_sims else 0,
            'valid_poses_count': len([d for d in self.interaction_data.values() if d['rmsd'] <= 2.0])
        }