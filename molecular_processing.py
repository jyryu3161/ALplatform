"""
분자 처리 및 변환을 위한 통합 모듈

이 모듈은 SMILES 변환, 형태 생성, 핑거프린트 계산 등
분자 처리와 관련된 모든 기능을 제공합니다.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Any
from tqdm import tqdm
import gc

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit import DataStructs
from rdkit import __version__ as rdkit_version

from fingerprint import get_morgan_fingerprint_safe
from exceptions import MolecularProcessingError, DataValidationError


class ConformerGenerator:
    """분자 형태(conformer) 생성을 담당하는 클래스"""
    
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
                raise MolecularProcessingError(f"Failed to create molecule from SMILES for {cid}")

            mol = Chem.AddHs(mol)
            
            # 기본 형태 생성 테스트
            test_mol = Chem.Mol(mol)
            embed_result = AllChem.EmbedMolecule(test_mol)
            if embed_result != 0:
                logging.warning(f"Failed to embed molecule for {cid}")
                return None
            
            # 다중 형태 생성
            conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=self.conf_num)
            
            if not conf_ids:
                logging.warning(f"Failed to generate conformers for {cid}")
                return [test_mol]
                
            # MMFF 최적화
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
            
            logging.debug(f"Generated {len(conformers)} conformers for {cid}")
            return conformers
            
        except Exception as e:
            raise MolecularProcessingError(f"Error generating conformers for {cid}: {str(e)}")
            
        finally:
            # 메모리 정리
            try:
                if mol is not None:
                    del mol
                if test_mol is not None:
                    del test_mol
            except:
                pass  
            
            self._conformer_count += 1
            if self._conformer_count % 100 == 0:
                gc.collect()


class FingerprintComputer:
    """분자 핑거프린트 계산을 담당하는 클래스"""
    
    def __init__(self, radius: int = 2, n_bits: int = 1024):
        """
        Args:
            radius: Morgan 핑거프린트 반지름
            n_bits: 핑거프린트 비트 수
        """
        self.radius = radius
        self.n_bits = n_bits
        self.fingerprint_cache: Dict[str, np.ndarray] = {}
        self.fingerprint_computed = False
    
    def compute_fingerprint(self, cid: str, smiles: str) -> Optional[np.ndarray]:
        """
        단일 분자의 핑거프린트를 계산합니다.
        
        Args:
            cid: 화합물 ID
            smiles: SMILES 문자열
            
        Returns:
            핑거프린트 배열 또는 None
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            fp = get_morgan_fingerprint_safe(mol, self.radius, self.n_bits)
            if fp is None:
                return None
                
            fp_array = np.array(list(fp), dtype=np.float32)
            self.fingerprint_cache[cid] = fp_array
            return fp_array
            
        except Exception as e:
            logging.debug(f"Error computing fingerprint for {cid}: {str(e)}")
            return None
    
    def precompute_all_fingerprints(self, structure_info: Dict[str, str]) -> bool:
        """
        모든 분자의 핑거프린트를 사전 계산합니다.
        
        Args:
            structure_info: {cid: smiles} 딕셔너리
            
        Returns:
            성공 여부
        """
        if self.fingerprint_computed:
            logging.info("Fingerprints already computed, skipping...")
            return True
            
        total_compounds = len(structure_info)
        logging.info(f"Pre-computing ECFP fingerprints for {total_compounds:,} compounds...")
        
        successful_count = 0
        failed_count = 0
        batch_size = 1000
        
        cids = list(structure_info.keys())
        
        with tqdm(range(0, len(cids), batch_size), 
                desc="Computing ECFP fingerprints", 
                unit="batches",
                ncols=100) as pbar:
            
            for start_idx in pbar:
                end_idx = min(start_idx + batch_size, len(cids))
                batch_cids = cids[start_idx:end_idx]
                
                batch_success = 0
                batch_failed = 0
                
                for cid in batch_cids:
                    smiles = structure_info[cid]
                    fp = self.compute_fingerprint(cid, smiles)
                    
                    if fp is not None:
                        batch_success += 1
                    else:
                        batch_failed += 1
                
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
    
    def get_fingerprint(self, cid: str) -> Optional[np.ndarray]:
        """
        캐시된 핑거프린트를 반환합니다.
        
        Args:
            cid: 화합물 ID
            
        Returns:
            핑거프린트 배열 또는 None
        """
        if not self.fingerprint_computed:
            raise RuntimeError("Fingerprints not pre-computed. Call precompute_all_fingerprints() first.")
        
        return self.fingerprint_cache.get(cid)
    
    def get_fingerprints_batch(self, cid_list: List[str]) -> Tuple[np.ndarray, List[int]]:
        """
        여러 화합물의 핑거프린트를 배치로 반환합니다.
        
        Args:
            cid_list: 화합물 ID 리스트
            
        Returns:
            (핑거프린트 배열, 유효한 인덱스 리스트)
        """
        fingerprints = []
        valid_indices = []
        
        for i, cid in enumerate(cid_list):
            fp = self.get_fingerprint(cid)
            if fp is not None:
                fingerprints.append(fp)
                valid_indices.append(i)
        
        if fingerprints:
            return np.array(fingerprints, dtype=np.float32), valid_indices
        else:
            return np.array([]), valid_indices
    
    def clear_cache(self, keep_active_cids: Optional[set] = None):
        """
        핑거프린트 캐시를 정리합니다.
        
        Args:
            keep_active_cids: 유지할 화합물 ID 집합
        """
        if keep_active_cids is None:
            self.fingerprint_cache.clear()
            logging.info("Cleared all fingerprint cache")
        else:
            cache_keys_to_remove = []
            for cid in self.fingerprint_cache:
                if cid not in keep_active_cids:
                    cache_keys_to_remove.append(cid)
            
            for cid in cache_keys_to_remove:
                del self.fingerprint_cache[cid]
            
            if cache_keys_to_remove:
                logging.info(f"Cleaned {len(cache_keys_to_remove)} unused fingerprints from cache")


class SimilarityCalculator:
    """분자 유사도 계산을 담당하는 클래스"""
    
    def __init__(self, reference_smiles: str, similarity_threshold: float = 0.2):
        """
        Args:
            reference_smiles: 기준 분자의 SMILES
            similarity_threshold: 유사도 임계값
        """
        self.reference_smiles = reference_smiles
        self.similarity_threshold = similarity_threshold
        
        # 기준 분자의 핑거프린트 계산
        ref_mol = Chem.MolFromSmiles(reference_smiles)
        if ref_mol is None:
            raise DataValidationError(f"Invalid reference SMILES: {reference_smiles}")
        
        self.reference_fp = get_morgan_fingerprint_safe(ref_mol, 2, 1024)
        if self.reference_fp is None:
            raise MolecularProcessingError("Failed to generate reference fingerprint")
    
    def calculate_similarity(self, smiles: str) -> float:
        """
        기준 분자와의 Tanimoto 유사도를 계산합니다.
        
        Args:
            smiles: 비교할 분자의 SMILES
            
        Returns:
            Tanimoto 유사도 (0-1)
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0
            
            mol_fp = get_morgan_fingerprint_safe(mol, 2, 1024)
            if mol_fp is None:
                return 0.0
            
            return DataStructs.TanimotoSimilarity(self.reference_fp, mol_fp)
            
        except Exception as e:
            logging.debug(f"Error calculating similarity for {smiles}: {str(e)}")
            return 0.0
    
    def filter_by_similarity(self, molecule_dict: Dict[str, str]) -> Dict[str, str]:
        """
        유사도 임계값을 기준으로 분자를 필터링합니다.
        
        Args:
            molecule_dict: {cid: smiles} 딕셔너리
            
        Returns:
            필터링된 {cid: smiles} 딕셔너리
        """
        filtered_dict = {}
        fingerprint_calculated = 0
        fingerprint_failed = 0
        
        for cid, smiles in tqdm(molecule_dict.items(), desc="Similarity filtering"):
            similarity = self.calculate_similarity(smiles)
            
            if similarity > 0:
                fingerprint_calculated += 1
                if similarity >= self.similarity_threshold:
                    filtered_dict[cid] = smiles
            else:
                fingerprint_failed += 1
        
        logging.info(f"Similarity filtering: {len(molecule_dict):,} → {len(filtered_dict):,} compounds")
        logging.info(f"Fingerprints calculated: {fingerprint_calculated:,}, Failed: {fingerprint_failed:,}")
        
        return filtered_dict


class MolecularProcessor:
    """분자 처리를 위한 통합 인터페이스"""
    
    def __init__(self, conf_num: int = 1, fp_radius: int = 2, fp_bits: int = 1024):
        """
        Args:
            conf_num: 생성할 형태의 개수
            fp_radius: 핑거프린트 반지름
            fp_bits: 핑거프린트 비트 수
        """
        self.conformer_generator = ConformerGenerator(conf_num)
        self.fingerprint_computer = FingerprintComputer(fp_radius, fp_bits)
        
        # 처리 통계
        self.processing_stats = {
            'conformers_generated': 0,
            'fingerprints_computed': 0,
            'similarity_filtered': 0,
            'processing_errors': 0
        }
        
    def set_reference_molecule(self, reference_smiles: str, similarity_threshold: float = 0.2):
        """기준 분자를 설정합니다."""
        self.similarity_calculator = SimilarityCalculator(reference_smiles, similarity_threshold)
    
    def process_molecules(self, molecule_dict: Dict[str, str], 
                         filter_by_similarity: bool = True) -> Dict[str, str]:
        """
        분자 딕셔너리를 처리합니다.
        
        Args:
            molecule_dict: {cid: smiles} 딕셔너리
            filter_by_similarity: 유사도 필터링 수행 여부
            
        Returns:
            처리된 {cid: smiles} 딕셔너리
        """
        processed_dict = molecule_dict.copy()
        
        if filter_by_similarity and hasattr(self, 'similarity_calculator'):
            original_count = len(processed_dict)
            processed_dict = self.similarity_calculator.filter_by_similarity(processed_dict)
            self.processing_stats['similarity_filtered'] += original_count - len(processed_dict)
        
        return processed_dict
    
    def batch_generate_conformers(self, molecule_dict: Dict[str, str]) -> Dict[str, List[Chem.Mol]]:
        """
        배치로 형태를 생성합니다.
        
        Args:
            molecule_dict: {cid: smiles} 딕셔너리
            
        Returns:
            {cid: conformers_list} 딕셔너리
        """
        conformers_dict = {}
        
        for cid, smiles in tqdm(molecule_dict.items(), desc="Generating conformers"):
            try:
                conformers = self.conformer_generator.generate_conformers(cid, smiles)
                if conformers:
                    conformers_dict[cid] = conformers
                    self.processing_stats['conformers_generated'] += len(conformers)
                else:
                    self.processing_stats['processing_errors'] += 1
            except Exception as e:
                logging.debug(f"Error generating conformers for {cid}: {str(e)}")
                self.processing_stats['processing_errors'] += 1
        
        return conformers_dict
    
    def precompute_fingerprints(self, structure_info: Dict[str, str]) -> bool:
        """핑거프린트를 사전 계산합니다."""
        success = self.fingerprint_computer.precompute_all_fingerprints(structure_info)
        if success:
            self.processing_stats['fingerprints_computed'] = len(structure_info)
        return success
    
    def get_fingerprint_batch(self, cid_list: List[str]) -> Tuple[np.ndarray, List[int]]:
        """배치로 핑거프린트를 가져옵니다."""
        return self.fingerprint_computer.get_fingerprints_batch(cid_list)
    
    def clear_fingerprint_cache(self, keep_active_cids: Optional[set] = None):
        """핑거프린트 캐시를 정리합니다."""
        self.fingerprint_computer.clear_cache(keep_active_cids)
    
    def validate_smiles_batch(self, molecule_dict: Dict[str, str]) -> Tuple[Dict[str, str], List[str]]:
        """
        SMILES 배치를 검증합니다.
        
        Args:
            molecule_dict: {cid: smiles} 딕셔너리
            
        Returns:
            (유효한 {cid: smiles} 딕셔너리, 무효한 SMILES 리스트)
        """
        valid_dict = {}
        invalid_smiles = []
        
        for cid, smiles in tqdm(molecule_dict.items(), desc="Validating SMILES"):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # 추가 검증: 분자량, 원자 수 등
                    if self._is_reasonable_molecule(mol):
                        valid_dict[cid] = smiles
                    else:
                        invalid_smiles.append(smiles)
                        self.processing_stats['processing_errors'] += 1
                else:
                    invalid_smiles.append(smiles)
                    self.processing_stats['processing_errors'] += 1
            except Exception as e:
                logging.debug(f"Error validating SMILES for {cid}: {str(e)}")
                invalid_smiles.append(smiles)
                self.processing_stats['processing_errors'] += 1
        
        logging.info(f"SMILES validation: {len(molecule_dict)} → {len(valid_dict)} valid, "
                    f"{len(invalid_smiles)} invalid")
        
        return valid_dict, invalid_smiles
    
    def _is_reasonable_molecule(self, mol: Chem.Mol) -> bool:
        """분자가 합리적인 범위 내인지 확인합니다."""
        try:
            # 분자량 체크 (50-1000 Da)
            mw = Chem.Descriptors.MolWt(mol)
            if mw < 50 or mw > 1000:
                return False
            
            # 원자 수 체크 (5-100개)
            num_atoms = mol.GetNumAtoms()
            if num_atoms < 5 or num_atoms > 100:
                return False
            
            # 회전 가능 결합 수 체크 (0-20개)
            rot_bonds = Chem.Descriptors.NumRotatableBonds(mol)
            if rot_bonds > 20:
                return False
            
            return True
            
        except Exception:
            return False
    
    def calculate_molecular_descriptors(self, molecule_dict: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """
        분자 기술자를 계산합니다.
        
        Args:
            molecule_dict: {cid: smiles} 딕셔너리
            
        Returns:
            {cid: {descriptor_name: value}} 딕셔너리
        """
        descriptors = {}
        
        for cid, smiles in tqdm(molecule_dict.items(), desc="Computing molecular descriptors"):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                
                desc_dict = {}
                desc_dict['molecular_weight'] = Chem.Descriptors.MolWt(mol)
                desc_dict['logp'] = Chem.Descriptors.MolLogP(mol)
                desc_dict['num_atoms'] = mol.GetNumAtoms()
                desc_dict['num_bonds'] = mol.GetNumBonds()
                desc_dict['num_rings'] = Chem.rdMolDescriptors.CalcNumRings(mol)
                desc_dict['rotatable_bonds'] = Chem.Descriptors.NumRotatableBonds(mol)
                desc_dict['hbd'] = Chem.Descriptors.NumHDonors(mol)
                desc_dict['hba'] = Chem.Descriptors.NumHAcceptors(mol)
                desc_dict['tpsa'] = Chem.Descriptors.TPSA(mol)
                
                descriptors[cid] = desc_dict
                
            except Exception as e:
                logging.debug(f"Error computing descriptors for {cid}: {str(e)}")
                continue
        
        logging.info(f"Computed descriptors for {len(descriptors)} compounds")
        return descriptors
    
    def filter_by_drug_likeness(self, molecule_dict: Dict[str, str]) -> Dict[str, str]:
        """
        Lipinski의 5 규칙을 기반으로 약물 유사성 필터링을 수행합니다.
        
        Args:
            molecule_dict: {cid: smiles} 딕셔너리
            
        Returns:
            필터링된 {cid: smiles} 딕셔너리
        """
        filtered_dict = {}
        
        for cid, smiles in tqdm(molecule_dict.items(), desc="Drug-likeness filtering"):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                
                # Lipinski의 5 규칙
                mw = Chem.Descriptors.MolWt(mol)
                logp = Chem.Descriptors.MolLogP(mol)
                hbd = Chem.Descriptors.NumHDonors(mol)
                hba = Chem.Descriptors.NumHAcceptors(mol)
                
                # 규칙 확인
                if (mw <= 500 and 
                    logp <= 5 and 
                    hbd <= 5 and 
                    hba <= 10):
                    filtered_dict[cid] = smiles
                
            except Exception as e:
                logging.debug(f"Error filtering {cid}: {str(e)}")
                continue
        
        original_count = len(molecule_dict)
        filtered_count = len(filtered_dict)
        removed_count = original_count - filtered_count
        
        logging.info(f"Drug-likeness filtering: {original_count} → {filtered_count} compounds")
        logging.info(f"Removed {removed_count} compounds ({removed_count/original_count*100:.1f}%)")
        
        return filtered_dict
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """처리 통계를 반환합니다."""
        return self.processing_stats.copy()
    
    def reset_statistics(self):
        """처리 통계를 초기화합니다."""
        self.processing_stats = {
            'conformers_generated': 0,
            'fingerprints_computed': 0,
            'similarity_filtered': 0,
            'processing_errors': 0
        }