"""
데이터 로딩 및 전처리 모듈

CSV 파일로부터 화학 구조 데이터를 로딩하고,
필터링, 중복제거, 유사도 계산 등의 전처리를 수행합니다.
"""

import os
import logging
import gc
from typing import Dict, Optional, Tuple, List
import pandas as pd
import numpy as np
from tqdm import tqdm

from rdkit import Chem, DataStructs
from oddt import toolkit

from molecule_filter import MoleculeFilter, MoleculeDeduplicator
from fingerprint import get_morgan_fingerprint_safe
from exceptions import DataLoadingError, DataValidationError


class SimilarityFilter:
    """분자 유사도 필터링 클래스"""
    
    def __init__(self, reference_smiles: str, similarity_threshold: float = 0.2):
        """
        Args:
            reference_smiles: 기준 분자의 SMILES
            similarity_threshold: 유사도 임계값
        """
        self.reference_smiles = reference_smiles
        self.similarity_threshold = similarity_threshold
        self.invalid_smiles = []
        
        # 기준 분자의 핑거프린트 계산
        self.reference_fp = self._compute_reference_fingerprint()
    
    def _compute_reference_fingerprint(self):
        """기준 분자의 핑거프린트를 계산합니다."""
        try:
            ref_mol = Chem.MolFromSmiles(self.reference_smiles)
            if ref_mol is None:
                raise DataValidationError(f"Invalid reference SMILES: {self.reference_smiles}")
            
            ref_fp = get_morgan_fingerprint_safe(ref_mol, 2, 1024)
            if ref_fp is None:
                raise DataValidationError("Failed to generate reference fingerprint")
            
            return ref_fp
            
        except Exception as e:
            raise DataLoadingError(f"Error computing reference fingerprint: {str(e)}")
    
    def filter_by_similarity(self, molecule_dict: Dict[str, str]) -> Dict[str, str]:
        """
        유사도 기준으로 분자를 필터링합니다.
        
        Args:
            molecule_dict: {cid: smiles} 딕셔너리
            
        Returns:
            필터링된 {cid: smiles} 딕셔너리
        """
        filtered_dict = {}
        fingerprint_calculated = 0
        fingerprint_failed = 0
        
        for cid, smiles in tqdm(molecule_dict.items(), desc="Similarity filtering"):
            similarity = self._calculate_similarity(smiles)
            
            if similarity > 0:
                fingerprint_calculated += 1
                if similarity >= self.similarity_threshold:
                    filtered_dict[cid] = smiles
            else:
                fingerprint_failed += 1
                self.invalid_smiles.append(smiles)
        
        logging.info(f"Similarity filtering: {len(molecule_dict):,} → {len(filtered_dict):,} compounds")
        logging.info(f"Fingerprints calculated: {fingerprint_calculated:,}, Failed: {fingerprint_failed:,}")
        
        return filtered_dict
    
    def _calculate_similarity(self, smiles: str) -> float:
        """분자의 유사도를 계산합니다."""
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


class CSVDataLoader:
    """CSV 파일 데이터 로더"""
    
    def __init__(self, 
                 enable_element_filter: bool = True,
                 enable_deduplication: bool = True):
        """
        Args:
            enable_element_filter: 원소 필터링 활성화 여부
            enable_deduplication: 중복제거 활성화 여부
        """
        self.enable_element_filter = enable_element_filter
        self.enable_deduplication = enable_deduplication
        
        # 필터 및 중복제거기 초기화
        self.molecule_filter = MoleculeFilter() if enable_element_filter else None
        self.deduplicator = MoleculeDeduplicator() if enable_deduplication else None
        
        # 통계 저장
        self.loading_stats = {}
        
        if enable_deduplication:
            logging.info("SMILES-based deduplication enabled")
        if enable_element_filter:
            logging.info("Element filtering enabled")
    
    def load_csv(self, csv_file: str, library_name: str = "main") -> Dict[str, str]:
        """
        CSV 파일을 로딩하고 기본 전처리를 수행합니다.
        
        Args:
            csv_file: CSV 파일 경로
            library_name: 라이브러리 이름 (로깅용)
            
        Returns:
            {cid: smiles} 딕셔너리
            
        Raises:
            DataLoadingError: 파일 로딩 실패 시
        """
        if not os.path.exists(csv_file):
            raise DataLoadingError(f"CSV file not found: {csv_file}")
        
        try:
            df = pd.read_csv(csv_file)
            
            # 필수 컬럼 확인
            required_columns = ['Catalog ID', 'smiles']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise DataValidationError(f"Missing required columns: {missing_columns}")
            
            # 결측값 제거
            df = df.dropna(subset=required_columns)
            original_count = len(df)
            
            if original_count == 0:
                raise DataLoadingError(f"No valid data found in {csv_file}")
            
            logging.info(f"{library_name} library: Starting with {original_count:,} compounds")
            
            # 딕셔너리 변환
            molecule_dict = {}
            for _, row in df.iterrows():
                cid = str(row['Catalog ID'])
                smiles = row['smiles']
                molecule_dict[cid] = smiles
            
            # 통계 저장
            self.loading_stats[library_name] = {
                'original_count': original_count,
                'loaded_count': len(molecule_dict)
            }
            
            logging.info(f"{library_name} library loaded: {len(molecule_dict):,} compounds")
            
            del df
            gc.collect()
            
            return molecule_dict
            
        except pd.errors.EmptyDataError:
            raise DataLoadingError(f"Empty CSV file: {csv_file}")
        except Exception as e:
            raise DataLoadingError(f"Error loading CSV {csv_file}: {str(e)}")
    
    def apply_deduplication(self, molecule_dict: Dict[str, str], library_name: str = "library") -> Dict[str, str]:
        """SMILES 기반 중복제거를 적용합니다."""
        if not self.enable_deduplication or self.deduplicator is None:
            logging.info(f"{library_name} deduplication disabled")
            return molecule_dict
        
        original_count = len(molecule_dict)
        logging.info(f"Applying SMILES deduplication to {library_name}...")
        
        deduplicated_dict, duplicate_count = self.deduplicator.deduplicate_dictionary(molecule_dict)
        
        dedup_rate = (duplicate_count / original_count * 100) if original_count > 0 else 0
        logging.info(f"{library_name} deduplication: {original_count:,} → {len(deduplicated_dict):,} "
                    f"({dedup_rate:.1f}% duplicates removed)")
        
        # 통계 업데이트
        if library_name in self.loading_stats:
            self.loading_stats[library_name]['after_dedup'] = len(deduplicated_dict)
            self.loading_stats[library_name]['duplicates_removed'] = duplicate_count
        
        return deduplicated_dict
    
    def apply_element_filter(self, molecule_dict: Dict[str, str], library_name: str = "library") -> Dict[str, str]:
        """원소 필터링을 적용합니다."""
        if not self.enable_element_filter or self.molecule_filter is None:
            logging.info(f"{library_name} element filtering disabled")
            return molecule_dict
        
        original_count = len(molecule_dict)
        logging.info(f"Applying element filtering to {library_name} ({original_count:,} compounds)...")
        
        filtered_dict, filtered_count = self.molecule_filter.filter_molecule_dict(molecule_dict)
        
        filter_rate = (filtered_count / original_count * 100) if original_count > 0 else 0
        logging.info(f"{library_name} element filtering: {original_count:,} → {len(filtered_dict):,} "
                    f"({filter_rate:.1f}% filtered out)")
        
        # 통계 업데이트
        if library_name in self.loading_stats:
            self.loading_stats[library_name]['after_element_filter'] = len(filtered_dict)
            self.loading_stats[library_name]['element_filtered'] = filtered_count
        
        return filtered_dict
    
    def get_loading_statistics(self) -> Dict[str, Dict]:
        """로딩 통계를 반환합니다."""
        return self.loading_stats.copy()
    
    def save_deduplication_report(self, output_dir: str):
        """중복제거 리포트를 저장합니다."""
        if self.enable_deduplication and self.deduplicator is not None:
            self.deduplicator.save_report(output_dir)


class MolecularDataLoader:
    """
    분자 데이터 로딩을 위한 통합 인터페이스
    
    메인 라이브러리와 초기 CSV를 로딩하고 통합 전처리를 수행합니다.
    """
    
    def __init__(self,
                 ligand_file: str,
                 similarity_threshold: float = 0.2,
                 enable_element_filter: bool = True,
                 enable_deduplication: bool = True):
        """
        Args:
            ligand_file: 기준 리간드 PDB 파일
            similarity_threshold: 유사도 임계값
            enable_element_filter: 원소 필터링 활성화 여부
            enable_deduplication: 중복제거 활성화 여부
        """
        self.ligand_file = ligand_file
        self.similarity_threshold = similarity_threshold
        
        # CSV 로더 초기화
        self.csv_loader = CSVDataLoader(enable_element_filter, enable_deduplication)
        
        # 기준 분자의 SMILES 추출
        self.reference_smiles = self._extract_reference_smiles()
        
        # 유사도 필터 초기화
        self.similarity_filter = SimilarityFilter(self.reference_smiles, similarity_threshold)
        
        # 결합된 데이터 저장
        self.combined_data = {}
        self.invalid_smiles = []
    
    def _extract_reference_smiles(self) -> str:
        """기준 리간드 파일에서 SMILES를 추출합니다."""
        try:
            ref_mol = next(toolkit.readfile('pdb', self.ligand_file))
            ref_smiles = ref_mol.write('smi').strip()
            logging.info(f"Reference SMILES extracted: {ref_smiles}")
            return ref_smiles
        except Exception as e:
            raise DataLoadingError(f"Failed to extract reference SMILES from {self.ligand_file}: {str(e)}")
    
    def load_and_process(self, 
                        main_csv: str, 
                        initial_csv: Optional[str] = None) -> Tuple[Dict[str, str], List[str]]:
        """
        메인 라이브러리와 초기 CSV를 로딩하고 통합 전처리를 수행합니다.
        
        Args:
            main_csv: 메인 라이브러리 CSV 파일
            initial_csv: 초기 화합물 CSV 파일 (선택사항)
            
        Returns:
            (통합된 {cid: smiles} 딕셔너리, 무효한 SMILES 리스트)
        """
        # 메인 라이브러리 처리
        main_library = self._process_library(main_csv, "main")
        
        # 초기 CSV 처리 (있는 경우)
        initial_library = {}
        if initial_csv and os.path.exists(initial_csv):
            initial_library = self._process_library(initial_csv, "initial")
        
        # 라이브러리 통합
        combined_library = self._combine_libraries(main_library, initial_library)
        
        # 무효한 SMILES 수집
        all_invalid_smiles = self.similarity_filter.invalid_smiles.copy()
        
        # 최종 통계 로깅
        self._log_final_statistics(main_library, initial_library, combined_library)
        
        self.combined_data = combined_library
        self.invalid_smiles = all_invalid_smiles
        
        return combined_library, all_invalid_smiles
    
    def _process_library(self, csv_file: str, library_name: str) -> Dict[str, str]:
        """단일 라이브러리를 최적화된 순서로 처리합니다."""
        # 1단계: CSV 로딩
        raw_data = self.csv_loader.load_csv(csv_file, library_name)
        
        # 2단계: 중복제거 (가장 효율적인 첫 번째 필터)
        dedup_data = self.csv_loader.apply_deduplication(raw_data, library_name)
        
        # 3단계: 유사도 필터링 (중복 제거 후 더 적은 수에 대해)
        similar_data = self.similarity_filter.filter_by_similarity(dedup_data)
        
        # 4단계: 원소 필터링 (가장 빠른 마지막 필터)
        final_data = self.csv_loader.apply_element_filter(similar_data, library_name)
        
        # 메모리 정리
        del raw_data, dedup_data, similar_data
        gc.collect()
        
        return final_data
    
    def _combine_libraries(self, main_library: Dict[str, str], 
                          initial_library: Dict[str, str]) -> Dict[str, str]:
        """라이브러리들을 통합합니다."""
        combined = main_library.copy()
        overlap_count = 0
        
        for cid, smiles in initial_library.items():
            if cid in combined:
                overlap_count += 1
            else:
                combined[cid] = smiles
        
        logging.info(f"Combined library: {len(combined)} compounds")
        logging.info(f"  - From main library: {len(main_library)}")
        logging.info(f"  - From initial CSV (unique): {len(initial_library) - overlap_count}")
        logging.info(f"  - Overlapping compounds: {overlap_count}")
        
        return combined
    
    def _log_final_statistics(self, main_library: Dict[str, str], 
                            initial_library: Dict[str, str], 
                            combined_library: Dict[str, str]):
        """최종 처리 통계를 로깅합니다."""
        stats = self.csv_loader.get_loading_statistics()
        
        total_original = sum(stat.get('original_count', 0) for stat in stats.values())
        final_total = len(combined_library)
        
        logging.info("=" * 60)
        logging.info("OPTIMIZED FILTERING PIPELINE COMPLETED")
        logging.info("=" * 60)
        logging.info(f"Combined Library Final Statistics:")
        logging.info(f"  Total Original: {total_original:,} compounds")
        logging.info(f"  Final Combined: {final_total:,} compounds")
        
        if total_original > 0:
            efficiency = (final_total / total_original * 100)
            logging.info(f"  Overall Efficiency: {efficiency:.1f}% retained")
        
        logging.info(f"  Processing Order: Deduplication → Similarity → Element Filtering")
        logging.info("=" * 60)
    
    def save_deduplication_report(self, output_dir: str):
        """중복제거 리포트를 저장합니다."""
        self.csv_loader.save_deduplication_report(output_dir)
    
    def get_statistics(self) -> Dict:
        """로딩 통계를 반환합니다."""
        return {
            'csv_loading_stats': self.csv_loader.get_loading_statistics(),
            'combined_compounds': len(self.combined_data),
            'invalid_smiles_count': len(self.invalid_smiles),
            'similarity_threshold': self.similarity_threshold,
            'reference_smiles': self.reference_smiles
        }