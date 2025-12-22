"""
데이터 로딩 및 전처리 모듈

CSV 파일로부터 화학 구조 데이터를 로딩하고,
필터링, 중복제거, 유사도 계산 등의 전처리를 수행합니다.
"""

import os
import logging
import gc
import glob
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
        
        filtered_dict, filter_stats = self.molecule_filter.filter_molecule_dict(molecule_dict)
        
        # filter_stats는 딕셔너리이므로 필요한 값을 추출
        filtered_count = filter_stats.get('total_filtered', 0) if isinstance(filter_stats, dict) else 0
        filter_rate = filter_stats.get('filter_rate', 0) if isinstance(filter_stats, dict) else 0
        
        logging.info(f"{library_name} element filtering: {original_count:,} → {len(filtered_dict):,} "
                    f"({filter_rate:.1f}% filtered out)")
        
        # 통계 업데이트
        if library_name in self.loading_stats:
            self.loading_stats[library_name]['after_element_filter'] = len(filtered_dict)
            self.loading_stats[library_name]['element_filtered'] = filtered_count
            # 상세한 필터링 통계도 저장
            if isinstance(filter_stats, dict):
                self.loading_stats[library_name]['filter_stats'] = filter_stats
        
        return filtered_dict
    
    def get_loading_statistics(self) -> Dict[str, Dict]:
        """로딩 통계를 반환합니다."""
        # 데이터 검증 및 정리
        validated_stats = {}
        for lib_name, stat in self.loading_stats.items():
            if isinstance(stat, dict):
                # 필수 키들의 타입 검증
                validated_stat = {}
                for key in ['original_count', 'loaded_count']:
                    value = stat.get(key, 0)
                    if isinstance(value, (int, float)):
                        validated_stat[key] = int(value)
                    else:
                        logging.warning(f"Invalid {key} type in {lib_name}: {type(value)}, setting to 0")
                        validated_stat[key] = 0
                
                # 추가 통계 정보 복사
                for key, value in stat.items():
                    if key not in validated_stat:
                        validated_stat[key] = value
                        
                validated_stats[lib_name] = validated_stat
            else:
                logging.warning(f"Invalid statistics format for {lib_name}: {type(stat)}")
                validated_stats[lib_name] = {'original_count': 0, 'loaded_count': 0}
        
        return validated_stats
    
    def save_deduplication_report(self, output_dir: str):
        """중복제거 리포트를 저장합니다."""
        if self.enable_deduplication and self.deduplicator is not None:
            self.deduplicator.save_report(output_dir)

    def load_directory(self, directory_path: str, library_name: str = "main") -> Dict[str, str]:
        """
        디렉터리 내 모든 CSV 파일을 로딩하고 병합합니다.

        Args:
            directory_path: CSV 파일들이 있는 디렉터리 경로
            library_name: 라이브러리 이름 (로깅용)

        Returns:
            병합된 {cid: smiles} 딕셔너리

        Raises:
            DataLoadingError: 디렉터리가 없거나 CSV 파일이 없는 경우
        """
        if not os.path.exists(directory_path):
            raise DataLoadingError(f"Directory not found: {directory_path}")

        if not os.path.isdir(directory_path):
            raise DataLoadingError(f"Path is not a directory: {directory_path}")

        # 디렉터리 내 모든 CSV 파일 검색
        csv_files = glob.glob(os.path.join(directory_path, "*.csv"))

        if not csv_files:
            raise DataLoadingError(f"No CSV files found in directory: {directory_path}")

        csv_files = sorted(csv_files)  # 일관된 순서로 처리
        logging.info(f"Loading chemical database from directory: {directory_path}")
        logging.info(f"Found {len(csv_files)} CSV files to process")

        # 각 CSV 파일을 개별 처리
        libraries = []
        file_stats = {}

        for idx, csv_file in enumerate(csv_files, 1):
            file_basename = os.path.basename(csv_file)
            logging.info(f"Processing file {idx}/{len(csv_files)}: {file_basename}")

            file_library_name = f"{library_name}_{idx}"

            try:
                # CSV 로딩
                molecule_dict = self.load_csv(csv_file, file_library_name)
                libraries.append(molecule_dict)

                # 파일별 통계 저장
                file_stats[file_basename] = {
                    'original_count': len(molecule_dict),
                    'library_name': file_library_name
                }

            except Exception as e:
                logging.warning(f"Failed to load {file_basename}: {str(e)}")
                continue

        if not libraries:
            raise DataLoadingError(f"Failed to load any valid CSV files from {directory_path}")

        # 라이브러리 병합
        logging.info(f"Merging {len(libraries)} processed libraries...")
        merged_library = self.merge_libraries(libraries)

        # 병합 통계 업데이트
        self.loading_stats[library_name] = {
            'directory': directory_path,
            'num_files': len(csv_files),
            'num_loaded_files': len(libraries),
            'file_stats': file_stats,
            'merged_count': len(merged_library)
        }

        logging.info(f"{library_name} directory loaded: {len(merged_library):,} compounds from {len(libraries)} files")

        return merged_library

    def merge_libraries(self, libraries: List[Dict[str, str]]) -> Dict[str, str]:
        """
        여러 라이브러리 딕셔너리를 병합하면서 중복을 처리합니다.

        Args:
            libraries: {cid: smiles} 딕셔너리들의 리스트

        Returns:
            병합된 {cid: smiles} 딕셔너리
        """
        if not libraries:
            return {}

        if len(libraries) == 1:
            return libraries[0]

        merged = {}
        duplicate_cids = 0
        total_before_merge = sum(len(lib) for lib in libraries)

        for library in libraries:
            for cid, smiles in library.items():
                if cid in merged:
                    duplicate_cids += 1
                    # 중복된 CID가 있으면 첫 번째 것을 유지
                    logging.debug(f"Duplicate CID found during merge: {cid}")
                else:
                    merged[cid] = smiles

        logging.info(f"Global merge: {total_before_merge:,} → {len(merged):,} compounds")

        if duplicate_cids > 0:
            logging.info(f"  - Cross-file duplicates removed: {duplicate_cids:,}")

        return merged


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

    def _detect_path_type(self, path: str) -> Tuple[str, str]:
        """
        경로가 파일인지 디렉터리인지 감지합니다.

        Args:
            path: 검사할 경로

        Returns:
            ('file', path) 또는 ('directory', path) 튜플

        Raises:
            DataLoadingError: 경로가 존재하지 않거나 유효하지 않은 경우
        """
        if not os.path.exists(path):
            raise DataLoadingError(f"Path does not exist: {path}")

        if os.path.isfile(path):
            return ('file', path)
        elif os.path.isdir(path):
            return ('directory', path)
        else:
            raise DataLoadingError(f"Path is neither a file nor a directory: {path}")

    def load_and_process(self,
                        main_csv: str,
                        initial_csv: Optional[str] = None) -> Tuple[Dict[str, str], List[str]]:
        """
        메인 라이브러리와 초기 CSV를 로딩하고 통합 전처리를 수행합니다.

        메인 라이브러리는 CSV 파일 또는 CSV 파일들이 있는 디렉터리로 제공할 수 있습니다.

        Args:
            main_csv: 메인 라이브러리 CSV 파일 또는 디렉터리 경로
            initial_csv: 초기 화합물 CSV 파일 (선택사항, 파일만 가능)

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
        try:
            logging.debug(f"Before _log_final_statistics, loading_stats: {self.csv_loader.loading_stats}")
            self._log_final_statistics(main_library, initial_library, combined_library)
        except Exception as e:
            logging.error(f"Critical error in _log_final_statistics: {str(e)}")
            logging.debug(f"Stats content: {self.csv_loader.loading_stats}")
            raise
        
        self.combined_data = combined_library
        self.invalid_smiles = all_invalid_smiles
        
        return combined_library, all_invalid_smiles
    
    def _process_library(self, path: str, library_name: str) -> Dict[str, str]:
        """
        라이브러리를 최적화된 순서로 처리합니다.

        파일 또는 디렉터리에서 데이터를 로딩하고 필터링 파이프라인을 적용합니다.

        Args:
            path: CSV 파일 경로 또는 CSV 파일들이 있는 디렉터리 경로
            library_name: 라이브러리 이름 (로깅 및 통계용)

        Returns:
            필터링된 {cid: smiles} 딕셔너리
        """
        # 1단계: 경로 유형 감지 및 CSV 로딩
        path_type, validated_path = self._detect_path_type(path)

        if path_type == 'file':
            logging.info(f"Loading {library_name} library from CSV file: {os.path.basename(path)}")
            raw_data = self.csv_loader.load_csv(validated_path, library_name)
        else:  # path_type == 'directory'
            logging.info(f"Loading {library_name} library from directory: {path}")
            raw_data = self.csv_loader.load_directory(validated_path, library_name)

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
        try:
            stats = self.csv_loader.get_loading_statistics()
            
            # 타입 안전성을 보장하는 통계 계산
            total_original = 0
            for lib_name, stat in stats.items():
                if isinstance(stat, dict):
                    original_count = stat.get('original_count', 0)
                    if isinstance(original_count, (int, float)):
                        total_original += int(original_count)
                    else:
                        logging.warning(f"Invalid original_count type in {lib_name}: {type(original_count)}")
                else:
                    logging.warning(f"Invalid stat type for {lib_name}: {type(stat)}")
            
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
            else:
                logging.warning("  Cannot calculate efficiency: total_original is 0")
            
            logging.info(f"  Processing Order: Deduplication → Similarity → Element Filtering")
            logging.info("=" * 60)
            
        except Exception as e:
            logging.error(f"Error logging final statistics: {str(e)}")
            logging.info("=" * 60)
            logging.info("OPTIMIZED FILTERING PIPELINE COMPLETED")
            logging.info("=" * 60)
            logging.info(f"Final Combined Library: {len(combined_library):,} compounds")
            logging.info("(Detailed statistics unavailable due to error)")
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