# Mixin class for saving results
import os
import csv
import logging
import numpy as np

class ResultSaverMixin:
    def __init__(self):
        super().__init__()
        
    def save_cycle_detailed_results(self, cycle_number, cycle_compounds):
        """기본 사이클 상세 결과 저장 (예측 정보 없이)"""
        cycle_dir = os.path.join(self.output_dir, f'cycle_{cycle_number}')
        os.makedirs(cycle_dir, exist_ok=True)

        detailed_file = os.path.join(cycle_dir, 'detailed_results_before_filtering.csv')
        with open(detailed_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'CID', 'SMILES', 'qVina_affinity', 
                'MCS_size', 'MCS_similarity', 'RMSD', 'HBond_similarity'
            ])
            
            for cid in cycle_compounds:
                if cid in self.docking_engine.docking_scores and cid in self.docking_engine.interaction_data:
                    rmsd = self.docking_engine.interaction_data[cid]['rmsd']
                    hbond_sim = self.docking_engine.interaction_data[cid]['hbond_similarity']
                    
                    writer.writerow([
                        cid,
                        self.structure_info[cid],
                        f"{self.docking_engine.docking_scores[cid]:.3f}",
                        self.docking_engine.interaction_data[cid]['mcs_size'],
                        f"{self.docking_engine.interaction_data[cid]['mcs_similarity']:.3f}",
                        f"{rmsd:.3f}",
                        f"{hbond_sim:.3f}"
                    ])

        logging.info(f"Saved detailed cycle {cycle_number} results to {detailed_file}")

    def save_cycle_detailed_results_with_predictions(self, cycle_number, cycle_compounds):
        """예측 정보가 포함된 상세 결과 저장"""
        cycle_dir = os.path.join(self.output_dir, f'cycle_{cycle_number}')
        os.makedirs(cycle_dir, exist_ok=True)

        # 예측 정보가 포함된 상세 결과 파일
        detailed_file = os.path.join(cycle_dir, 'detailed_results_with_predictions.csv')
        with open(detailed_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'CID', 'SMILES', 'Predicted_Score', 'Prediction_Uncertainty', 
                'Acquisition_Score', 'Exploitation_Score', 'Exploration_Score',
                'Actual_qVina_affinity', 'MCS_size', 'MCS_similarity', 'RMSD', 'HBond_similarity',
                'Docking_Status', 'Prediction_Error'
            ])
            
            for cid in cycle_compounds:
                smiles = self.structure_info.get(cid, 'N/A')
                
                # 예측 정보 가져오기
                cycle_pred_data = self.sample_acquisition.cycle_predictions.get(cycle_number, {}).get(cid, {})
                predicted_score = cycle_pred_data.get('predicted_score', 'N/A')
                uncertainty = cycle_pred_data.get('uncertainty', 'N/A')
                acquisition_score = cycle_pred_data.get('acquisition_score', 'N/A')
                exploitation_score = cycle_pred_data.get('exploitation_score', 'N/A')
                exploration_score = cycle_pred_data.get('exploration_score', 'N/A')
                
                # 실제 도킹 결과
                if cid in self.docking_engine.docking_scores and cid in self.docking_engine.interaction_data:
                    actual_score = self.docking_engine.docking_scores[cid]
                    rmsd = self.docking_engine.interaction_data[cid]['rmsd']
                    hbond_sim = self.docking_engine.interaction_data[cid]['hbond_similarity']
                    mcs_size = self.docking_engine.interaction_data[cid]['mcs_size']
                    mcs_sim = self.docking_engine.interaction_data[cid]['mcs_similarity']
                    docking_status = 'Success'
                    
                    # 예측 오차 계산
                    if predicted_score != 'N/A':
                        prediction_error = abs(float(predicted_score) - actual_score)
                    else:
                        prediction_error = 'N/A'
                else:
                    actual_score = 'Failed'
                    rmsd = 'N/A'
                    hbond_sim = 'N/A'
                    mcs_size = 'N/A'
                    mcs_sim = 'N/A'
                    docking_status = 'Failed'
                    prediction_error = 'N/A'
                
                writer.writerow([
                    cid, smiles, 
                    f"{predicted_score:.3f}" if predicted_score != 'N/A' else predicted_score,
                    f"{uncertainty:.3f}" if uncertainty != 'N/A' else uncertainty,
                    f"{acquisition_score:.3f}" if acquisition_score != 'N/A' else acquisition_score,
                    f"{exploitation_score:.3f}" if exploitation_score != 'N/A' else exploitation_score,
                    f"{exploration_score:.3f}" if exploration_score != 'N/A' else exploration_score,
                    f"{actual_score:.3f}" if actual_score != 'Failed' else actual_score,
                    mcs_size, 
                    f"{mcs_sim:.3f}" if mcs_sim != 'N/A' else mcs_sim,
                    f"{rmsd:.3f}" if rmsd != 'N/A' else rmsd,
                    f"{hbond_sim:.3f}" if hbond_sim != 'N/A' else hbond_sim,
                    docking_status,
                    f"{prediction_error:.3f}" if prediction_error != 'N/A' else prediction_error
                ])

        logging.info(f"Saved detailed cycle {cycle_number} results with predictions to {detailed_file}")

    def save_prediction_analysis(self):
        """모든 예측 정보를 종합하는 최종 분석 파일 저장"""
        analysis_file = os.path.join(self.output_dir, 'prediction_analysis.csv')
        
        with open(analysis_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'CID', 'SMILES', 'Predicted_Score', 'Actual_Score', 'Prediction_Error',
                'Prediction_Uncertainty', 'Final_Acquisition_Score', 'Was_Selected'
            ])
            
            # 실제 도킹된 분자들에 대한 예측 성능 분석
            for cid in self.docking_engine.docking_scores:
                if cid in self.sample_acquisition.predicted_scores:  # 예측이 있는 경우만
                    predicted = self.sample_acquisition.predicted_scores[cid]
                    actual = self.docking_engine.docking_scores[cid]
                    error = abs(predicted - actual)
                    uncertainty = self.sample_acquisition.prediction_uncertainty.get(cid, 'N/A')
                    acq_score = self.sample_acquisition.acquisition_scores.get(cid, 'N/A')
                    
                    writer.writerow([
                        cid,
                        self.structure_info.get(cid, 'N/A'),
                        f"{predicted:.3f}",
                        f"{actual:.3f}",
                        f"{error:.3f}",
                        f"{uncertainty:.3f}" if uncertainty != 'N/A' else uncertainty,
                        f"{acq_score:.3f}" if acq_score != 'N/A' else acq_score,
                        'Yes'
                    ])
        
        logging.info(f"Prediction analysis saved to {analysis_file}")
        
        # 예측 성능 통계 계산 및 출력
        self.log_prediction_statistics()

    def log_prediction_statistics(self):
        """예측 성능 통계 출력"""
        if not self.sample_acquisition.predicted_scores:
            logging.info("No prediction data available for analysis")
            return
        
        # 실제 도킹된 분자 중 예측이 있는 것들만 분석
        valid_predictions = []
        for cid in self.docking_engine.docking_scores:
            if cid in self.sample_acquisition.predicted_scores:
                predicted = self.sample_acquisition.predicted_scores[cid]
                actual = self.docking_engine.docking_scores[cid]
                valid_predictions.append((predicted, actual))
        
        if len(valid_predictions) < 2:
            logging.info("Insufficient prediction data for statistical analysis")
            return
        
        predicted_values = [p[0] for p in valid_predictions]
        actual_values = [p[1] for p in valid_predictions]
        errors = [abs(p - a) for p, a in valid_predictions]
        
        # 통계 계산
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean([(p - a)**2 for p, a in valid_predictions]))
        
        correlation = np.corrcoef(predicted_values, actual_values)[0, 1]
        
        logging.info("\n=== Prediction Performance Analysis ===")
        logging.info(f"Total predictions made: {len(self.sample_acquisition.predicted_scores):,}")
        logging.info(f"Predictions with actual docking: {len(valid_predictions)}")
        logging.info(f"Prediction MAE: {mae:.3f}")
        logging.info(f"Prediction RMSE: {rmse:.3f}")
        logging.info(f"Prediction-Actual Correlation: {correlation:.3f}")
        logging.info(f"Best prediction error: {min(errors):.3f}")
        logging.info(f"Worst prediction error: {max(errors):.3f}")
        logging.info("=======================================")

    def save_cycle_results(self, cycle_number):
        cycle_dir = os.path.join(self.output_dir, f'cycle_{cycle_number}')
        os.makedirs(cycle_dir, exist_ok=True)

        results_file = os.path.join(cycle_dir, 'cycle_results.csv')
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'CID', 'SMILES', 'qVina_affinity', 
                'MCS_size', 'MCS_similarity', 'RMSD', 'HBond_similarity'
            ])
            
            for idx in self.training_indices:
                cid = self.cids[idx]
                if (cid in self.docking_engine.docking_scores and 
                    cid in self.docking_engine.interaction_data and 
                    self.docking_engine.interaction_data[cid]['rmsd'] <= 2.0 and
                    self.docking_engine.interaction_data[cid]['hbond_similarity'] >= self.hbond_similarity_threshold):
                    
                    writer.writerow([
                        cid,
                        self.structure_info[cid],
                        f"{self.docking_engine.docking_scores[cid]:.3f}",
                        self.docking_engine.interaction_data[cid]['mcs_size'],
                        f"{self.docking_engine.interaction_data[cid]['mcs_similarity']:.3f}",
                        f"{self.docking_engine.interaction_data[cid]['rmsd']:.3f}",
                        f"{self.docking_engine.interaction_data[cid]['hbond_similarity']:.3f}"
                    ])

        logging.info(f"Saved cycle {cycle_number} results to {results_file}")

    def save_final_results(self):
        results_file = os.path.join(self.output_dir, 'final_results.csv')
        
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'CID', 'SMILES', 'qVina_affinity', 
                'MCS_size', 'MCS_similarity', 'RMSD', 'HBond_similarity'
            ])
            
            for cid, score in sorted(self.docking_engine.docking_scores.items(), key=lambda x: x[1]):
                if cid not in self.structure_info:
                    logging.debug(f"Skipping removed CID: {cid}")
                    continue

                if (cid in self.docking_engine.interaction_data and 
                    self.docking_engine.interaction_data[cid]['rmsd'] <= 2.0 and
                    self.docking_engine.interaction_data[cid]['hbond_similarity'] >= self.hbond_similarity_threshold):
                    
                    writer.writerow([
                        cid,
                        self.structure_info[cid],
                        f"{score:.3f}",
                        self.docking_engine.interaction_data[cid]['mcs_size'],
                        f"{self.docking_engine.interaction_data[cid]['mcs_similarity']:.3f}",
                        f"{self.docking_engine.interaction_data[cid]['rmsd']:.3f}",
                        f"{self.docking_engine.interaction_data[cid]['hbond_similarity']:.3f}"
                    ])

        logging.info(f"Final results saved to {results_file}")