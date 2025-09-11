import os
import csv
import time
import psutil
import gc
import logging
import datetime
from typing import Dict, Any, Optional


class CyclePerformanceMonitor:    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.cycle_data = []
        self.cycle_start_time = None
        self.current_cycle = None
        self.summary_file = os.path.join(output_dir, 'cycle_performance_summary.csv')
        self._initialize_summary_file()
        
    def _initialize_summary_file(self):
        with open(self.summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Cycle', 'Duration_Seconds', 'Duration_Minutes', 'Compounds_Processed',
                'Processing_Rate_CPM', 'Training_R2', 'Training_MAE', 'Training_RMSE', 'Training_Samples'
            ])
    
    def start_cycle(self, cycle_number: int):
        self.current_cycle = cycle_number
        self.cycle_start_time = time.time()
        
        logging.info(f"Cycle {cycle_number} started")
        
        self.cycle_data.append({
            'cycle': cycle_number,
            'start_time': self.cycle_start_time,
            'compounds_processed': 0
        })
    
    def end_cycle(self, compounds_processed: int = 0, metrics: dict = None):
        if self.cycle_start_time is None or self.current_cycle is None:
            logging.warning("Cycle not properly started")
            return
        
        cycle_duration = time.time() - self.cycle_start_time
        
        processing_rate = 0.0
        if cycle_duration > 0 and compounds_processed > 0:
            processing_rate = compounds_processed / (cycle_duration / 60)
        
        current_data = self.cycle_data[-1]
        current_data.update({
            'duration': cycle_duration,
            'compounds_processed': compounds_processed,
            'processing_rate': processing_rate,
            'metrics': metrics
        })
        
        logging.info(f"Cycle {self.current_cycle} completed:")
        logging.info(f"  - Duration: {cycle_duration:.1f} seconds ({cycle_duration/60:.1f} minutes)")
        
        # 성능 지표 출력
        if metrics:
            logging.info(f"  - Training R²: {metrics.get('r2', 'N/A'):.4f}")
            logging.info(f"  - Training MAE: {metrics.get('mae', 'N/A'):.4f}")
            logging.info(f"  - Training RMSE: {metrics.get('rmse', 'N/A'):.4f}")
            logging.info(f"  - Training samples: {metrics.get('n_samples', 'N/A')}")
        
        logging.info(f"  - Compounds processed: {compounds_processed}")
        if processing_rate > 0:
            logging.info(f"  - Processing rate: {processing_rate:.1f} compounds/minute")
        
        # 메모리 사용량 로깅 추가
        if self.current_cycle % 5 == 0:  # 5사이클마다 상세 메모리 정보
            log_memory_usage(f"cycle {self.current_cycle} end", detailed=True)
        else:
            log_memory_usage(f"cycle {self.current_cycle} end")
        
        self._save_cycle_to_csv(current_data)
        self.cycle_start_time = None
        self.current_cycle = None

    def _save_cycle_to_csv(self, cycle_data: Dict[str, Any]):
        with open(self.summary_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # metrics에서 성능 지표 추출 (None 안전 처리)
            metrics = cycle_data.get('metrics', {})
            if metrics is None:
                metrics = {}
            training_r2 = metrics.get('r2', '')
            training_mae = metrics.get('mae', '')
            training_rmse = metrics.get('rmse', '')
            training_samples = metrics.get('n_samples', '')
            
            writer.writerow([
                cycle_data['cycle'],
                f"{cycle_data['duration']:.1f}",
                f"{cycle_data['duration']/60:.1f}",
                cycle_data['compounds_processed'],
                f"{cycle_data['processing_rate']:.1f}",
                f"{training_r2:.4f}" if training_r2 != '' else "",
                f"{training_mae:.4f}" if training_mae != '' else "",
                f"{training_rmse:.4f}" if training_rmse != '' else "",
                training_samples if training_samples != '' else ""
            ])
    
    def get_cycle_summary(self) -> Dict[str, Any]:
        if not self.cycle_data:
            return {}
        
        completed_cycles = [data for data in self.cycle_data if 'duration' in data]
        
        if not completed_cycles:
            return {}
        
        total_cycles = len(completed_cycles)
        total_time = sum(data['duration'] for data in completed_cycles)
        total_compounds = sum(data['compounds_processed'] for data in completed_cycles)
        
        avg_time_per_cycle = total_time / total_cycles
        avg_compounds_per_cycle = total_compounds / total_cycles
        overall_processing_rate = total_compounds / (total_time / 60) if total_time > 0 else 0
        
        return {
            'total_cycles': total_cycles,
            'total_time_seconds': total_time,
            'total_time_minutes': total_time / 60,
            'total_compounds': total_compounds,
            'avg_time_per_cycle': avg_time_per_cycle,
            'avg_compounds_per_cycle': avg_compounds_per_cycle,
            'overall_processing_rate': overall_processing_rate
        }
    
    def log_final_summary(self):
        summary = self.get_cycle_summary()
        
        if not summary:
            logging.info("No cycle data available for summary")
            return
        
        logging.info("\n=== Cycle Performance Summary ===")
        logging.info(f"Total cycles completed: {summary['total_cycles']}")
        logging.info(f"Total execution time: {summary['total_time_seconds']:.1f} seconds ({summary['total_time_minutes']:.1f} minutes)")
        logging.info(f"Average time per cycle: {summary['avg_time_per_cycle']:.1f} seconds")
        logging.info(f"Total compounds processed: {summary['total_compounds']}")
        logging.info(f"Average compounds per cycle: {summary['avg_compounds_per_cycle']:.1f}")
        logging.info(f"Overall processing rate: {summary['overall_processing_rate']:.1f} compounds/minute")
        logging.info("================================")
        logging.info(f"Cycle performance report saved to: {self.summary_file}")

def log_memory_usage(message: str = "", detailed: bool = False):
    """향상된 메모리 로깅"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        if detailed:
            # 시스템 메모리 정보
            system_memory = psutil.virtual_memory()
            logging.info(f"Memory usage {message}:")
            logging.info(f"  Process memory: {memory_mb:.1f} MB")
            logging.info(f"  System memory: {system_memory.percent:.1f}% used")
            logging.info(f"  Available: {system_memory.available / 1024 / 1024:.1f} MB")
        else:
            logging.debug(f"Memory usage {message}: {memory_mb:.1f} MB")
            
    except Exception as e:
        logging.debug(f"Failed to log memory usage: {e}")

if __name__ == "__main__":
    import tempfile
    import shutil
    
    test_dir = tempfile.mkdtemp()
    
    try:
        monitor = CyclePerformanceMonitor(test_dir)
        
        for cycle in range(3):
            monitor.start_cycle(cycle + 1)
            
            time.sleep(1.5)  
            
            monitor.end_cycle(compounds_processed=10 * (cycle + 1), validation_r2=0.5 + (cycle * 0.1))
        
        monitor.log_final_summary()
        
        print(f"Test completed. Check file: {monitor.summary_file}")
        
        with open(monitor.summary_file, 'r') as f:
            print("\nGenerated CSV content:")
            print(f.read())
        
    finally:
        shutil.rmtree(test_dir)
