import os
import sys
import argparse
import logging
import yaml
from datetime import datetime
from Activedocking import ActiveLearningWithConformers

def load_yaml_config(config_file):
    """YAML 설정 파일을 로드합니다."""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_file}")
    except yaml.YAMLError as e:
        raise ValueError(f"YAML 파일 파싱 오류: {e}")
    except Exception as e:
        raise Exception(f"설정 파일 로드 오류: {e}")

def setup_logging(output_dir):
    log_file = os.path.join(output_dir, 'active_learning.log')
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def main():
    parser = argparse.ArgumentParser(
        description="Active Learning for Molecular Docking with TabM Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python main.py -o results -l ligand.pdb -p protein.pdb -d database.csv

  # TabM (deep learning) with auto-tuning
  python main.py -o results -l ligand.pdb -p protein.pdb -d database.csv \\
    --auto_tune --epochs 150

  # Custom parameters with tuning
  python main.py -o results -l ligand.pdb -p protein.pdb -d database.csv \\
    --auto_tune --batch_size 20
        """
    )
    
    # ===== 설정 파일 옵션 =====
    parser.add_argument('--config', type=str, 
                       help="YAML 설정 파일 경로")
    
    # ===== 필수 파라미터 =====
    required = parser.add_argument_group('Required Arguments')
    required.add_argument('-o', '--output_dir', default=None,
                         help="Output directory for results")
    required.add_argument('-l', '--ligand', default=None,
                         help="Reference ligand PDB file") 
    required.add_argument('-p', '--protein', default=None,
                         help="Target protein PDB file") 
    required.add_argument('-d', '--database', default=None,
                         help="Chemical database (CSV file or directory containing CSV files)")
    
    # ===== 튜닝 옵션 =====
    tuning_group = parser.add_argument_group('Hyperparameter Tuning')
    tuning_group.add_argument('--auto_tune', action='store_true', 
                             help="Enable automatic hyperparameter tuning")
    tuning_group.add_argument('--tuning_method', type=str, default='random',
                             choices=['grid', 'random'],
                             help="Tuning method: grid or random search (default: random)")
    
    # ===== Active Learning 파라미터 =====
    al_group = parser.add_argument_group('Active Learning Parameters')
    al_group.add_argument('-i', '--initial_csv', 
                         help="Initial training set CSV file (optional)")
    al_group.add_argument('-s', '--similarity', type=float, default=0.2, 
                         help="Molecular similarity threshold (default: 0.2)")
    al_group.add_argument('--hbond_similarity', type=float, default=0.3,
                         help="H-bond similarity threshold (default: 0.3)")
    al_group.add_argument('--rmsd_threshold', type=float, default=2.0,
                         help="RMSD threshold for valid poses (default: 2.0)")
    al_group.add_argument('--budget', type=int, default=100000,
                         help="Total compound budget (default: 100000)")
    al_group.add_argument('--batch_size', type=int, default=10,
                         help="Compounds per cycle (default: 10)")
    al_group.add_argument('--n_cycles', type=int, default=10,
                         help="Number of AL cycles (default: 10)")
    al_group.add_argument('--conf_num', type=int, default=1,
                         help="Conformers per molecule (default: 1)")
    
    # TabM 전용 파라미터
    tabm_params = parser.add_argument_group('TabM (Deep Learning) Parameters')
    tabm_params.add_argument('--learning_rate', type=float, default=0.001,
                            help="Learning rate for TabM (default: 0.001)")
    tabm_params.add_argument('--dropout', type=float, default=0.2,
                            help="Dropout rate for TabM (default: 0.2)")
    tabm_params.add_argument('--epochs', type=int, default=100,
                            help="Training epochs for TabM (default: 100)")
    tabm_params.add_argument('--batch_size_nn', type=int, default=256,
                            help="Neural network batch size for TabM (default: 256)")
    tabm_params.add_argument('--mc_dropout_samples', type=int, default=50,
                            help="Monte Carlo dropout samples for uncertainty (default: 50)")
    
    # ===== 시스템 설정 =====
    system_group = parser.add_argument_group('System Settings')
    system_group.add_argument('--max_threads', type=int, default=8,
                             help="Maximum CPU threads (default: 8)")
    system_group.add_argument('--disable_element_filter', action='store_true',
                             help="Disable element filtering for docking compatibility")
    system_group.add_argument('--disable_deduplication', action='store_true',
                             help="Disable SMILES-based deduplication")
    
    # ===== Exploration 파라미터 =====
    exploration_group = parser.add_argument_group('Exploration Strategy')
    exploration_group.add_argument('--initial_exploration_ratio', type=float, default=0.8,
                                  help="Initial exploration ratio (default: 0.8)")
    exploration_group.add_argument('--final_exploration_ratio', type=float, default=0.2,
                                  help="Final exploration ratio (default: 0.2)")
    
    # 명령행에서 실제로 제공된 인수들 기록
    provided_args = set()
    for action in parser._actions:
        for option_string in action.option_strings:
            if option_string in sys.argv:
                provided_args.add(action.dest)
    
    args = parser.parse_args()
    
    # ===== YAML 설정 파일 처리 =====
    if args.config:
        try:
            config = load_yaml_config(args.config)
            
            # 명령행에서 제공되지 않은 값들을 YAML에서 가져오기
            for key, value in config.items():
                if hasattr(args, key):
                    was_provided = key in provided_args
                    if not was_provided:
                        setattr(args, key, value)
                    
        except Exception as e:
            print(f"오류: {e}")
            return 1
    
    # ===== 필수 매개변수 검증 =====
    required_params = {
        'output_dir': '-o/--output_dir',
        'ligand': '-l/--ligand',
        'protein': '-p/--protein', 
        'database': '-d/--database'
    }
    
    missing_params = []
    for param, flag in required_params.items():
        if not getattr(args, param):
            missing_params.append(flag)
    
    if missing_params:
        print("오류: 다음 필수 매개변수들이 누락되었습니다:")
        for param in missing_params:
            print(f"  {param}")
        if args.config:
            print(f"설정 파일 '{args.config}'에서 이 값들을 제공해주세요.")
        return 1
    
    setup_logging(args.output_dir)
    
    start_time = datetime.now()
    
    logging.info("=" * 60)
    logging.info(f"ACTIVE LEARNING FOR MOLECULAR DOCKING")
    logging.info("=" * 60)
    
    logging.info(f"INPUT FILES:")
    logging.info(f"   Protein: {args.protein}")
    logging.info(f"   Ligand: {args.ligand}")
    logging.info(f"   Database: {args.database}")
    if args.initial_csv:
        logging.info(f"   Initial set: {args.initial_csv}")
    logging.info(f"   Output: {args.output_dir}")
    
    logging.info(f"MODEL CONFIGURATION:")
    logging.info(f"   Type: TABM")
    
    if args.auto_tune:
        logging.info("   Auto-tuning: ENABLED")
        logging.info(f"   Tuning method: {args.tuning_method}")
    else:
        logging.info("   Auto-tuning: DISABLED")
    
    logging.info(f"ACTIVE LEARNING SETUP:")
    logging.info(f"   Budget: {args.budget:,} compounds")
    logging.info(f"   Batch size: {args.batch_size}")
    logging.info(f"   Cycles: {args.n_cycles}")
    logging.info(f"   Similarity threshold: {args.similarity}")
    logging.info(f"   H-bond threshold: {args.hbond_similarity}")
    logging.info(f"   RMSD threshold: {args.rmsd_threshold}")
    
    logging.info(f"NEURAL NETWORK PARAMETERS:")
    logging.info(f"   Learning rate: {args.learning_rate}")
    logging.info(f"   Dropout: {args.dropout}")
    logging.info(f"   Epochs: {args.epochs}")
    logging.info(f"   Batch size: {args.batch_size_nn}")
    logging.info(f"   MC samples: {args.mc_dropout_samples}")
    
    logging.info(f"SYSTEM SETTINGS:")
    logging.info(f"   CPU threads: {args.max_threads}")
    logging.info(f"   Element filter: {'DISABLED' if args.disable_element_filter else 'ENABLED'}")
    logging.info(f"   Deduplication: {'DISABLED' if args.disable_deduplication else 'ENABLED'}")
    
    logging.info("=" * 60)
    logging.info(f"Starting at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("=" * 60)
    
    try:
        al = ActiveLearningWithConformers(
            csv_file=args.database,
            protein_file=args.protein,
            ligand_file=args.ligand,
            output_dir=args.output_dir,
            similarity_threshold=args.similarity,
            hbond_similarity_threshold=args.hbond_similarity,
            rmsd_threshold=args.rmsd_threshold,
            initial_csv=args.initial_csv,
            enable_element_filter=not args.disable_element_filter,
            enable_deduplication=not args.disable_deduplication,
            budget=args.budget,
            batch_size=args.batch_size,
            n_cycles=args.n_cycles,
            conf_num=args.conf_num,
            # 모델 파라미터
            auto_tune=args.auto_tune,
            tuning_method=args.tuning_method,
            learning_rate=args.learning_rate,
            dropout=args.dropout,
            epochs=args.epochs,
            model_batch_size=getattr(args, 'batch_size_nn', 256),
            mc_dropout_samples=args.mc_dropout_samples,
            max_threads=args.max_threads,
            # Exploration 파라미터
            initial_exploration_ratio=args.initial_exploration_ratio,
            final_exploration_ratio=args.final_exploration_ratio
        )

        results, model = al.run()

        end_time = datetime.now()
        total_time = end_time - start_time
        
        logging.info("=" * 60)
        logging.info(f"FINAL RESULTS")
        logging.info("=" * 60)
        
        if results:
            logging.info(f"SUCCESS! Found {len(results)} valid compounds")
            logging.info(f"Best docking score: {results[0][1]:.3f} (CID: {results[0][0]})")
            
            logging.info(f"TOP 5 COMPOUNDS:")
            for i, (cid, score) in enumerate(results[:5], 1):
                logging.info(f"   {i}. CID {cid}: {score:.3f}")
            
            if hasattr(model, 'get_model_info'):
                model_info = model.get_model_info()
                if model_info.get('best_params'):
                    logging.info("OPTIMAL HYPERPARAMETERS:")
                    for param, value in model_info['best_params'].items():
                        logging.info(f"   {param}: {value}")
        else:
            logging.warning("No valid compounds found")
            logging.info("💡 Try adjusting similarity thresholds or increasing budget")
        
        logging.info(f"PERFORMANCE STATISTICS:")
        logging.info(f"   Total runtime: {total_time}")
        logging.info(f"   Average per cycle: {total_time / args.n_cycles if args.n_cycles > 0 else 'N/A'}")
        
        logging.info(f"OUTPUT FILES:")
        logging.info(f"   Final results: {args.output_dir}/final_results.csv")
        logging.info(f"   Cycle results: {args.output_dir}/cycle_*/")
        logging.info(f"   Docking poses: {args.output_dir}/pose/")
        logging.info(f"   Log file: {args.output_dir}/active_learning.log")
        
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
        return 1
    except Exception as e:
        logging.error("=" * 60)
        logging.error(f"ERROR OCCURRED")
        logging.error("=" * 60)
        logging.error(f"Error: {str(e)}")
        
        import traceback
        logging.error("Detailed traceback:")
        logging.error(traceback.format_exc())
        
        logging.error("Troubleshooting tips:")
        logging.error("   1. Check if all input files exist and are readable")
        logging.error("   2. Verify CSV file format (must have 'Catalog ID' and 'smiles' columns)")
        logging.error("   3. Ensure sufficient disk space and memory")
        logging.error("   4. Try reducing batch_size if memory issues occur")
        
        return 1
    
    finally:
        end_time = datetime.now()
        logging.info("=" * 60)
        logging.info(f"Completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("=" * 60)
    
    return 0

if __name__ == '__main__':
    exit(main())
