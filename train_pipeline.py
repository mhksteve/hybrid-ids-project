"""
Main Training Pipeline for Hybrid IDS
three scenarios
- Scenario A: Standalone Machine Learning (RF, XGBoost, SVM)
- Scenario B: Standalone Deep Learning (1D-CNN, LSTM)
- Scenario C: Hybrid Model (CNN Feature Extractor + RF Classifier)

Supports multiple datasets via command-line arguments:
- cicids: python train_pipeline.py --dataset cicids --data-path data/cicids/
- nsl-kdd: python train_pipeline.py --dataset nslkdd --train-path data/nslkdd/KDDTrain+.txt --test-path data/nslkdd/KDDTest+.txt
"""


import os
import argparse

from src.models_ml import MLModels
from src.models_dl import DLModels
from src.hybrid import HybridModel


def run_scenario_a(data, output_dir='models'):
    """
    Scenario A: Standalone Machine Learning
    Train and evaluate RF, XGBoost, and SVM models
    """
    print("\n\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "SCENARIO A: STANDALONE MACHINE LEARNING" + " "*19 + "║")
    print("╚" + "="*78 + "╝")
    
    ml_models = MLModels()
    
    # Build models
    ml_models.build_random_forest(n_estimators=100)
    ml_models.build_xgboost(n_estimators=100)
    ml_models.build_svm(kernel='rbf')
    
    # Train models
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Random Forest
    ml_models.train_model('RandomForest', X_train, y_train, X_val, y_val)
    rf_results = ml_models.evaluate_model('RandomForest', X_test, y_test, data['label_encoder'])
    ml_models.save_model('RandomForest', os.path.join(output_dir, 'random_forest.pkl'))
    
    # XGBoost
    ml_models.train_model('XGBoost', X_train, y_train, X_val, y_val)
    xgb_results = ml_models.evaluate_model('XGBoost', X_test, y_test, data['label_encoder'])
    ml_models.save_model('XGBoost', os.path.join(output_dir, 'xgboost.pkl'))
    
    # SVM
    ml_models.train_model('SVM', X_train, y_train, X_val, y_val, use_subset_for_svm=True, svm_subset_size=10000)
    svm_results = ml_models.evaluate_model('SVM', X_test, y_test, data['label_encoder'])
    ml_models.save_model('SVM', os.path.join(output_dir, 'svm.pkl'))
    
    # Output summary
    print("\n" + "="*60)
    print("SCENARIO A SUMMARY")
    print("="*60)
    print(f"Random Forest - Accuracy: {rf_results['accuracy']*100:.2f}%, Latency: {rf_results['avg_latency_ms']:.2f} ms")
    print(f"XGBoost       - Accuracy: {xgb_results['accuracy']*100:.2f}%, Latency: {xgb_results['avg_latency_ms']:.2f} ms")
    print(f"SVM           - Accuracy: {svm_results['accuracy']*100:.2f}%, Latency: {svm_results['avg_latency_ms']:.2f} ms")
    print("="*60)
    
    return ml_models, {
        'RandomForest': rf_results,
        'XGBoost': xgb_results,
        'SVM': svm_results
    }


def run_scenario_b(data, output_dir='models'):
    """
    Scenario B: Standalone Deep Learning
    Train and evaluate 1D-CNN and LSTM models
    """
    print("\n\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "SCENARIO B: STANDALONE DEEP LEARNING" + " "*22 + "║")
    print("╚" + "="*78 + "╝")
    
    dl_models = DLModels()
    
    # Prepare data
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    input_shape = (data['n_features'], 1)
    n_classes = data['n_classes']
    
    # Build and train 1D-CNN
    dl_models.build_1d_cnn(input_shape, n_classes, filters=[64, 128, 256])
    dl_models.train_model('1D-CNN', X_train, y_train, X_val, y_val, epochs=20, batch_size=256, patience=10) #set batch size epoch 5
    cnn_results = dl_models.evaluate_model('1D-CNN', X_test, y_test, data['label_encoder'])
    dl_models.save_model('1D-CNN', os.path.join(output_dir, '1d_cnn.h5'))
    
    # Build and train LSTM
    dl_models.build_lstm(input_shape, n_classes, lstm_units=[128, 64])
    dl_models.train_model('LSTM', X_train, y_train, X_val, y_val, epochs=20, batch_size=256, patience=10) #set batch size epoch 5
    lstm_results = dl_models.evaluate_model('LSTM', X_test, y_test, data['label_encoder'])
    dl_models.save_model('LSTM', os.path.join(output_dir, 'lstm.h5'))
    
    # Summary
    print("\n" + "="*60)
    print("SCENARIO B SUMMARY")
    print("="*60)
    print(f"1D-CNN - Accuracy: {cnn_results['accuracy']*100:.2f}%, Latency: {cnn_results['avg_latency_ms']:.2f} ms")
    print(f"LSTM   - Accuracy: {lstm_results['accuracy']*100:.2f}%, Latency: {lstm_results['avg_latency_ms']:.2f} ms")
    print("="*60)
    
    return dl_models, {
        '1D-CNN': cnn_results,
        'LSTM': lstm_results
    }


def run_scenario_c(data, dl_models, output_dir='models'):
    """
    Scenario C: Hybrid Model
    Combine CNN feature extraction with RF classification
    """
    print("\n\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*26 + "SCENARIO C: HYBRID MODEL" + " "*28 + "║")
    print("╚" + "="*78 + "╝")
    
    hybrid = HybridModel()
    
    # Build hybrid from pre-trained CNN
    cnn_model = dl_models.models['1D-CNN']
    hybrid.build_from_pretrained_cnn(cnn_model, feature_layer_name='dense_2', n_estimators=100)
    
    # Train hybrid model
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    hybrid.train_hybrid(X_train, y_train, X_val, y_val)
    
    # Evaluate
    hybrid_results = hybrid.evaluate(X_test, y_test, data['label_encoder'])
    
    # Save hybrid model with dataset-specific paths
    hybrid.save_models(
        os.path.join(output_dir, 'cnn_feature_extractor.h5'),
        os.path.join(output_dir, 'hybrid_rf.pkl')
    )
    
    # Summary
    print("\n" + "="*60)
    print("SCENARIO C SUMMARY")
    print("="*60)
    print(f"Hybrid Model - Accuracy: {hybrid_results['accuracy']*100:.2f}%, Latency: {hybrid_results['avg_latency_ms']:.2f} ms")
    print("="*60)
    
    return hybrid, hybrid_results


def main():
    """
    Main execution function
    Run all three scenarios sequentially with dataset-specific preprocessing
    """
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Train Hybrid IDS on CICIDS2017 or NSL-KDD dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on CICIDS2017
  python train_pipeline.py --dataset cicids --data-path data/cicids2017/
  
  # Train on NSL-KDD
  python train_pipeline.py --dataset nslkdd --train-path data/nslkdd/KDDTrain+.txt --test-path data/nslkdd/KDDTest+.txt
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['cicids', 'nslkdd'],
        help='Dataset to use: cicids (CICIDS2017) or nslkdd (NSL-KDD)'
    )
    
    # CICIDS2017 arguments
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/cicids2017/',
        help='Path to folder containing CICIDS2017 CSV files (default: data/cicids2017/)'
    )
    
    # NSL-KDD arguments
    parser.add_argument(
        '--train-path',
        type=str,
        default='data/nslkdd/KDDTrain+.txt',
        help='Path to NSL-KDD training file (default: data/nslkdd/KDDTrain+.txt)'
    )
    
    parser.add_argument(
        '--test-path',
        type=str,
        default='data/nslkdd/KDDTest+.txt',
        help='Path to NSL-KDD test file (default: data/nslkdd/KDDTest+.txt)'
    )
    
    args = parser.parse_args()
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*15 + "HYBRID AI-BASED INTRUSION DETECTION SYSTEM" + " "*20 + "║")
    print("║" + " "*25 + "Training Pipeline" + " "*36 + "║")
    print("╚" + "="*78 + "╝")
    
    print(f"\nDataset: {args.dataset.upper()}")
    print("="*80)
    
    # Phase 1: Data Preprocessing (dataset-specific)
    if args.dataset == 'cicids':
        # Import CICIDS2017 preprocessing
        from src.preprocess_cicids import process_cicids
        
        print(f"\nUsing CICIDS2017 preprocessing pipeline")
        print(f"Data path: {args.data_path}")
        
        # Check if path exists
        if not os.path.exists(args.data_path):
            print(f"\n✗ ERROR: Data path not found: {args.data_path}")
            print("\nPlease ensure your CICIDS2017 CSV files are placed in the correct location.")
            print("Expected structure:")
            print("  data/cicids2017/")
            print("    ├── Monday-WorkingHours.pcap_ISCX.csv")
            print("    ├── Tuesday-WorkingHours.pcap_ISCX.csv")
            print("    └── ... (other CSV files)")
            return
        
        # Process CICIDS2017 data
        output_dir = 'models/cicids'
        data = process_cicids(args.data_path, output_dir=output_dir)
        
    elif args.dataset == 'nslkdd':
        # Import NSL-KDD preprocessing
        from src.preprocess_nslkdd import process_nsl_kdd
        
        print(f"\nUsing NSL-KDD preprocessing pipeline")
        print(f"Train path: {args.train_path}")
        print(f"Test path: {args.test_path}")
        
        # Check if paths exist
        if not os.path.exists(args.train_path):
            print(f"\n✗ ERROR: Training file not found: {args.train_path}")
            print("\nPlease ensure your NSL-KDD files are placed in the correct location.")
            print("Expected structure:")
            print("  data/nslkdd/")
            print("    ├── KDDTrain+.txt")
            print("    └── KDDTest+.txt")
            return
        
        if not os.path.exists(args.test_path):
            print(f"\n✗ ERROR: Test file not found: {args.test_path}")
            return
        
        # Process NSL-KDD data
        output_dir = 'models/nslkdd'
        data = process_nsl_kdd(args.train_path, args.test_path, output_dir=output_dir)
    
    if data is None:
        print("\n✗ Data loading failed. Exiting...")
        return
    
    # Update model paths to use dataset-specific directories
    print(f"\n✓ Models will be saved to: {output_dir}/")
    
    # Phase 2: Model Training
    
    # Scenario A: ML Models
    ml_models, scenario_a_results = run_scenario_a(data, output_dir)
    
    # Scenario B: DL Models
    dl_models, scenario_b_results = run_scenario_b(data, output_dir)
    
    # Scenario C: Hybrid Model
    hybrid_model, scenario_c_results = run_scenario_c(data, dl_models, output_dir)
    
    # Final Summary
    print("\n\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*28 + "FINAL SUMMARY" + " "*37 + "║")
    print("╚" + "="*78 + "╝")
    
    print("\n" + "="*80)
    print(f"MODEL COMPARISON - {data['dataset_name'].upper()}")
    print("="*80)
    
    print("\nScenario A - Machine Learning:")
    for model_name, results in scenario_a_results.items():
        print(f"  {model_name:15} - Accuracy: {results['accuracy']*100:6.2f}%, Latency: {results['avg_latency_ms']:6.2f} ms")
    
    print("\nScenario B - Deep Learning:")
    for model_name, results in scenario_b_results.items():
        print(f"  {model_name:15} - Accuracy: {results['accuracy']*100:6.2f}%, Latency: {results['avg_latency_ms']:6.2f} ms")
    
    print("\nScenario C - Hybrid Model:")
    print(f"  {'Hybrid Model':15} - Accuracy: {scenario_c_results['accuracy']*100:6.2f}%, Latency: {scenario_c_results['avg_latency_ms']:6.2f} ms")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nAll models have been saved to '{output_dir}/' directory.")
    print("\nNext Steps:")
    print(f"  1. Run the dashboard: streamlit run app.py -- --dataset {args.dataset}")
    print("  2. The dashboard will use the Hybrid Model for real-time detection")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
