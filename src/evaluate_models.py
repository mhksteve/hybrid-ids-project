"""
Model Evaluation Script for Hybrid IDS
Generates formal metrics, confusion matrices, ROC curves, and feature importance plots
for academic dissertation/publication

Usage:
    python src/evaluate_models.py --dataset cicids
    python src/evaluate_models.py --dataset nslkdd
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow import keras
import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_data_and_artifacts(dataset_dir):
    """
    Load test data and preprocessing artifacts
    
    Args:
        dataset_dir: Path to model directory (e.g., 'models/cicids')
    
    Returns:
        X_test, y_test, label_encoder
    """
    print(f"\n{'='*80}")
    print(f"LOADING DATA FROM: {dataset_dir}")
    print(f"{'='*80}\n")
    
    try:
        X_test = np.load(f'{dataset_dir}/X_test.npy')
        y_test = np.load(f'{dataset_dir}/y_test.npy')
        label_encoder = joblib.load(f'{dataset_dir}/label_encoder.pkl')
        
        print(f"✓ Test data loaded: {X_test.shape}")
        print(f"✓ Test labels loaded: {y_test.shape}")
        print(f"✓ Classes: {label_encoder.classes_}")
        print(f"✓ Number of classes: {len(label_encoder.classes_)}")
        
        return X_test, y_test, label_encoder
    
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        sys.exit(1)


def load_model(model_name, dataset_dir):
    """
    Load a specific model
    
    Args:
        model_name: Name of the model
        dataset_dir: Path to model directory
    
    Returns:
        Loaded model(s) or None if failed
    """
    try:
        if model_name == 'Random Forest':
            return joblib.load(f'{dataset_dir}/random_forest.pkl')
        
        elif model_name == 'XGBoost':
            return joblib.load(f'{dataset_dir}/xgboost.pkl')
        
        elif model_name == 'SVM':
            return joblib.load(f'{dataset_dir}/svm.pkl')
        
        elif model_name == '1D-CNN':
            return keras.models.load_model(f'{dataset_dir}/1d_cnn.h5')
        
        elif model_name == 'LSTM':
            return keras.models.load_model(f'{dataset_dir}/lstm.h5')
        
        elif model_name == 'Hybrid Model':
            cnn_extractor = keras.models.load_model(f'{dataset_dir}/cnn_feature_extractor.h5')
            rf_classifier = joblib.load(f'{dataset_dir}/hybrid_rf.pkl')
            return (cnn_extractor, rf_classifier)
        
        else:
            return None
    
    except Exception as e:
        print(f"  ✗ Failed to load {model_name}: {e}")
        return None


def predict_model(model, X_test, model_name):
    """
    Make predictions with proper input reshaping for different model types
    
    Args:
        model: Loaded model
        X_test: Test features
        model_name: Name of the model
    
    Returns:
        predictions, probabilities (or None), inference_time
    """
    start_time = time.time()
    
    # Deep Learning models need 3D input
    if model_name in ['1D-CNN', 'LSTM']:
        X_test_3d = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        y_pred_probs = model.predict(X_test_3d, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Hybrid Model
    elif model_name == 'Hybrid Model':
        cnn_extractor, rf_classifier = model
        X_test_3d = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Extract deep features
        deep_features = cnn_extractor.predict(X_test_3d, verbose=0)
        
        # Classify with RF
        y_pred = rf_classifier.predict(deep_features)
        y_pred_probs = rf_classifier.predict_proba(deep_features)
    
    # Machine Learning models
    else:
        y_pred = model.predict(X_test)
        
        # SVM may not have predict_proba if probability=False
        try:
            y_pred_probs = model.predict_proba(X_test)
        except AttributeError:
            print(f"  ⚠ {model_name} does not support predict_proba (skipping ROC curve)")
            y_pred_probs = None
    
    inference_time = time.time() - start_time
    
    return y_pred, y_pred_probs, inference_time


def evaluate_all_models(X_test, y_test, label_encoder, dataset_dir):
    """
    Evaluate all 6 models and collect metrics
    
    Args:
        X_test: Test features
        y_test: Test labels
        label_encoder: Label encoder for class names
        dataset_dir: Path to model directory
    
    Returns:
        Dictionary of results for each model
    """
    print(f"\n{'='*80}")
    print("EVALUATING ALL MODELS")
    print(f"{'='*80}\n")
    
    model_names = ['Random Forest', 'XGBoost', 'SVM', '1D-CNN', 'LSTM', 'Hybrid Model']
    results = {}
    
    for model_name in model_names:
        print(f"[{model_name}]")
        
        # Load model
        model = load_model(model_name, dataset_dir)
        if model is None:
            print(f"  ✗ Skipping {model_name} (not found)\n")
            continue
        
        print(f"  ✓ Model loaded")
        
        # Make predictions
        y_pred, y_pred_probs, inference_time = predict_model(model, X_test, model_name)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        avg_latency = (inference_time / len(X_test)) * 1000  # ms per sample
        
        print(f"  ✓ Accuracy: {accuracy*100:.2f}%")
        print(f"  ✓ Precision: {precision:.4f}")
        print(f"  ✓ Recall: {recall:.4f}")
        print(f"  ✓ F1-Score: {f1:.4f}")
        print(f"  ✓ Avg Latency: {avg_latency:.2f} ms\n")
        
        # Store results
        results[model_name] = {
            'model': model,
            'y_pred': y_pred,
            'y_pred_probs': y_pred_probs,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'inference_time': inference_time,
            'avg_latency': avg_latency
        }
    
    return results


def save_metrics_csv(results, output_dir):
    """
    Save evaluation metrics to CSV file
    
    Args:
        results: Dictionary of model results
        output_dir: Directory to save the CSV
    """
    print(f"\n{'='*80}")
    print("SAVING METRICS TO CSV")
    print(f"{'='*80}\n")
    
    metrics_data = []
    
    for model_name, result in results.items():
        metrics_data.append({
            'Model': model_name,
            'Accuracy': f"{result['accuracy']*100:.2f}%",
            'Precision (Macro)': f"{result['precision']:.4f}",
            'Recall (Macro)': f"{result['recall']:.4f}",
            'F1-Score (Macro)': f"{result['f1_score']:.4f}",
            'Avg Latency (ms)': f"{result['avg_latency']:.2f}",
            'Total Inference Time (s)': f"{result['inference_time']:.2f}"
        })
    
    df = pd.DataFrame(metrics_data)
    csv_path = f'{output_dir}/evaluation_metrics.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"✓ Metrics saved to: {csv_path}\n")
    print(df.to_string(index=False))

def plot_standalone_model_comparison(results, output_dir, dataset_name):
    """
    Generate a grouped bar chart for standalone model comparison.
    This is intended for dissertation reporting, especially Chapter 4.2.
    It compares Accuracy (%) and Macro F1-score (%) for:
    Random Forest, XGBoost, SVM, 1D-CNN, and LSTM.
    """
    print(f"\n{'='*80}")
    print("GENERATING STANDALONE MODEL COMPARISON BAR CHART")
    print(f"{'='*80}\n")

    # ADDED: keep the order consistent with the dissertation text
    standalone_models = ['Random Forest', 'XGBoost', 'SVM', '1D-CNN', 'LSTM']

    # ADDED: only keep models that were successfully evaluated
    available_models = [m for m in standalone_models if m in results]

    if not available_models:
        print(" ⚠ No standalone models available for comparison plot\n")
        return

    # ADDED: convert values to percentages for easier visual comparison
    accuracy_values = [results[m]['accuracy'] * 100 for m in available_models]
    f1_values = [results[m]['f1_score'] * 100 for m in available_models]

    x = np.arange(len(available_models))
    width = 0.35

    plt.figure(figsize=(10, 6))

    # ADDED: first bar group = accuracy
    bars1 = plt.bar(
        x - width / 2,
        accuracy_values,
        width,
        label='Accuracy (%)',
        color='steelblue',
        edgecolor='black'
    )

    # ADDED: second bar group = macro F1
    bars2 = plt.bar(
        x + width / 2,
        f1_values,
        width,
        label='Macro F1-score (%)',
        color='darkorange',
        edgecolor='black'
    )

    plt.xticks(x, available_models, rotation=15)
    plt.ylabel('Score (%)', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.title(
        f'Standalone Model Comparison - {dataset_name.upper()}',
        fontsize=14,
        fontweight='bold'
    )
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # ADDED: label the bars for easier dissertation use
    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontsize=9
        )

    # ADDED: save with dataset-specific filename
    filename = f'{output_dir}/standalone_model_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f" ✓ Saved: {filename}\n")

def save_classification_reports(results, y_test, label_encoder, output_dir):
    """
    Save detailed per-class classification reports to text files

    Args:
        results: Dictionary of model results
        y_test: True labels
        label_encoder: Label encoder for class names
        output_dir: Directory to save reports
    """
    print(f"\n{'=' * 80}")
    print("GENERATING CLASSIFICATION REPORTS")
    print(f"{'=' * 80}\n")

    class_names = label_encoder.classes_

    for model_name, result in results.items():
        print(f"  Saving {model_name}...")

        # Generate classification report
        report = classification_report(
            y_test,
            result['y_pred'],
            target_names=class_names,
            digits=4,
            zero_division=0
        )

        # Save to text file
        filename = f'{output_dir}/{model_name.lower().replace(" ", "_")}_classification_report.txt'

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Classification Report - {model_name}\n")
            f.write("=" * 80 + "\n\n")
            f.write(report)
            f.write("\n\n")
            f.write(f"Overall Metrics:\n")
            f.write(f"  Accuracy: {result['accuracy'] * 100:.2f}%\n")
            f.write(f"  Inference Time: {result['inference_time']:.2f}s\n")
            f.write(f"  Avg Latency: {result['avg_latency']:.2f}ms\n")

        print(f"    ✓ Saved: {filename}")

    print()

def plot_confusion_matrices(results, y_test, label_encoder, output_dir):
    """
    Generate and save confusion matrix heatmaps for all models
    
    Args:
        results: Dictionary of model results
        y_test: True labels
        label_encoder: Label encoder for class names
        output_dir: Directory to save plots
    """
    print(f"\n{'='*80}")
    print("GENERATING CONFUSION MATRICES")
    print(f"{'='*80}\n")
    
    class_names = label_encoder.classes_
    
    for model_name, result in results.items():
        print(f"  Plotting {model_name}...")
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, result['y_pred'])
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'}
        )
        
        plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save
        filename = f'{output_dir}/{model_name.lower().replace(" ", "_")}_confusion_matrix.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ Saved: {filename}")
    
    print()


def plot_combined_roc_curve(results, y_test, label_encoder, output_dir):
    """
    Generate combined macro-average ROC curve for all models
    
    Args:
        results: Dictionary of model results
        y_test: True labels
        label_encoder: Label encoder for class names
        output_dir: Directory to save plot
    """
    print(f"\n{'='*80}")
    print("GENERATING COMBINED ROC CURVE")
    print(f"{'='*80}\n")
    
    # Binarize the labels for multi-class ROC
    n_classes = len(label_encoder.classes_)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Colors for different models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for idx, (model_name, result) in enumerate(results.items()):
        # Skip if no probabilities (e.g., SVM without probability=True)
        if result['y_pred_probs'] is None:
            print(f"  ⚠ Skipping {model_name} (no probabilities available)")
            continue
        
        y_pred_probs = result['y_pred_probs']
        
        # Compute ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute macro-average ROC curve
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        
        mean_tpr /= n_classes
        
        # Calculate macro AUC
        macro_auc = auc(all_fpr, mean_tpr)
        
        # Plot
        plt.plot(
            all_fpr,
            mean_tpr,
            color=colors[idx % len(colors)],
            lw=2,
            label=f'{model_name} (AUC = {macro_auc:.3f})'
        )
        
        print(f"  ✓ {model_name}: Macro AUC = {macro_auc:.3f}")
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.500)')
    
    # Formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Macro-Average ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    filename = f'{output_dir}/combined_roc_curve.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n  ✓ Saved: {filename}\n")


def plot_feature_importance(results, output_dir, top_n=20):
    """
    Plot feature importance for Random Forest and XGBoost
    
    Args:
        results: Dictionary of model results
        output_dir: Directory to save plots
        top_n: Number of top features to display
    """
    print(f"\n{'='*80}")
    print("GENERATING FEATURE IMPORTANCE PLOTS")
    print(f"{'='*80}\n")
    
    for model_name in ['Random Forest', 'XGBoost']:
        if model_name not in results:
            print(f"  ⚠ Skipping {model_name} (not found)")
            continue
        
        print(f"  Plotting {model_name}...")
        
        model = results[model_name]['model']
        
        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Get top N features
            indices = np.argsort(importances)[::-1][:top_n]
            top_importances = importances[indices]
            
            # Feature names (use indices if names not available)
            feature_names = [f'Feature {i}' for i in indices]
            
            # Create figure
            plt.figure(figsize=(10, 8))
            
            # Plot horizontal bar chart
            y_pos = np.arange(len(top_importances))
            plt.barh(y_pos, top_importances, align='center', color='steelblue', edgecolor='black')
            plt.yticks(y_pos, feature_names)
            plt.xlabel('Importance Score', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.title(f'Top {top_n} Feature Importances - {model_name}', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()  #
            plt.tight_layout()
            
            # Save
            filename = f'{output_dir}/{model_name.lower().replace(" ", "_")}_feature_importance.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    ✓ Saved: {filename}")
        
        else:
            print(f"    ⚠ {model_name} does not have feature_importances_ attribute")
    
    print()


def main():
    """
    Main evaluation pipeline
    """
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Evaluate trained IDS models and generate publication-quality plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python src/evaluate_models.py --dataset cicids
    python src/evaluate_models.py --dataset nslkdd
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['cicids', 'nslkdd'],
        help='Dataset to evaluate (cicids or nslkdd)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    dataset_dir = f'models/{args.dataset}'
    output_dir = f'results/{args.dataset}'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("HYBRID IDS - MODEL EVALUATION")
    print("="*80)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Output Directory: {output_dir}")
    print("="*80)
    
    # Load data
    X_test, y_test, label_encoder = load_data_and_artifacts(dataset_dir)
    
    # Evaluate all models
    results = evaluate_all_models(X_test, y_test, label_encoder, dataset_dir)
    
    if not results:
        print("\n✗ No models were successfully evaluated. Exiting.")
        sys.exit(1)
    
    # Generate outputs
    save_metrics_csv(results, output_dir)
    plot_standalone_model_comparison(results, output_dir, args.dataset)
    save_classification_reports(results, y_test, label_encoder, output_dir)
    plot_confusion_matrices(results, y_test, label_encoder, output_dir)
    plot_combined_roc_curve(results, y_test, label_encoder, output_dir)
    plot_feature_importance(results, output_dir, top_n=20)
    
    # Final summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}/")
    print("\nGenerated files:")
    print(f"  • evaluation_metrics.csv")
    print(f" • standalone_model_comparison.png")
    print(f"  • {len(results)}x confusion matrix plots")
    print(f"  • combined_roc_curve.png")
    print(f"  • 2x feature importance plots (RF, XGBoost)")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
