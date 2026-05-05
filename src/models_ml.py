"""
Machine Learning Models Module
Random Forest, XGBoost, and SVM implementations
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import time
import numpy as np


class MLModels:
    """
    Container class for Standalone Machine Learning models
    Scenario A: Random Forest, XGBoost, and SVM
    """
    
    def __init__(self):
        self.models = {}
        self.training_times = {}
        self.predictions = {}
        self.metrics = {}
    
    def build_random_forest(self, n_estimators=100, max_depth=30, criterion='gini', random_state=42):
        """
        Build Random Forest classifier
        
        Args:
            n_estimators: Number of trees (default: 100)
            max_depth: Maximum tree depth
            criterion: Split criterion (default: 'gini')
            random_state: Random seed
        
        Returns:
            RandomForestClassifier instance
        """
        print("\n[Random Forest] Building model...")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            criterion=criterion,
            random_state=random_state,
            n_jobs=-1,  # Use all CPU cores
            verbose=0
        )
        self.models['RandomForest'] = model
        print(f"✓ Random Forest created with {n_estimators} trees")
        return model
    
    def build_xgboost(self, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42):
        """
        Build XGBoost classifier
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Step size shrinkage
            random_state: Random seed
        
        Returns:
            XGBClassifier instance
        """
        print("\n[XGBoost] Building model...")
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        self.models['XGBoost'] = model
        print(f"✓ XGBoost created with {n_estimators} estimators")
        return model
    
    def build_svm(self, kernel='rbf', C=1.0, random_state=42):
        """
        Build SVM classifier
        
        Args:
            kernel: Kernel type (default: 'rbf')
            C: Regularization parameter
            random_state: Random seed
        
        Returns:
            SVC instance
        """
        print("\n[SVM] Building model...")
        print("⚠ Note: SVM training may be slow on large datasets")
        model = SVC(
            kernel=kernel,
            C=C,
            probability=True,
            random_state=random_state,
            verbose=False
        )
        self.models['SVM'] = model
        print(f"✓ SVM created with {kernel} kernel")
        return model
    
    def train_model(self, model_name, X_train, y_train, X_val=None, y_val=None, use_subset_for_svm=True, svm_subset_size=10000):
        """
        Args:
            model_name: Name of the model ('RandomForest', 'XGBoost', 'SVM')
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            use_subset_for_svm: If True, use subset for SVM training
            svm_subset_size: Size of subset for SVM
        
        Returns:
            Trained model
        """
        if model_name not in self.models:
            print(f"✗ Model '{model_name}' not found. Please build it first.")
            return None
        
        model = self.models[model_name]
        
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        
        # Use subset for SVM
        if model_name == 'SVM' and use_subset_for_svm and X_train.shape[0] > svm_subset_size:
            print(f"⚠ Using subset of {svm_subset_size:,} samples for SVM training (full dataset: {X_train.shape[0]:,})")
            indices = np.random.choice(X_train.shape[0], svm_subset_size, replace=False)
            X_train_subset = X_train[indices]
            y_train_subset = y_train[indices]
        else:
            X_train_subset = X_train
            y_train_subset = y_train
        
        # Model training
        start_time = time.time()
        print(f"Training on {X_train_subset.shape[0]:,} samples...")
        
        try:
            model.fit(X_train_subset, y_train_subset)
            training_time = time.time() - start_time
            self.training_times[model_name] = training_time
            
            print(f"✓ Training completed in {training_time:.2f} seconds")
            
            # Evaluate on training set
            train_pred = model.predict(X_train_subset)
            train_acc = accuracy_score(y_train_subset, train_pred)
            print(f"Training Accuracy: {train_acc*100:.2f}%")
            
            # Evaluate on validation set
            if X_val is not None and y_val is not None:
                val_pred = model.predict(X_val)
                val_acc = accuracy_score(y_val, val_pred)
                print(f"Validation Accuracy: {val_acc*100:.2f}%")
                
                self.metrics[model_name] = {
                    'train_accuracy': train_acc,
                    'val_accuracy': val_acc,
                    'training_time': training_time
                }
            else:
                self.metrics[model_name] = {
                    'train_accuracy': train_acc,
                    'training_time': training_time
                }
            
            return model
            
        except Exception as e:
            print(f"✗ Training failed: {e}")
            return None
    
    def evaluate_model(self, model_name, X_test, y_test, label_encoder=None):
        """
        Comprehensive evaluation of a model
        
        Args:
            model_name: Name of the model
            X_test: Test features
            y_test: Test labels
            label_encoder: LabelEncoder instance for class names
        
        Returns:
            Dictionary of evaluation metrics
        """
        if model_name not in self.models:
            print(f"✗ Model '{model_name}' not found.")
            return None
        
        model = self.models[model_name]
        
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name} on Test Set")
        print(f"{'='*60}")
        
        # Prediction with timing
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nTest Accuracy: {accuracy*100:.2f}%")
        print(f"Inference Time: {inference_time:.4f} seconds ({X_test.shape[0]} samples)")
        print(f"Average Latency: {(inference_time/X_test.shape[0])*1000:.2f} ms/sample")
        
        # Classification report
        if label_encoder is not None:
            target_names = label_encoder.classes_
        else:
            target_names = [f"Class_{i}" for i in range(len(np.unique(y_test)))]
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Store results
        self.predictions[model_name] = y_pred
        
        evaluation_results = {
            'accuracy': accuracy,
            'inference_time': inference_time,
            'avg_latency_ms': (inference_time/X_test.shape[0])*1000,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
        
        return evaluation_results
    
    def save_model(self, model_name, filepath):
        """
        Save a trained model to disk
        
        Args:
            model_name: Name of the model
            filepath: Path to save the model
        """
        if model_name not in self.models:
            print(f"✗ Model '{model_name}' not found.")
            return False
        
        try:
            joblib.dump(self.models[model_name], filepath)
            print(f"✓ {model_name} saved to {filepath}")
            return True
        except Exception as e:
            print(f"✗ Failed to save {model_name}: {e}")
            return False
    
    def load_model(self, model_name, filepath):
        """
        Load a trained model from disk
        
        Args:
            model_name: Name to assign to the loaded model
            filepath: Path to the saved model
        """
        try:
            model = joblib.load(filepath)
            self.models[model_name] = model
            print(f"✓ {model_name} loaded from {filepath}")
            return model
        except Exception as e:
            print(f"✗ Failed to load {model_name}: {e}")
            return None
