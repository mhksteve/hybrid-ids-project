"""
Hybrid Model
Combines Deep Learning feature extraction (1D-CNN) with Machine Learning classification (Random Forest)
Scenario C: The Core Hybrid Solution
"""

import numpy as np
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import time


class HybridModel:
    """
    Hybrid AI Model: CNN Feature Extractor + Random Forest Classifier
    """
    
    def __init__(self):
        self.cnn_extractor = None
        self.rf_classifier = None
        self.feature_layer_name = 'dense_2'  # Layer to extract features from
        self.metrics = {}
    
    def build_from_pretrained_cnn(self, cnn_model, feature_layer_name='dense_2', n_estimators=100):
        """
        Args:
            cnn_model: Trained Keras 1D-CNN model
            feature_layer_name: Name of the layer to extract features from
            n_estimators: Number of trees for Random Forest
        
        Returns:
            Tuple of (feature_extractor, rf_classifier)
        """
        print("\n" + "="*60)
        print("BUILDING HYBRID MODEL")
        print("="*60)
        
        # step 1: Create feature extractor from CNN
        print(f"\n[1/2] Creating CNN Feature Extractor...")
        print(f"Extracting features from layer: '{feature_layer_name}'")
        
        try:
            # output of the specified layer
            feature_layer = cnn_model.get_layer(feature_layer_name)
            
            # create a new model that outputs the features
            self.cnn_extractor = keras.Model(
                inputs=cnn_model.input,
                outputs=feature_layer.output,
                name='CNN_Feature_Extractor'
            )
            
            print(f"✓ Feature Extractor created successfully")
            print(f"Input shape: {self.cnn_extractor.input_shape}")
            print(f"Output shape: {self.cnn_extractor.output_shape}")
            
        except Exception as e:
            print(f"✗ Failed to create feature extractor: {e}")
            print(f"Available layers: {[layer.name for layer in cnn_model.layers]}")
            return None
        
        # step 2: Initialize Random Forest classifier
        print(f"\n[2/2] Initializing Random Forest Classifier...")
        self.rf_classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion='gini',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        print(f"✓ Random Forest initialized with {n_estimators} trees")
        
        print("\n" + "="*60)
        print("HYBRID MODEL READY")
        print("="*60)
        
        return self.cnn_extractor, self.rf_classifier
    
    def extract_deep_features(self, X):
        """
        Args:
            X: Input data (2D or 3D array)
        
        Returns:
            Deep features (2D array)
        """
        # Ensure data is 3D for CNN
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Extract features
        deep_features = self.cnn_extractor.predict(X, verbose=0)
        
        return deep_features
    
    def train_hybrid(self, X_train, y_train, X_val=None, y_val=None):
        """
        Args:
            X_train: Training features (raw)
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        
        Returns:
            Trained RF classifier
        """
        print("\n" + "="*60)
        print("TRAINING HYBRID MODEL")
        print("="*60)
        
        # step 1: Extract deep features
        print(f"\n[1/2] Extracting deep features from training data...")
        print(f"Input shape: {X_train.shape}")
        
        start_time = time.time()
        X_train_deep = self.extract_deep_features(X_train)
        feature_extraction_time = time.time() - start_time
        
        print(f"✓ Deep features extracted in {feature_extraction_time:.2f} seconds")
        print(f"Deep feature shape: {X_train_deep.shape}")
        
        # step 2: Train Random Forest on deep features
        print(f"\n[2/2] Training Random Forest on deep features...")
        print(f"Training on {X_train_deep.shape[0]:,} samples with {X_train_deep.shape[1]} deep features")
        
        start_time = time.time()
        self.rf_classifier.fit(X_train_deep, y_train)
        rf_training_time = time.time() - start_time
        
        print(f"✓ Random Forest trained in {rf_training_time:.2f} seconds")
        
        # Evaluate on training set
        train_pred = self.rf_classifier.predict(X_train_deep)
        train_acc = accuracy_score(y_train, train_pred)
        print(f"Training Accuracy: {train_acc*100:.2f}%")
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            print(f"\nValidating on validation set...")
            X_val_deep = self.extract_deep_features(X_val)
            val_pred = self.rf_classifier.predict(X_val_deep)
            val_acc = accuracy_score(y_val, val_pred)
            print(f"Validation Accuracy: {val_acc*100:.2f}%")
            
            self.metrics['train_accuracy'] = train_acc
            self.metrics['val_accuracy'] = val_acc
        else:
            self.metrics['train_accuracy'] = train_acc
        
        self.metrics['feature_extraction_time'] = feature_extraction_time
        self.metrics['rf_training_time'] = rf_training_time
        self.metrics['total_training_time'] = feature_extraction_time + rf_training_time
        
        print("\n" + "="*60)
        print("HYBRID MODEL TRAINING COMPLETE")
        print("="*60)
        
        return self.rf_classifier
    
    def predict(self, X):
        """
        Args:
            X: Input features (raw)
        
        Returns:
            Predictions
        """
        # Extract deep features
        X_deep = self.extract_deep_features(X)
        
        # Classify using Random Forest
        predictions = self.rf_classifier.predict(X_deep)
        
        return predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the hybrid model
        
        Args:
            X: Input features (raw)
        
        Returns:
            Class probabilities
        """
        # Extract deep features
        X_deep = self.extract_deep_features(X)
        
        # Get probabilities from Random Forest
        probabilities = self.rf_classifier.predict_proba(X_deep)
        
        return probabilities
    
    def evaluate(self, X_test, y_test, label_encoder=None):
        """
        Args:
            X_test: Test features
            y_test: Test labels
            label_encoder: LabelEncoder instance for class names
        
        Returns:
            Dictionary of evaluation metrics
        """
        print("\n" + "="*60)
        print("EVALUATING HYBRID MODEL ON TEST SET")
        print("="*60)
        
        # Extract features and predict with timing
        print(f"\nExtracting features from {X_test.shape[0]:,} test samples...")
        
        start_time = time.time()
        X_test_deep = self.extract_deep_features(X_test)
        feature_extraction_time = time.time() - start_time
        
        print(f"Classifying with Random Forest...")
        start_time = time.time()
        y_pred = self.rf_classifier.predict(X_test_deep)
        classification_time = time.time() - start_time
        
        total_inference_time = feature_extraction_time + classification_time
        
        # metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nTest Accuracy: {accuracy*100:.2f}%")
        print(f"\nTiming Breakdown:")
        print(f"  Feature Extraction: {feature_extraction_time:.4f} seconds")
        print(f"  Classification:     {classification_time:.4f} seconds")
        print(f"  Total Inference:    {total_inference_time:.4f} seconds")
        print(f"  Average Latency:    {(total_inference_time/X_test.shape[0])*1000:.2f} ms/sample")
        
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
        
        evaluation_results = {
            'accuracy': accuracy,
            'feature_extraction_time': feature_extraction_time,
            'classification_time': classification_time,
            'total_inference_time': total_inference_time,
            'avg_latency_ms': (total_inference_time/X_test.shape[0])*1000,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
        
        print("\n" + "="*60)
        
        return evaluation_results
    
    def save_models(self, cnn_path='models/cnn_feature_extractor.h5', rf_path='models/hybrid_rf.pkl'):
        """
        Args:
            cnn_path: Path to save CNN feature extractor
            rf_path: Path to save Random Forest classifier
        """
        print(f"\nSaving hybrid model components...")
        
        try:
            # Save CNN feature extractor
            self.cnn_extractor.save(cnn_path)
            print(f"✓ CNN Feature Extractor saved to {cnn_path}")
            
            # Save Random Forest classifier
            joblib.dump(self.rf_classifier, rf_path)
            print(f"✓ Random Forest Classifier saved to {rf_path}")
            
            print("✓ Hybrid model saved successfully!")
            return True
            
        except Exception as e:
            print(f"✗ Failed to save hybrid model: {e}")
            return False
    
    def load_models(self, cnn_path='models/cnn_feature_extractor.h5', rf_path='models/hybrid_rf.pkl'):
        """
        Args:
            cnn_path: Path to CNN feature extractor
            rf_path: Path to Random Forest classifier
        """
        print(f"\nLoading hybrid model components...")
        
        try:
            # Load CNN feature extractor
            self.cnn_extractor = keras.models.load_model(cnn_path)
            print(f"✓ CNN Feature Extractor loaded from {cnn_path}")
            
            # Load Random Forest classifier
            self.rf_classifier = joblib.load(rf_path)
            print(f"✓ Random Forest Classifier loaded from {rf_path}")
            
            print("✓ Hybrid model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load hybrid model: {e}")
            return False
