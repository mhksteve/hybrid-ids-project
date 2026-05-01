"""
Deep Learning Models Module
Contains 1D-CNN and LSTM implementations using TensorFlow/Keras
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time


class DLModels:
    """
    Container class for Standalone Deep Learning models
    Scenario B: 1D-CNN and LSTM
    """
    
    def __init__(self):
        self.models = {}
        self.training_histories = {}
        self.metrics = {}
    
    def build_1d_cnn(self, input_shape, n_classes, filters=[64, 128, 256], kernel_size=3, dropout_rate=0.3):
        """
        Build 1D-CNN architecture for feature extraction and classification
        
        Args:
            input_shape: Shape of input (n_features, 1)
            n_classes: Number of output classes
            filters: List of filter sizes for Conv1D layers
            kernel_size: Size of convolution kernel
            dropout_rate: Dropout rate
        
        Returns:
            Keras model
        """
        print("\n[1D-CNN] Building model...")
        
        model = models.Sequential(name='1D_CNN')
        
        # Input layer
        model.add(layers.Input(shape=input_shape))
        
        # Convolutional blocks
        for i, num_filters in enumerate(filters):
            model.add(layers.Conv1D(
                filters=num_filters,
                kernel_size=kernel_size,
                activation='relu',
                padding='same',
                name=f'conv1d_{i+1}'
            ))
            model.add(layers.MaxPooling1D(pool_size=2, name=f'maxpool_{i+1}'))
            model.add(layers.Dropout(dropout_rate, name=f'dropout_{i+1}'))
        
        # Flatten and dense layers
        model.add(layers.Flatten(name='flatten'))
        model.add(layers.Dense(128, activation='relu', name='dense_1'))
        model.add(layers.Dropout(dropout_rate, name='dropout_final'))
        model.add(layers.Dense(64, activation='relu', name='dense_2'))
        
        # Output layer
        model.add(layers.Dense(n_classes, activation='softmax', name='output'))
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.models['1D-CNN'] = model
        
        print("✓ 1D-CNN architecture:")
        model.summary()
        
        return model
    
    def build_lstm(self, input_shape, n_classes, lstm_units=[128, 64], dropout_rate=0.3):
        """
        Build LSTM architecture for sequential pattern recognition
        
        Args:
            input_shape: Shape of input (n_features, 1)
            n_classes: Number of output classes
            lstm_units: List of LSTM layer units
            dropout_rate: Dropout rate
        
        Returns:
            Keras model
        """
        print("\n[LSTM] Building model...")
        
        model = models.Sequential(name='LSTM')
        
        # Input layer
        model.add(layers.Input(shape=input_shape))
        
        # LSTM layers
        for i, units in enumerate(lstm_units[:-1]):
            model.add(layers.LSTM(
                units=units,
                return_sequences=True,  # Return sequences for stacked LSTM
                dropout=dropout_rate,
                name=f'lstm_{i+1}'
            ))
        
        # Final LSTM layer (no return_sequences)
        model.add(layers.LSTM(
            units=lstm_units[-1],
            return_sequences=False,
            dropout=dropout_rate,
            name=f'lstm_{len(lstm_units)}'
        ))
        
        # Dense layers
        model.add(layers.Dense(64, activation='relu', name='dense_1'))
        model.add(layers.Dropout(dropout_rate, name='dropout'))
        
        # Output layer
        model.add(layers.Dense(n_classes, activation='softmax', name='output'))
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.models['LSTM'] = model
        
        print("✓ LSTM architecture:")
        model.summary()
        
        return model
    
    def train_model(self, model_name, X_train, y_train, X_val, y_val, epochs=50, batch_size=128, patience=10):
        """
        Train a deep learning model with callbacks
        
        Args:
            model_name: Name of the model ('1D-CNN' or 'LSTM')
            X_train: Training features (must be reshaped to 3D)
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Maximum number of epochs
            batch_size: Batch size
            patience: Early stopping patience
        
        Returns:
            Training history
        """
        if model_name not in self.models:
            print(f"✗ Model '{model_name}' not found. Please build it first.")
            return None
        
        model = self.models[model_name]
        
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        
        # Ensure data is 3D (samples, timesteps, features)
        if len(X_train.shape) == 2:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            print(f"Reshaped input: {X_train.shape}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                verbose=1,
                min_lr=1e-7
            )
        ]
        
        # Train
        print(f"\nTraining on {X_train.shape[0]:,} samples for up to {epochs} epochs...")
        start_time = time.time()
        
        try:
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            training_time = time.time() - start_time
            
            print(f"\n✓ Training completed in {training_time:.2f} seconds")
            
            # Store history
            self.training_histories[model_name] = history
            
            # Evaluate on training and validation sets
            train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
            val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
            
            print(f"\nFinal Metrics:")
            print(f"Training   - Loss: {train_loss:.4f}, Accuracy: {train_acc*100:.2f}%")
            print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc*100:.2f}%")
            
            self.metrics[model_name] = {
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'training_time': training_time,
                'epochs_trained': len(history.history['loss'])
            }
            
            return history
            
        except Exception as e:
            print(f"✗ Training failed: {e}")
            return None
    
    def evaluate_model(self, model_name, X_test, y_test, label_encoder=None):
        """
        Comprehensive evaluation of a deep learning model
        
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
        
        # Ensure data is 3D
        if len(X_test.shape) == 2:
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Evaluate
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        # Prediction with timing
        start_time = time.time()
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        inference_time = time.time() - start_time
        
        print(f"\nTest Accuracy: {test_acc*100:.2f}%")
        print(f"Test Loss: {test_loss:.4f}")
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
        
        evaluation_results = {
            'accuracy': test_acc,
            'loss': test_loss,
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
            filepath: Path to save the model (.h5 format)
        """
        if model_name not in self.models:
            print(f"✗ Model '{model_name}' not found.")
            return False
        
        try:
            self.models[model_name].save(filepath)
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
            model = keras.models.load_model(filepath)
            self.models[model_name] = model
            print(f"✓ {model_name} loaded from {filepath}")
            return model
        except Exception as e:
            print(f"✗ Failed to load {model_name}: {e}")
            return None
