"""
CICIDS2017 Dataset Preprocessing Module
Handles loading, cleaning, encoding, scaling, and balancing for CICIDS2017 data
"""

import numpy as np
import pandas as pd
import glob
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import joblib


def process_cicids(folder_path, output_dir='models/cicids', test_size=0.30, val_split=0.5, random_state=42):
    """
    Complete preprocessing pipeline for CICIDS2017 dataset
    
    Args:
        folder_path: Path to folder containing CICIDS2017 CSV files
        output_dir: Directory to save preprocessing artifacts
        test_size: Proportion for test+validation (default: 0.30 for 70% train)
        val_split: Split of test_size between validation and test (0.5 = equal split)
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary containing processed train, validation, and test sets
    """
    
    print("="*80)
    print("CICIDS2017 DATASET PREPROCESSING")
    print("="*80)
    
    # Step 1: Load all CSV files from the folder
    print(f"\n[1/8] Loading CSV files from: {folder_path}")
    
    try:
        # Find all CSV files in the folder
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        
        if not csv_files:
            print(f"✗ ERROR: No CSV files found in {folder_path}")
            return None
        
        print(f"Found {len(csv_files)} CSV file(s):")
        for csv_file in csv_files:
            print(f"  - {os.path.basename(csv_file)}")
        
        # Load and concatenate all CSV files
        print(f"\nLoading and concatenating files...")
        df_list = []
        
        for csv_file in csv_files:
            try:
                temp_df = pd.read_csv(csv_file)
                df_list.append(temp_df)
                print(f"  ✓ Loaded {os.path.basename(csv_file)}: {temp_df.shape[0]:,} rows")
            except Exception as e:
                print(f"  ✗ Failed to load {os.path.basename(csv_file)}: {e}")
                continue
        
        if not df_list:
            print(f"✗ ERROR: No files could be loaded successfully")
            return None
        
        # Concatenate all dataframes
        df = pd.concat(df_list, ignore_index=True)
        print(f"\n✓ Successfully concatenated {len(df_list)} file(s)")
        print(f"Combined dataset shape: {df.shape} (Rows: {df.shape[0]:,}, Columns: {df.shape[1]})")
        
    except Exception as e:
        print(f"✗ ERROR loading data: {e}")
        return None
    
    # Step 2: Fix column names (CICIDS has leading spaces)
    print(f"\n[2/8] Cleaning column names...")
    print(f"Original columns (first 5): {df.columns.tolist()[:5]}")
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    print(f"Cleaned columns (first 5): {df.columns.tolist()[:5]}")
    
    # Visual Check
    print(f"\n[3/8] Dataset Overview:")
    print(f"Shape: {df.shape} (Rows: {df.shape[0]:,}, Columns: {df.shape[1]})")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nColumn names:")
    print(df.columns.tolist())
    
    # Step 3: Identify target column
    print(f"\n[4/8] Identifying target column...")
    target_column = 'Label'
    
    if target_column not in df.columns:
        # Try alternative names
        possible_targets = ['Label', 'label', ' Label', 'Attack Label', 'attack_label']
        for alt_target in possible_targets:
            if alt_target in df.columns:
                target_column = alt_target
                print(f"⚠ Target column found as: '{target_column}'")
                break
        else:
            print(f"✗ ERROR: Target column 'Label' not found in dataset")
            print(f"Available columns: {df.columns.tolist()}")
            return None
    else:
        print(f"✓ Target column 'Label' found")
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    print(f"\nTarget distribution:")
    print(y.value_counts())
    
    # Step 4: Cleaning
    print(f"\n[5/8] Cleaning data...")
    print(f"Initial NaN count: {X.isnull().sum().sum()}")
    print(f"Initial Inf count: {np.isinf(X.select_dtypes(include=[np.number])).sum().sum()}")
    
    # Get numeric columns only
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    
    # Replace infinite values with column max
    for col in numeric_columns:
        if np.isinf(X[col]).any():
            max_val = X[col][~np.isinf(X[col])].max()
            if pd.isna(max_val):  # If all values are inf
                max_val = 0
            X[col].replace([np.inf, -np.inf], max_val, inplace=True)
    
    # Fill NaN with median
    for col in numeric_columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            if pd.isna(median_val):  # If all values are NaN
                median_val = 0
            X[col].fillna(median_val, inplace=True)
    
    print(f"✓ After cleaning - NaN: {X.isnull().sum().sum()}, Inf: {np.isinf(X).sum().sum()}")
    
    # Step 5: Encoding
    print(f"\n[6/8] Encoding target labels...")
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Save label encoder
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(label_encoder, os.path.join(output_dir, 'label_encoder.pkl'))
    print(f"✓ Label Encoder saved to {output_dir}/label_encoder.pkl")
    print(f"Encoded classes: {label_encoder.classes_}")
    print(f"Encoded distribution: {np.bincount(y_encoded)}")
    
    # Step 6: Scaling
    print(f"\n[7/8] Scaling features to [0, 1]...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    print(f"✓ MinMaxScaler saved to {output_dir}/scaler.pkl")
    print(f"Feature range: [{X_scaled.min():.4f}, {X_scaled.max():.4f}]")
    
    # Step 7: Splitting
    print(f"\n[8/8] Splitting data (70% Train, 15% Val, 15% Test)...")
    
    # First split: 70% train, 30% temp (which will be split into val and test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    # Second split: Split temp into validation and test (50-50 of the 30%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_split, random_state=random_state, stratify=y_temp
    )
    
    print(f"✓ Train set: {X_train.shape[0]:,} samples")
    print(f"✓ Validation set: {X_val.shape[0]:,} samples")
    print(f"✓ Test set: {X_test.shape[0]:,} samples")

    # Step 8: Balancing with SMOTE (Fix for Windows joblib/loky crash & memory issues)
    print(f"\n[9/9] Applying balanced sampling strategy to training set...")
    print(f"Before balancing - Train set distribution: {np.bincount(y_train)}")

    # Fix 1: Windows joblib/loky workaround
    os.environ['LOKY_MAX_CPU_COUNT'] = '4'

    try:
        # Fix 2: Undersample massive majority classes first
        undersample_strategy = {
            0: 150000,  # Cap largest majority class
            4: 100000,  # Cap second largest
            10: 80000,  # Cap third largest
            2: 80000  # Cap fourth largest
        }

        # Fix 3: Oversample minority classes with custom targets
        smote_strategy = {
            1: 10000,  # Boost rare class
            3: 8000,  # Boost rare class
            5: 6000,  # Boost rare class
            6: 5000,  # Boost rare class
            7: 6000,  # Boost rare class
            8: 1000,  # Boost rare class
            9: 1000,  # Boost rare class
            11: 5000,  # Boost very rare class
            12: 2000,  # Boost very rare class (Heartbleed has only 8 samples)
            13: 1000,
            14: 2000
        }

        # Fix 4: Set k_neighbors=3 to handle Heartbleed (8 samples) safely
        # Pipeline: First undersample, then oversample
        sampling_pipeline = ImbPipeline([
            ('undersample', RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=random_state)),
            ('oversample', SMOTE(sampling_strategy=smote_strategy, k_neighbors=3, random_state=random_state))
        ])

        X_train_balanced, y_train_balanced = sampling_pipeline.fit_resample(X_train, y_train)

        print(f"After balancing - Train set distribution: {np.bincount(y_train_balanced)}")
        print(f"✓ Balanced sampling applied successfully!")
        print(f"New train set size: {X_train_balanced.shape[0]:,} samples")

    except Exception as e:
        print(f"⚠ Balanced sampling failed: {e}")
        print("Proceeding with imbalanced data...")
        X_train_balanced = X_train
        y_train_balanced = y_train
    
    # Prepare return dictionary
    processed_data = {
        'X_train': X_train_balanced,
        'y_train': y_train_balanced,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': X.columns.tolist(),
        'n_features': X_scaled.shape[1],
        'n_classes': len(label_encoder.classes_),
        'label_encoder': label_encoder,
        'scaler': scaler,
        'dataset_name': 'CICIDS2017'
    }
    
    # Save test data for dashboard simulation
    print(f"\nSaving test data for dashboard simulation...")
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    print(f"✓ Test data saved to {output_dir}/")
    
    print("\n" + "="*80)
    print("CICIDS2017 PREPROCESSING COMPLETE")
    print("="*80)
    print(f"Features: {processed_data['n_features']}")
    print(f"Classes: {processed_data['n_classes']}")
    print(f"Ready for model training!")
    print("="*80 + "\n")
    
    return processed_data
