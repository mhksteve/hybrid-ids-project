"""
NSL-KDD Dataset Preprocessing Module
"""

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib


def process_nsl_kdd(train_path, test_path, output_dir='models/nslkdd', random_state=42):
    """
    Args:
        train_path: Path to KDDTrain+.txt file
        test_path: Path to KDDTest+.txt file
        output_dir: Directory to save preprocessing artifacts
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary containing processed train, validation, and test sets
    """
    
    print("="*80)
    print("NSL-KDD DATASET PREPROCESSING")
    print("="*80)
    
    # exact column names for NSL-KDD (41 features/target/difficulty)
    kdd_cols = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'target', 'difficulty_level'
    ]
    
    # step 1: Load train data
    print(f"\n[1/9] Loading training data from: {train_path}")
    
    try:
        train_df = pd.read_csv(train_path, names=kdd_cols, header=None)
        print(f"✓ Training data loaded successfully")
        print(f"Shape: {train_df.shape} (Rows: {train_df.shape[0]:,}, Columns: {train_df.shape[1]})")
    except FileNotFoundError:
        print(f"✗ ERROR: Training file not found at {train_path}")
        return None
    except Exception as e:
        print(f"✗ ERROR loading training data: {e}")
        return None
    
    # step 2: Load test data
    print(f"\n[2/9] Loading test data from: {test_path}")
    
    try:
        test_df = pd.read_csv(test_path, names=kdd_cols, header=None)
        print(f"✓ Test data loaded successfully")
        print(f"Shape: {test_df.shape} (Rows: {test_df.shape[0]:,}, Columns: {test_df.shape[1]})")
    except FileNotFoundError:
        print(f"✗ ERROR: Test file not found at {test_path}")
        return None
    except Exception as e:
        print(f"✗ ERROR loading test data: {e}")
        return None
    
    # Visual Check
    print(f"\n[3/9] Dataset Overview:")
    print(f"Training set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    print(f"\nFirst 5 rows of training data:")
    print(train_df.head())
    print(f"\nColumn names:")
    print(train_df.columns.tolist())
    
    # step 3: Drop difficulty_level column
    print(f"\n[4/9] Dropping 'difficulty_level' column...")
    train_df = train_df.drop(columns=['difficulty_level'])
    test_df = test_df.drop(columns=['difficulty_level'])
    print(f"✓ Column dropped")
    print(f"New shapes - Train: {train_df.shape}, Test: {test_df.shape}")

    # step 4: Separate features and target with attack category mapping
    print(f"\n[5/10] Separating features and target...")

    X_train = train_df.drop(columns=['target'])
    y_train_raw = train_df['target']
    X_test = test_df.drop(columns=['target'])
    y_test_raw = test_df['target']

    print(f"✓ Features and target separated")

    # Remove trailing dots from attack names
    y_train_clean = y_train_raw.str.rstrip('.')
    y_test_clean = y_test_raw.str.rstrip('.')

    print(f"\nOriginal unique attacks in training: {y_train_clean.nunique()}")
    print(f"Original unique attacks in test: {y_test_clean.nunique()}")

    # attack category mapping (40+ attacks → 5 categories)
    attack_mapping = {
        # Normal
        'normal': 'Normal',

        # DoS attacks
        'back': 'DoS',
        'land': 'DoS',
        'neptune': 'DoS',
        'pod': 'DoS',
        'smurf': 'DoS',
        'teardrop': 'DoS',
        'mailbomb': 'DoS',
        'processtable': 'DoS',
        'udpstorm': 'DoS',
        'apache2': 'DoS',
        'worm': 'DoS',

        # Probe attacks
        'ipsweep': 'Probe',
        'nmap': 'Probe',
        'portsweep': 'Probe',
        'satan': 'Probe',
        'mscan': 'Probe',
        'saint': 'Probe',

        # R2L (Remote to Local) attacks
        'ftp_write': 'R2L',
        'guess_passwd': 'R2L',
        'imap': 'R2L',
        'multihop': 'R2L',
        'phf': 'R2L',
        'spy': 'R2L',
        'warezclient': 'R2L',
        'warezmaster': 'R2L',
        'sendmail': 'R2L',
        'named': 'R2L',
        'snmpgetattack': 'R2L',
        'snmpguess': 'R2L',
        'xlock': 'R2L',
        'xsnoop': 'R2L',
        'httptunnel': 'R2L',

        # U2R (User to Root) attacks
        'buffer_overflow': 'U2R',
        'loadmodule': 'U2R',
        'perl': 'U2R',
        'rootkit': 'U2R',
        'ps': 'U2R',
        'sqlattack': 'U2R',
        'xterm': 'U2R'
    }

    # apply mapping (fallback to original if not in mapping)
    y_train = y_train_clean.map(attack_mapping).fillna(y_train_clean)
    y_test = y_test_clean.map(attack_mapping).fillna(y_test_clean)

    print(f"\n✓ Attack categories mapped to 5 main classes")
    print(f"Training target distribution (5 categories):")
    print(y_train.value_counts())
    print(f"\nTest target distribution (5 categories):")
    print(y_test.value_counts())
    
    # step 5: One-Hot Encoding for categorical features
    print(f"\n[6/9] Applying One-Hot Encoding to categorical features...")
    
    categorical_cols = ['protocol_type', 'service', 'flag']
    print(f"Categorical columns: {categorical_cols}")
    
    # one-hot encoding - training data
    X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, prefix=categorical_cols)
    print(f"✓ Training data encoded - Shape: {X_train_encoded.shape}")
    
    # one-hot encoding - test data
    X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, prefix=categorical_cols)
    print(f"✓ Test data encoded - Shape: {X_test_encoded.shape}")

    print(f"\nAligning train and test columns...")

    train_cols = set(X_train_encoded.columns)
    test_cols = set(X_test_encoded.columns)
    
    # Find missing columns
    missing_in_test = train_cols - test_cols
    missing_in_train = test_cols - train_cols
    
    # add missing columns with zeros
    for col in missing_in_test:
        X_test_encoded[col] = 0
    
    for col in missing_in_train:
        X_train_encoded[col] = 0
    
    # Ensure columns are in the same order
    X_test_encoded = X_test_encoded[X_train_encoded.columns]
    
    print(f"✓ Columns aligned")
    print(f"Final shapes - Train: {X_train_encoded.shape}, Test: {X_test_encoded.shape}")
    
    # step 6: Label Encoding for target
    print(f"\n[7/9] Encoding target labels...")
    
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # save label encoder
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(label_encoder, os.path.join(output_dir, 'label_encoder.pkl'))
    print(f"✓ Label Encoder saved to {output_dir}/label_encoder.pkl")
    print(f"Encoded classes: {label_encoder.classes_}")
    print(f"Training distribution: {np.bincount(y_train_encoded)}")
    print(f"Test distribution: {np.bincount(y_test_encoded)}")
    
    # step 7: Scaling
    print(f"\n[8/9] Scaling features to [0, 1]...")
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)
    
    # save scaler
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    print(f"✓ MinMaxScaler saved to {output_dir}/scaler.pkl")
    print(f"Training feature range: [{X_train_scaled.min():.4f}, {X_train_scaled.max():.4f}]")
    print(f"Test feature range: [{X_test_scaled.min():.4f}, {X_test_scaled.max():.4f}]")
    
    # step 8: validation set from training data (15% of training)
    print(f"\n[9/9] Creating validation set from training data...")
    
    from sklearn.model_selection import train_test_split
    
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_scaled, y_train_encoded, test_size=0.15, random_state=random_state, stratify=y_train_encoded
    )
    
    print(f"✓ Train set: {X_train_final.shape[0]:,} samples")
    print(f"✓ Validation set: {X_val.shape[0]:,} samples")
    print(f"✓ Test set: {X_test_scaled.shape[0]:,} samples")

    # step 9: Balanced Sampling Pipeline (Undersampling + SMOTE)
    print(f"\n[10/10] Applying balanced sampling pipeline to training set...")
    print(f"Before balancing - Train set distribution:")

    # current distribution
    unique_classes, class_counts = np.unique(y_train_final, return_counts=True)
    for cls, count in zip(unique_classes, class_counts):
        print(f"  {cls}: {count:,}")

    os.environ['LOKY_MAX_CPU_COUNT'] = '4'

    try:
        target_samples = 15000

        # Build undersampling strategy
        under_strategy = {}
        smote_strategy = {}

        print(f"\nTarget samples per class: {target_samples:,}")
        print(f"\nStrategy:")

        for cls, count in zip(unique_classes, class_counts):
            if count > target_samples:
                # Undersample large classes
                under_strategy[cls] = target_samples
                print(f"  {cls}: Undersample {count:,} → {target_samples:,}")
            elif count >= 6 and count < target_samples:
                # Oversample small classes (only if >= 6 samples for SMOTE)
                smote_strategy[cls] = target_samples
                print(f"  {cls}: Oversample {count:,} → {target_samples:,}")
            elif count < 6:
                # Too few samples for SMOTE
                print(f"  {cls}: Leave unchanged ({count} samples - too few for SMOTE)")
            else:
                # Exactly at target
                print(f"  {cls}: Already at target ({count:,} samples)")

        # create balanced sampling pipeline
        if under_strategy or smote_strategy:
            steps = []

            # undersample if needed
            if under_strategy:
                steps.append(('undersample', RandomUnderSampler(
                    sampling_strategy=under_strategy,
                    random_state=random_state
                )))

            # oversampling if needed
            if smote_strategy:
                steps.append(('oversample', SMOTE(
                    sampling_strategy=smote_strategy,
                    k_neighbors=5,
                    random_state=random_state
                )))

            # apply
            if steps:
                sampling_pipeline = ImbPipeline(steps)
                X_train_balanced, y_train_balanced = sampling_pipeline.fit_resample(X_train_final, y_train_final)

                print(f"\n✓ Balanced sampling pipeline applied successfully!")
                print(f"\nAfter balancing - Train set distribution:")
                unique_balanced, counts_balanced = np.unique(y_train_balanced, return_counts=True)
                for cls, count in zip(unique_balanced, counts_balanced):
                    print(f"  {cls}: {count:,}")

                print(f"\nNew train set size: {X_train_balanced.shape[0]:,} samples")
            else:
                print(f"\n⚠ No balancing needed (all classes within acceptable range)")
                X_train_balanced = X_train_final
                y_train_balanced = y_train_final
        else:
            print(f"\n⚠ No balancing strategy needed")
            X_train_balanced = X_train_final
            y_train_balanced = y_train_final

    except Exception as e:
        print(f"⚠ Balanced sampling failed: {e}")
        print("Proceeding with imbalanced data...")
        X_train_balanced = X_train_final
        y_train_balanced = y_train_final
    
    # return dictionary
    processed_data = {
        'X_train': X_train_balanced,
        'y_train': y_train_balanced,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test_scaled,
        'y_test': y_test_encoded,
        'feature_names': X_train_encoded.columns.tolist(),
        'n_features': X_train_scaled.shape[1],
        'n_classes': len(label_encoder.classes_),
        'label_encoder': label_encoder,
        'scaler': scaler,
        'dataset_name': 'NSL-KDD'
    }
    
    # Save test data for dashboard simulation
    print(f"\nSaving test data for dashboard simulation...")
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test_scaled)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test_encoded)
    print(f"✓ Test data saved to {output_dir}/")
    
    print("\n" + "="*80)
    print("NSL-KDD PREPROCESSING COMPLETE")
    print("="*80)
    print(f"Features: {processed_data['n_features']}")
    print(f"Classes: {processed_data['n_classes']}")
    print(f"Ready for model training!")
    print("="*80 + "\n")
    
    return processed_data
