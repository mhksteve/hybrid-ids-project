# EXECUTION GUIDE - Hybrid AI-Based IDS

This guide provides step-by-step instructions to set up, train, and run the Hybrid Intrusion Detection System.

---

## PREREQUISITES

Before you begin, ensure you have:

- **Python 3.10** should be installed
- **activate .venv**
- **pip --version** to check if pip installed
- **Storage**: ~2-5GB for datasets and models
---

## STEP 1: INSTALLATION
### Using pip (Recommended)
```bash
cd hybrid-ids-project
# Install all dependencies
pip install -r requirements.txt
```

### Verify Installation
Run verify_setup file to check dependencies required all installed
```bash
# Run the verification script
python verify_setup.py
```
---

## STEP 2: PREPARE YOUR DATASET

### Download the Dataset

1. **CICIDS2017**
   - https://www.unb.ca/cic/datasets/ids-2017.html
   - Download the CSV version
   - Size: ~500MB - 2GB (depending on the subset)

2. **NSL-KDD**
   - https://www.unb.ca/cic/datasets/nsl.html
   - Download KDDTrain+ and KDDTest+ sets

### Place the Dataset
Place dataset to `data` folder:
- CICIDS2017 folder renamed to `cicids2017`
- NSL-KDD folder renamed to `nslkdd`
---

## STEP 3: TRAIN THE MODELS

### Run the Training Pipeline

```bash
  Train on CICIDS2017
  python train_pipeline.py --dataset cicids 
  
  Train on NSL-KDD
  python train_pipeline.py --dataset nslkdd 
```

### What Happens During Training?

#### **Phase 1: Data Preprocessing** (2-5 minutes)
```
PHASE 1: DATA INGESTION & PREPROCESSING
========================================
[1/7] Loading data from: data/CICIDS2017.csv
✓ Data loaded successfully!

[2/7] Dataset Overview:
Shape: (2830743, 79)
...
```

#### **Phase 2: Scenario A - Machine Learning** (10-20 minutes)
```
SCENARIO A: STANDALONE MACHINE LEARNING
========================================
[Random Forest] Building model...
✓ Random Forest created with 100 trees
...
```

#### **Phase 3: Scenario B - Deep Learning** (20-60 minutes)
```
SCENARIO B: STANDALONE DEEP LEARNING
====================================
[1D-CNN] Building model...
...
Epoch 1/50
███████████████████ 1234/1234 [======] - 45s
...
```
**Note**: This phase takes the longest. You'll see progress bars for each epoch.

#### **Phase 4: Scenario C - Hybrid Model** (5-10 minutes)
```
SCENARIO C: HYBRID MODEL
========================
BUILDING HYBRID MODEL
...
TRAINING HYBRID MODEL
...
```

### Expected Total Time

| Dataset Size | Approximate Time |
|--------------|------------------|
| Small (<100K samples) | 15-30 minutes |
| Medium (100K-500K) | 30-90 minutes |
| Large (>500K) | 1-3 hours |

### Monitor Progress

### Final Output

```
╔══════════════════════════════════════════════════════════════╗
║                        FINAL SUMMARY                          ║
╚══════════════════════════════════════════════════════════════╝

MODEL COMPARISON
================
Scenario A - Machine Learning:
  RandomForest    - Accuracy:  97.23%, Latency:   1.45 ms
  XGBoost         - Accuracy:  98.12%, Latency:   2.31 ms
  SVM             - Accuracy:  95.67%, Latency:   5.89 ms

Scenario B - Deep Learning:
  1D-CNN          - Accuracy:  98.45%, Latency:  12.34 ms
  LSTM            - Accuracy:  97.89%, Latency:  18.67 ms

Scenario C - Hybrid Model:
  Hybrid Model    - Accuracy:  98.78%, Latency:   3.21 ms

TRAINING COMPLETE!
All models have been saved to the 'models/' directory.
```

### Saved Artifacts

After training, you'll have in `models/`:
```
models/
├── cnn_feature_extractor.h5  
├── hybrid_rf.pkl              
├── random_forest.pkl          
├── xgboost.pkl               
├── svm.pkl                   
├── 1d_cnn.h5                 
├── lstm.h5                  
├── scaler.pkl                
├── label_encoder.pkl         
├── X_test.npy                
└── y_test.npy                
```
---

## STEP 4: DASHBOARD

### Start the Dashboard
```bash
streamlit run app.py
```
### Expected Output

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.100:8501
```
The dashboard will automatically open in your default browser.
