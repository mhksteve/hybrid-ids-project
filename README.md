# Hybrid AI-Based Intrusion Detection System

This project implements a hybrid intrusion detection system that combines deep learning feature extraction with machine learning classification. The system was developed for dissertation research and includes a Streamlit dashboard for simulated near real-time monitoring.

## Project Overview

The project evaluates three modelling approaches:

1. Standalone machine learning models:
   - Random Forest
   - XGBoost
   - SVM

2. Standalone deep learning models:
   - 1D-CNN
   - LSTM

3. Hybrid model:
   - 1D-CNN feature extractor
   - Random Forest classifier

The aim of the project is to compare these approaches in terms of detection performance, interpretability, and near real-time responsiveness.


## Project Structure

```
hybrid-ids-project/
├── data/
│   ├── cicids2017/
│   └── nslkdd/
├── models/
├── results/
│   ├── cicids/
│   └── nslkdd/
├── src/
│   ├── __init__.py
│   ├── preprocess_cicids.py
│   ├── preprocess_nslkdd.py
│   ├── models_ml.py
│   ├── models_dl.py
│   ├── hybrid.py
│   └── evaluate_models.py
├── train_pipeline.py
├── app.py
├── verify_setup.py
├── requirements.txt
└── README.md
```

## Main Components

### Data

The `data/` directory contains the dataset folders used for training and testing:

- `data/cicids2017/` - https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset/data
- `data/nslkdd/` - https://www.kaggle.com/datasets/hassan06/nslkdd

These are handled separately because the datasets have different structures and require different preprocessing steps.

### Results

The `results/` directory stores outputs generated during evaluation:

- `results/cicids/`
- `results/nslkdd/`

These folders contain dataset-specific results such as plots, metrics, reports, and evaluation outputs.

### Preprocessing

The project uses separate preprocessing pipelines for the two datasets:

- `src/preprocess_cicids.py`
- `src/preprocess_nslkdd.py`

These scripts handle cleaning, encoding, scaling, splitting, and class balancing.

### Machine Learning Models

Implemented in `src/models_ml.py`:

- Random Forest
- XGBoost
- SVM

### Deep Learning Models

Implemented in `src/models_dl.py`:

- 1D-CNN
- LSTM

### Hybrid Model

Implemented in `src/hybrid.py`:

- feature extraction using a trained 1D-CNN
- final classification using Random Forest

### Evaluation

Implemented in `src/evaluate_models.py`:

- accuracy
- precision
- recall
- F1-score
- confusion matrix
- ROC/AUC
- inference latency

### Dashboard

Implemented in `app.py` using Streamlit.

The dashboard replays saved test samples in a simulated near real-time setting and displays:

- predictions
- cumulative accuracy
- latency
- attack distribution
- model comparison views

## Requirements

This project was developed using:

- Python 3.1
- TensorFlow / Keras
- scikit-learn
- XGBoost
- imbalanced-learn
- Streamlit
- Plotly
- NumPy
- pandas
- Matplotlib
- Seaborn

Install dependencies with:

```bash
pip install -r requirements.txt
```

**Note**: If you're using a GPU, install TensorFlow GPU version:
```bash
pip install tensorflow-gpu==2.13.0
```

## Dataset Preparation

This project was developed using
- CICIDS2017
- NSL-KDD

Place the dataset files inside the `data/` directory before training.

Notes:
- CICIDS2017 is handled through the `data/cicids2017/` folder
- NSL-KDD is handled through the `data/nslkdd/` folder
- preprocessing for the two datasets is separate because the data structures are different

## Training

To train the models on CICIDS2017, run:
```bash
python train_pipeline.py --dataset cicids
```
To train the models on NSL-KDD, run:
```bash
python train_pipeline.py --dataset nslkdd
```

**The training process:**:
- preprocesses the selected dataset
- trains the standalone machine learning models 
- trains the standalone deep learning models
- builds the hybrid model
- saves model artefacts 
- stores evaluation outputs for later use in the dashboard

With the current full setup, training can take up to 3 hours depending on the dataset size and available hardware.

## Quick Test Setup ##
For a quicker verification run, the project settings can be reduced temporarily before running the usual training commands. This is useful if a supervisor or examiner wants to confirm that the pipeline works without waiting for the full training cycle.

**A short test setup can use**:
- n_estimators = 10 for Random Forest and XGBoost
- batch_size = 64 for the deep learning models
- epochs = 2 for the deep learning models

These changes should be made in train_pipeline.py.

### Where to change the values ###

**Scenario A: Standalone Machine Learning**
In train_pipeline.py, inside run_scenario_a():
```bash
ml_models.build_random_forest(n_estimators=10) #set 100 to 10
ml_models.build_xgboost(n_estimators=10) #set 100 to 10
```
You can also reduce the SVM subset size for a quicker check.
```bash
ml_models.train_model('SVM', X_train, y_train, X_val, y_val, use_subset_for_svm=True, svm_subset_size=2000)
```
**Scenario B: Standalone Deep Learning**
In train_pipeline.py, inside run_scenario_b():
```bash
dl_models.train_model('1D-CNN', X_train, y_train, X_val, y_val, epochs=20, batch_size=256, patience=10)
dl_models.train_model('LSTM', X_train, y_train, X_val, y_val, epochs=20, batch_size=256, patience=10)
```
**Scenario C: Hybrid Model**
In train_pipeline.py, inside run_scenario_c():
```bash
hybrid.build_from_pretrained_cnn(cnn_model, feature_layer_name='dense_2', n_estimators=10)
```

### Dashboard 
After training, launch the dashboard with:
```bash
streamlit run app.py
```
The dashboard opens in the browser at `http://localhost:8501` and provides a simulated near real-time replay of saved test data.

**Dashboard Features**:
- dataset selection
- model selection
- replay speed adjustment
- prediction monitoring
- latency display
- attack distribution charts
- benchmarking and model comparison

This is not a live packet capture system. It uses stored test samples to simulate monitoring behaviour.

### Notes on the Implementation

A few points are important for understanding the project correctly:

- the hybrid model is specifically a 1D-CNN feature extractor with a Random Forest classifier
- the dashboard operates in a simulated near real-time setting
- the project is intended for research and educational use
- the implementation is based on benchmark datasets rather than live production traffic
- CICIDS2017 and NSL-KDD are trained and evaluated through separate preprocessing pipelines

## Troubleshooting

### Issue: Dataset File Not Found
**Solution**: Make sure the required dataset files are placed in the correct dataset folders inside `data/`.

### "Out of memory" error
**Solution**: 
- Reduce batch size in training: `batch_size=64`
- Use SVM subset: `use_subset_for_svm=True`
- Train on a subset of data for testing

### Issue: "Target column not found"
**Solution**: The system tries common names automatically (`Label`, `label`, `Attack Label`, etc.). If your column has a different name, specify it in `load_and_process_data()`.

### Issue: Dashboard shows "Models not loaded"
**Solution**: Run `python train_pipeline.py` first to train and save the models.

### Issue: TensorFlow GPU not working
**Solution**: 
- Check CUDA installation: `nvidia-smi`
- Install compatible TensorFlow GPU version
- Verify CUDA and cuDNN versions match TensorFlow requirements

## Model Evaluation
After training, detailed evaluation reports are printed:
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1-Score**: Per-class metrics
- **Confusion Matrix**: Classification breakdown
- **Inference Time**: Total time for all test samples
- **Average Latency**: Time per sample (ms)

## References

**Datasets**:
- [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html): Canadian Institute for Cybersecurity
- [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html): Network Security Laboratory

## License

This project is for educational and research purposes.


