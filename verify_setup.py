"""
Setup Verification
Run to verify if all dependencies are installed
"""

import sys
import importlib

def check_package(package_name, display_name=None):
    """
    Check if a package is installed
    """
    if display_name is None:
        display_name = package_name
    
    try:
        importlib.import_module(package_name)
        print(f"✓ {display_name:20} - Installed")
        return True
    except ImportError:
        print(f"✗ {display_name:20} - NOT INSTALLED")
        return False


def verify_setup():
    """
    Verify all dependencies are installed
    """
    print("="*60)
    print("HYBRID IDS - SETUP VERIFICATION")
    print("="*60)

    print("\n[1/3] Checking Python environment...")
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")

    #Check if Python 3.10 is being used
    if python_version.major == 3 and python_version.minor == 10:
        print("✓ Python 3.10 detected")
    else:
        print(
            f"⚠ WARNING: You are using Python {python_version.major}.{python_version.minor}. This project requires Python 3.10.")

    # Check if a virtual environment is active
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if in_venv:
        print("✓ Virtual environment is active")
    else:
        print("⚠ WARNING: You are NOT running inside a virtual environment!")
    
    print("\n[2/3] Checking required packages...")
    
    packages = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'imblearn': 'Imbalanced-learn',
        'tensorflow': 'TensorFlow',
        'keras': 'Keras',
        'xgboost': 'XGBoost',
        'streamlit': 'Streamlit',
        'plotly': 'Plotly',
        'joblib': 'Joblib'
    }
    
    all_installed = True
    for package, display_name in packages.items():
        if not check_package(package, display_name):
            all_installed = False
    
    print("\n[3/3] Checking project structure...")
    
    import os
    
    required_dirs = ['data', 'models', 'src']
    required_files = [
        'src/preprocess_cicids.py',
        'src/preprocess_nslkdd.py',
        'src/models_ml.py',
        'src/models_dl.py',
        'src/hybrid.py',
        'train_pipeline.py',
        'app.py',
        'requirements.txt'
    ]
    
    structure_ok = True
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✓ Directory: {directory}")
        else:
            print(f"✗ Directory: {directory} - NOT FOUND")
            structure_ok = False
    
    for filepath in required_files:
        if os.path.exists(filepath):
            print(f"✓ File: {filepath}")
        else:
            print(f"✗ File: {filepath} - NOT FOUND")
            structure_ok = False
    
    print("\n" + "="*60)
    print("VERIFICATION RESULTS")
    print("="*60)
    
    if all_installed and structure_ok:
        print("✅ ALL CHECKS PASSED!")
        print("\nYou are ready to proceed!")
        print("\nNext Steps:")
        print("1. Place your dataset in data/CICIDS2017.csv")
        print("2. Run training: python train_pipeline.py")
        print("3. Launch dashboard: streamlit run app.py")
    else:
        print("❌ SETUP INCOMPLETE")
        
        if not all_installed:
            print("\n⚠ Missing packages detected!")
            print("Install missing packages with:")
            print("  pip install -r requirements.txt")
        
        if not structure_ok:
            print("\n⚠ Project structure incomplete!")
            print("Ensure all required files and directories are present.")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    verify_setup()
