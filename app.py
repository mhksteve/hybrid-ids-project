"""
Streamlit Dashboard for Hybrid AI-Based Intrusion Detection System
Real-time visualization and monitoring interface with multi-model support
"""

import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import joblib
from tensorflow import keras
from sklearn.metrics import accuracy_score


script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, 'src')
sys.path.insert(0, src_path)

# Page configuration
st.set_page_config(
    page_title="Hybrid IDS - Real-time Monitoring",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .status-normal {
        color: #28a745;
        font-weight: bold;
    }
    .status-attack {
        color: #dc3545;
        font-weight: bold;
        animation: blink 1s infinite;
    }
    @keyframes blink {
        0%, 50%, 100% { opacity: 1; }
        25%, 75% { opacity: 0.5; }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_active_model(dataset_dir, model_name):
    """
    Load the specified model and preprocessing artifacts
    
    Args:
        dataset_dir: 'cicids' or 'nslkdd'
        model_name: 'Random Forest', 'XGBoost', 'SVM', '1D-CNN', 'LSTM', or 'Hybrid Model'
    
    Returns:
        Tuple of (model(s), scaler, label_encoder, success)
    """
    try:
        base_path = f'models/{dataset_dir}'
        
        # Load preprocessing artifacts
        scaler = joblib.load(f'{base_path}/scaler.pkl')
        label_encoder = joblib.load(f'{base_path}/label_encoder.pkl')
        
        # Load model based on type
        if model_name == 'Random Forest':
            model = joblib.load(f'{base_path}/random_forest.pkl')
            return model, scaler, label_encoder, True
            
        elif model_name == 'XGBoost':
            model = joblib.load(f'{base_path}/xgboost.pkl')
            return model, scaler, label_encoder, True

        elif model_name == 'SVM':
            model = joblib.load(f'{base_path}/svm.pkl')
            return model, scaler, label_encoder, True

        elif model_name == '1D-CNN':
            model = keras.models.load_model(f'{base_path}/1d_cnn.h5', compile=False)
            return model, scaler, label_encoder, True

        elif model_name == 'LSTM':
            model = keras.models.load_model(f'{base_path}/lstm.h5', compile=False)
            return model, scaler, label_encoder, True

        elif model_name == 'Hybrid Model':
            cnn_extractor = keras.models.load_model(f'{base_path}/cnn_feature_extractor.h5', compile=False)
            rf_classifier = joblib.load(f'{base_path}/hybrid_rf.pkl')
            return (cnn_extractor, rf_classifier), scaler, label_encoder, True

        else:
            st.error(f"Unknown model: {model_name}")
            return None, None, None, False

    except Exception as e:
        st.error(f"Error loading {model_name}: {e}")
        return None, None, None, False


@st.cache_data
def load_test_data(dataset_dir):
    """
    Load test data for the selected dataset
    """
    try:
        X_test = np.load(f'models/{dataset_dir}/X_test.npy')
        y_test = np.load(f'models/{dataset_dir}/y_test.npy')
        return X_test, y_test, True
    except Exception as e:
        st.error(f"Error loading test data: {e}")
        return None, None, False


def make_prediction(sample, model, scaler, label_encoder, model_name):
    """
    Make prediction using the specified model

    Args:
        sample: Single sample (1D array)
        model: Loaded model (or tuple of models for Hybrid)
        scaler: MinMaxScaler
        label_encoder: LabelEncoder
        model_name: Name of the model

    Returns:
        prediction, probability, latency_ms
    """
    start_time = time.time()
    
    # Ensure sample is 2D
    if len(sample.shape) == 1:
        sample = sample.reshape(1, -1)

    # Deep Learning models (1D-CNN, LSTM) need 3D input
    if model_name in ['1D-CNN', 'LSTM']:
        sample_3d = sample.reshape(sample.shape[0], sample.shape[1], 1)
        prediction_probs = model.predict(sample_3d, verbose=0)
        prediction = np.argmax(prediction_probs, axis=1)[0]
        probability = prediction_probs[0]
        
    # Hybrid Model
    elif model_name == 'Hybrid Model':
        cnn_extractor, rf_classifier = model  # Unpack tuple
        sample_3d = sample.reshape(sample.shape[0], sample.shape[1], 1)
        
        # Extract deep features with CNN
        deep_features = cnn_extractor.predict(sample_3d, verbose=0)
        
        # Classify with Random Forest
        prediction = rf_classifier.predict(deep_features)[0]
        probability = rf_classifier.predict_proba(deep_features)[0]
        
    # Machine Learning models (RF, XGBoost, SVM)
    else:
        prediction = model.predict(sample)[0]

        if model_name == 'SVM':
            probability = np.zeros(len(label_encoder.classes_))
            probability[prediction] = 1.0
        else:
            probability = model.predict_proba(sample)[0]
    
    latency = (time.time() - start_time) * 1000
    
    return prediction, probability, latency


def main():
    """
    Main dashboard application
    """
    
    # Header
    st.markdown('<p class="main-header"> Hybrid AI-Based Intrusion Detection System</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Configuration")

    # Dataset selector
    dataset_choice = st.sidebar.selectbox(
        "Select Dataset",
        options=["CICIDS2017", "NSL-KDD"],
        index=0
    )

    # Initialize dataset tracker if not exists
    if 'current_dataset' not in st.session_state:
        st.session_state.current_dataset = dataset_choice

    # Check if dataset changed
    if st.session_state.current_dataset != dataset_choice:
        # Dataset changed - reset all session state variables
        st.session_state.current_dataset = dataset_choice
        st.session_state.simulation_running = False
        st.session_state.packet_index = 0
        st.session_state.total_packets = 0
        st.session_state.attack_count = 0
        st.session_state.traffic_history = []
        st.session_state.accuracy_history = []
        st.session_state.latency_history = []

        # Delete attack_distribution so it rebuilds with new label_encoder.classes_
        if 'attack_distribution' in st.session_state:
            del st.session_state.attack_distribution

        # Trigger rerun to refresh UI with new dataset
        st.rerun()

    # Set dataset directory based on selection
    dataset_dir = 'cicids' if dataset_choice == "CICIDS2017" else 'nslkdd'
    
    # Model selector
    model_choice = st.sidebar.selectbox(
        "Select Active Model",
        options=["Hybrid Model", "Random Forest", "XGBoost", "SVM", "1D-CNN", "LSTM"],
        index=0
    )

    st.sidebar.markdown("---")

    # Load selected model
    model, scaler, label_encoder, models_loaded = load_active_model(dataset_dir, model_choice)


    if not models_loaded:
        st.warning(f"{model_choice} not loaded. Please train the models first by running: `python train_pipeline.py --dataset {dataset_dir}`")
        return
    
    # Load test data
    X_test, y_test, data_loaded = load_test_data(dataset_dir)
    
    if not data_loaded:
        st.warning(f"Test data not loaded for {dataset_choice}. Please run the training pipeline first.")
        return



    # 2 tabs streamlit
    if st.session_state.get('simulation_running', False):
        st.warning(
            "**Simulation is currently running!** Model Comparison is disabled. Please stop the simulation in the 'Live Traffic Feed' tab to enable benchmarking.")
        st.markdown("---")


    tab1, tab2 = st.tabs(["🔴 Live Traffic Simulation", "Model Comparison"])
    
    # TAB 1: LIVE SIMULATION
    with tab1:
        # Initialize session state
        if 'simulation_running' not in st.session_state:
            st.session_state.simulation_running = False
        if 'packet_index' not in st.session_state:
            st.session_state.packet_index = 0
        if 'total_packets' not in st.session_state:
            st.session_state.total_packets = 0
        if 'attack_count' not in st.session_state:
            st.session_state.attack_count = 0
        if 'traffic_history' not in st.session_state:
            st.session_state.traffic_history = []
        if 'attack_distribution' not in st.session_state:
            st.session_state.attack_distribution = {label: 0 for label in label_encoder.classes_}
        if 'accuracy_history' not in st.session_state:
            st.session_state.accuracy_history = []
        if 'latency_history' not in st.session_state:
            st.session_state.latency_history = []
        
        # Simulation Controls
        st.sidebar.subheader("Simulation Controls")
        
        col1, col2, col3 = st.sidebar.columns(3)
        with col1:
            if st.button("Start", disabled=st.session_state.simulation_running):
                st.session_state.simulation_running = True
                st.rerun()
        
        with col2:
            if st.button("Stop", disabled=not st.session_state.simulation_running):
                st.session_state.simulation_running = False
                st.rerun()

        with col3:
            if st.button("Reset"):
                st.session_state.packet_index = 0
                st.session_state.total_packets = 0
                st.session_state.attack_count = 0
                st.session_state.traffic_history = []
                st.session_state.attack_distribution = {label: 0 for label in label_encoder.classes_}
                st.session_state.accuracy_history = []
                st.session_state.latency_history = []
                st.rerun()
        
        # Speed control
        st.sidebar.subheader("Speed Control")
        packets_per_second = st.sidebar.slider("Packets per Second", min_value=1, max_value=20, value=5)
        delay = 1.0 / packets_per_second
        
        # Model Info
        st.sidebar.subheader("Active Model")
        st.sidebar.info(f"""
        **Model:** {model_choice}  
        **Dataset:** {dataset_choice}  
        **Status:** Online ✅
        """)
        
        st.sidebar.subheader("Dataset Info")
        st.sidebar.info(f"""
        **Total Samples:** {len(X_test):,}  
        **Features:** {X_test.shape[1]}  
        **Classes:** {len(label_encoder.classes_)}
        """)
        
        # Main panel - Metrics Row
        st.subheader("📈 Live Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Packets Processed",
                value=f"{st.session_state.total_packets:,}",
                delta=f"+1" if st.session_state.simulation_running else None
            )
        
        with col2:
            if st.session_state.total_packets > 0:
                current_status = "🔴 ATTACK DETECTED" if st.session_state.traffic_history[-1]['is_attack'] else "🟢 Normal"
                status_class = "status-attack" if st.session_state.traffic_history[-1]['is_attack'] else "status-normal"
            else:
                current_status = "⚪ Idle"
                status_class = ""
            
            st.metric(
                label="Current Status",
                value=current_status
            )
        
        with col3:
            if len(st.session_state.accuracy_history) > 0:
                current_accuracy = st.session_state.accuracy_history[-1]
            else:
                current_accuracy = 0.0
            
            st.metric(
                label="Accuracy",
                value=f"{current_accuracy*100:.1f}%"
            )
        
        with col4:
            if len(st.session_state.latency_history) > 0:
                current_latency = st.session_state.latency_history[-1]
            else:
                current_latency = 0.0
            
            st.metric(
                label="Latency",
                value=f"{current_latency:.2f} ms"
            )
        
        st.markdown("---")
        
        # Charts Row
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.subheader("Live Traffic Feed")
            
            if st.session_state.traffic_history:
                # Show last 10 packets
                recent_traffic = st.session_state.traffic_history[-10:][::-1]  # Reverse to show newest first
                
                traffic_df = pd.DataFrame(recent_traffic)
                
                # Style the dataframe
                def highlight_attacks(row):
                    if row['Status'] == 'ATTACK':
                        return ['background-color: #ffcccc; color: #8b0000; font-weight: bold'] * len(row)
                    else:
                        return ['background-color: #ccffcc; color: #006400; font-weight: bold'] * len(row)
                
                styled_df = traffic_df.style.apply(highlight_attacks, axis=1)
                
                st.dataframe(styled_df, use_container_width=True, height=400)
            else:
                st.info("Waiting for traffic data... Click 'Start' to begin simulation.")
        
        with col_right:
            st.subheader("Attack Distribution")
            
            if st.session_state.total_packets > 0:
                # Distribution chart
                labels = list(st.session_state.attack_distribution.keys())
                values = list(st.session_state.attack_distribution.values())
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=labels,
                        y=values,
                        marker=dict(
                            color=values,
                            colorscale='Reds',
                            showscale=False
                        ),
                        text=values,
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title="Traffic Classification Distribution",
                    xaxis_title="Attack Type",
                    yaxis_title="Count",
                    height=400,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Waiting for traffic data... Click 'Start' to begin simulation.")
        
        # Performance Charts
        st.markdown("---")
        st.subheader("📉 Performance Metrics Over Time")
        
        col_perf1, col_perf2 = st.columns(2)
        
        with col_perf1:
            if len(st.session_state.accuracy_history) > 0:
                acc_fig = go.Figure()
                acc_fig.add_trace(go.Scatter(
                    y=st.session_state.accuracy_history,
                    mode='lines+markers',
                    name='Accuracy',
                    line=dict(color='blue', width=2)
                ))
                acc_fig.update_layout(
                    title="Cumulative Accuracy",
                    xaxis_title="Packet Number",
                    yaxis_title="Accuracy",
                    yaxis_range=[0, 1],
                    height=300,
                    template="plotly_white"
                )
                st.plotly_chart(acc_fig, use_container_width=True)
        
        with col_perf2:
            if len(st.session_state.latency_history) > 0:
                lat_fig = go.Figure()
                lat_fig.add_trace(go.Scatter(
                    y=st.session_state.latency_history,
                    mode='lines+markers',
                    name='Latency',
                    line=dict(color='orange', width=2)
                ))
                lat_fig.update_layout(
                    title="Inference Latency",
                    xaxis_title="Packet Number",
                    yaxis_title="Latency (ms)",
                    height=300,
                    template="plotly_white"
                )
                st.plotly_chart(lat_fig, use_container_width=True)
        
        # Simulation loop
        if st.session_state.simulation_running:
            if st.session_state.packet_index < len(X_test):
                # Get current sample
                current_sample = X_test[st.session_state.packet_index]
                true_label = y_test[st.session_state.packet_index]
                
                # Make prediction using the unified function
                prediction, probability, latency = make_prediction(current_sample, model, scaler, label_encoder, model_choice)
                
                # Decode labels
                predicted_label = label_encoder.inverse_transform([prediction])[0]
                true_label_name = label_encoder.inverse_transform([true_label])[0]
                
                # Update counters
                st.session_state.total_packets += 1
                is_attack = predicted_label.lower() != 'benign' and predicted_label.lower() != 'normal'
                
                if is_attack:
                    st.session_state.attack_count += 1
                
                # Update distribution
                st.session_state.attack_distribution[predicted_label] += 1
                
                # Update traffic history
                traffic_entry = {
                    'Packet #': st.session_state.total_packets,
                    'Predicted': predicted_label,
                    'Actual': true_label_name,
                    'Confidence': f"{max(probability)*100:.1f}%",
                    'Status': 'ATTACK' if is_attack else 'Normal',
                    'Latency (ms)': f"{latency:.2f}",
                    'is_attack': is_attack
                }
                st.session_state.traffic_history.append(traffic_entry)
                
                # Update accuracy
                correct_predictions = sum(1 for entry in st.session_state.traffic_history 
                                         if entry['Predicted'] == entry['Actual'])
                current_accuracy = correct_predictions / st.session_state.total_packets
                st.session_state.accuracy_history.append(current_accuracy)
                
                # Update latency
                st.session_state.latency_history.append(latency)
                
                # Move to next packet
                st.session_state.packet_index += 1
                
                # Wait and rerun
                time.sleep(delay)
                st.rerun()
            
            else:
                st.session_state.simulation_running = False
                st.success("Simulation completed! All test samples processed.")
                st.balloons()
    
    # ========== TAB 2: MODEL COMPARISON ==========
    with tab2:
        st.subheader("Model Performance Comparison")
        st.markdown("Benchmark all 6 trained models on test samples")

        # Benchmark settings
        col1, col2 = st.columns([3, 1])
        with col1:
            benchmark_samples = st.slider("Number of samples to benchmark", 100, 5000, 1000, 100)
        with col2:
            # Disable tab 2 if simulation is running
            simulation_active = st.session_state.get('simulation_running', False)
            run_benchmark = st.button(
                "Run Benchmark",
                type="primary",
                disabled=simulation_active,
                help="Stop the simulation first to run benchmark" if simulation_active else "Click to benchmark all models"
            )
        
        if run_benchmark:
            st.markdown("---")
            
            # All model names
            all_models = ["Random Forest", "XGBoost", "SVM", "1D-CNN", "LSTM", "Hybrid Model"]
            
            # Results storage
            results = []
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Benchmark each model
            for idx, model_name in enumerate(all_models):
                status_text.text(f"Benchmarking {model_name}... ({idx+1}/{len(all_models)})")
                
                try:
                    # Load model
                    model_obj, scaler_obj, label_encoder_obj, success = load_active_model(dataset_dir, model_name)
                    
                    if not success:
                        st.warning(f"⚠️ {model_name} could not be loaded. Skipping...")
                        continue
                    
                    # Get benchmark samples
                    X_benchmark = X_test[:benchmark_samples]
                    y_benchmark = y_test[:benchmark_samples]
                    
                    # Run predictions
                    predictions = []
                    latencies = []
                    
                    for i in range(benchmark_samples):
                        pred, prob, lat = make_prediction(
                            X_benchmark[i], model_obj, scaler_obj, label_encoder_obj, model_name
                        )
                        predictions.append(pred)
                        latencies.append(lat)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_benchmark, predictions) * 100
                    avg_latency = np.mean(latencies)
                    
                    # Store results
                    results.append({
                        'Model': model_name,
                        'Accuracy (%)': round(accuracy, 2),
                        'Avg Latency (ms)': round(avg_latency, 2),
                        'Total Time (s)': round(sum(latencies) / 1000, 2)
                    })
                    
                except Exception as e:
                    st.error(f"Error benchmarking {model_name}: {e}")
                
                # Update progress
                progress_bar.progress((idx + 1) / len(all_models))
            
            status_text.text("✅ Benchmark complete!")
            progress_bar.empty()
            
            # Display results
            if results:
                st.markdown("---")
                st.subheader("📈 Benchmark Results")
                
                # Results dataframe
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Accuracy bar chart
                    fig_acc = go.Figure(data=[
                        go.Bar(
                            x=df_results['Model'],
                            y=df_results['Accuracy (%)'],
                            marker=dict(
                                color=df_results['Accuracy (%)'],
                                colorscale='RdYlGn',
                                showscale=True,
                                colorbar=dict(title="Accuracy %")
                            ),
                            text=df_results['Accuracy (%)'],
                            textposition='outside'
                        )
                    ])
                    
                    fig_acc.update_layout(
                        title="Model Accuracy Comparison",
                        xaxis_title="Model",
                        yaxis_title="Accuracy (%)",
                        yaxis_range=[0, 100],
                        height=400,
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig_acc, use_container_width=True)
                
                with col2:
                    # Latency vs Accuracy scatter
                    fig_scatter = go.Figure(data=[
                        go.Scatter(
                            x=df_results['Avg Latency (ms)'],
                            y=df_results['Accuracy (%)'],
                            mode='markers+text',
                            marker=dict(
                                size=15,
                                color=df_results['Accuracy (%)'],
                                colorscale='RdYlGn',
                                showscale=True,
                                colorbar=dict(title="Accuracy %")
                            ),
                            text=df_results['Model'],
                            textposition='top center',
                            textfont=dict(size=10)
                        )
                    ])
                    
                    fig_scatter.update_layout(
                        title="Latency vs Accuracy Trade-off",
                        xaxis_title="Average Latency (ms)",
                        yaxis_title="Accuracy (%)",
                        yaxis_range=[0, 100],
                        height=400,
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Best model highlight
                best_acc_model = df_results.loc[df_results['Accuracy (%)'].idxmax()]
                best_speed_model = df_results.loc[df_results['Avg Latency (ms)'].idxmin()]
                
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success(f"Highest Accuracy**: {best_acc_model['Model']} ({best_acc_model['Accuracy (%)']}%)")
                
                with col2:
                    st.info(f"Fastest Model**: {best_speed_model['Model']} ({best_speed_model['Avg Latency (ms)']} ms)")


if __name__ == "__main__":
    main()
