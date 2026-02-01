import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

def train_and_eval():
    df = pd.read_csv('data/processed_training_data.csv')
    
    # Feature engineering for ML
    # Convert categorical to numerical
    le_backend = LabelEncoder()
    df['backend_code'] = le_backend.fit_transform(df['backend'])
    
    le_precision = LabelEncoder()
    df['precision_code'] = le_precision.fit_transform(df['precision'])
    
    feature_cols = [
        'n_qubits', 'n_gates', 'depth', 'n_2q_gates', 'n_1q_gates',
        'treewidth_heuristic', 'avg_degree', 'algebraic_connectivity',
        'spectral_gap', 'magic_density', 'volume', 'n_lines',
        'max_degree', 'clustering_coeff', 'num_components',
        'max_component_size', 'avg_span', 'max_span',
        'backend_code', 'precision_code'
    ]
    
    X = df[feature_cols].copy()
    
    # 1. Threshold Prediction (Classification)
    y_thresh = df['selected_threshold']
    
    print("Threshold Model Evaluation (Accuracy)")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    rfc = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
    rfc_scores = cross_val_score(rfc, X, y_thresh, cv=kf)
    print(f"RF Threshold Accuracy: {rfc_scores.mean():.4f} (+/- {rfc_scores.std():.4f})")
    
    # Train final model
    best_thresh_model = rfc.fit(X, y_thresh)
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_thresh_model, 'models/threshold_model.joblib')
    joblib.dump(le_backend, 'models/le_backend.joblib')
    joblib.dump(le_precision, 'models/le_precision.joblib')
    joblib.dump(feature_cols, 'models/feature_cols.joblib')
    
    print("\nThreshold model saved to models/threshold_model.joblib")

if __name__ == "__main__":
    train_and_eval()
