import json
import os
import pandas as pd
import numpy as np
from scripts.features import extract_features

def prepare_dataset(data_path, circuits_dir, output_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Create mapping of file to circuit features
    circuit_files = [c['file'] for c in data['circuits']]
    features_list = []
    
    print(f"Extracting features for {len(circuit_files)} circuits...")
    for f_name in circuit_files:
        qasm_path = os.path.join(circuits_dir, f_name)
        if not os.path.exists(qasm_path):
            print(f"Warning: {qasm_path} not found.")
            continue
        feats = extract_features(qasm_path)
        if feats:
            feats['file'] = f_name
            features_list.append(feats)
    
    df_features = pd.DataFrame(features_list)
    
    # Extract labels from results
    results_list = []
    for res in data['results']:
        if res['status'] == 'ok' or res['status'] == 'no_threshold_met':
            entry = {
                'file': res['file'],
                'backend': res['backend'],
                'precision': res['precision'],
                'selected_threshold': res['selection']['selected_threshold'],
            }
            results_list.append(entry)
            
    df_results = pd.DataFrame(results_list)
    
    # Merge
    df_final = pd.merge(df_results, df_features, on='file', how='inner')
    
    # Save
    df_final.to_csv(output_path, index=False)
    print(f"Prepared dataset with {len(df_final)} entries saved to {output_path}")

if __name__ == "__main__":
    DATA_PATH = "data/hackathon_public.json"
    CIRCUITS_DIR = "circuits"
    OUTPUT_PATH = "data/processed_training_data.csv"
    
    prepare_dataset(DATA_PATH, CIRCUITS_DIR, OUTPUT_PATH)
