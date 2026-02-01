import argparse
import json
import os
import joblib
import pandas as pd
import numpy as np
from scripts.features import extract_features

THRESHOLD_LADDER = [1, 2, 4, 8, 16, 32, 64, 128, 256]

def get_next_rung(rung):
    """Returns the next higher rung in the ladder, if available."""
    try:
        idx = THRESHOLD_LADDER.index(rung)
        if idx < len(THRESHOLD_LADDER) - 1:
            return THRESHOLD_LADDER[idx + 1]
    except ValueError:
        pass
    return rung

def main():
    parser = argparse.ArgumentParser(description="Predict QOSHER scores for QASM circuits.")
    parser.add_argument("--tasks", type=str, help="Path to the holdout task list")
    parser.add_argument("--circuits", type=str, help="Directory containing QASM files")
    parser.add_argument("--id-map", type=str, help="JSON mapping from task id to QASM filename")
    parser.add_argument("--output", type=str, help="Path to output JSON file")
    args = parser.parse_args()

    # Load artifacts
    try:
        threshold_model = joblib.load('models/threshold_model.joblib')
        le_backend = joblib.load('models/le_backend.joblib')
        le_precision = joblib.load('models/le_precision.joblib')
        feature_cols = joblib.load('models/feature_cols.joblib')
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    tasks = json.load(open(args.tasks))['tasks']
    id_map = {entry["id"]: entry["qasm_file"] for entry in json.load(open(args.id_map))['entries']}
    
    output_json = []
    
    # Pre-cache features for unique circuits to save time
    unique_circuits = set(id_map.values())
    circuit_features = {}
    print(f"Extracting features for {len(unique_circuits)} circuits...")
    for qasm_file in unique_circuits:
        qasm_path = os.path.join(args.circuits, qasm_file)
        feats = extract_features(qasm_path)
        if feats:
            circuit_features[qasm_file] = feats

    for task in tasks:
        task_id = task["id"]
        qasm_file = id_map.get(task_id)
        
        if not qasm_file or qasm_file not in circuit_features:
            # Fallback
            output_json.append({
                "id": task_id,
                "predicted_threshold_min": 1,
                "predicted_forward_wall_s": 100.0
            })
            continue
            
        feats = circuit_features[qasm_file].copy()
        
        # Add task-specific features
        feats['backend_code'] = le_backend.transform([task['processor']])[0]
        feats['precision_code'] = le_precision.transform([task['precision']])[0]
        
        # Convert to DataFrame for prediction
        X_thresh = pd.DataFrame([feats])[feature_cols]
        
        # Predict Threshold with Confidence Logic
        probs = threshold_model.predict_proba(X_thresh)[0]
        classes = threshold_model.classes_
        top_idx = np.argmax(probs)
        pred_thresh = int(classes[top_idx])
        confidence = probs[top_idx]
        
        # Intelligent Scoring Optimization:
        # If confidence is low (< 60%) and there's a significant chance (> 20%) 
        # that the true threshold is higher, we bump to the next rung to avoid a 0 score.
        risk_of_underprediction = sum(probs[i] for i, c in enumerate(classes) if c > pred_thresh)
        
        if confidence < 0.6 and risk_of_underprediction > 0.2:
            pred_thresh = get_next_rung(pred_thresh)
        
        # Ensure it's in the ladder (failsafe)
        if pred_thresh not in THRESHOLD_LADDER:
            pred_thresh = min([r for r in THRESHOLD_LADDER if r >= pred_thresh] or [256])
        
        # Use a conservative baseline for runtime (e.g. 100s)
        pred_runtime = 100.0

        output_json.append({
            "id": task_id,
            "predicted_threshold_min": pred_thresh,
            "predicted_forward_wall_s": pred_runtime
        })

    # write to output file
    with open(args.output, "w") as f:
        json.dump(output_json, f, indent=2)
    
    print(f"Predictions saved to {args.output}")

if __name__ == "__main__":
    main()