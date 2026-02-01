import argparse
import json
from qiskit import qasm2
import os

def main():
    parser = argparse.ArgumentParser(description="Predict QOSHER scores for QASM circuits.")
    parser.add_argument("--tasks", type=str, help="Path to the public holdout task list (IDs + CPU/GPU + precision)")
    parser.add_argument("--circuits", type=str, help="Directory containing hidden holdout QASM files (provided by organizers)")
    parser.add_argument("--id-map", type=str, help="JSON mapping from public task id to QASM filename")
    parser.add_argument("--output", type=str, help="Path to output JSON file")
    args = parser.parse_args()

    tasks = json.load(open(args.tasks))['tasks']
    id_map = {entry["id"]: entry["qasm_file"] for entry in json.load(open(args.id_map))['entries']}
    
    output_json = []
    for task in tasks:
        # get qasm file from id_map
        qasm_file = id_map[task["id"]]
        qasm_path = os.path.join(args.circuits, qasm_file)
        circuit = qasm2.load(qasm_path)

        output_json.append({
            "id": task["id"],
            "predicted_threshold_min": 1,
            "predicted_forward_wall_s": 100
        })

    # write to output file
    with open(args.output, "w") as f:
        json.dump(output_json, f)

    
if __name__ == "__main__":
    main()