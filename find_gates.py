from qiskit import QuantumCircuit
import os
import glob

circuit_dir = "circuits"
files = glob.glob(os.path.join(circuit_dir, "*.qasm"))

all_ops = set()

for f in files:
    try:
        qc = QuantumCircuit.from_qasm_file(f)
        ops = qc.count_ops()
        all_ops.update(ops.keys())
    except Exception as e:
        print(f"Error reading {f}: {e}")

print("All found operations:", all_ops)
