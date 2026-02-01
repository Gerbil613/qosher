import json
import os
import glob
from qiskit import QuantumCircuit

def get_unique_gates(circuit_dir):
    files = glob.glob(os.path.join(circuit_dir, "*.qasm"))
    all_ops = set()
    for f in files:
        try:
            qc = QuantumCircuit.from_qasm_file(f)
            all_ops.update(qc.count_ops().keys())
        except Exception as e:
            print(f"Error reading {f}: {e}")
    return all_ops

def calculate_duration(qc, gate_specs):
    # Map from qubit index to the time it becomes free
    qubit_finish_times = {q: 0.0 for q in qc.qubits}
    
    # Iterate over instructions in topological order (which qc.data typically is)
    for instruction in qc.data:
        op = instruction.operation
        qubits = instruction.qubits
        
        gate_name = op.name
        duration = gate_specs.get(gate_name, 1.0) # Default to 1.0 if unknown
        
        # Find start time: max finish time of involved qubits
        start_time = 0.0
        for q in qubits:
            start_time = max(start_time, qubit_finish_times[q])
            
        end_time = start_time + duration
        
        # Update finish times
        for q in qubits:
            qubit_finish_times[q] = end_time
            
    return max(qubit_finish_times.values()) if qubit_finish_times else 0.0

if __name__ == "__main__":
    circuit_dir = "circuits"
    print("Finding unique gates...")
    gates = get_unique_gates(circuit_dir)
    print("Unique gates:", gates)
    
    # Define dummy specs
    gate_specs = {g: 1.0 for g in gates}
    
    # Test on one circuit
    test_circuit = "circuits/ae_indep_qiskit_20.qasm"
    if os.path.exists(test_circuit):
        qc = QuantumCircuit.from_qasm_file(test_circuit)
        duration = calculate_duration(qc, gate_specs)
        print(f"Duration of {test_circuit} (all gates=1.0): {duration}")
    else:
        print(f"Test circuit {test_circuit} not found.")
