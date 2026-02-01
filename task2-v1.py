from scripts.features import *
import json
import matplotlib.pyplot as plt
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile
from qiskit.transpiler import InstructionDurations, CouplingMap
import networkx as nx

with open("data/hackathon_public.json", "r") as f:
    data = json.load(f)

filename = "circuits/grover-v-chain_indep_qiskit_7.qasm"
circuit = QuantumCircuit.from_qasm_file(filename)

# qreg_q = QuantumRegister(3, 'q')
# creg_c = ClassicalRegister(4, 'c')
# circuit = QuantumCircuit(qreg_q, creg_c)

# circuit.cx(qreg_q[0], qreg_q[1])
# circuit.h(qreg_q[1])
# circuit.cx(qreg_q[1], qreg_q[2])

# Define allowed gates and their durations
gate_specs = [
    ('h', 1),
    ('cx', 1),
    ('u3', 1),
    ('u', 1),
    ('u1', 1),
    ('u2', 1),
    ('p', 1),
    ('rx', 1),
    ('ry', 1),
    ('x', 1),
    ('cz', 1),
    ('swap', 1),
    ('rzz', 1),
    ('cu1', 1),
    ('cp', 1),
    ('cry', 1),
    ('rccx', 1),
    ('ccx', 1),
    ('cswap', 1),
]

valid_gate_names = {name for name, _ in gate_specs}

# Filter circuit to remove unsupported operations (e.g. measure, barrier)
new_circuit = QuantumCircuit(circuit.qubits, circuit.clbits)
for instr in circuit.data:
    if instr.operation.name in valid_gate_names:
        new_circuit.append(instr.operation, instr.qubits, instr.clbits)

circuit = new_circuit

durations = InstructionDurations(
    [(name, None, duration) for name, duration in gate_specs]
)

coupling_map = CouplingMap.from_full(circuit.num_qubits)

scheduled_qc = transpile(circuit, scheduling_method='alap', instruction_durations=durations, basis_gates=list(valid_gate_names), coupling_map=coupling_map, optimization_level=0)

print(f"Total duration: {scheduled_qc.duration} dt")
