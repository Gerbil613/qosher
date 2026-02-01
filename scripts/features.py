import json
import os
import argparse
import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
import warnings
import re

warnings.filterwarnings('ignore')

def build_interaction_graph(qc):
    """
    Builds graph where nodes are qubits and edges are 2-qubit gates.
    """
    G = nx.Graph()
    G.add_nodes_from(range(qc.num_qubits))
    
    for instruction in qc.data:
        qargs = instruction.qubits
        if len(qargs) == 2:
            u = qargs[0]._index
            v = qargs[1]._index
            if u != v:
                G.add_edge(u, v)            
    return G

def get_treewidth(G):
    """
    Estimates treewidth using the min-fill-in heuristic.
    This is a standard proxy for tensor network contraction complexity.
    """
    if G.number_of_nodes() == 0:
        return 0
    try:
        width, _ = nx.algorithms.approximation.treewidth_min_fill_in(G)
        return width
    except Exception:
        return 0

def get_spectral_gap(G):
    """
    Computes the spectral gap and algebraic connectivity of the Laplacian.
    """
    if G.number_of_nodes() < 2:
        return 0.0, 0.0
        
    try:
        spectrum = nx.laplacian_spectrum(G)
        
        # Algebraic Connectivity is the second smallest eigenvalue
        fiedler = spectrum[1] if len(spectrum) > 1 else 0.0
        
        # Spectral Gap: Difference between largest and second largest
        gap = spectrum[-1] - spectrum[-2] if len(spectrum) > 1 else 0.0
        
        return fiedler, gap
    except Exception:
        return 0.0, 0.0

def get_magic_density(qc):
    """
    Calculates the density of non-Clifford gates (T, Toffoli, rotations).
    """
    non_clifford_ops = {'t', 'tdg', 'ccx', 'rx', 'ry', 'rz', 'cp', 'u1', 'u2', 'u3'}
    count = 0
    for instr in qc.data:
        if instr.operation.name in non_clifford_ops:
            count += 1
            
    # Normalize by circuit volume (qubits * depth) or just gate count
    total_gates = len(qc.data)
    if total_gates == 0:
        return 0.0
    
    return count / total_gates

def extract_features(qasm_path):
    try:
        qc = QuantumCircuit.from_qasm_file(qasm_path)
    except Exception as e:
        print(f"Error reading {qasm_path}: {e}")
        return None

    features = {}
    
    # Basic Metrics
    features['n_qubits'] = qc.num_qubits
    features['n_gates'] = len(qc.data)
    features['depth'] = qc.depth()
    
    # Count specific gates
    ops = qc.count_ops()
    features['n_2q_gates'] = sum(ops[k] for k in ops if k in ['cx', 'cz', 'swap', 'ecr', 'rzz'])
    features['n_1q_gates'] = features['n_gates'] - features['n_2q_gates']
    
    G = build_interaction_graph(qc)
    
    features['treewidth_heuristic'] = get_treewidth(G)
    features['avg_degree'] = np.mean([d for _, d in G.degree()]) if G.number_of_nodes() > 0 else 0
    
    fiedler, gap = get_spectral_gap(G)
    features['algebraic_connectivity'] = fiedler
    features['spectral_gap'] = gap
    
    # Quantum-Specific Metrics
    features['magic_density'] = get_magic_density(qc)
    # Estimate 'active' volume (sum of qubits used in each layer)
    features['volume'] = qc.num_qubits * qc.depth()
    
    # Structural & Regex Features
    try:
        with open(qasm_path, 'r') as f:
            text = f.read()
        features['n_lines'] = sum(1 for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("//"))
        features['n_meas_regex'] = len(re.findall(r"\bmeasure\b", text))
        features['n_cx_regex'] = len(re.findall(r"\bcx\b", text))
        features['n_cz_regex'] = len(re.findall(r"\bcz\b", text))
    except Exception:
        features['n_lines'] = 0
        
    # Graph Structure
    if G.number_of_nodes() > 0:
        features['max_degree'] = max([d for _, d in G.degree()])
        features['clustering_coeff'] = nx.average_clustering(G)
        features['num_components'] = nx.number_connected_components(G)
        features['max_component_size'] = len(max(nx.connected_components(G), key=len)) if features['num_components'] > 0 else 0
    else:
        features['max_degree'] = 0
        features['clustering_coeff'] = 0
        features['num_components'] = 0
        features['max_component_size'] = 0
        
    # Span / Locality
    spans = []
    for instr in qc.data:
         if len(instr.qubits) == 2:
             q1 = instr.qubits[0]._index
             q2 = instr.qubits[1]._index
             spans.append(abs(q1 - q2))
             
    features['avg_span'] = np.mean(spans) if spans else 0
    features['max_span'] = np.max(spans) if spans else 0

    return features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from QASM circuits")
    parser.add_argument("--input_dir", required=True, help="Directory containing QASM files")
    parser.add_argument("--output", required=True, help="Output JSON file")
    
    args = parser.parse_args()
    
    results = {}
    
    if not os.path.exists(args.input_dir):
        print(f"Directory {args.input_dir} does not exist.")
        exit(1)
        
    files = [f for f in os.listdir(args.input_dir) if f.endswith('.qasm')]
    print(f"Found {len(files)} QASM files in {args.input_dir}")
    
    for f in files:
        path = os.path.join(args.input_dir, f)
        feats = extract_features(path)
        if feats:
            results[f] = feats
            
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Saved features for {len(results)} circuits to {args.output}")
