import json
import os
import glob
import random
import copy
from qiskit import QuantumCircuit
import networkx as nx

def build_graph(circuit, current_specs):
    # This rebuilds the graph structure every time, or at least updates weights
    # To be efficient, we should build the structure once and only update weights,
    # but for simplicity let's stick to the structure requested.
    # To optimize this, later we can just update edge weights.
    
    graph = nx.DiGraph()
    n = circuit.num_qubits
    graph.add_nodes_from(range(n))
    
    current_nodes = list(range(n))
    next_node = n

    for instruction in circuit.data:
        op_name = instruction.operation.name
        qargs = instruction.qubits
        
        # Get weight or default
        weight = current_specs.get(op_name, 0.02)
        
        if len(qargs) == 1:
            try:
                # Use find_bit for Qiskit 1.0 compatibility
                u_idx = circuit.find_bit(qargs[0]).index
            except AttributeError:
                # Fallback for older Qiskit
                u_idx = qargs[0].index
            
            u_in = current_nodes[u_idx]
            u_out = next_node
            next_node += 1
            
            # Using POSITIVE weights for standard max-path algorithms
            # The original code used negative weights to find min path (most negative).
            # We will use positive weights and find the longest path (max weight).
            graph.add_edge(u_in, u_out, weight=weight)
            
            current_nodes[u_idx] = u_out

        elif len(qargs) >= 2:
            # Handle multi-qubit gates (mostly 2-qubit)
            indices = []
            for q in qargs:
                try:
                    indices.append(circuit.find_bit(q).index)
                except AttributeError:
                    indices.append(q.index)
            
            # Create a central node for the gate interaction
            # Inputs -> Mid -> Outputs
            mid_node = next_node
            next_node += 1
            
            # Distribute weight? 
            # Original code split weight by 2 for edges involving mid node?
            # It did: u_in->mid (-w/2), v_in->mid (-w/2), mid->u_out (-w/2), mid->v_out (-w/2)
            # This results in a path u_in -> mid -> u_out having total weight -w.
            # So effectively the gate adds 'w' to the path length on any wire involved.
            # We will replicate this with positive weights.
            
            w_segment = weight / 2.0
            
            gate_out_nodes = []
            for _ in indices:
                gate_out_nodes.append(next_node)
                next_node += 1
                
            for i, idx in enumerate(indices):
                u_in = current_nodes[idx]
                u_out = gate_out_nodes[i]
                
                graph.add_edge(u_in, mid_node, weight=w_segment)
                graph.add_edge(mid_node, u_out, weight=w_segment)
                
                current_nodes[idx] = u_out

    return graph

def get_critical_path_duration(graph):
    # Calculate the longest path in the DAG.
    # Since it's a DAG, we can use dag_longest_path_length.
    # Note: Using positive weights here.
    if graph.number_of_nodes() == 0:
        return 0.0
    try:
        return nx.dag_longest_path_length(graph, weight='weight')
    except nx.NetworkXUnfeasible:
        # Fallback if cycles exist (should not happen in circuit)
        print("Cycle detected!")
        return 0.0



def load_data():
    data_path = "data/hackathon_public.json"
    if not os.path.exists(data_path):
        print("Data file not found.")
        return []
        
    with open(data_path, "r") as f:
        data = json.load(f)
        
    training_data = []
    if 'results' in data:
        for res in data['results']:
            circuit_file = res.get('file')
            # Look for threshold_sweep results
            if circuit_file and 'threshold_sweep' in res:
                 for point in res['threshold_sweep']:
                     if 'threshold' in point and 'run_wall_s' in point:
                         t = point['threshold']
                         target = point['run_wall_s']
                         if target is None:
                             continue
                         
                         training_data.append({
                            'file': circuit_file,
                            'threshold': t,
                            'target': target
                        })
    return training_data

def precompute_graphs(training_data):
    cached_data = [] # List of (G, edges_by_gate, topo, gate_set, threshold, target)
    
    unique_files = set(d['file'] for d in training_data)
    print(f"Pre-computing graphs for {len(unique_files)} unique circuits...")
    
    circuit_map = {} # fname -> (G, edges_by_gate, topo_order, gate_set)
    
    for fname in unique_files:
        path = os.path.join("circuits", fname)
        if not os.path.exists(path):
            continue
            
        try:
            qc = QuantumCircuit.from_qasm_file(path)
            
            G = nx.DiGraph()
            n = qc.num_qubits
            G.add_nodes_from(range(n))
            
            current_nodes = list(range(n))
            next_node = n
            
            # gate_specs is not defined here, assuming it's a global or imported dict
            # For precompute, we just need to know which gates are present
            # and map edges to them.
            # Let's define a dummy gate_specs for this scope if it's not global
            # or assume it's available.
            # Based on the original code, gate_specs is used in learn_weights,
            # so here we just need to collect gates.
            
            # To make this self-contained for precompute, we need a list of known gates
            # or dynamically add to edges_by_gate. The latter is safer.
            edges_by_gate = {} # Dynamically add gates
            present_gates = set()
            
            for instruction in qc.data:
                op_name = instruction.operation.name
                qargs = instruction.qubits
                present_gates.add(op_name)
                
                if op_name not in edges_by_gate:
                    edges_by_gate[op_name] = []

                if len(qargs) == 1:
                    try:
                        u_idx = qc.find_bit(qargs[0]).index
                    except AttributeError:
                        u_idx = qargs[0].index
                    
                    u_in = current_nodes[u_idx]
                    u_out = next_node
                    next_node += 1
                    
                    G.add_edge(u_in, u_out)
                    edges_by_gate[op_name].append((u_in, u_out))
                    current_nodes[u_idx] = u_out

                elif len(qargs) >= 2:
                    indices = []
                    for q in qargs:
                        try:
                            indices.append(qc.find_bit(q).index)
                        except AttributeError:
                            indices.append(q.index)
                    
                    mid_node = next_node
                    next_node += 1
                    
                    indices_out_nodes = []
                    for _ in indices:
                        indices_out_nodes.append(next_node)
                        next_node += 1
                        
                    for i, idx in enumerate(indices):
                        u = current_nodes[idx]
                        v = indices_out_nodes[i]
                        G.add_edge(u, mid_node)
                        edges_by_gate[op_name].append((u, mid_node))
                        G.add_edge(mid_node, v)
                        edges_by_gate[op_name].append((mid_node, v))
                        current_nodes[idx] = v
            
            try:
                topo_order = list(nx.topological_sort(G))
            except nx.NetworkXUnfeasible:
                print(f"Cycle needed in {fname}, skipping topo sort optimization.")
                topo_order = []
                
            circuit_map[fname] = (G, edges_by_gate, topo_order, present_gates)
            
        except Exception as e:
            print(f"Error parsing {fname}: {e}")
            
    # Link data 
    for item in training_data:
        fname = item['file']
        if fname in circuit_map:
            # item has threshold, target
            cached_data.append({
                'fname': fname,
                'graph_data': circuit_map[fname],
                'threshold': item['threshold'],
                'target': item['target']
            })
            
    return cached_data

def update_graph_weights(G, edges_by_gate, specs, gate_to_update=None):
    gates_loop = [gate_to_update] if gate_to_update else edges_by_gate.keys()
    multi_qubit_gates = {'cx', 'cz', 'swap', 'rzz', 'cu1', 'cp', 'cry', 'rccx', 'ccx', 'cswap'}
    
    for gate in gates_loop:
        if gate not in edges_by_gate or not edges_by_gate[gate]:
            continue
        weight = specs.get(gate, 0.0)
        final_weight = weight / 2.0 if gate in multi_qubit_gates else weight
        for u, v in edges_by_gate[gate]:
            G[u][v]['weight'] = final_weight

def fast_longest_path(G, topo_order):
    if not topo_order:
        return nx.dag_longest_path_length(G, weight='weight')
    dist = {node: 0.0 for node in G}
    for u in topo_order:
        d_u = dist[u]
        for v in G[u]:
            w = G[u][v].get('weight', 0.0)
            if d_u + w > dist[v]:
                dist[v] = d_u + w
    return max(dist.values()) if dist else 0.0

# Define gate_specs globally or pass it around.
# Assuming a default gate_specs for initialization if not loaded.
gate_specs = {
    'id': 0.01, 'u1': 0.01, 'u2': 0.01, 'u3': 0.01, 'x': 0.01, 'y': 0.01, 'z': 0.01,
    'h': 0.01, 's': 0.01, 'sdg': 0.01, 't': 0.01, 'tdg': 0.01, 'rx': 0.01, 'ry': 0.01,
    'rz': 0.01, 'sx': 0.01, 'sxdg': 0.01, 'p': 0.01, 'r': 0.01, 'reset': 0.01,
    'cx': 0.05, 'cy': 0.05, 'cz': 0.05, 'swap': 0.05, 'ccx': 0.05, 'cswap': 0.05,
    'cu1': 0.05, 'cu3': 0.05, 'rxx': 0.05, 'ryy': 0.05, 'rzz': 0.05, 'rzx': 0.05,
    'ecr': 0.05, 'dcx': 0.05, 'iswap': 0.05, 'ms': 0.05, 'rccx': 0.05, 'rc3x': 0.05,
    'cp': 0.05, 'crx': 0.05, 'cry': 0.05, 'crz': 0.05, 'csx': 0.05, 'mcx': 0.05,
    'mcy': 0.05, 'mcz': 0.05, 'mcrx': 0.05, 'mcry': 0.05, 'mcrz': 0.05, 'mcsx': 0.05,
    'mcu1': 0.05, 'mcu3': 0.05, 'mcswap': 0.05, 'barrier': 0.0
}


def learn_weights():
    # Load and Precompute
    print("Loading data...")
    training_data = load_data()
    print(f"Loaded {len(training_data)} training examples.")
    
    print("Pre-computing optimization graphs...")
    cached_training_data = precompute_graphs(training_data)
    print(f"Ready to train on {len(cached_training_data)} examples.")
    
    # Initialize specs
    alpha = 2.0  # Default initialization
    
    if os.path.exists("gate_specs.json"):
        print("Loading existing gate specs from gate_specs.json...")
        with open("gate_specs.json", "r") as f:
            data = json.load(f)
            # data could be just specs or {'specs': ..., 'alpha': ...}
            # Handle backward compatibility
            if 'alpha' in data:
               current_specs = data['specs']
               alpha = data['alpha']
            else:
               current_specs = data 
               # Ensure keys
    else:
        print("Initializing random gate specs...")
        current_specs = gate_specs.copy()
        
    # Ensure current_specs has all keys
    for op in gate_specs:
        if op not in current_specs:
             current_specs[op] = random.uniform(0.05, 0.20)
        # If freshly initialized from old file, might need random init for new keys
        if current_specs[op] == 0.0 and op != 'barrier': # Just in case
             current_specs[op] = random.uniform(0.05, 0.20)

    # Map gate -> indices
    gate_to_indices = {g: [] for g in current_specs}
    for idx, item in enumerate(cached_training_data):
        present_gates = item['graph_data'][3]
        for g in present_gates:
            if g in gate_to_indices:
                gate_to_indices[g].append(idx)

    # Initial Pass - Calculate Path Lengths ONLY (unscaled)
    # We need to distinguish between raw path length (sum of gate times) and final prediction
    # Pred = Path * log2(threshold) / alpha
    
    current_path_lengths = []
    
    print("Calculating initial error...")
    total_error = 0.0
    
    for item in cached_training_data:
        G, edges_map, topo, _ = item['graph_data']
        update_graph_weights(G, edges_map, current_specs)
        path_len = fast_longest_path(G, topo)
        current_path_lengths.append(path_len)
        
        # Calculate Prediction
        import math
        threshold = item['threshold']
        if threshold <= 0: threshold = 1 # Safety
        log_t = math.log2(threshold)
        
        # Avoid division by zero
        safe_alpha = alpha if abs(alpha) > 1e-6 else 1e-6
        pred = path_len * log_t / safe_alpha
        
        total_error += abs(pred - item['target'])
        
    print(f"Initial Error: {total_error}")
    
    keys = sorted([k for k in current_specs.keys() if k != 'barrier'])
    
    # Optimizer Params
    max_total_iter = 3000
    total_iter = 0
    initial_lr = 0.5
    
    # Helper to save
    def save_state():
        with open("gate_specs.json", "w") as f:
            json.dump({'specs': current_specs, 'alpha': alpha}, f, indent=2)

    # Helper for Alpha Error
    def calc_error_for_alpha(test_alpha):
        err = 0.0
        safe_a = test_alpha if abs(test_alpha) > 1e-6 else 1e-6
        for idx, item in enumerate(cached_training_data):
            path_len = current_path_lengths[idx]
            threshold = item['threshold']
            log_t = math.log2(threshold) if threshold > 0 else 0
            pred = path_len * log_t / safe_a
            err += abs(pred - item['target'])
        return err

    # Helper for Gate Error
    def calc_partial_gate(gate, val, current_total_error):
        indices = gate_to_indices.get(gate, [])
        current_specs[gate] = val
        
        partial_err_diff = 0.0
        new_path_lens = {}
        
        safe_alpha = alpha if abs(alpha) > 1e-6 else 1e-6

        for idx in indices:
            item = cached_training_data[idx]
            G, edges_map, topo, _ = item['graph_data']
            target = item['target']
            threshold = item['threshold']
            log_t = math.log2(threshold) if threshold > 0 else 0

            # Update Graph
            update_graph_weights(G, edges_map, current_specs, gate_to_update=gate)
            
            # New Path Length
            new_path_len = fast_longest_path(G, topo)
            new_path_lens[idx] = new_path_len
            
            # Prediction
            old_pred = current_path_lengths[idx] * log_t / safe_alpha
            new_pred = new_path_len * log_t / safe_alpha
            
            old_err = abs(old_pred - target)
            new_err = abs(new_pred - target)
            
            partial_err_diff += (new_err - old_err)
            
        return current_total_error + partial_err_diff, new_path_lens


    print("Starting optimization loop (Coordinate Descent)...")
    
    while total_iter < max_total_iter:
        # Include 'ALPHA' in optimization, but strictly optimize it at the start 
        # AND after every gate change? 
        # User asked for "alpha learn as well in every episode".
        # Let's iterate gates, but optimize alpha after each gate improvement.
        
        # Optimize Alpha First
        while True:
            progress = total_iter / max_total_iter
            lr = initial_lr * (1 - 0.9 * progress) * 0.1
            val_up = alpha + alpha * lr
            val_down = alpha - alpha * lr
            err_up = calc_error_for_alpha(val_up)
            err_down = calc_error_for_alpha(val_down)
            
            improved_alpha = False
            if err_up < total_error and err_up < err_down:
                total_error = err_up
                alpha = val_up
                improved_alpha = True
                print(f"Iter {total_iter}: ALPHA IMPROVED (Up) to {alpha:.5f}. Error: {total_error:.4f}")
                save_state()
            elif err_down < total_error and err_down <= err_up:
                total_error = err_down
                alpha = val_down
                improved_alpha = True
                print(f"Iter {total_iter}: ALPHA IMPROVED (Down) to {alpha:.5f}. Error: {total_error:.4f}")
                save_state()
            if not improved_alpha: break
        
        random.shuffle(keys)
        any_improvement_in_pass = False
        
        for gate in keys:
            if total_iter >= max_total_iter: break
            
            indices = gate_to_indices.get(gate, [])
            if not indices: continue
            
            while True:
                if total_iter >= max_total_iter: break
                progress = total_iter / max_total_iter
                lr = initial_lr * (1 - 0.9 * progress)
                lr = max(lr, 0.001)
                
                original_val = current_specs[gate]
                perturb = original_val * lr
                val_up = original_val + perturb
                val_down = max(1e-6, original_val - perturb)
                
                err_up, paths_up = calc_partial_gate(gate, val_up, total_error)
                err_down, paths_down = calc_partial_gate(gate, val_down, total_error)
                
                improved = False
                
                if err_up < total_error and err_up < err_down:
                    total_error = err_up
                    current_specs[gate] = val_up
                    for idx, p in paths_up.items():
                        current_path_lengths[idx] = p
                    for idx in indices:
                         G, edges_map, _, _ = cached_training_data[idx]['graph_data']
                         update_graph_weights(G, edges_map, current_specs, gate_to_update=gate)
                    
                    # Update Bias output logic
                    bias_sum = 0
                    for r in range(len(cached_training_data)):
                         th = cached_training_data[r]['threshold']
                         lt = math.log2(th) if th > 0 else 0
                         pr = current_path_lengths[r] * lt / alpha
                         bias_sum += (pr - cached_training_data[r]['target'])
                    
                    print(f"Iter {total_iter}: {gate} IMPROVED (Up) to {val_up:.5f}. Error: {total_error:.4f} Bias: {bias_sum:.4f}")
                    save_state()
                    improved = True
                    
                elif err_down < total_error and err_down <= err_up:
                    total_error = err_down
                    current_specs[gate] = val_down
                    for idx, p in paths_down.items():
                        current_path_lengths[idx] = p
                    for idx in indices:
                         G, edges_map, _, _ = cached_training_data[idx]['graph_data']
                         update_graph_weights(G, edges_map, current_specs, gate_to_update=gate)
                    
                    bias_sum = 0
                    for r in range(len(cached_training_data)):
                         th = cached_training_data[r]['threshold']
                         lt = math.log2(th) if th > 0 else 0
                         pr = current_path_lengths[r] * lt / alpha
                         bias_sum += (pr - cached_training_data[r]['target'])

                    print(f"Iter {total_iter}: {gate} IMPROVED (Down) to {val_down:.5f}. Error: {total_error:.4f} Bias: {bias_sum:.4f}")
                    save_state()
                    improved = True
                
                else:
                    current_specs[gate] = original_val
                    for idx in indices:
                        G, edges_map, _, _ = cached_training_data[idx]['graph_data']
                        update_graph_weights(G, edges_map, current_specs, gate_to_update=gate)
                
                if improved:
                    any_improvement_in_pass = True
                    total_iter += 1
                    
                    # --- AUTO OPTIMIZE ALPHA AFTER GATE CHANGE ---
                    while True:
                         val_up = alpha + alpha * 0.05
                         val_down = alpha - alpha * 0.05
                         err_up_a = calc_error_for_alpha(val_up)
                         err_down_a = calc_error_for_alpha(val_down)
                         
                         imp_a = False
                         if err_up_a < total_error and err_up_a < err_down_a:
                             total_error = err_up_a
                             alpha = val_up
                             imp_a = True
                             print(f"Iter {total_iter}: ALPHA ADAPTED (Up) to {alpha:.5f}. Error: {total_error:.4f}")
                             save_state()
                         elif err_down_a < total_error and err_down_a < err_up_a:
                             total_error = err_down_a
                             alpha = val_down
                             imp_a = True
                             print(f"Iter {total_iter}: ALPHA ADAPTED (Down) to {alpha:.5f}. Error: {total_error:.4f}")
                             save_state()
                         if not imp_a: break
                    # ---------------------------------------------
                else:
                    break
        
        if not any_improvement_in_pass:
             print("Convergence reached.")
             break
             
    print("Final Error:", total_error)
    save_state()
    
if __name__ == "__main__":
    learn_weights()
