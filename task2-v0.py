from scripts.features import *
import json
import matplotlib.pyplot as plt
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
import networkx as nx

gate_specs = {
    'h': 1,
    'cx': 1,
    'u3': 1,
    'u': 1,
    'u1': 1,
    'u2': 1,
    'p': 1,
    'rx': 1,
    'ry': 1,
    'x': 1,
    'cz': 1,
    'swap': 1,
    'rzz': 1,
    'cu1': 1,
    'cp': 1,
    'cry': 1,
    'rccx': 1,
    'ccx': 1,
    'cswap': 1,
}

def build_graph(circuit):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(circuit.num_qubits))
    
    current_nodes = list(range(circuit.num_qubits))
    next_node = circuit.num_qubits

    for instruction in circuit.data:
        qargs = instruction.qubits
        if len(qargs) == 1:
            u_idx = qargs[0]._index
            weight = gate_specs[instruction.operation.name]
            
            u_in = current_nodes[u_idx]
            u_out = next_node
            next_node += 1
            
            graph.add_edge(u_in, u_out, weight=weight)
            
            current_nodes[u_idx] = u_out

        if len(qargs) == 2:
            # get the type of gate and set the weights equal to the gate_specs
            gate_type = instruction.operation.name
            weight = gate_specs[gate_type]
            u_idx = qargs[0]._index
            v_idx = qargs[1]._index
            
            if u_idx != v_idx:
                u_in = current_nodes[u_idx]
                v_in = current_nodes[v_idx]
                
                u_out = next_node
                next_node += 1
                v_out = next_node
                next_node += 1
                
                mid_node = next_node
                next_node += 1
                
                # Directed edges: Inputs -> Middle -> Outputs
                graph.add_edge(u_in, mid_node, weight=weight/2)
                graph.add_edge(v_in, mid_node, weight=weight/2)
                graph.add_edge(mid_node, u_out, weight=weight/2)
                graph.add_edge(mid_node, v_out, weight=weight/2)
                
                current_nodes[u_idx] = u_out
                current_nodes[v_idx] = v_out

    # pos = nx.spring_layout(graph)
    # nx.draw(graph, pos, with_labels=True)
    # edge_labels = {
    #     (u, v): d['weight'] for u, v, d in graph.edges(data=True)
    # }

    # # 5. Draw the edge labels
    # nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)   
    # plt.show()       
    return graph

with open("data/hackathon_public.json", "r") as f:
    data = json.load(f)

filename = "circuits/grover-v-chain_indep_qiskit_7.qasm"
#circuit = QuantumCircuit.from_qasm_file(filename)

qreg_q = QuantumRegister(3, 'q')
creg_c = ClassicalRegister(4, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.cx(qreg_q[0], qreg_q[1])
circuit.h(qreg_q[1])
circuit.cx(qreg_q[1], qreg_q[2])

graph = build_graph(circuit)
path = dict(nx.algorithms.shortest_paths.all_pairs_shortest_path(graph))
print(path)
for i in range(graph.number_of_nodes()):    
    for j in range(graph.number_of_nodes()):
        try:
            print(path[i][j])
        except KeyError:
            print(f"No path from {i} to {j}")
        try:
            print(path[j][i])
        except KeyError:
            print(f"No path from {j} to {i}")