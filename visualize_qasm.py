import argparse
from qiskit import qasm2
from qiskit.circuit import QuantumCircuit
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Visualize a QASM file using Qiskit.")
    parser.add_argument("file", help="Path to the QASM file to visualize.")
    parser.add_argument("--output", "-o", default="text", help="Output visualization style (e.g., text, mpl, latex). Default is 'text'.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found.")
        sys.exit(1)

    try:
        # Try loading directly
        try:
            circuit = qasm2.load(args.file)
        except Exception as qasm2_error:
            # If it fails, it might be due to missing 'u' gate or other issues.
            # We will try to read the content and inject the 'u' gate definition 
            # which is often missing in some export formats but used as U3.
            # We also ensure qelib1.inc resolution if possible, although loads() 
            # should handle it if present.
            
            with open(args.file, 'r') as f:
                qasm_content = f.read()
            
            # Simple heuristic: if 'u(' is used but not defined, inject it.
            # Definition: gate u(theta,phi,lambda) q { U(theta,phi,lambda) q; }
            # We prepend it after 'OPENQASM 2.0;'.
            
            header = "OPENQASM 2.0;"
            if header in qasm_content and "gate u(" not in qasm_content:
                # Inject u definition
                u_def = "\ngate u(theta,phi,lambda) q { U(theta,phi,lambda) q; }\n"
                # Insert after the header and include
                # Find the position of 'include "qelib1.inc";' or header
                if 'include "qelib1.inc";' in qasm_content:
                    idx = qasm_content.find('include "qelib1.inc";') + len('include "qelib1.inc";')
                    qasm_content = qasm_content[:idx] + u_def + qasm_content[idx:]
                else:
                    idx = qasm_content.find(header) + len(header)
                    qasm_content = qasm_content[:idx] + u_def + qasm_content[idx:]
            
            # Qiskit doesn't automatically provide it for qasm2.load unless in path.
            # We can try to rely on the fact that if it failed above, we are retrying.
            
            try:
                circuit = qasm2.loads(qasm_content)
            except Exception as qasm2_error:
                # If qasm2 fails (e.g. undefined 'cp' or 'u'), try the legacy parser
                print(f"qasm2 parser failed ({qasm2_error}), trying legacy parser...")
                try:
                    circuit = QuantumCircuit.from_qasm_file(args.file)
                except Exception as legacy_error:
                    # If legacy also fails, raise the original or new error
                    raise Exception(f"Both parsers failed. qasm2: {qasm2_error}, legacy: {legacy_error}")
        
        print(f"Visualizing '{args.file}' with style '{args.output}':")
        print(circuit.draw(output=args.output))
        
    except Exception as e:
        print(f"Error processing QASM file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
