import pennylane as qml
import os
import sys
import json

try:
    from pennylane_qiskit.converter import circuit_to_qiskit
except ImportError:
    sys.exit(1)

# Plantilla que simula ser un código generado por el translator del scheduler
TEMPLATE_FINAL = """
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile
from qiskit_aer import AerSimulator
import json
import numpy as np

# Reconstruimos el circuito desde QASM incrustado
# Esto es infalible porque evita problemas de sintaxis de Python
qasm_data = \"\"\"{qasm_string}\"\"\"

try:
    # Qiskit < 1.0
    circuit = QuantumCircuit.from_qasm_str(qasm_data)
except AttributeError:
    # Qiskit >= 1.0
    from qiskit import qasm2
    circuit = qasm2.loads(qasm_data)

# Ejecución estándar
backend = AerSimulator()
qc_compiled = transpile(circuit, backend)
job = backend.run(qc_compiled, shots={shots})
result = job.result()
counts = result.get_counts()

print(json.dumps(counts))
"""

def tape_to_qiskit_script(tape, filename, shots=1024):
    try:
        # 1. Expansión y Limpieza
        expanded_tape = tape.expand()
        for op in expanded_tape.operations:
            new_params = []
            if hasattr(op, 'data'):
                for p in op.data:
                    if hasattr(p, 'detach'):
                        val = p.detach().cpu().numpy()
                        new_params.append(float(val) if val.ndim == 0 else val)
                    else:
                        new_params.append(p)
                op.data = tuple(new_params)

        # 2. Convertir a Qiskit
        num_qubits = len(expanded_tape.wires)
        qc = circuit_to_qiskit(expanded_tape, num_qubits)
        
        # 3. Obtener QASM
        try:
            from qiskit import qasm2
            qasm_str = qasm2.dumps(qc)
        except ImportError:
            qasm_str = qc.qasm()

        # 4. Generar Script Final
        # Aquí está la magia: incrustamos el QASM dentro del script .py
        # Así el Scheduler recibe un .py válido que sabe ejecutarse a sí mismo.
        full_script = TEMPLATE_FINAL.format(qasm_string=qasm_str, shots=shots)
        
        os.makedirs("generated_circuits", exist_ok=True)
        filepath = os.path.join("generated_circuits", filename)
        with open(filepath, "w") as f:
            f.write(full_script)
            
        return filepath
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None