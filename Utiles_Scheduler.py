import pennylane as qml
import os
import sys
import json

# --- IMPORTACIÓN ESPECÍFICA PARA TU VERSIÓN ---
try:
    from pennylane_qiskit.converter import circuit_to_qiskit
except ImportError:
    print("❌ ERROR CRÍTICO: No se encuentra 'circuit_to_qiskit'.")
    sys.exit(1)

# Plantilla que se enviará al Scheduler
TEMPLATE_QISKIT = """
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import json

# --- 1. RECONSTRUIR CIRCUITO DESDE QASM ---
qasm_code = \"\"\"{qasm_string}\"\"\"
circuit = QuantumCircuit.from_qasm_str(qasm_code)

# --- 2. EJECUCIÓN ---
backend = AerSimulator()
qc_compiled = transpile(circuit, backend)

# Ejecución
job = backend.run(qc_compiled, shots={shots})
result = job.result()
counts = result.get_counts()

# Salida JSON estándar para tu Scheduler
print(json.dumps(counts))
"""

def tape_to_qiskit_script(tape, filename, shots=1024):
    """
    Convierte un QuantumTape de PennyLane a un script .py de Qiskit.
    """
    try:
        # 1. Obtener el número de qubits (register_size)
        # El tape sabe cuántos cables se están usando.
        num_qubits = len(tape.wires)
        
        # 2. Convertir Tape a objeto QuantumCircuit de Qiskit
        # AHORA PASAMOS EL SEGUNDO ARGUMENTO OBLIGATORIO
        qiskit_circuit = circuit_to_qiskit(tape, num_qubits)
        
        # 3. Obtener string QASM
        qasm_str = qiskit_circuit.qasm()
        
        # 4. Rellenar plantilla
        full_script = TEMPLATE_QISKIT.format(
            qasm_string=qasm_str,
            shots=shots
        )
        
        # 5. Guardar en disco
        os.makedirs("generated_circuits", exist_ok=True)
        filepath = os.path.join("generated_circuits", filename)
        with open(filepath, "w") as f:
            f.write(full_script)
            
        return filepath
        
    except Exception as e:
        print(f"❌ Error convirtiendo tape {filename}: {e}")
        import traceback
        traceback.print_exc()
        return None