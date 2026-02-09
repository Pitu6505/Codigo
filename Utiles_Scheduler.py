import pennylane as qml
import os
import sys
import json
import numpy as np

# --- IMPORTACIÓN ESPECÍFICA PARA TU VERSIÓN ---
try:
    from pennylane_qiskit.converter import circuit_to_qiskit
except ImportError:
    print("❌ ERROR CRÍTICO: No se encuentra 'circuit_to_qiskit'.")
    sys.exit(1)

# Importamos QASM2 explícitamente para Qiskit 1.0+
try:
    from qiskit import qasm2
except ImportError:
    # Fallback para versiones viejas (aunque tu error dice que estás en la nueva)
    qasm2 = None

# Plantilla que se enviará al Scheduler (Actualizada para Qiskit 1.0)
TEMPLATE_QISKIT = """
from qiskit import QuantumCircuit, transpile
from qiskit import qasm2
from qiskit_aer import AerSimulator
import json

# --- 1. RECONSTRUIR CIRCUITO DESDE QASM ---
# En Qiskit 1.0 se usa qasm2.loads
qasm_code = \"\"\"{qasm_string}\"\"\"
circuit = qasm2.loads(qasm_code)

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
    Convierte un QuantumTape de PennyLane a un script .py de Qiskit 1.0+.
    Incluye limpieza de Tensores y uso de QASM2.
    """
    try:
        # 1. EXPANSIÓN
        expanded_tape = tape.expand()

        # 2. LIMPIEZA DE TENSORES
        # Convertimos torch.Tensor a float python puro
        for op in expanded_tape.operations:
            new_params = []
            if hasattr(op, 'data'): # Protección extra
                for p in op.data:
                    if hasattr(p, 'detach'):
                        val = p.detach().cpu().numpy()
                        if val.ndim == 0:
                            new_params.append(float(val))
                        else:
                            new_params.append(val)
                    else:
                        new_params.append(p)
                # Reasignamos usando tuple()
                op.data = tuple(new_params)

        # 3. Obtener el número de qubits
        num_qubits = len(expanded_tape.wires)
        
        # 4. Convertir Tape limpio a Qiskit
        qiskit_circuit = circuit_to_qiskit(expanded_tape, num_qubits)
        
        # 5. Obtener string QASM (Adaptado a Qiskit 1.0)
        if qasm2:
            qasm_str = qasm2.dumps(qiskit_circuit)
        else:
            # Fallback por si acaso
            try:
                qasm_str = qiskit_circuit.qasm()
            except AttributeError:
                raise ImportError("Necesitas qiskit >= 1.0 con soporte qasm2, o volver a qiskit 0.46")
        
        # 6. Rellenar plantilla
        full_script = TEMPLATE_QISKIT.format(
            qasm_string=qasm_str,
            shots=shots
        )
        
        # 7. Guardar en disco
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