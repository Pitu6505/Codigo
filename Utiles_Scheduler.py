import pennylane as qml
import os
import sys
import json

CIRCUITS_DIR = "circuitos"


def ensure_circuits_dir():
    os.makedirs(CIRCUITS_DIR, exist_ok=True)
    return CIRCUITS_DIR


def circuit_path(filename):
    return os.path.join(ensure_circuits_dir(), filename)

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
        
        filepath = circuit_path(filename)
        with open(filepath, "w") as f:
            f.write(full_script)
            
        return filepath
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    
def tape_to_braket_script(tape, filename, shots):
    """
    Convierte un QuantumTape de PennyLane a un script de AWS Braket SDK.
    Compatible con tensores de PyTorch y el Scheduler de multiplexación.
    """
    # 1. Cabecera exacta que espera tu Scheduler
    lineas_codigo = [
        "import numpy as np",
        "from braket.devices import LocalSimulator",
        "from braket.circuits import Circuit",
        "from collections import Counter",
        "",
        f"shots = {shots}",
        "",
        "circuit = Circuit()"
    ]
    
    # 2. Desglose de puertas
    for op in tape.operations:
        nombre_puerta = op.name
        qubits = op.wires.tolist()
        parametros_crudos = op.parameters
        
        # EXTRACCIÓN SEGURA: Convertir tensores de PyTorch a floats puros de Python
        p_vals = []
        for p in parametros_crudos:
            if hasattr(p, 'item'): # Si es un tensor de PyTorch
                p_vals.append(float(p.item()))
            else: # Si ya es un número normal
                p_vals.append(float(p))
        
        # 3. Mapeo a sintaxis de AWS Braket
        if nombre_puerta == "RX":
            lineas_codigo.append(f"circuit.rx({qubits[0]}, {p_vals[0]:.6f})")
        elif nombre_puerta == "RY":
            lineas_codigo.append(f"circuit.ry({qubits[0]}, {p_vals[0]:.6f})")
        elif nombre_puerta == "RZ":
            lineas_codigo.append(f"circuit.rz({qubits[0]}, {p_vals[0]:.6f})")
        elif nombre_puerta == "CNOT":
            lineas_codigo.append(f"circuit.cnot({qubits[0]}, {qubits[1]})")
        elif nombre_puerta == "Hadamard":
            lineas_codigo.append(f"circuit.h({qubits[0]})")
        elif nombre_puerta == "PauliX":
            lineas_codigo.append(f"circuit.x({qubits[0]})")
        elif nombre_puerta == "PauliZ":
            lineas_codigo.append(f"circuit.z({qubits[0]})")
        else:
            lineas_codigo.append(f"# WARNING: Puerta no soportada - {nombre_puerta}")

    # 4. Guardar archivo
    filepath = circuit_path(filename)
    with open(filepath, "w") as f:
        f.write("\n".join(lineas_codigo))

    return filepath