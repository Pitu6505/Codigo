import sys
import pennylane as qml

print(f"Python Version: {sys.version}")
print(f"PennyLane Version: {qml.version()}")

try:
    import pennylane_qiskit
    print(f"\n✅ pennylane_qiskit importado correctamente.")
    print(f"Versión del plugin: {pennylane_qiskit.__version__}")
    
    # Vamos a buscar dónde está la función 'to_qiskit' o 'convert'
    print("\n🔍 Buscando funciones de conversión disponibles:")
    
    # Chequeo nivel 1: Top Level
    if hasattr(pennylane_qiskit, 'to_qiskit'):
        print("  -> Encontrado: pennylane_qiskit.to_qiskit")
    
    # Chequeo nivel 2: Dentro de converter
    if hasattr(pennylane_qiskit, 'converter'):
        print("  -> Módulo 'converter' existe. Contenido:")
        attrs = dir(pennylane_qiskit.converter)
        found = [a for a in attrs if 'qiskit' in a or 'convert' in a]
        print(f"     {found}")
        
    # Chequeo nivel 3: PennyLane Namespace
    try:
        if hasattr(qml, 'qiskit'):
            print(f"  -> qml.qiskit existe. Atributos: {[a for a in dir(qml.qiskit) if 'to_' in a]}")
    except:
        pass

except ImportError:
    print("\n❌ CRÍTICO: No se puede importar 'pennylane_qiskit'.")
    print("Asegúrate de ejecutar: pip install pennylane-qiskit")