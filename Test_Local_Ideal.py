import pennylane as qml
import torch
import numpy as np
import matplotlib.pyplot as plt

PESOS_FILE = "pesos_two_moons.pth"
N_QUBITS = 2

# --- 1. CONFIGURACIÓN DEL SIMULADOR ---
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch")
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))

# --- 2. CARGA DE MODELO Y DATOS ---
print(f"📂 Cargando modelo de {PESOS_FILE}...")
# Usamos weights_only=False por seguridad de PyTorch 2.6
checkpoint = torch.load(PESOS_FILE, weights_only=False)
pesos = checkpoint['weights']
bias_val = checkpoint['bias'].item()
X_min = checkpoint['X_min']
X_max = checkpoint['X_max']
X_test_raw = checkpoint['X_test_raw']
y_test_real = checkpoint['y_test_raw']

# Normalizamos igual que en el entrenamiento
X_test_norm = np.pi * (X_test_raw - X_min) / (X_max - X_min)
X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32)

# --- 3. EVALUACIÓN LOCAL (INFERENCIA) ---
aciertos = 0
total_muestras = len(X_test_tensor)
predicciones_array = np.zeros(total_muestras)

print("🚀 Ejecutando inferencia en Simulador Ideal (Local)...")
for k in range(total_muestras):
    # Predicción pura del simulador matemático
    valor_esperado = qnode(X_test_tensor[k], pesos).item()
    prediccion_cruda = valor_esperado + bias_val
    
    clase_predicha = 1 if prediccion_cruda > 0 else 0
    # Ajustamos y_test_real que estaba en formato (-1, 1) para compararlo
    clase_real = 1 if y_test_real[k] > 0 else 0
    
    predicciones_array[k] = clase_predicha
    
    if clase_predicha == clase_real:
        aciertos += 1

precision = (aciertos / total_muestras) * 100

print("=" * 50)
print("🏆 RESULTADOS DE INFERENCIA EN SIMULADOR IDEAL")
print("=" * 50)
print(f"Muestras:        {total_muestras}")
print(f"Precisión Final: {precision:.2f}%")
print("=" * 50)

# --- 4. GENERACIÓN DE LA GRÁFICA ---
print("\n📊 Generando gráfica de resultados ideales...")
plt.figure(figsize=(12, 5))

# Convertimos a 0 y 1 para simplificar el dibujo de las reales
y_test_dibujo = np.where(y_test_real > 0, 1, 0)

# Subplot 1: Datos Reales
plt.subplot(1, 2, 1)
plt.scatter(X_test_raw[y_test_dibujo==0, 0], X_test_raw[y_test_dibujo==0, 1], color='red', label='Clase 0', edgecolor='k')
plt.scatter(X_test_raw[y_test_dibujo==1, 0], X_test_raw[y_test_dibujo==1, 1], color='blue', label='Clase 1', edgecolor='k')
plt.title("Valores Reales (Dataset Original)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()

# Subplot 2: Predicciones del Simulador
plt.subplot(1, 2, 2)
plt.scatter(X_test_raw[predicciones_array==0, 0], X_test_raw[predicciones_array==0, 1], color='red', label='Predicción 0', edgecolor='k')
plt.scatter(X_test_raw[predicciones_array==1, 0], X_test_raw[predicciones_array==1, 1], color='blue', label='Predicción 1', edgecolor='k')

# Marcamos los errores con una 'X' grande verde
errores = (y_test_dibujo != predicciones_array)
if np.any(errores):
    plt.scatter(X_test_raw[errores, 0], X_test_raw[errores, 1], color='lime', marker='x', s=100, label='Error de Predicción')

plt.title(f"Predicción en SIMULADOR IDEAL (Acc: {precision:.2f}%)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()

plt.tight_layout()
nombre_imagen = "Resultados_Simulador_Ideal.png"
plt.savefig(nombre_imagen, dpi=300)
print(f"✅ Gráfica guardada como: {nombre_imagen}")
plt.show()