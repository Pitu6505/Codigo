import pennylane as qml
import torch
import numpy as np
import os

# --- CONFIGURACIÓN ---
N_QUBITS = 6
# CAMBIA ESTO POR EL MODELO QUE QUIERAS PROBAR:
MODEL_PATH = "Modelos/3ºModelo.pth" 

print(f"--- 1. Preparando Datos para evaluar {MODEL_PATH} ---")
X = np.load('ArchivoRed/X_data.npy', allow_pickle=True)
y = np.load('ArchivoRed/y_data.npy', allow_pickle=True)
y = y.flatten().astype(int)
y = np.array([0 if val <= 0 else 1 for val in y])

TRAIN_SIZE = 128
X_test = torch.tensor(X[TRAIN_SIZE:], dtype=torch.float32)
y_test = torch.tensor(y[TRAIN_SIZE:], dtype=torch.long)

if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR: No se encuentra el modelo en {MODEL_PATH}")
    exit()

checkpoint = torch.load(MODEL_PATH)
pesos_entrenados = checkpoint['weights']
bias_entrenado = checkpoint['bias']

# 🟢 NORMALIZACIÓN AUTOMÁTICA (Usando los valores guardados en el entrenamiento)
if 'x_min' in checkpoint and 'x_max' in checkpoint:
    x_min = checkpoint['x_min']
    x_max = checkpoint['x_max']
    X_test = np.pi * (X_test - x_min) / (x_max - x_min)
    print("✅ Normalización aplicada (Modelos 2 y 3 detectados).")
else:
    print("⚠️ No se encontró normalización en el checkpoint (Modelo 1 detectado).")

# --- 2. DETECCIÓN DE ARQUITECTURA ---
dev = qml.device("default.qubit", wires=N_QUBITS)

# Detectamos si es M1/M2 (Strongly) o M3 (Basic) viendo las dimensiones de los pesos
if len(pesos_entrenados.shape) == 3:
    print("🏗️ Arquitectura detectada: StronglyEntanglingLayers (Modelo 1 o 2)")
    @qml.qnode(dev)
    def qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        return qml.expval(qml.PauliZ(0))
else:
    print("🏗️ Arquitectura detectada: BasicEntanglerLayers (Modelo 3)")
    @qml.qnode(dev)
    def qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
        qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
        return qml.expval(qml.PauliZ(0))

# --- 3. EVALUACIÓN (TESTING) ---
print("\n--- 3. Iniciando Evaluación ---")
correctos = 0
total = len(X_test)
clases_predichas = []

with torch.no_grad():
    for i in range(total):
        # Forward pass (Simulador local súper rápido)
        pred_cruda = qnode(X_test[i], pesos_entrenados) + bias_entrenado
        
        # Clasificación
        clase_predicha = 1 if pred_cruda.item() > 0 else 0
        clase_real = y_test[i].item()
        clases_predichas.append(clase_predicha)
        
        if clase_predicha == clase_real:
            correctos += 1

# --- 4. RESULTADOS FINALES ---
precision = correctos / total
print("\n" + "="*50)
print(f"🏆 RESULTADOS FINALES ({MODEL_PATH})")
print("="*50)
print(f"Total de muestras evaluadas : {total}")
print(f"Precisión (Accuracy)        : {precision * 100:.2f}%")
print("="*50)

pred_unos = sum(clases_predichas)
pred_ceros = len(clases_predichas) - pred_unos
print(f"📊 Distribución de predicciones: {pred_ceros} Normales (0) | {pred_unos} Ataques (1)")