import pennylane as qml
import torch
import numpy as np
import os

# --- CONFIGURACIÓN ---
N_QUBITS = 6
# Cambia esto si guardaste el archivo con otro nombre (ej. "model_epoch_15.pth")
MODEL_PATH = "Modelos/1ºModelo.pth" 

print("--- 1. Preparando Datos de Prueba ---")
# Cargar todo el dataset
X = np.load('ArchivoRed/X_data.npy', allow_pickle=True)
y = np.load('ArchivoRed/y_data.npy', allow_pickle=True)
y = y.flatten().astype(int)
y = np.array([0 if val <= 0 else 1 for val in y])

# SEPARAMOS LOS DATOS: Usamos solo los que NO se usaron en el entrenamiento
TRAIN_SIZE = 128
X_test = torch.tensor(X[TRAIN_SIZE:], dtype=torch.float32)
y_test = torch.tensor(y[TRAIN_SIZE:], dtype=torch.long)

# Si tu dataset de test es muy grande, puedes limitar cuántos quieres probar
TEST_SIZE = len(X_test) # O pon un número menor, ej. 200, si tarda mucho
X_test = X_test[:TEST_SIZE]
y_test = y_test[:TEST_SIZE]

print(f"✅ Datos cargados. Se evaluarán {len(X_test)} muestras nuevas.")

# --- 2. RECONSTRUIR EL CIRCUITO ---
print("\n--- 2. Cargando el Circuito y el Modelo ---")
dev = qml.device("default.qubit", wires=N_QUBITS)

# ESTO DEBE SER EXACTAMENTE IGUAL AL ENTRENAMIENTO
@qml.qnode(dev)
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))

# Cargar pesos y bias
if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR: No se encuentra el modelo en {MODEL_PATH}")
    exit()

checkpoint = torch.load(MODEL_PATH)

# Usamos torch.no_grad() porque aquí NO vamos a entrenar, solo a inferir
with torch.no_grad():
    pesos_entrenados = checkpoint['weights']
    bias_entrenado = checkpoint['bias']
    
print(f"✅ Modelo cargado correctamente (Epoch {checkpoint.get('epoch', 'N/A')}).")


# --- 3. EVALUACIÓN (TESTING) ---
print("\n--- 3. Iniciando Evaluación ---")
correctos = 0
total = len(X_test)

# Listas para guardar métricas si quieres graficar luego
predicciones_crudas = []
clases_predichas = []

with torch.no_grad():
    for i in range(total):
        # 1. Pase hacia adelante (Forward pass) usando el simulador local de PennyLane
        # Esto será rapidísimo porque se ejecuta en tu CPU local, no va a IBM
        pred_cruda = qnode(X_test[i], pesos_entrenados) + bias_entrenado
        
        # 2. Clasificación:
        # El valor de expectación va de [-1 a 1]. 
        # Si es mayor que 0, lo clasificamos como 1 (Ataque). Si es menor, como 0 (Normal).
        clase_predicha = 1 if pred_cruda.item() > 0 else 0
        clase_real = y_test[i].item()
        
        # 3. Guardar resultados
        predicciones_crudas.append(pred_cruda.item())
        clases_predichas.append(clase_predicha)
        
        if clase_predicha == clase_real:
            correctos += 1
            
        # Imprimimos las primeras 15 muestras para ver cómo se comporta
        if i < 15:
            resultado = "✅ BIEN" if clase_predicha == clase_real else "❌ MAL"
            print(f"Muestra {i+1:02d} | Real: {clase_real} | Predicho: {clase_predicha} (Crudo: {pred_cruda.item():.4f}) -> {resultado}")


# --- 4. RESULTADOS FINALES ---
precision = correctos / total
print("\n" + "="*50)
print("🏆 RESULTADOS FINALES DE LA PRUEBA (TESTING)")
print("="*50)
print(f"Total de muestras evaluadas : {total}")
print(f"Predicciones correctas      : {correctos}")
print(f"Predicciones incorrectas    : {total - correctos}")
print(f"Precisión (Accuracy)        : {precision * 100:.2f}%")
print("="*50)

# Opcional: Contar cuántos 0s y 1s predijo para ver si el modelo colapsó
pred_unos = sum(clases_predichas)
pred_ceros = len(clases_predichas) - pred_unos
print(f"📊 Distribución de predicciones: {pred_ceros} Normales (0) | {pred_unos} Ataques (1)")