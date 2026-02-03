import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import time

# --- 1. CARGA DE DATOS ---
print("--- Cargando datos... ---")
X = np.load('ArchivoRed/X_data.npy', allow_pickle=True, requires_grad=False)
y = np.load('ArchivoRed/y_data.npy', allow_pickle=True, requires_grad=False)

# Asegurar tipos
y = y.flatten().astype(int)
y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

n_qubits = X.shape[1]
print(f"Entrenando con {n_qubits} Qubits (Modo Optimizado)")

# --- 2. CIRCUITO AVANZADO ---
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def circuit(weights, features):
    # Angle Encoding
    qml.AngleEmbedding(features, wires=range(n_qubits), rotation='X')
    
    # CAMBIO 1: StronglyEntanglingLayers
    # Esta capa entrelaza todos los qubits con todos, creando un "cerebro" más denso
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    
    return qml.expval(qml.PauliZ(0))

def variational_classifier(weights, bias, x):
    return circuit(weights, x) + bias

def cost(weights, bias, X_batch, y_batch):
    predictions = [variational_classifier(weights, bias, x) for x in X_batch]
    return np.mean((y_batch - np.stack(predictions)) ** 2)

# --- 3. ENTRENAMIENTO REFINADO ---
np.random.seed(42) # Semilla fija para reproducibilidad

# Configuración para StronglyEntanglingLayers
n_layers = 3 # Aumentamos una capa para más "inteligencia"
# La forma de los pesos en StronglyEntangling es (n_layers, n_qubits, 3)
weights_init = 0.01 * np.random.randn(n_layers, n_qubits, 3, requires_grad=True)
bias_init = np.array(0.0, requires_grad=True)

# CAMBIO 2: Learning rate más suave (0.01)
opt = qml.AdamOptimizer(stepsize=0.01)

weights = weights_init
bias = bias_init
batch_size = 10
epochs = 30 # CAMBIO 3: Más tiempo para aprender

print("\n--- Iniciando Entrenamiento Optimizado ---")
loss_history = []
start_time = time.time()

for epoch in range(epochs):
    indices = np.random.permutation(len(X_train))
    X_train_sh = X_train[indices]
    y_train_sh = y_train[indices]
    
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train_sh[i:i + batch_size]
        y_batch = y_train_sh[i:i + batch_size]
        weights, bias, _, _ = opt.step(cost, weights, bias, X_batch, y_batch)

    # Monitorización
    if (epoch + 1) % 5 == 0 or epoch == 0:
        # Evaluamos en una muestra pequeña para ver el progreso sin perder tiempo
        current_cost = cost(weights, bias, X_train[:50], y_train[:50])
        print(f"Epoch {epoch+1}/{epochs} | Coste: {current_cost:.4f} | Tiempo: {time.time()-start_time:.1f}s")
        loss_history.append(current_cost)

# --- 4. EVALUACIÓN ---
print("\n--- Evaluando Modelo Optimizado ---")
preds_raw = [variational_classifier(weights, bias, x) for x in X_test]
preds_bin = np.array([1 if p > 0 else -1 for p in preds_raw]).astype(int)
y_test = y_test.astype(int)

acc = accuracy_score(y_test, preds_bin)
print(f"\nACCURACY MEJORADO: {acc:.2%}")

# Matriz
print(classification_report(y_test, preds_bin, labels=[-1, 1], target_names=['Benigno', 'Ataque']))

cm = confusion_matrix(y_test, preds_bin, labels=[-1, 1])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Benigno', 'Ataque'], yticklabels=['Benigno', 'Ataque'])
plt.title(f'QML Optimizado ({n_qubits} Qubits)')
plt.show()