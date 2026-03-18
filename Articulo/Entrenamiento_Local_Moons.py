import pennylane as qml
import torch
import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# --- 1. GENERACIÓN DEL DATASET "TWO MOONS" ---
print("--- Generando Dataset Two Moons ---")
# Generamos 200 puntos con un poco de ruido para que sea realista
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

# Convertir etiquetas (0, 1) a (-1, 1) para la medida PauliZ
y_pm = np.where(y == 0, -1, 1)

# Normalizar los datos de entrada (X, Y) entre 0 y PI para las puertas de rotación
X_min, X_max = X.min(axis=0), X.max(axis=0)
X_norm = np.pi * (X - X_min) / (X_max - X_min)

# Dividir en Train (150) y Test (50)
X_train = torch.tensor(X_norm[:150], dtype=torch.float32)
y_train = torch.tensor(y_pm[:150], dtype=torch.float32)
X_test = torch.tensor(X_norm[150:], dtype=torch.float32)
y_test = torch.tensor(y_pm[150:], dtype=torch.float32)

print(f"✅ Datos listos. Entrenamiento: 150, Testeo: 50. Qubits necesarios: 2")

# --- 2. DEFINICIÓN DEL CIRCUITO CUÁNTICO (2 QUBITS) ---
n_qubits = 2
Capas_entrelazamiento = 6
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def qnode(inputs, weights):
    # Incrustación de los 2 datos (X, Y) en los 2 qubits
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    
    # Capas de entrelazamiento (Shallow Circuit)
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    
    # Medimos el valor esperado en el qubit 0
    return qml.expval(qml.PauliZ(0))

# Inicializamos los pesos (Capas_entrelazamiento x 2 qubits = 12 pesos)
init_weights = 0.1 * torch.randn(Capas_entrelazamiento, n_qubits)
weights = torch.tensor(init_weights, requires_grad=True, dtype=torch.float32)
bias = torch.tensor(0.0, requires_grad=True, dtype=torch.float32)

# --- 3. BUCLE DE ENTRENAMIENTO LOCAL ---
optimizer = torch.optim.Adam([weights, bias], lr=0.1)
epochs = 15
batch_size = 15

print("\n🚀 Iniciando Entrenamiento en Simulador Local...")
for epoch in range(epochs):
    loss_epoch = 0
    
    # Mezclar datos en cada época
    perm = torch.randperm(X_train.size(0))
    
    for i in range(0, X_train.size(0), batch_size):
        idx = perm[i:i+batch_size]
        X_batch, y_batch = X_train[idx], y_train[idx]
        
        def closure():
            optimizer.zero_grad()
            preds = torch.stack([qnode(x, weights) + bias for x in X_batch])
            # Error Cuadrático Medio (MSE)
            loss = torch.mean((preds - y_batch)**2)
            loss.backward()
            return loss
            
        loss_batch = optimizer.step(closure)
        loss_epoch += loss_batch.item()
        
    print(f"Epoch {epoch+1}/{epochs} | Loss promedio: {loss_epoch / (X_train.size(0)//batch_size):.4f}")

# --- 4. GUARDAR EL MODELO PARA ENVIARLO AL SCHEDULER ---
print("\n💾 Guardando pesos perfectos...")
torch.save({
    'weights': weights.detach(),
    'bias': bias.detach(),
    'X_min': X_min,
    'X_max': X_max,
    'X_test_raw': X[150:], # Guardamos los de test crudos para evaluarlos luego en la nube
    'y_test_raw': y[150:]
}, "pesos_two_moons.pth")
print("✅ Archivo 'pesos_two_moons.pth' generado. ¡Listo para la nube!")