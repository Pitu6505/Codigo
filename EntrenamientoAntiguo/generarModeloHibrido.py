import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

print("--- 1. Preparando Entrenamiento para Guardar ---")

# Cargar Datos
X = np.load('ArchivoRed/X_data.npy', allow_pickle=True)
y = np.load('ArchivoRed/y_data.npy', allow_pickle=True)

# Ajustar etiquetas
y = y.flatten().astype(int)
y = np.array([0 if val <= 0 else 1 for val in y])

# Convertir a Torch
X_torch = torch.tensor(X, dtype=torch.float32)
y_torch = torch.tensor(y, dtype=torch.long)

# Configuración
n_qubits = 4
n_layers = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

quantum_layer = qml.qnn.TorchLayer(qnode, {"weights": (n_layers, n_qubits, 3)})

class HybridNN(nn.Module):
    def __init__(self):
        super(HybridNN, self).__init__()
        self.cl_layer_1 = nn.Linear(6, n_qubits)
        self.q_layer = quantum_layer
        self.cl_layer_2 = nn.Linear(n_qubits, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.cl_layer_1(x)
        x = self.relu(x)
        x = x * (np.pi / 2.0)
        x = self.q_layer(x)
        x = self.cl_layer_2(x)
        return self.softmax(x)

# Entrenar
model = HybridNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

print("--- Entrenando modelo (Versión Express)... ---")
epochs = 30 # Suficiente para que guarde algo decente
batch_size = 16

for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_torch.size()[0])
    
    epoch_loss = 0
    for i in range(0, len(X_torch), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_torch[indices], y_torch[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = loss_fn(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(X_torch):.4f}")

# --- EL PASO CLAVE: GUARDAR ---
print("\n--- Guardando Modelo en Disco ---")
ruta_archivo = "modelo_hibrido_96acc.pth"
torch.save(model.state_dict(), ruta_archivo)

if os.path.exists(ruta_archivo):
    print(f"✅ ¡ÉXITO! Archivo generado: {ruta_archivo}")
    print("Ahora sí puedes ejecutar Run_IBM.py")
else:
    print("❌ Error: No se pudo crear el archivo.")