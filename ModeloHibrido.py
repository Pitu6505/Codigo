import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# --- 1. PREPARACIÓN DE DATOS (Estilo PyTorch) ---
print("--- Cargando datos para Modelo Híbrido ---")
X = np.load('ArchivoRed/X_data.npy', allow_pickle=True)
y = np.load('ArchivoRed/y_data.npy', allow_pickle=True)

# Ajuste de etiquetas: PyTorch quiere 0 y 1 (no -1 y 1)
y = y.flatten().astype(int)
y = np.where(y == -1, 0, y) # Si usabas -1, pasalo a 0. Si ya es 0/1, déjalo.
# Aseguramos que solo hay 0 y 1
y = np.array([0 if val <= 0 else 1 for val in y])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertir a Tensores de PyTorch
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.long)
X_test_torch = torch.tensor(X_test, dtype=torch.float32)
y_test_torch = torch.tensor(y_test, dtype=torch.long)

n_qubits = 4 # Usaremos 4 qubits en el "cuello de botella" cuántico
dev = qml.device("default.qubit", wires=n_qubits)

# --- 2. CAPA CUÁNTICA (QNode) ---
@qml.qnode(dev)
def qnode(inputs, weights):
    # Cargar datos (Angle Encoding)
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    # Capa variacional
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    # Medir todos los qubits para obtener características cuánticas
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Convertir QNode a capa de Keras/Torch
n_layers = 2
weight_shapes = {"weights": (n_layers, n_qubits, 3)}
quantum_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

# --- 3. ARQUITECTURA HÍBRIDA ---
class HybridNN(nn.Module):
    def __init__(self):
        super(HybridNN, self).__init__()
        # 1. Capa Clásica de Entrada (Comprime o expande las features originales)
        # Entrada: 6 features (tus datos) -> Salida: 4 (para los 4 qubits)
        self.cl_layer_1 = nn.Linear(X.shape[1], n_qubits)
        
        # 2. Capa Cuántica (Procesa la información en espacio de Hilbert)
        self.q_layer = quantum_layer
        
        # 3. Capa Clásica de Salida
        # Entrada: 4 (salida de los qubits) -> Salida: 2 (Clases: Benigno/Ataque)
        self.cl_layer_2 = nn.Linear(n_qubits, 2)
        
        # Funciones de activación
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.cl_layer_1(x)
        x = self.relu(x)
        
        # Un truco técnico: Escalamos los datos entre 0 y Pi antes de entrar al cuántico
        x = x * (np.pi / 2.0) 
        
        x = self.q_layer(x)
        
        x = self.cl_layer_2(x)
        return self.softmax(x)

# --- 4. ENTRENAMIENTO ---
model = HybridNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

epochs = 50
batch_size = 16
loss_list = []

print(f"--- Entrenando Híbrido (PyTorch + QML) por {epochs} épocas ---")

for epoch in range(epochs):
    model.train()
    
    # Shuffle manual simple
    permutation = torch.randperm(X_train_torch.size()[0])
    
    epoch_loss = 0
    batches = 0
    
    for i in range(0, len(X_train_torch), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train_torch[indices], y_train_torch[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = loss_fn(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        batches += 1
        
    avg_loss = epoch_loss / batches
    loss_list.append(avg_loss)
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

# --- 5. EVALUACIÓN ---
model.eval()
with torch.no_grad():
    outputs = model(X_test_torch)
    _, preds = torch.max(outputs, 1)

preds_np = preds.numpy()
y_test_np = y_test_torch.numpy()

acc = accuracy_score(y_test_np, preds_np)
print(f"\nACCURACY HÍBRIDO: {acc:.2%}")

print(classification_report(y_test_np, preds_np, target_names=['Benigno', 'Ataque']))

# Gráfica Loss
plt.plot(loss_list)
plt.title('Convergencia Modelo Híbrido')
plt.show()

# Matriz Confusión
cm = confusion_matrix(y_test_np, preds_np)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
plt.title('Matriz Confusión Híbrida')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.show()

import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

# Definimos un circuito pequeño para verlo
dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def circuito_visual(features, weights):
    qml.AngleEmbedding(features, wires=range(4))
    qml.StronglyEntanglingLayers(weights, wires=range(4))
    return qml.expval(qml.PauliZ(0))

# Datos falsos solo para generar el dibujo
features = np.array([0.1, 0.2, 0.3, 0.4])
weights = 0.1 * np.random.randn(3, 4, 3) # 3 capas

# Dibujar
print(qml.draw(circuito_visual)(features, weights))

# O versión gráfica bonita (si tienes matplotlib bien configurado)
qml.draw_mpl(circuito_visual)(features, weights)
plt.show()