import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns

# 1. Cargar los MISMOS datos que usó el cuántico
print("--- Cargando datos para Benchmark Clásico ---")
X = np.load('ArchivoRed/X_data.npy', allow_pickle=True)
y = np.load('ArchivoRed/y_data.npy', allow_pickle=True)

# Asegurar formato correcto (igual que hiciste en el script cuántico)
y = y.flatten().astype(int)
y = np.where(y == 0, -1, 1) # Usamos -1 y 1 para ser justos

# Split idéntico al cuántico (random_state=42 es la clave)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Datos cargados. Entrenando con {X.shape[1]} características (Qubits equivalentes).")

# --- MODELO 1: SVM (Support Vector Machine) ---
# Es el rival directo de los clasificadores cuánticos
print("\n1. Entrenando SVM Clásica...")
svm = SVC(kernel='rbf') # Kernel radial estándar
svm.fit(X_train, y_train)
preds_svm = svm.predict(X_test)
acc_svm = accuracy_score(y_test, preds_svm)
print(f"SVM Accuracy: {acc_svm:.2%}")

# --- MODELO 2: Random Forest ---
# El estándar en la industria de ciberseguridad
print("2. Entrenando Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
preds_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, preds_rf)
print(f"Random Forest Accuracy: {acc_rf:.2%}")

# --- REPORTE DETALLADO (Del mejor clásico) ---
best_model_name = "SVM" if acc_svm > acc_rf else "Random Forest"
best_preds = preds_svm if acc_svm > acc_rf else preds_rf

print(f"\n--- REPORTE DEL MEJOR CLÁSICO ({best_model_name}) ---")
print(classification_report(y_test, best_preds, target_names=['Benigno', 'Ataque']))

# Visualización comparativa
modelos = ['QML (Tu modelo)', 'SVM Clásica', 'Random Forest']
accuracies = [0.8433, acc_svm, acc_rf] # 0.8433 es tu resultado anterior

plt.figure(figsize=(8, 5))
barras = plt.bar(modelos, accuracies, color=['purple', 'orange', 'green'])
plt.ylim(0.5, 1.0)
plt.title('Comparativa Preliminar: Cuántico vs Clásico (Mismos datos)')
plt.ylabel('Accuracy')

# Poner el número encima de la barra
for rect in barras:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, f'{height:.2%}', ha='center', va='bottom')

plt.show()