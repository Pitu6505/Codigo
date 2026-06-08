import numpy as np
import matplotlib.pyplot as plt

# --- 1. PARÁMETROS DEL MODELO MATEMÁTICO (Basado en IQM Garnet) ---
N_total = 50           # Total de circuitos a evaluar
T_base = 0.30          # Precio por tarea en AWS ($)
S = 1000               # Número de shots
C_shot = 0.00145       # Precio por shot en IQM Garnet ($)

Q_QPU = 20             # Qubits totales de IQM Garnet
Q_circ = 2             # Qubits por circuito (VQC)

# Calculamos el factor de multiplexación (K)
K = np.floor(Q_QPU / Q_circ)  # Para IQM: 20/2 = 10 circuitos por tarea

# Eje X: de 1 a 50 circuitos
N_array = np.arange(1, N_total + 1)

# --- 2. APLICACIÓN DE LAS FÓRMULAS ---
# Coste por tarea completa = T_base + (S * C_shot) = 0.30 + 1.45 = 1.75$
coste_tarea_unica = T_base + (S * C_shot)

# Función Lineal Tradicional
C_std = N_array * coste_tarea_unica

# Función Escalonada (QCRAFT Scheduler)
# np.ceil redondea hacia arriba (ej: circuito 1 al 10 = 1 lote. Circuito 11 = 2 lotes)
C_qcraft = np.ceil(N_array / K) * coste_tarea_unica

# --- 3. CONFIGURACIÓN Y GENERACIÓN DE LA GRÁFICA ---
plt.figure(figsize=(10, 6))

# Pintamos la línea de coste estándar (Roja)
plt.plot(N_array, C_std, color='#d62728', linewidth=2.5, label='Standard Execution (Sequential)')

# Pintamos el área bajo la curva roja para enfatizar el sobrecoste
plt.fill_between(N_array, C_std, C_qcraft, color='#ff9896', alpha=0.3, label='Unnecessary Overhead')

# Pintamos la línea del Scheduler (Escalonada, Azul)
plt.step(N_array, C_qcraft, where='post', color='#1f77b4', linewidth=3, label='QCRAFT Scheduler (Multiplexed)')

# Pintamos el área bajo la curva azul para mostrar el precio real pagado
plt.fill_between(N_array, C_qcraft, 0, step="post", color='#aec7e8', alpha=0.5)

# --- 4. FORMATO DE LA GRÁFICA (Estilo Académico) ---
plt.xlabel('Number of Evaluated Circuits ($N$)', fontsize=12, fontweight='bold')
plt.ylabel('Cumulative Cost in USD ($)', fontsize=12, fontweight='bold')

# Límites y rejilla
plt.xlim(1, 50)
plt.ylim(0, max(C_std) + 5)
plt.grid(True, linestyle='--', alpha=0.6)

# Leyenda en inglés, sin título (ideal para papers)
plt.legend(loc='upper left', fontsize=11, framealpha=1, edgecolor='black')

# Guardar y mostrar
nombre_archivo = "Cost_Comparison_Chart.png"
plt.tight_layout()
plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
print(f"✅ Gráfica generada con éxito: {nombre_archivo}")
plt.show()