import numpy as np
import matplotlib.pyplot as plt

# --- 1. DATOS DEL EXPERIMENTO (50 Muestras) ---
maquinas = ['Rigetti Ankaa-3','Rigetti Cepheus', 'IQM Garnet', 'IonQ Forte\n(100 shots)']

# Coste sin Scheduler (1 circuito = 1 tarea)
coste_tradicional = [60.00, 36.25, 87.50, 415.00]

# Coste con QCRAFT Scheduler (Multiplexado)
coste_qcraft = [2.40, 0.73, 8.75, 16.60]

# Posiciones para las barras
x = np.arange(len(maquinas))
width = 0.30  # Ancho de las barras

# --- 2. CREACIÓN DE LA GRÁFICA ---
fig, ax = plt.subplots(figsize=(10, 6))

# Dibujar las barras
barras_tradicional = ax.bar(x - width/2, coste_tradicional, width, label='Standard Execution (Sequential)', color='#d62728', edgecolor='black')
barras_qcraft = ax.bar(x + width/2, coste_qcraft, width, label='QCRAFT Scheduler (Multiplexed)', color='#1f77b4', edgecolor='black')

# --- 3. AÑADIR ETIQUETAS DE TEXTO ENCIMA DE LAS BARRAS ---
def autolabel(rects):
    """Añade una etiqueta de texto encima de cada barra con su altura."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'${height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 puntos de desplazamiento vertical
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=16, fontweight='bold')

autolabel(barras_tradicional)
autolabel(barras_qcraft)

# --- 4. FORMATO ACADÉMICO ---
ax.set_ylabel('Total Execution Cost in USD ($)', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(maquinas, fontsize=16, fontweight='bold')
ax.legend(fontsize=16, loc='upper left', framealpha=1, edgecolor='black')

# Añadir una cuadrícula horizontal sutil para facilitar la lectura
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

# Ajustar límite Y para que las etiquetas más altas (como los $415) no se corten
ax.set_ylim(0, max(coste_tradicional) * 1.15)

# Guardar y mostrar
nombre_archivo = "BarChart_Total_Costs.png"
plt.tight_layout()
plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
print(f"✅ Gráfica de barras generada con éxito: {nombre_archivo}")
plt.show()