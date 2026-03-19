import json
import torch
import numpy as np
import matplotlib.pyplot as plt

PESOS_FILE = "pesos_two_moons.pth"
ARCHIVOS_JSON = ["results1.json", "results2.json"]

def extraer_probabilidades(ruta_json):
    """Lee el JSON de AWS Braket y extrae el diccionario de probabilidades."""
    try:
        with open(ruta_json, 'r') as f:
            data = json.load(f)
            # AWS Braket guarda las probabilidades en "measurementProbabilities"
            return data.get("measurementProbabilities", {})
    except Exception as e:
        print(f"❌ Error leyendo {ruta_json}: {e}")
        return {}

def calcular_expval_multiplexado(probs_dict, bit_index):
    """
    Calcula el valor esperado del observable PauliZ para un qubit específico.
    PauliZ = P(0) - P(1)
    """
    p0 = 0.0
    p1 = 0.0
    for bitstring, prob in probs_dict.items():
        # Braket devuelve strings de 36 caracteres. Leemos el bit específico.
        if bitstring[bit_index] == '0':
            p0 += prob
        else:
            p1 += prob
    return p0 - p1

def main():
    print(f"📂 Cargando modelo y datos base de {PESOS_FILE}...")
    checkpoint = torch.load(PESOS_FILE, weights_only=False)
    bias_val = checkpoint['bias'].item()
    X_test_raw = checkpoint['X_test_raw']
    y_test_real = checkpoint['y_test_raw']  # Formato: 0 y 1

    # Cargar los JSONs recuperados
    probs_t1 = extraer_probabilidades(ARCHIVOS_JSON[0])
    probs_t2 = extraer_probabilidades(ARCHIVOS_JSON[1])

    if not probs_t1 or not probs_t2:
        print("⚠️ No se pudieron cargar los JSONs. Revisa los nombres.")
        return

    # Vamos a almacenar las predicciones (36 en total)
    muestras_recuperadas = 36
    predicciones = np.zeros(len(y_test_real))
    predicciones.fill(np.nan) # Llenamos de NaN para identificar los faltantes
    
    aciertos = 0
    print("\n🚀 Analizando resultados multiplexados de IonQ Forte...")
    
    # --- Procesar Tarea 1 (Muestras 0 a 17) ---
    for i in range(18):
        bit_objetivo = i * 2  # Como cada circuito usa 2 qubits, el observable PauliZ(0) cae en el qubit par
        expval = calcular_expval_multiplexado(probs_t1, bit_objetivo)
        prediccion_cruda = expval + bias_val
        clase_predicha = 1 if prediccion_cruda > 0 else 0
        predicciones[i] = clase_predicha
        
        if clase_predicha == (1 if y_test_real[i] > 0 else 0):
            aciertos += 1

    # --- Procesar Tarea 2 (Muestras 18 a 35) ---
    for i in range(18):
        indice_global = i + 18
        bit_objetivo = i * 2
        expval = calcular_expval_multiplexado(probs_t2, bit_objetivo)
        prediccion_cruda = expval + bias_val
        clase_predicha = 1 if prediccion_cruda > 0 else 0
        predicciones[indice_global] = clase_predicha
        
        if clase_predicha == (1 if y_test_real[indice_global] > 0 else 0):
            aciertos += 1

    precision = (aciertos / muestras_recuperadas) * 100

    print("=" * 50)
    print("🏆 RESULTADOS RECUPERADOS: IONQ FORTE (Iones Atrapados)")
    print("=" * 50)
    print(f"Muestras evaluadas: {muestras_recuperadas} / 50")
    print(f"Muestras perdidas:  14 (Tarea 3 no recibida)")
    print(f"Precisión Física:   {precision:.2f}%")
    print("=" * 50)

    # --- GENERAR GRÁFICA ---
    print("\n📊 Generando gráfica normalizada...")
    plt.figure(figsize=(12, 5))
    y_test_dibujo = np.where(y_test_real > 0, 1, 0)

    # Subplot 1: Datos Reales
    plt.subplot(1, 2, 1)
    plt.scatter(X_test_raw[y_test_dibujo==0, 0], X_test_raw[y_test_dibujo==0, 1], color='red', label='Clase 0', edgecolor='k')
    plt.scatter(X_test_raw[y_test_dibujo==1, 0], X_test_raw[y_test_dibujo==1, 1], color='blue', label='Clase 1', edgecolor='k')
    plt.title("Valores Reales (Dataset Original)")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()

    # Subplot 2: Predicciones IonQ
    plt.subplot(1, 2, 2)
    
    # 1. Pintar los puntos predecidos por IonQ
    idx_evaluados = ~np.isnan(predicciones)
    plt.scatter(X_test_raw[idx_evaluados & (predicciones==0), 0], X_test_raw[idx_evaluados & (predicciones==0), 1], color='red', label='Predicción 0 (IonQ)', edgecolor='k')
    plt.scatter(X_test_raw[idx_evaluados & (predicciones==1), 0], X_test_raw[idx_evaluados & (predicciones==1), 1], color='blue', label='Predicción 1 (IonQ)', edgecolor='k')
    
    # 2. Pintar los puntos faltantes en gris
    idx_perdidos = np.isnan(predicciones)
    if np.any(idx_perdidos):
        plt.scatter(X_test_raw[idx_perdidos, 0], X_test_raw[idx_perdidos, 1], color='lightgray', label='No Evaluado (Timeout)', edgecolor='k', alpha=0.6)

    # 3. Marcar Errores
    errores = idx_evaluados & (y_test_dibujo != predicciones)
    if np.any(errores):
        plt.scatter(X_test_raw[errores, 0], X_test_raw[errores, 1], color='lime', marker='x', s=100, label='Error de IonQ')

    plt.title(f"Predicción IonQ Forte (Acc: {precision:.2f}% en 36 pts)")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()

    plt.tight_layout()
    plt.savefig("Resultados_Recuperados_IonQ.png", dpi=300)
    print("✅ Gráfica guardada como: Resultados_Recuperados_IonQ.png")
    plt.show()

if __name__ == "__main__":
    main()