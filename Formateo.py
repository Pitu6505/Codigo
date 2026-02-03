import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import os

CSV_PATH = 'Datasets/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv' 

N_QUBITS = 6         # Número de características finales (y qubits que usarás), cuanto mas alto, mas costoso sera la ejecucion que se tiene que realizar
TOTAL_SAMPLES = 1000 # Tamaño total del dataset reducido (para que la simulación no tarde días)
TEST_SIZE = 0.2      # 20% para testeo

def procesar_datos():
    print(f"--- Cargando dataset desde {CSV_PATH} ---")
    
  
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print("ERROR: No se encuentra el archivo CSV. Revisa la ruta.")
        return


    df.columns = df.columns.str.strip()
    
    print(f"Dimensiones originales: {df.shape}")


    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # 3. Gestión de Etiquetas (Target)
    # Asumimos que la columna se llama 'Label'. En CIC-IDS2017 suele ser así.
    if 'Label' not in df.columns:
        print("ERROR: No encuentro la columna 'Label'. Las columnas son:", df.columns)
        return

    # Convertir a Binario: BENIGN = 0, Cualquier Ataque = 1
    df['Label_Binary'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    
    # Separar Benignos y Maliciosos para balancear
    df_benign = df[df['Label_Binary'] == 0]
    df_attack = df[df['Label_Binary'] == 1]
    
    print(f"Benignos originales: {len(df_benign)} | Ataques originales: {len(df_attack)}")

    # 4. Undersampling (Balanceo y Reducción de tamaño)
    samples_per_class = TOTAL_SAMPLES // 2
    
    # Cogemos muestras aleatorias (asegurando que no pedimos más de las que hay)
    n_benign = min(len(df_benign), samples_per_class)
    n_attack = min(len(df_attack), samples_per_class)
    
    df_benign_sample = df_benign.sample(n=n_benign, random_state=42)
    df_attack_sample = df_attack.sample(n=n_attack, random_state=42)
    
    df_balanced = pd.concat([df_benign_sample, df_attack_sample])
    
    # Mezclar filas (Shuffle)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Dataset balanceado creado: {df_balanced.shape} filas.")

    # 5. Separar Features (X) y Target (y)
    # Quitamos las columnas de etiquetas y cualquier otra no numérica (IPs, Timestamps si las hubiera)
    # En CIC-IDS2017 casi todo es numérico excepto Label.
    y = df_balanced['Label_Binary'].values
    X = df_balanced.drop(['Label', 'Label_Binary'], axis=1)
    
    # Asegurarnos de que solo hay números
    X = X.select_dtypes(include=[np.number])

    # 6. Reducción de Dimensionalidad (PCA) para QML
    # Primero estandarizamos (Media 0, Varianza 1) -> Vital para que el PCA funcione bien
    scaler_std = StandardScaler()
    X_std = scaler_std.fit_transform(X)
    
    print(f"Aplicando PCA para reducir de {X.shape[1]} features a {N_QUBITS} qubits...")
    pca = PCA(n_components=N_QUBITS)
    X_pca = pca.fit_transform(X_std)
    
    # Explicación de varianza (Dato importante para tu TFM)
    varianza = np.sum(pca.explained_variance_ratio_)
    print(f"¡Atención! Con {N_QUBITS} qubits conservas el {varianza:.2%} de la información original.")

    # 7. Escalado para Ángulos (0 a PI)
    # Los qubits funcionan con rotaciones. Si metes un valor 500, da vueltas a lo loco.
    # Escalamos todo entre 0 y PI (3.1415...)
    scaler_minmax = MinMaxScaler(feature_range=(0, np.pi))
    X_final = scaler_minmax.fit_transform(X_pca)

    # 8. Guardar los datos procesados
    # Los guardamos en formato numpy (.npy) que es rápido de cargar después
    np.save('ArchivoRed/X_data.npy', X_final)
    np.save('ArchivoRed/y_data.npy', y)
    
    print("\n--- ¡PROCESO COMPLETADO! ---")
    print("Archivos generados: 'X_data.npy' y 'y_data.npy'")
    print("Ahora puedes cargar estos archivos directamente en tu script de PennyLane.")

if __name__ == "__main__":
    procesar_datos()