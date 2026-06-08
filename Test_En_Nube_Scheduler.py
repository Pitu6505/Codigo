import pennylane as qml
import torch
import numpy as np
import asyncio
from aiohttp import web
import aiohttp
import os
import time
import matplotlib.pyplot as plt
from Utiles_Scheduler import circuit_path, ensure_circuits_dir, tape_to_qiskit_script, tape_to_braket_script
# --- 1. CONFIGURACIÓN DEL EXPERIMENTO ---
SCHEDULER_URL = "http://localhost:8082/"
MY_LOCAL_IP = "http://localhost:5000"
SHOTS = 1000
N_QUBITS = 2
PESOS_FILE = "pesos_two_moons.pth"

# 🟢 CAMBIA ESTO PARA TESTEAR EN DIFERENTES MÁQUINAS 🟢
PROVEEDOR_DESTINO = ['ibm'] 

# Variables Globales del Servidor
results_storage = {}
batch_event = asyncio.Event()
total_circuitos_enviados = 0

# --- 2. SERVIDOR PUENTE ---
def counts_to_expval(counts):
    zeros = 0
    ones = 0
    total = 0
    for k, v in counts.items():
        bit = k[-1] 
        if bit == '0': zeros += v
        else: ones += v
        total += v
    if total == 0: return 0
    return (zeros - ones) / total

async def handle_callback(request):
    try:
        data = await request.json()
        name = data.get("circuit_name")
        raw_counts = data.get("results")

        val = counts_to_expval(raw_counts)
        results_storage[name] = val
        
        if len(results_storage) >= total_circuitos_enviados:
            batch_event.set()
        return web.Response(text="OK")
    except Exception as e:
        print(f"Error en callback: {e}")
        return web.Response(status=500)

async def handle_file(request):
    name = request.match_info.get('name', "Anon")
    path = circuit_path(name)
    if os.path.exists(path):
        return web.FileResponse(path)
    return web.Response(status=404)

# --- 3. FUNCIÓN PRINCIPAL DE EVALUACIÓN ---
async def evaluar_en_nube():
    global total_circuitos_enviados
    
    app = web.Application()
    app.router.add_get('/circuits/{name}', handle_file)
    app.router.add_post('/callback', handle_callback)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 5000)
    await site.start()
    print("🌐 Servidor Puente listo en puerto 5000")

    ensure_circuits_dir()

    print(f"\n📂 Cargando pesos de {PESOS_FILE}...")
    # Usamos weights_only=False por seguridad de PyTorch 2.6
    checkpoint = torch.load(PESOS_FILE, weights_only=False)
    pesos = checkpoint['weights'].detach().numpy() 
    X_min = checkpoint['X_min']
    X_max = checkpoint['X_max']
    X_test_raw = checkpoint['X_test_raw']
    y_test_real = checkpoint['y_test_raw']

    X_test_norm = np.pi * (X_test_raw - X_min) / (X_max - X_min)
    total_muestras = len(X_test_norm)
    total_circuitos_enviados = total_muestras
    
    print(f"✅ Vamos a evaluar {total_muestras} muestras en: {PROVEEDOR_DESTINO}")
    
    tapes_enviados = []
    for i in range(total_muestras):
        x_val = X_test_norm[i]
        with qml.tape.QuantumTape() as tape:
            qml.AngleEmbedding(x_val, wires=range(N_QUBITS))
            qml.BasicEntanglerLayers(pesos, wires=range(N_QUBITS))
            qml.expval(qml.PauliZ(0))

        tape = tape.expand(depth=2)
        tapes_enviados.append(tape)


    results_storage.clear()
    batch_event.clear()
    
    start_time = time.time()
    
    print(f"🚀 Enviando {total_circuitos_enviados} circuitos al Scheduler...")
    async with aiohttp.ClientSession() as session:
        tasks = []
        for k, tape in enumerate(tapes_enviados):
            fname = f"test_moons_t{k}.py"
            proveedor_str = PROVEEDOR_DESTINO[0].lower()
            if "aws" in proveedor_str:
                tape_to_braket_script(tape, fname, SHOTS)
            else:
                tape_to_qiskit_script(tape, fname, SHOTS)            
            with open(circuit_path(fname), "r") as circuit_file:
                code = circuit_file.read()
            payload = {
                "url": f"{MY_LOCAL_IP}/circuits/{fname}", 
                "shots": SHOTS,
                "provider": PROVEEDOR_DESTINO,
                "policy": "time",
                "callback_url": f"{MY_LOCAL_IP}/callback",
                "circuit_name": fname,
                "code": code
            }
            task = session.post(SCHEDULER_URL + 'circuit', json=payload)
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    print("⏳ Esperando resultados del ordenador cuántico...")
    await batch_event.wait()
    
    tiempo_total = time.time() - start_time
    print(f"\n✅ ¡Resultados recibidos en {tiempo_total:.2f} segundos!")

    # 4. Calcular la Precisión y Preparar datos para la Gráfica
    aciertos = 0
    bias_val = checkpoint['bias'].item()
    predicciones_array = np.zeros(total_muestras)
    
    for k in range(total_muestras):
        fname = f"test_moons_t{k}.py"
        valor_esperado = results_storage.get(fname, 0.0)
        
        prediccion_cruda = valor_esperado + bias_val
        clase_predicha = 1 if prediccion_cruda > 0 else 0
        
        # Ajustamos y_test_real que estaba en formato (-1, 1) para compararlo
        clase_real = 1 if y_test_real[k] > 0 else 0
        
        predicciones_array[k] = clase_predicha
        
        if clase_predicha == clase_real:
            aciertos += 1

    precision = (aciertos / total_muestras) * 100
    
    print("=" * 50)
    print(f"🏆 RESULTADOS DE INFERENCIA EN {PROVEEDOR_DESTINO[0].upper()}")
    print("=" * 50)
    print(f"Muestras:        {total_muestras}")
    print(f"Precisión Final: {precision:.2f}%")
    print("=" * 50)

    # --- 5. GENERACIÓN DE LA GRÁFICA ---
    print("\n📊 Generando gráfica de resultados...")
    plt.figure(figsize=(12, 5))

    # Convertimos a 0 y 1 para simplificar el dibujo
    y_test_dibujo = np.where(y_test_real > 0, 1, 0)

    # Subplot 1: Datos Reales
    plt.subplot(1, 2, 1)
    plt.scatter(X_test_raw[y_test_dibujo==0, 0], X_test_raw[y_test_dibujo==0, 1], color='red', label='Clase 0', edgecolor='k')
    plt.scatter(X_test_raw[y_test_dibujo==1, 0], X_test_raw[y_test_dibujo==1, 1], color='blue', label='Clase 1', edgecolor='k')
    plt.title("Valores Reales (Dataset Original)")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()

    # Subplot 2: Predicciones Cuánticas
    plt.subplot(1, 2, 2)
    plt.scatter(X_test_raw[predicciones_array==0, 0], X_test_raw[predicciones_array==0, 1], color='red', label='Predicción 0', edgecolor='k')
    plt.scatter(X_test_raw[predicciones_array==1, 0], X_test_raw[predicciones_array==1, 1], color='blue', label='Predicción 1', edgecolor='k')
    
    # Marcamos los errores con una 'X' grande verde para que se vean bien
    errores = (y_test_dibujo != predicciones_array)
    if np.any(errores):
        plt.scatter(X_test_raw[errores, 0], X_test_raw[errores, 1], color='lime', marker='x', s=100, label='Error de Predicción')

    plt.title(f"Predicción en {PROVEEDOR_DESTINO[0].upper()} (Acc: {precision:.2f}%)")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()

    plt.tight_layout()
    nombre_imagen = f"Resultados_{PROVEEDOR_DESTINO[0]}.png"
    plt.savefig(nombre_imagen, dpi=300)
    print(f"✅ Gráfica guardada como: {nombre_imagen}")
    plt.show()

    await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(evaluar_en_nube())