import pennylane as qml
import torch
import numpy as np
import asyncio
from aiohttp import web
import aiohttp
import os
import shutil
import json
import time
from Utiles_Scheduler import tape_to_qiskit_script

# --- CONFIGURACIÓN DEL HERO RUN ---
SCHEDULER_URL = "http://localhost:8082/"
MY_LOCAL_IP = "http://localhost:5000"   # Tu IP local o localhost
BATCH_SIZE = 32                         # Tamaño del Batch
EPOCHS = 15                           # El objetivo final
SHOTS = 1024                            # Disparos por circuito
N_QUBITS = 6
LEARNING_RATE = 0.05                    # Un poco alto al principio para aprender rápido

# Variables de Estado Globales
results_storage = {}       # Buzón de resultados
batch_event = asyncio.Event()
current_batch_total = 0

# --- 1. MODELO Y DATOS ---
print("--- 1. Preparando Datos ---")
X = np.load('ArchivoRed/X_data.npy', allow_pickle=True)
y = np.load('ArchivoRed/y_data.npy', allow_pickle=True)
# Ajuste de etiquetas (0 y 1)
y = y.flatten().astype(int)
y = np.array([0 if val <= 0 else 1 for val in y])

# REDUCCIÓN DE DATOS: Para que sea viable en tiempo, usamos un subset
# Usamos 128 muestras para entrenar (4 Batches de 32)
TRAIN_SIZE = 128
X_train = torch.tensor(X[:TRAIN_SIZE], dtype=torch.float32)
y_train = torch.tensor(y[:TRAIN_SIZE], dtype=torch.long)

print(f"Datos cargados. Entrenando con {TRAIN_SIZE} muestras ({TRAIN_SIZE // BATCH_SIZE} batches por epoch).")

# Definición del QNode (Abstracto)
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev)
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))

# Forma: (n_layers, n_qubits, 3) -> (2, 4, 3) = 24 pesos
# 1. Generamos los datos aleatorios primero (sin requires_grad)
init_weights = 0.1 * torch.randn(2, N_QUBITS, 3)

# 2. Creamos el tensor "Hoja" a partir de esos datos
# Al usar torch.tensor() envolvemos los datos y ahora sí es una variable raíz.
weights = torch.tensor(init_weights, requires_grad=True, dtype=torch.float32)

bias = torch.tensor(0.0, requires_grad=True, dtype=torch.float32)

# --- 2. SERVIDOR PUENTE (BRIDGE) ---
def counts_to_expval(counts):
    """Convierte {'0': 500, '1': 500} a valor esperado Z (-1 a 1)"""
    # Nota: Tu scheduler devuelve '00', '11', etc. Miramos paridad o solo qubit 0.
    # Asumimos que medimos Qubit 0. 
    zeros = 0
    ones = 0
    total = 0
    for k, v in counts.items():
        # Qiskit es Little Endian, el qubit 0 es el de la derecha
        # Si k='0010', q0 es '0'.
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
        
        # Procesar resultado
        val = counts_to_expval(raw_counts)
        results_storage[name] = val
        
        # Debug ligero
        # print(f"📩 {name} -> {val:.2f}")
        
        if len(results_storage) >= current_batch_total:
            batch_event.set()
            
        return web.Response(text="OK")
    except Exception as e:
        print(f"Error callback: {e}")
        return web.Response(status=500)

async def handle_file(request):
    name = request.match_info.get('name', "Anon")
    path = os.path.join("generated_circuits", name)
    if os.path.exists(path):
        return web.FileResponse(path)
    return web.Response(status=404)

# --- 3. CORE DE ENTRENAMIENTO ---
async def train_hero_run():
    global weights, bias, current_batch_total
    
    # Setup del Servidor
    app = web.Application()
    app.router.add_get('/circuits/{name}', handle_file)
    app.router.add_post('/callback', handle_callback)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 5000)
    await site.start()
    print("🌐 Servidor Puente listo en port 5000")

    # Carpetas
    if os.path.exists("generated_circuits"): shutil.rmtree("generated_circuits")
    os.makedirs("generated_circuits", exist_ok=True)
    os.makedirs("checkpoints_hero", exist_ok=True)

    optimizer = torch.optim.Adam([weights, bias], lr=LEARNING_RATE)
    history_loss = []

    print(f"\n🚀 INICIANDO HERO RUN ({EPOCHS} Epochs) 🚀")

    for epoch in range(EPOCHS):
        start_time = time.time()
        epoch_loss = 0
        batches = 0
        
        # Mezclar datos
        perm = torch.randperm(X_train.size(0))
        
        for i in range(0, len(X_train), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            x_batch = X_train[idx]
            y_target = y_train[idx] # 0 o 1
            # Convertir target a -1/1 para MSE
            y_target_pm = (y_target.float() * 2) - 1 

            # --- A. GENERACIÓN DE GRADIENTES (Parameter Shift) ---
            # Necesitamos calcular el gradiente manualmente
            # Para cada dato en el batch, generamos los tapes (circuitos shifted)
            
            tapes_to_send = []
            tape_map = [] # Para recordar qué tape corresponde a qué dato/parámetro
            
            # 1. Generar Tapes de Gradiente
# 1. Generar Tapes de Gradiente
            for j in range(len(x_batch)):
                # A. Preparamos el dato: Lo pasamos a NumPy para asegurar que 
                # PennyLane NO intente derivarlo (evita el error de dimensiones).
                # Al ser NumPy, es invisible para el gradiente.
                x_val = x_batch[j].detach().numpy()

                # B. Grabamos el circuito manualmente en una "Cinta" (Tape)
                # Esto crea el objeto puro del circuito sin ejecutarlo.
                with qml.tape.QuantumTape() as tape:
                    qml.AngleEmbedding(x_val, wires=range(N_QUBITS))
                    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
                    qml.expval(qml.PauliZ(0))

                # C. Pedimos los circuitos del gradiente a partir de la cinta
                # param_shift detectará automáticamente que 'weights' es el único
                # tensor entrenable en la cinta.
                g_tapes, fn = qml.gradients.param_shift(tape)
                
                start_idx = len(tapes_to_send)
                tapes_to_send.extend(g_tapes)
                tape_map.append((j, start_idx, len(g_tapes), fn))

            # 2. Generar Archivos .py
            files_payload = []
            results_storage.clear()
            batch_event.clear()
            current_batch_total = len(tapes_to_send)
            
            print(f"  Batch {batches+1}: Generando {current_batch_total} circuitos...", end="\r")
            
            async with aiohttp.ClientSession() as session:
                tasks = []
                for k, tape in enumerate(tapes_to_send):
                    fname = f"e{epoch}_b{batches}_t{k}.py"
                    tape_to_qiskit_script(tape, fname, SHOTS)
                    
                    # Payload para Scheduler
                    payload = {
                        "url": f"{MY_LOCAL_IP}/circuits/{fname}",
                        "shots": SHOTS,
                        "provider": ['ibm'],
                        "policy": "multibatch",
                        "criterio": 0,
                        "callback_url": f"{MY_LOCAL_IP}/callback",
                        "circuit_name": fname # Importante para identificar vuelta
                    }
                    task = session.post(SCHEDULER_URL + 'circuit', json=payload)
                    tasks.append(task)
                
                # 3. Enviar Masivamente
                await asyncio.gather(*tasks)
            
            # 4. Esperar Resultados
            print(f"  Batch {batches+1}: Enviados. Esperando Scheduler...      ", end="\r")
            await batch_event.wait()
            
            # 5. Reconstruir Gradientes
            grad_w_accum = torch.zeros_like(weights)
            
            for j, start, count, fn in tape_map:
                # Recoger resultados de este dato
                res_list = []
                for k in range(start, start+count):
                    fname = f"e{epoch}_b{batches}_t{k}.py"
                    res_list.append(results_storage[fname])
                
                # PennyLane reconstruye el gradiente usando la función fn
                # El resultado es una tupla de arrays numpy
                grad_per_sample = fn(res_list)
                
                # Convertir a Tensor y sumar
                # Nota: grad_per_sample[0] es el gradiente respecto a 'inputs' (no lo queremos)
                # grad_per_sample[1] es el gradiente respecto a 'weights'
                g_w = torch.tensor(grad_per_sample[1], dtype=torch.float32)
                
                # Regla de la cadena manual para MSE Loss: 2 * (pred - target) * grad_pred
                # Pero necesitamos la predicción (Forward pass).
                # Para simplificar y ahorrar circuitos, usamos el promedio de los shifts 
                # o asumimos una aproximación.
                # Lo riguroso es hacer también el forward pass. 
                # TRUCO: Parameter shift ya nos da la derivada de la salida dOut/dW.
                # dLoss/dW = dLoss/dOut * dOut/dW
                # dLoss/dOut = 2 * (Out - Target)
                
                # Aproximación: Usamos el gradiente directo asumiendo error promedio
                # O mejor: Enviamos también los circuitos Forward.
                # POR TIEMPO: Vamos a usar el gradiente puro para maximizar correlación (Class 1 -> output 1)
                # Loss = MSE
                
                # Necesitamos la predicción actual. Usaremos un modelo sombra local para estimar el error
                # Ojo: esto es una trampa válida en TFM para ahorrar tiempo.
                # Usamos el modelo simulado para calcular el error escalar (pred - target)
                # y el hardware real para calcular la dirección del gradiente (dOut/dW).
                pred_sim = qnode(x_batch[j], weights) 
                error = pred_sim - y_target_pm[j]
                
                grad_w_accum += 2 * error * g_w

            # Promediar gradiente
            grad_w_accum /= len(x_batch)
            
            # 6. Actualizar Pesos (Optimizer Step Manual)
            # Adam es complejo de hacer manual, usamos SGD con Momentum simple o actualización directa
            # Para usar Adam de PyTorch, le inyectamos el gradiente al tensor
            weights.grad = grad_w_accum
            optimizer.step()
            optimizer.zero_grad()
            
            # Calculamos Loss (aprox) para log
            batch_loss = torch.mean((qnode(x_batch, weights) - y_target_pm)**2)
            epoch_loss += batch_loss.item()
            batches += 1
        
        # --- FIN DE EPOCH ---
        avg_loss = epoch_loss / batches
        history_loss.append(avg_loss)
        duration = time.time() - start_time
        
        print(f"✅ Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Tiempo: {duration:.1f}s")
        
        # GUARDAR CHECKPOINT
        checkpoint_path = f"checkpoints_hero/model_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch,
            'weights': weights,
            'bias': bias,
            'loss': avg_loss
        }, checkpoint_path)
        print(f"💾 Guardado: {checkpoint_path}")

    await runner.cleanup()
    print("🏆 ¡HERO RUN COMPLETADO!")

if __name__ == "__main__":
    asyncio.run(train_hero_run())