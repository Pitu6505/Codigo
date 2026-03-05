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
MY_LOCAL_IP = "http://localhost:5000"
BATCH_SIZE = 32
EPOCHS = 15
SHOTS = 1024
N_QUBITS = 6
LEARNING_RATE = 0.05
CHECKPOINT_FILE = "checkpoints_hero/checkpoint_latest.pth" # Archivo maestro de guardado

# Variables Globales
results_storage = {}
batch_event = asyncio.Event()
current_batch_total = 0

# --- 1. MODELO Y DATOS ---
print("--- 1. Preparando Datos ---")
X = np.load('ArchivoRed/X_data.npy', allow_pickle=True)
y = np.load('ArchivoRed/y_data.npy', allow_pickle=True)
y = y.flatten().astype(int)
y = np.array([0 if val <= 0 else 1 for val in y])

TRAIN_SIZE = 128
X_train = torch.tensor(X[:TRAIN_SIZE], dtype=torch.float32)
y_train = torch.tensor(y[:TRAIN_SIZE], dtype=torch.long)

# 🟢 NORMALIZACIÓN: Escalar X_train entre 0 y Pi para AngleEmbedding
x_min = X_train.min(dim=0, keepdim=True)[0]
x_max = X_train.max(dim=0, keepdim=True)[0]
# Evitar división por cero
x_max = torch.where(x_max == x_min, x_max + 1e-8, x_max)
X_train = np.pi * (X_train - x_min) / (x_max - x_min)

print(f"✅ Datos cargados y normalizados. {TRAIN_SIZE} muestras.")

# Definición QNode
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev)
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))

# Inicialización de Pesos y Bias
init_weights = 0.1 * torch.randn(2, N_QUBITS, 3)
weights = torch.tensor(init_weights, requires_grad=True, dtype=torch.float32)
bias = torch.tensor(0.0, requires_grad=True, dtype=torch.float32)

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
        
        if len(results_storage) >= current_batch_total:
            batch_event.set()
        return web.Response(text="OK")
    except Exception as e:
        print(f"Error callback: {e}")
        return web.Response(status=500)

# 🟢 SERVIR ARCHIVOS: Permite al Scheduler descargar el archivo .py generado
async def handle_file(request):
    name = request.match_info.get('name', "Anon")
    path = os.path.join("generated_circuits", name)
    if os.path.exists(path):
        return web.FileResponse(path)
    return web.Response(status=404)

# --- 3. FUNCIONES DE CHECKPOINT ---
def save_checkpoint(epoch, batch_idx, optimizer, loss):
    """Guarda el estado actual para poder reanudar si se va la luz."""
    torch.save({
        'epoch': epoch,
        'batch_idx': batch_idx,
        'weights': weights, 
        'bias': bias,
        'optimizer_state': optimizer.state_dict(),
        'loss': loss,
        'x_min': x_min, # Guardamos factores de normalización por seguridad
        'x_max': x_max
    }, CHECKPOINT_FILE)

def load_checkpoint(optimizer):
    """Intenta cargar un entrenamiento previo."""
    if os.path.exists(CHECKPOINT_FILE):
        print(f"🔄 Encontrado checkpoint previo en {CHECKPOINT_FILE}")
        checkpoint = torch.load(CHECKPOINT_FILE)
        
        with torch.no_grad():
            weights.data = checkpoint['weights'].data
            bias.data = checkpoint['bias'].data
        
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch']
        start_batch = checkpoint['batch_idx'] + 1 
        
        print(f"⏩ Reanudando entrenamiento desde Epoch {start_epoch}, Batch {start_batch}")
        return start_epoch, start_batch
    else:
        print("🆕 Iniciando entrenamiento desde cero.")
        return 0, 0

# --- 4. CORE DE ENTRENAMIENTO ---
async def train_hero_run():
    global weights, bias, current_batch_total
    
    # Setup Servidor
    app = web.Application()
    app.router.add_get('/circuits/{name}', handle_file)
    app.router.add_post('/callback', handle_callback)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 5000)
    await site.start()
    print("🌐 Servidor Puente listo en port 5000")

    os.makedirs("generated_circuits", exist_ok=True)
    os.makedirs("checkpoints_hero", exist_ok=True)

    optimizer = torch.optim.Adam([weights, bias], lr=LEARNING_RATE)
    
    # INTENTO DE CARGA DE CHECKPOINT
    start_epoch, start_batch_global = load_checkpoint(optimizer)

    print(f"\n🚀 INICIANDO HERO RUN ({EPOCHS} Epochs) 🚀")
    print(f"📊 Pesos iniciales - Min: {weights.min().item():.6f}, Max: {weights.max().item():.6f}, Media: {weights.mean().item():.6f}")
    print(f"📊 Bias inicial: {bias.item():.6f}")

    for epoch in range(start_epoch, EPOCHS):
        start_time = time.time()
        epoch_loss = 0
        batches_done = 0
        
        perm = torch.randperm(X_train.size(0))
        batch_counter = 0
        batch_start_from = start_batch_global if epoch == start_epoch else 0

        for i in range(0, len(X_train), BATCH_SIZE):
            
            if batch_counter < batch_start_from:
                print(f"⏩ Saltando Batch {batch_counter+1} (Ya procesado anteriormente)")
                batch_counter += 1
                continue 

            idx = perm[i:i+BATCH_SIZE]
            x_batch = X_train[idx]
            y_target = y_train[idx]
            y_target_pm = (y_target.float() * 2) - 1 

            tapes_to_send = []
            tape_map = [] 
            
            # 1. Generar Tapes 
            for j in range(len(x_batch)):
                x_val = x_batch[j].detach().numpy()
                with qml.tape.QuantumTape() as tape:
                    qml.AngleEmbedding(x_val, wires=range(N_QUBITS))
                    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
                    qml.expval(qml.PauliZ(0))

                g_tapes, fn = qml.gradients.param_shift(tape)
                
                start_idx = len(tapes_to_send)
                tapes_to_send.extend(g_tapes)
                tape_map.append((j, start_idx, len(g_tapes), fn))

            # 2. Enviar al Scheduler
            results_storage.clear()
            batch_event.clear()
            current_batch_total = len(tapes_to_send)
            
            print(f"  Epoch {epoch+1} - Batch {batch_counter+1}: Generando {current_batch_total} circuitos...", end="\r")
            
            async with aiohttp.ClientSession() as session:
                tasks = []
                for k, tape in enumerate(tapes_to_send):
                    fname = f"e{epoch}_b{batch_counter}_t{k}.py"
                    tape_to_qiskit_script(tape, fname, SHOTS)
                    
                    payload = {
                        "url": f"{MY_LOCAL_IP}/circuits/{fname}", 
                        "shots": SHOTS,
                        "provider": ['ibm'],
                        "policy": "multibatch",
                        "criterio": 0,
                        "callback_url": f"{MY_LOCAL_IP}/callback",
                        "circuit_name": fname,
                        "code": open(f"generated_circuits/{fname}", "r").read()
                    }
                    task = session.post(SCHEDULER_URL + 'circuit', json=payload)
                    tasks.append(task)
                
                await asyncio.gather(*tasks)
            
            print(f"  Epoch {epoch+1} - Batch {batch_counter+1}: Esperando resultados IBM...      ", end="\r")
            await batch_event.wait()
            
            # 3. Calcular Gradientes y Actualizar
            grad_w_accum = torch.zeros_like(weights)
            grad_b_accum = torch.tensor(0.0) 
            
            for j, start, count, fn in tape_map:
                res_list = []
                for k in range(start, start+count):
                    fname = f"e{epoch}_b{batch_counter}_t{k}.py"
                    val = results_storage.get(fname, 0.0) 
                    res_list.append(val)
                
                grad_per_sample = fn(res_list)
                g_w = torch.tensor(grad_per_sample[0], dtype=torch.float32)
                
                pred_sim = qnode(x_batch[j], weights) + bias 
                error = pred_sim - y_target_pm[j]
                
                grad_w_accum += 2 * error * g_w
                grad_b_accum += 2 * error 

            grad_w_accum /= len(x_batch)
            grad_b_accum /= len(x_batch) 
            
            weights.grad = grad_w_accum
            bias.grad = grad_b_accum 
            
            # 🟢 CORRECCIÓN: Guardar estado antes de actualizar para calcular el cambio
            weights_before = weights.clone().detach()

            optimizer.step()
            optimizer.zero_grad()
            
            # 4. Calcular métricas y mostrar depuración
            # 🟢 CORRECCIÓN: Calcular la pérdida de forma segura sumando el bias
            with torch.no_grad():
                preds_batch = torch.stack([qnode(x_batch[j], weights) + bias for j in range(len(x_batch))])
                batch_loss = torch.mean((preds_batch - y_target_pm)**2)
            
            epoch_loss += batch_loss.item()
            
            weight_change = (weights - weights_before).abs().mean().item()
            grad_norm = grad_w_accum.norm().item()
            
            print(f"\n  📈 Epoch {epoch+1} - Batch {batch_counter+1}:")
            print(f"     Loss: {batch_loss.item():.6f}")
            print(f"     Grad norm: {grad_norm:.6f}")
            print(f"     Cambio pesos: {weight_change:.6f}")
            print(f"     Pesos - Min: {weights.min().item():.6f}, Max: {weights.max().item():.6f}, Media: {weights.mean().item():.6f}")
            print(f"     Bias: {bias.item():.6f}")
            
            save_checkpoint(epoch, batch_counter, optimizer, batch_loss.item())
            
            # Limpiar circuitos generados de este batch
            for k in range(len(tapes_to_send)):
                fname = f"e{epoch}_b{batch_counter}_t{k}.py"
                fpath = os.path.join("generated_circuits", fname)
                if os.path.exists(fpath):
                    os.remove(fpath)
            print(f"     🗑️  {len(tapes_to_send)} circuitos limpiados")
            
            batches_done += 1
            batch_counter += 1
        
        # Fin de Epoch
        avg_loss = epoch_loss / batches_done if batches_done > 0 else 0
        duration = time.time() - start_time
        print(f"\n✅ Epoch {epoch+1} Completada | Loss promedio: {avg_loss:.6f} | Tiempo: {duration:.1f}s")
        print(f"📊 Pesos finales epoch - Min: {weights.min().item():.6f}, Max: {weights.max().item():.6f}, Media: {weights.mean().item():.6f}")
        print("=" * 80)

    await runner.cleanup()
    print("🏆 ¡HERO RUN COMPLETADO!")

if __name__ == "__main__":
    asyncio.run(train_hero_run())