import numpy as np
import threading
import queue
import time
import os
import sys
from config import get_config
from model import transformer_block, rms_norm
from quantize import dequantize_q4_0
from tokenizer import BPETokenizer

model_dir = './q4_model'

def load_layer_from_disk(layer_idx, config):
    path = os.path.join(model_dir, f'layer_{layer_idx}.npz')
    data = np.load(path)
    weights = {}
    block_size = config.get('block_size', 32)
    for k in ['Wq', 'Wk', 'Wv', 'Wo', 'W_gate', 'W_up', 'W_down']:
        packed = data[f'{k}_packed']
        scale = data[f'{k}_scale']
        shape = tuple(data[f'{k}_shape'])
        weights[k] = dequantize_q4_0(packed, scale, shape, block_size=block_size)
    weights['attn_norm'] = data['attn_norm']
    weights['ffn_norm'] = data['ffn_norm']
    return weights

def layer_prefetcher(layer_idx, q, config):
    try:
        weights = load_layer_from_disk(layer_idx, config)
        q.put((layer_idx, weights))
    except Exception as e:
        q.put((layer_idx, e))

def generate(prompt, tokenizer, config, max_gen_len=100):
    # Verificación explícita de alineación del vocabulario para prevenir bugs silenciosos
    if tokenizer.vocab_size != config['vocab_size']:
        raise ValueError(
            f"ERROR DE ALINEACIÓN CRÍTICO: El vocabulario del tokenizador ({tokenizer.vocab_size}) "
            f"no coincide con la configuración del modelo ({config['vocab_size']}). "
            "Asegúrate de haber reentrenado el tokenizador con la configuración activa."
        )

    tokens = tokenizer.encode(prompt, add_bos=True)
    
    # Cargar pesos base del modelo (embeddings y norma final)
    base_path = os.path.join(model_dir, 'base.npz')
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"No se encontró base.npz en {model_dir}. ¿Ejecutaste el pipeline de entrenamiento primero?")
        
    base = np.load(base_path)
    embeddings = base['embeddings']
    final_norm = base['final_norm']
    
    # Preasignar la memoria fija para el KV-cache deslizante (FP16 para conservar aún más memoria RAM)
    kv_cache = []
    for _ in range(config['n_layers']):
        k_c = np.zeros((config['window_size'], config['n_kv_heads'], config['head_dim']), dtype=np.float16)
        v_c = np.zeros((config['window_size'], config['n_kv_heads'], config['head_dim']), dtype=np.float16)
        kv_cache.append((k_c, v_c))
        
    pos = 0
    x = embeddings[tokens]
    
    for step in range(max_gen_len):
        # Implementación de doble buffer: Prefetching asíncrono de la siguiente capa
        # mientras se computa la capa activa en el hilo principal
        layer_q = queue.Queue(maxsize=1)
        
        # Lanzar prefetch de la capa 0
        threading.Thread(target=layer_prefetcher, args=(0, layer_q, config)).start()
        
        for i in range(config['n_layers']):
            idx, res = layer_q.get()
            if isinstance(res, Exception):
                raise res
                
            weights = res
            
            # Lanzar prefetch de la capa i+1 si no estamos en la última capa
            if i < config['n_layers'] - 1:
                threading.Thread(target=layer_prefetcher, args=(i+1, layer_q, config)).start()
            
            # Convertir KV-cache de FP16 a FP32 para computar el bloque
            k_cache_fp32 = kv_cache[i][0].astype(np.float32)
            v_cache_fp32 = kv_cache[i][1].astype(np.float32)
            
            x, (k_out, v_out) = transformer_block(x, weights, (k_cache_fp32, v_cache_fp32), pos, config)
            
            # Almacenar de vuelta en FP16 para mantener el footprint de RAM extremadamente bajo
            kv_cache[i] = (k_out.astype(np.float16), v_out.astype(np.float16))
            
        # Proyectar logits usando el tied embedding
        x_norm = rms_norm(x, final_norm)
        logits = x_norm[-1:] @ embeddings.T
        
        next_token = int(np.argmax(logits, axis=-1)[0])
        yield next_token
        
        if next_token == tokenizer.special_tokens.get('<eos>', 257):
            break
            
        x = embeddings[[next_token]]
        pos += len(tokens) if step == 0 else 1
        tokens = [next_token]

if __name__ == '__main__':
    config = get_config()
    
    # Inicializar y cargar el tokenizador
    tok = BPETokenizer()
    tokenizer_file = 'tokenizer.json'
    if os.path.exists(tokenizer_file):
        tok.load(tokenizer_file)
    else:
        print("tokenizer.json no encontrado. Ejecutando entrenamiento rápido del pipeline...")
        from train import train_and_save_pipeline
        train_and_save_pipeline(config)
        tok.load(tokenizer_file)
        
    # Asegurar que el modelo cuantizado existe antes de iniciar
    if not os.path.exists(model_dir):
        print(f"Directorio {model_dir} no encontrado. Inicializando pesos base y cuantizando...")
        from quantize import quantize_model_directory
        quantize_model_directory('./fp32_model', model_dir, config)

    print("=== VERIFICACIÓN DEL ENTORNO DE EJECUCIÓN ===")
    np.show_config()
    print(f"Vocab size del modelo: {config['vocab_size']}")
    print(f"Vocab size del tokenizador: {tok.vocab_size}")
    print("=" * 45)
    
    prompt = "El mar es"
    print(f"Usuario: {prompt}")
    print("Transformer Edge: ", end='', flush=True)
    
    t0 = time.time()
    first_token_time = None
    gen_tokens = 0
    
    try:
        for token in generate(prompt, tok, config, max_gen_len=15):
            if first_token_time is None:
                first_token_time = time.time() - t0
            print(tok.decode([token]), end='', flush=True)
            gen_tokens += 1
            
        total_time = time.time() - t0
        print(f"\n\n[Stats] Tiempo hasta primer token: {first_token_time:.4f}s")
        if gen_tokens > 1:
            print(f"[Stats] Velocidad de generación: {gen_tokens / (total_time - first_token_time):.2f} tokens/s")
    except Exception as e:
        print(f"\nOcurrió un error en ejecución: {e}", file=sys.stderr)
