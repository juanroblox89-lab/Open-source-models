import numpy as np
import threading
import queue
import time
import os
import sys
from model import transformer_block, rms_norm
from quantize import dequantize_q4_0
from tokenizer import BPETokenizer

config = {
    'vocab_size': 32000,
    'd_model': 1280,
    'n_layers': 26,
    'n_heads': 20,
    'n_kv_heads': 5,
    'head_dim': 64,
    'ffn_hidden': 3520,
    'window_size': 512
}

model_dir = './q4_model'

def load_layer_from_disk(layer_idx):
    path = os.path.join(model_dir, f'layer_{layer_idx}.npz')
    data = np.load(path)
    weights = {}
    for k in ['Wq', 'Wk', 'Wv', 'Wo', 'W_gate', 'W_up', 'W_down']:
        packed = data[f'{k}_packed']
        scale = data[f'{k}_scale']
        shape = tuple(data[f'{k}_shape'])
        weights[k] = dequantize_q4_0(packed, scale, shape)
    weights['attn_norm'] = data['attn_norm']
    weights['ffn_norm'] = data['ffn_norm']
    return weights

def layer_prefetcher(layer_idx, q):
    weights = load_layer_from_disk(layer_idx)
    q.put((layer_idx, weights))

def generate(prompt, tokenizer, max_gen_len=100):
    tokens = tokenizer.encode(prompt, add_bos=True)
    
    # Load base weights
    base = np.load(os.path.join(model_dir, 'base.npz'))
    embeddings = base['embeddings']
    final_norm = base['final_norm']
    
    # Preallocate KV cache (FP32 here for simplicity, FP16 in production)
    kv_cache = []
    for _ in range(config['n_layers']):
        k_c = np.zeros((config['window_size'], config['n_kv_heads'], config['head_dim']), dtype=np.float32)
        v_c = np.zeros((config['window_size'], config['n_kv_heads'], config['head_dim']), dtype=np.float32)
        kv_cache.append((k_c, v_c))
        
    pos = 0
    x = embeddings[tokens]
    
    for step in range(max_gen_len):
        layer_q = queue.Queue(maxsize=1)
        # Prefetch layer 0
        threading.Thread(target=layer_prefetcher, args=(0, layer_q)).start()
        
        for i in range(config['n_layers']):
            idx, weights = layer_q.get()
            if i < config['n_layers'] - 1:
                threading.Thread(target=layer_prefetcher, args=(i+1, layer_q)).start()
            
            x, kv_cache[i] = transformer_block(x, weights, kv_cache[i], pos, config)
            
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
    # Initialize tokenizer
    tok = BPETokenizer()
    if os.path.exists('tokenizer.json'):
        tok.load('tokenizer.json')
    
    # Ensure model exists for test
    if not os.path.exists(model_dir):
        print("Generando pesos aleatorios de prueba...")
        os.system('python train.py')
        os.system('python quantize.py')
        from quantize import quantize_model_directory
        quantize_model_directory('./fp32_model', './q4_model')
        import shutil
        shutil.copy('./fp32_model/base.npz', './q4_model/base.npz')

    print("Verificando optimización BLAS:")
    np.show_config()
    print("-" * 50)
    
    prompt = "Escribe un poema sobre el mar:"
    print(f"Usuario: {prompt}")
    print("Fable 5 Edge: ", end='', flush=True)
    
    t0 = time.time()
    first_token_time = None
    gen_tokens = 0
    
    for token in generate(prompt, tok, max_gen_len=20):
        if first_token_time is None:
            first_token_time = time.time() - t0
        print(tok.decode([token]), end='', flush=True)
        gen_tokens += 1
        
    total_time = time.time() - t0
    print(f"\n\n[Stats] Time to first token: {first_token_time:.2f}s")
    print(f"[Stats] Generation speed: {gen_tokens / (total_time - first_token_time):.2f} tokens/s")
