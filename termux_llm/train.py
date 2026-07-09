import numpy as np

# NOTA DE DISEÑO: Entrenar un LLM de 500M de parámetros desde cero en NumPy puro
# en CPU usando diferenciación automática implementada a mano es prohibitivamente
# lento e inestable. Este script sirve como demostración conceptual de la API de pesos.
# Para entrenar el modelo real, se debe usar PyTorch en un servidor con GPU, 
# y exportar los pesos finales a los diccionarios .npz que lee nuestra inferencia.

def init_weights(config):
    weights = {}
    d_model = config['d_model']
    # Tied embeddings
    weights['embeddings'] = np.random.normal(0, 0.02, (config['vocab_size'], d_model)).astype(np.float32)
    weights['final_norm'] = np.ones(d_model, dtype=np.float32)
    
    for i in range(config['n_layers']):
        # Attention
        weights[f'layer_{i}_Wq'] = np.random.normal(0, 0.02, (d_model, d_model)).astype(np.float32)
        weights[f'layer_{i}_Wk'] = np.random.normal(0, 0.02, (d_model, config['n_kv_heads'] * config['head_dim'])).astype(np.float32)
        weights[f'layer_{i}_Wv'] = np.random.normal(0, 0.02, (d_model, config['n_kv_heads'] * config['head_dim'])).astype(np.float32)
        weights[f'layer_{i}_Wo'] = np.random.normal(0, 0.02, (d_model, d_model)).astype(np.float32)
        weights[f'layer_{i}_attn_norm'] = np.ones(d_model, dtype=np.float32)
        
        # FFN
        weights[f'layer_{i}_W_gate'] = np.random.normal(0, 0.02, (d_model, config['ffn_hidden'])).astype(np.float32)
        weights[f'layer_{i}_W_up'] = np.random.normal(0, 0.02, (d_model, config['ffn_hidden'])).astype(np.float32)
        weights[f'layer_{i}_W_down'] = np.random.normal(0, 0.02, (config['ffn_hidden'], d_model)).astype(np.float32)
        weights[f'layer_{i}_ffn_norm'] = np.ones(d_model, dtype=np.float32)
        
    return weights

def save_checkpoint_for_streaming(weights, config, output_dir):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save base weights
    base = {'embeddings': weights['embeddings'], 'final_norm': weights['final_norm']}
    np.savez(os.path.join(output_dir, 'base.npz'), **base)
    
    # Save each layer separately for streaming
    for i in range(config['n_layers']):
        layer_w = {k.replace(f'layer_{i}_', ''): v for k, v in weights.items() if k.startswith(f'layer_{i}_')}
        np.savez(os.path.join(output_dir, f'layer_{i}.npz'), **layer_w)
    
    print(f"Checkpoint FP32 guardado en {output_dir}")

if __name__ == '__main__':
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
    print("Inicializando pesos FP32 aleatorios (referencia)...")
    w = init_weights(config)
    save_checkpoint_for_streaming(w, config, './fp32_model')
