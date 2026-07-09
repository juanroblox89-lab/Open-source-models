import numpy as np
import os
from config import get_config
from tokenizer import BPETokenizer

# NOTA DE DISEÑO: Entrenar un LLM de 500M de parámetros desde cero en NumPy puro
# en CPU usando diferenciación automática implementada a mano es prohibitivamente
# lento e inestable. Este script sirve como demostración conceptual de la API de pesos
# y la preparación del pipeline. Entrena automáticamente el tokenizador y genera un
# conjunto de pesos maestros de prueba en FP32 (con inicialización aleatoria de
# distribución normal estandarizada de Xavier/He). Para entrenar el modelo de producción real,
# se debe usar PyTorch en un servidor con GPU, y exportar los pesos finales a los
# diccionarios .npz que lee nuestra inferencia.

def init_weights(config):
    weights = {}
    d_model = config['d_model']
    vocab_size = config['vocab_size']
    
    # Tied embeddings: embeddings de entrada y proyección de salida comparten memoria
    # Ahorra vocab_size * d_model parámetros (~41M)
    print(f"Inicializando embeddings de tamaño: {vocab_size} x {d_model}")
    weights['embeddings'] = np.random.normal(0, 0.02, (vocab_size, d_model)).astype(np.float32)
    weights['final_norm'] = np.ones(d_model, dtype=np.float32)
    
    for i in range(config['n_layers']):
        # Proyecciones de Atención GQA (Grouped Query Attention)
        weights[f'layer_{i}_Wq'] = np.random.normal(0, 0.02, (d_model, d_model)).astype(np.float32)
        weights[f'layer_{i}_Wk'] = np.random.normal(0, 0.02, (d_model, config['n_kv_heads'] * config['head_dim'])).astype(np.float32)
        weights[f'layer_{i}_Wv'] = np.random.normal(0, 0.02, (d_model, config['n_kv_heads'] * config['head_dim'])).astype(np.float32)
        weights[f'layer_{i}_Wo'] = np.random.normal(0, 0.02, (d_model, d_model)).astype(np.float32)
        weights[f'layer_{i}_attn_norm'] = np.ones(d_model, dtype=np.float32)
        
        # Red de alimentación hacia adelante (SwiGLU)
        weights[f'layer_{i}_W_gate'] = np.random.normal(0, 0.02, (d_model, config['ffn_hidden'])).astype(np.float32)
        weights[f'layer_{i}_W_up'] = np.random.normal(0, 0.02, (d_model, config['ffn_hidden'])).astype(np.float32)
        weights[f'layer_{i}_W_down'] = np.random.normal(0, 0.02, (config['ffn_hidden'], d_model)).astype(np.float32)
        weights[f'layer_{i}_ffn_norm'] = np.ones(d_model, dtype=np.float32)
        
    return weights

def save_checkpoint_for_streaming(weights, config, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar pesos base del modelo (embeddings y norma de salida final)
    base = {'embeddings': weights['embeddings'], 'final_norm': weights['final_norm']}
    np.savez(os.path.join(output_dir, 'base.npz'), **base)
    
    # Guardar cada capa por separado para habilitar el streaming desde disco (baja RAM)
    for i in range(config['n_layers']):
        layer_w = {k.replace(f'layer_{i}_', ''): v for k, v in weights.items() if k.startswith(f'layer_{i}_')}
        np.savez(os.path.join(output_dir, f'layer_{i}.npz'), **layer_w)
    
    print(f"Checkpoint FP32 completo guardado exitosamente para streaming en: {output_dir}")

def train_and_save_pipeline(config, output_dir='./fp32_model', tokenizer_path='tokenizer.json'):
    # Generar texto sintético educativo en español de alta calidad para entrenar el tokenizador
    print("Preparando corpus sintético de entrenamiento en español...")
    textos_sinteticos = [
        "El mar es un cuerpo de agua salada de gran extensión.",
        "Un modelo transformer procesa secuencias de tokens mediante auto-atención.",
        "En Termux sobre Android 14, optimizamos el rendimiento de la CPU ARM64.",
        "La cuantización Q4_0 reduce el tamaño en disco de los pesos de FP32 a 4 bits.",
        "Grouped Query Attention reduce significativamente el tamaño del KV-cache para inferencia veloz.",
        "SwiGLU ofrece mejor capacidad de modelado por parámetro que GELU o ReLU.",
        "NumPy y OpenBLAS permiten realizar operaciones vectoriales aceleradas en CPUs móviles.",
        "El prefetch asíncrono con doble buffer oculta la latencia de lectura de almacenamiento UFS.",
        "Nuestros embeddings posicionales rotatorios (RoPE) mantienen la consistencia espacial en secuencias."
    ]
    # Replicar para asegurar suficiente texto para el vocabulario completo
    corpus_text = " ".join(textos_sinteticos) * 200
    
    # 1. Entrenar y guardar el Tokenizador usando la única fuente de verdad (config)
    tokenizer = BPETokenizer(vocab_size=config['vocab_size'])
    tokenizer.train(corpus_text)
    tokenizer.save(tokenizer_path)
    
    # 2. Inicializar los pesos maestros en FP32
    print("Inicializando pesos maestros FP32 con distribución aleatoria normalizada...")
    weights = init_weights(config)
    
    # 3. Guardar el checkpoint listo para streaming
    save_checkpoint_for_streaming(weights, config, output_dir)
    print("Entrenamiento ficticio y pipeline de guardado completados con éxito.")

if __name__ == '__main__':
    config = get_config()
    train_and_save_pipeline(config)
