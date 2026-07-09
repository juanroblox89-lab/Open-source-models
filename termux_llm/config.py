# config.py
# Única fuente de verdad para todos los hiperparámetros del modelo y el tokenizador.

# Configuración del Modelo E (Elegida para producción en Termux)
MODEL_CONFIG = {
    'vocab_size': 32000,
    'd_model': 1280,
    'n_layers': 26,
    'n_heads': 20,
    'n_kv_heads': 5,          # Grouped Query Attention (GQA) con ratio 4:1
    'head_dim': 64,           # 1280 / 20 = 64
    'ffn_hidden': 3520,       # SwiGLU factor ~2.75x
    'window_size': 512,       # Sliding Window Attention
    'max_seq_len': 2048,
    'theta': 10000.0,         # RoPE base frequency
    'block_size': 32,         # Block size para cuantización Q4_0
}

# Configuración reducida para Pruebas de Integración y validación rápida
TEST_CONFIG = {
    'vocab_size': 512,
    'd_model': 128,
    'n_layers': 2,
    'n_heads': 4,
    'n_kv_heads': 2,          # GQA ratio 2:1
    'head_dim': 32,
    'ffn_hidden': 256,
    'window_size': 64,
    'max_seq_len': 128,
    'theta': 10000.0,
    'block_size': 32,
}

# Variable global para seleccionar la configuración activa
# Por defecto se usa la configuración de producción.
ACTIVE_CONFIG = MODEL_CONFIG

def set_active_config(use_test=False):
    global ACTIVE_CONFIG
    if use_test:
        ACTIVE_CONFIG = TEST_CONFIG
    else:
        ACTIVE_CONFIG = MODEL_CONFIG
    return ACTIVE_CONFIG

def get_config():
    return ACTIVE_CONFIG
