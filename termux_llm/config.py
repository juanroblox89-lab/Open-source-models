# config.py
# Única fuente de verdad para todos los hiperparámetros del modelo y el tokenizador.

import os
# Optimizar el subproceso de álgebra lineal (OpenBLAS/MKL/etc) para usar un número eficiente de hilos.
# Por defecto en el Redmi Note 14 (Dimensity 6100+) se prefiere usar 2-4 hilos para evitar
# que la sobrecarga de sincronización degrade el rendimiento en los núcleos Cortex-A55 pequeños.
# Configurando estas variables antes de que NumPy se importe por primera vez.
for env_var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    if env_var not in os.environ:
        os.environ[env_var] = "4"

# Configuración del Modelo E (Elegida para producción en Termux - Parámetros Cuatriplicados)
MODEL_CONFIG = {
    'vocab_size': 32000,
    'd_model': 2560,          # Duplicado de 1280 (parámetros escalados 4x)
    'n_layers': 26,
    'n_heads': 40,            # Duplicado de 20 para mantener head_dim de 64
    'n_kv_heads': 10,         # Grouped Query Attention (GQA) con ratio 4:1
    'head_dim': 64,           # 2560 / 40 = 64
    'ffn_hidden': 7040,       # SwiGLU factor ~2.75x (duplicado de 3520)
    'window_size': 512,       # Sliding Window Attention
    'max_seq_len': 2048,
    'theta': 10000.0,         # RoPE base frequency
    'block_size': 32,         # Block size para cuantización Q4_0
}

# Configuración reducida para Pruebas de Integración y validación rápida
TEST_CONFIG = {
    'vocab_size': 512,
    'd_model': 256,           # Duplicado de 128
    'n_layers': 4,            # Duplicado de 2
    'n_heads': 8,             # Duplicado de 4
    'n_kv_heads': 4,          # GQA ratio 2:1
    'head_dim': 32,
    'ffn_hidden': 512,        # Duplicado de 256
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
