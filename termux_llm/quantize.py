import numpy as np
import os
import shutil
from config import get_config

def quantize_q4_0(x, block_size=32):
    """
    Cuantización Q4_0 optimizada y vectorizada en NumPy puro.
    Divide la matriz en bloques independientes, calcula un factor de escala (float16) por bloque,
    escala los valores al rango [-8, 7], y los empaqueta en enteros de 4 bits.
    """
    assert x.shape[-1] % block_size == 0, f"La última dimensión de la matriz ({x.shape[-1]}) debe ser divisible por el tamaño del bloque ({block_size})"
    x_reshaped = x.reshape(-1, block_size)
    
    # Obtener el valor absoluto máximo por bloque para calcular la escala
    abs_max = np.max(np.abs(x_reshaped), axis=1, keepdims=True)
    scale = abs_max / 7.0
    scale[scale == 0] = 1e-5  # Prevenir división por cero
    
    # Escalar, redondear y recortar al rango de 4 bits con signo [-8, 7]
    x_q = np.round(x_reshaped / scale).astype(np.int8)
    x_q = np.clip(x_q, -8, 7)
    
    # Mapear del rango [-8, 7] a enteros sin signo [0, 15] para empaquetado seguro
    x_q_uint = (x_q + 8).astype(np.uint8)
    
    # Empaquetar dos valores de 4 bits en un solo byte (uint8)
    # Valores en posiciones pares ocupan los 4 bits bajos; impares los 4 bits altos
    packed = (x_q_uint[:, 0::2] | (x_q_uint[:, 1::2] << 4))
    
    return packed, scale.astype(np.float16)

def dequantize_q4_0(packed, scale, shape, block_size=32):
    """
    Descuantización ultra veloz de Q4_0 a FP32 usando operaciones de máscara de bits en NumPy.
    """
    # Desempaquetar los bits y restaurar el desplazamiento con signo (-8)
    unpacked_0 = (packed & 0x0F).astype(np.int8) - 8
    unpacked_1 = ((packed >> 4) & 0x0F).astype(np.int8) - 8
    
    unpacked = np.empty((packed.shape[0], block_size), dtype=np.int8)
    unpacked[:, 0::2] = unpacked_0
    unpacked[:, 1::2] = unpacked_1
    
    # Multiplicar por el factor de escala de bloque correspondiente y redimensionar a la forma original
    deq = unpacked.astype(np.float32) * scale.astype(np.float32)
    return deq.reshape(shape)

def quantize_model_directory(input_dir, output_dir, config):
    """
    Recorre el directorio del modelo FP32 base y cuantiza todas las capas al formato optimizado Q4_0.
    Las normas (RMSNorm) y embeddings base no se cuantizan para preservar la fidelidad de la señal.
    """
    os.makedirs(output_dir, exist_ok=True)
    block_size = config.get('block_size', 32)
    
    # Copiar base.npz (contiene embeddings y normas finales no cuantizadas por estabilidad)
    base_src = os.path.join(input_dir, 'base.npz')
    base_dst = os.path.join(output_dir, 'base.npz')
    if os.path.exists(base_src):
        shutil.copy(base_src, base_dst)
        print(f"Copiados embeddings y normas base a {output_dir}")
        
    for file in os.listdir(input_dir):
        if file.endswith('.npz') and 'layer' in file:
            data = np.load(os.path.join(input_dir, file))
            q_data = {}
            for k, v in data.items():
                if 'norm' not in k:  # Preservar estabilidad: las normas permanecen en FP32
                    packed, scale = quantize_q4_0(v, block_size=block_size)
                    q_data[f'{k}_packed'] = packed
                    q_data[f'{k}_scale'] = scale
                    q_data[f'{k}_shape'] = np.array(v.shape)
                else:
                    q_data[k] = v
            np.savez(os.path.join(output_dir, file), **q_data)
            print(f"Cuantizado exitosamente a Q4_0: {file}")

if __name__ == '__main__':
    config = get_config()
    print("Iniciando prueba de cuantización local de matriz aleatoria...")
    x = np.random.randn(config['d_model'], config['d_model']).astype(np.float32)
    packed, scale = quantize_q4_0(x, block_size=config['block_size'])
    y = dequantize_q4_0(packed, scale, x.shape, block_size=config['block_size'])
    error = np.mean(np.abs(x - y))
    print(f"Error de cuantización absoluto medio para matriz {x.shape}: {error:.6f}")
    
    # Si existen pesos del modelo base, los cuantizamos
    if os.path.exists('./fp32_model'):
        print("Directorio './fp32_model' detectado. Cuantizando modelo completo...")
        quantize_model_directory('./fp32_model', './q4_model', config)
