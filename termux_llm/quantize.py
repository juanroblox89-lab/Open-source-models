import numpy as np
import os

def quantize_q4_0(x, block_size=32):
    assert x.shape[-1] % block_size == 0
    x_reshaped = x.reshape(-1, block_size)
    
    abs_max = np.max(np.abs(x_reshaped), axis=1, keepdims=True)
    scale = abs_max / 7.0
    scale[scale == 0] = 1e-5
    
    x_q = np.round(x_reshaped / scale).astype(np.int8)
    x_q = np.clip(x_q, -8, 7)
    
    x_q_uint = (x_q + 8).astype(np.uint8)
    packed = (x_q_uint[:, 0::2] | (x_q_uint[:, 1::2] << 4))
    
    return packed, scale.astype(np.float16)

def dequantize_q4_0(packed, scale, shape, block_size=32):
    unpacked_0 = (packed & 0x0F).astype(np.int8) - 8
    unpacked_1 = ((packed >> 4) & 0x0F).astype(np.int8) - 8
    
    unpacked = np.empty((packed.shape[0], block_size), dtype=np.int8)
    unpacked[:, 0::2] = unpacked_0
    unpacked[:, 1::2] = unpacked_1
    
    deq = unpacked.astype(np.float32) * scale.astype(np.float32)
    return deq.reshape(shape)

def quantize_model_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.endswith('.npz') and 'layer' in file:
            data = np.load(os.path.join(input_dir, file))
            q_data = {}
            for k, v in data.items():
                if 'norm' not in k: # don't quantize norms
                    packed, scale = quantize_q4_0(v)
                    q_data[f'{k}_packed'] = packed
                    q_data[f'{k}_scale'] = scale
                    q_data[f'{k}_shape'] = np.array(v.shape)
                else:
                    q_data[k] = v
            np.savez(os.path.join(output_dir, file), **q_data)
            print(f"Quantized {file}")

if __name__ == '__main__':
    # Test
    x = np.random.randn(1280, 1280).astype(np.float32)
    p, s = quantize_q4_0(x)
    y = dequantize_q4_0(p, s, x.shape)
    error = np.mean(np.abs(x - y))
    print(f"Error medio de cuantización Q4_0: {error:.4f}")
