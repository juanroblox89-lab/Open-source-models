import numpy as np

def rms_norm(x, weight, eps=1e-5):
    variance = np.mean(x**2, axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(variance + eps)) * weight

def apply_rope(q, k, pos, theta=10000.0):
    seq_len, n_heads, head_dim = q.shape
    _, n_kv_heads, _ = k.shape
    
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2)[: (head_dim // 2)] / head_dim))
    pos_arr = np.arange(pos, pos + seq_len)
    freqs = np.outer(pos_arr, freqs)
    
    freqs_cos = np.cos(freqs)
    freqs_sin = np.sin(freqs)
    
    q_r, q_i = q[..., 0::2], q[..., 1::2]
    k_r, k_i = k[..., 0::2], k[..., 1::2]
    
    q_out = np.empty_like(q)
    q_out[..., 0::2] = q_r * np.expand_dims(freqs_cos, 1) - q_i * np.expand_dims(freqs_sin, 1)
    q_out[..., 1::2] = q_r * np.expand_dims(freqs_sin, 1) + q_i * np.expand_dims(freqs_cos, 1)
    
    k_out = np.empty_like(k)
    k_out[..., 0::2] = k_r * np.expand_dims(freqs_cos, 1) - k_i * np.expand_dims(freqs_sin, 1)
    k_out[..., 1::2] = k_r * np.expand_dims(freqs_sin, 1) + k_i * np.expand_dims(freqs_cos, 1)
    
    return q_out, k_out

def swiglu(x, W_gate, W_up, W_down):
    gate = x @ W_gate
    up = x @ W_up
    gate = gate * (1.0 / (1.0 + np.exp(-gate))) # SiLU activation
    return (gate * up) @ W_down

def gqa_attention(x, Wq, Wk, Wv, Wo, kv_cache, pos, config):
    seq_len = x.shape[0]
    n_heads = config['n_heads']
    n_kv_heads = config['n_kv_heads']
    head_dim = config['head_dim']
    window_size = config['window_size']
    theta = config.get('theta', 10000.0)
    
    q = x @ Wq
    k = x @ Wk
    v = x @ Wv
    
    q = q.reshape(seq_len, n_heads, head_dim)
    k = k.reshape(seq_len, n_kv_heads, head_dim)
    v = v.reshape(seq_len, n_kv_heads, head_dim)
    
    q, k = apply_rope(q, k, pos, theta=theta)
    
    if kv_cache is not None:
        cache_k, cache_v = kv_cache
        
        if seq_len > 1: # prefill mode
            k_use = np.concatenate([cache_k[:pos], k], axis=0)[-window_size:]
            v_use = np.concatenate([cache_v[:pos], v], axis=0)[-window_size:]
        else: # decode mode
            cache_k = np.roll(cache_k, -1, axis=0) if pos >= window_size else cache_k
            cache_v = np.roll(cache_v, -1, axis=0) if pos >= window_size else cache_v
            insert_pos = min(pos, window_size - 1)
            cache_k[insert_pos] = k[0]
            cache_v[insert_pos] = v[0]
            k_use = cache_k[:insert_pos+1]
            v_use = cache_v[:insert_pos+1]
            
        kv_cache_out = (cache_k, cache_v)
    else:
        k_use, v_use = k, v
        kv_cache_out = None

    repeats = n_heads // n_kv_heads
    k_rep = np.repeat(k_use, repeats, axis=1)
    v_rep = np.repeat(v_use, repeats, axis=1)
    
    scores = np.einsum('shd,chd->hsc', q, k_rep) / np.sqrt(head_dim)
    
    if seq_len > 1:
        mask = np.triu(np.full((seq_len, k_use.shape[0]), -np.inf), k=1 + (k_use.shape[0] - seq_len))
        scores += mask
        
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    out = np.einsum('hsc,chd->shd', attn_weights, v_rep)
    out = out.reshape(seq_len, n_heads * head_dim)
    
    return out @ Wo, kv_cache_out

def transformer_block(x, weights, kv_cache, pos, config):
    norm1 = rms_norm(x, weights['attn_norm'])
    attn_out, kv_cache_out = gqa_attention(
        norm1, weights['Wq'], weights['Wk'], weights['Wv'], weights['Wo'],
        kv_cache, pos, config
    )
    x = x + attn_out
    
    norm2 = rms_norm(x, weights['ffn_norm'])
    ffn_out = swiglu(norm2, weights['W_gate'], weights['W_up'], weights['W_down'])
    x = x + ffn_out
    
    return x, kv_cache_out
