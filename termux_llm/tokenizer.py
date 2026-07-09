import json
import collections
from config import get_config

class BPETokenizer:
    def __init__(self, vocab_size=None):
        config = get_config()
        self.vocab_size = vocab_size if vocab_size is not None else config['vocab_size']
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.special_tokens = {'<bos>': 256, '<eos>': 257, '<pad>': 258}
        self.vocab.update({v: k.encode('utf-8') for k, v in self.special_tokens.items()})
        self._next_id = 259

    def _get_stats(self, ids):
        counts = collections.Counter()
        for pair in zip(ids, ids[1:]):
            counts[pair] += 1
        return counts

    def _merge(self, ids, pair, idx):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text):
        print(f"Iniciando entrenamiento del tokenizador BPE (vocab_size: {self.vocab_size})...")
        tokens = list(text.encode('utf-8'))
        
        while self._next_id < self.vocab_size:
            stats = self._get_stats(tokens)
            if not stats:
                break
            best = max(stats, key=stats.get)
            self.merges[best] = self._next_id
            self.vocab[self._next_id] = self.vocab[best[0]] + self.vocab[best[1]]
            tokens = self._merge(tokens, best, self._next_id)
            self._next_id += 1
            if self._next_id % 1000 == 0 or self._next_id == self.vocab_size:
                print(f"Tamaño actual del vocabulario: {self._next_id}/{self.vocab_size}")

    def encode(self, text, add_bos=True):
        tokens = list(text.encode('utf-8'))
        while len(tokens) >= 2:
            stats = self._get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self._merge(tokens, pair, idx)
        
        if add_bos:
            tokens = [self.special_tokens['<bos>']] + tokens
        return tokens

    def decode(self, ids):
        text_bytes = b''.join([self.vocab[i] for i in ids if i in self.vocab and i not in self.special_tokens.values()])
        return text_bytes.decode('utf-8', errors='replace')

    def save(self, path):
        data = {
            'merges': {f"{k[0]},{k[1]}": v for k, v in self.merges.items()},
            'vocab_size': self.vocab_size
        }
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"Tokenizador guardado exitosamente en {path}")

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        self.vocab_size = data.get('vocab_size', self.vocab_size)
        self.merges = {tuple(map(int, k.split(','))): v for k, v in data['merges'].items()}
        for pair, idx in self.merges.items():
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self._next_id = max(self._next_id, idx + 1)
        print(f"Tokenizador cargado desde {path} con vocab_size={self.vocab_size}")
