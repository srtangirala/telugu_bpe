import logging
from typing import List

class Tokenizer:
    def __init__(self, vocab: dict, merges: dict):
        self.vocab = vocab
        self.merges = merges
        self.logger = logging.getLogger(__name__)

    def encode(self, text: str) -> List[int]:
        """Encode text to token ids."""
        words = text.split()
        encoded = []
        
        for word in words:
            tokens = list(word)
            while True:
                pairs = self._get_pairs(tokens)
                if not pairs:
                    break
                    
                pair = self._find_mergeable_pair(pairs)
                if not pair:
                    break
                    
                tokens = self._merge_pair(tokens, pair)
            
            encoded.extend([self.vocab.get(token, self.vocab['<unk>']) for token in tokens])
        
        return encoded

    def decode(self, ids: List[int]) -> str:
        """Decode token ids back to text."""
        inv_vocab = {v: k for k, v in self.vocab.items()}
        return ''.join(inv_vocab[id] for id in ids)

    def _get_pairs(self, tokens: List[str]) -> List[tuple]:
        """Get all adjacent pairs in token list."""
        return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]

    def _find_mergeable_pair(self, pairs: List[tuple]) -> tuple:
        """Find the first pair that can be merged according to learned merges."""
        for pair in pairs:
            if pair in self.merges:
                return pair
        return None

    def _merge_pair(self, tokens: List[str], pair: tuple) -> List[str]:
        """Merge all occurrences of pair in token list."""
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(self.merges[pair])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens 