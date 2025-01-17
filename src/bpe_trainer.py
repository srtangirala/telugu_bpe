import logging
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import json
import pickle
from pathlib import Path

class BPETrainer:
    def __init__(self, vocab_size: int = 8000, batch_size: int = 10000):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.merges = {}
        self.vocab = set()
        self.n_cores = mp.cpu_count()
        self.logger = logging.getLogger(__name__)

    def get_stats_chunk(self, sequences_chunk: List[List[str]]) -> Counter:
        """Process a chunk of sequences using Counter."""
        pairs = Counter()
        for seq in sequences_chunk:
            if len(seq) < 2:
                continue
            pairs.update(zip(seq[:-1], seq[1:]))
        return pairs

    def get_stats(self, sequences: List[List[str]]) -> Dict[Tuple[str, str], int]:
        """Fast parallel counting of pair frequencies."""
        stats = Counter()
        
        # Process entire sequence list at once
        chunk_size = max(1, len(sequences) // self.n_cores)
        sequence_chunks = [sequences[i:i + chunk_size] 
                          for i in range(0, len(sequences), chunk_size)]

        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            chunk_stats = executor.map(self.get_stats_chunk, sequence_chunks)
            
            # Combine results efficiently
            for chunk_stat in chunk_stats:
                stats.update(chunk_stat)

        return dict(stats)

    def merge_vocab_chunk(self, args: tuple) -> List[List[str]]:
        """Optimized chunk processing for merging."""
        sequences_chunk, pair, new_token = args
        return [
            self._fast_merge_sequence(seq, pair, new_token)
            for seq in sequences_chunk
        ]

    def _fast_merge_sequence(self, seq: List[str], pair: Tuple[str, str], new_token: str) -> List[str]:
        """Optimized single sequence merging."""
        if len(seq) < 2:
            return seq

        new_seq = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                new_seq.append(new_token)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        return new_seq

    def merge_vocab(self, sequences: List[List[str]], pair: Tuple[str, str]) -> List[List[str]]:
        """Fast parallel merging of sequences."""
        new_token = pair[0] + pair[1]
        
        # Split into larger chunks for better parallelization
        chunk_size = max(1, len(sequences) // (self.n_cores * 2))
        sequence_chunks = [sequences[i:i + chunk_size] 
                          for i in range(0, len(sequences), chunk_size)]
        
        # Prepare arguments
        chunk_args = [(chunk, pair, new_token) for chunk in sequence_chunks]
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            merged_chunks = list(executor.map(self.merge_vocab_chunk, chunk_args))
            
        return [seq for chunk in merged_chunks for seq in chunk]

    def train(self, text: str) -> Dict[str, int]:
        """Train BPE on input text with optimizations."""
        self.logger.info("Starting BPE training...")
        
        # Pre-process text into sequences once
        sequences = [[c for c in word] for word in text.split()]
        vocab = set(char for word in sequences for char in word)
        
        # Add special tokens
        special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>']
        for token in special_tokens:
            vocab.add(token)
        
        # Pre-allocate memory for frequent operations
        pbar = tqdm(total=self.vocab_size - len(vocab), desc="Training BPE")
        
        while len(vocab) < self.vocab_size:
            # Get statistics in parallel
            stats = self.get_stats(sequences)
            if not stats:
                break
            
            # Find best pair efficiently
            best_pair = max(stats.items(), key=lambda x: x[1])[0]
            
            # Merge sequences in parallel
            sequences = self.merge_vocab(sequences, best_pair)
            
            # Update vocabulary
            new_token = best_pair[0] + best_pair[1]
            vocab.add(new_token)
            self.merges[best_pair] = new_token
            
            pbar.update(1)
            
        pbar.close()
        vocab_list = special_tokens + list(vocab - set(special_tokens))
        self.vocab = {token: idx for idx, token in enumerate(vocab_list)}
        self.logger.info(f"BPE training completed. Vocabulary size: {len(self.vocab)}")
        
        return self.vocab

    def save_model(self, save_path: str):
        """Save the trained model to disk."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'vocab': self.vocab,
            'merges': self.merges,
            'vocab_size': self.vocab_size
        }
        
        # Save vocabulary (no need for str conversion since keys are already strings)
        with open(save_path / 'vocab.json', 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        
        # Save merges
        with open(save_path / 'merges.pkl', 'wb') as f:
            pickle.dump(self.merges, f)
            
        self.logger.info(f"Model saved to {save_path}")

    @classmethod
    def load_model(cls, model_path: str):
        """Load a trained model from disk."""
        model_path = Path(model_path)
        trainer = cls()
        
        # Load vocabulary
        with open(model_path / 'vocab.json', 'r', encoding='utf-8') as f:
            trainer.vocab = json.load(f)
            
        # Load merges
        with open(model_path / 'merges.pkl', 'rb') as f:
            trainer.merges = pickle.load(f)
            
        return trainer 