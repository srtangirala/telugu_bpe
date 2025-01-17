import logging
from src.tokenizer import Tokenizer
from src.bpe_trainer import BPETrainer
from src.utils import calculate_compression_ratio

def test_tokenize(text: str, model_path: str = 'models/telugu_bpe'):
    # Load the trained model
    trainer = BPETrainer.load_model(model_path)
    tokenizer = Tokenizer(trainer.vocab, trainer.merges)
    
    # Tokenize the text
    tokens = tokenizer.encode(text)
    
    # Calculate compression ratio
    ratio = calculate_compression_ratio(text, tokens)
    
    print(f"Input text: {text}")
    print(f"Tokens: {','.join(map(str, tokens))}")
    print(f"Compression ratio: {ratio:.2f}")
    
    return tokens, ratio

if __name__ == "__main__":
    # Example Telugu text - replace with your test text
    test_text = "నమస్కారం ఎలా ఉన్నారు"
    test_tokenize(test_text) 