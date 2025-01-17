import logging
from pathlib import Path
from src.bpe_trainer import BPETrainer
from src.tokenizer import Tokenizer
from src.utils import load_text, calculate_compression_ratio

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Load Telugu text
    text = load_text('data/telugu_text_2.txt')
    logger.info(f"Loaded text with {len(text)} characters")

    # Train BPE
    trainer = BPETrainer(vocab_size=8000)
    vocab = trainer.train(text)
    
    # Save the model
    trainer.save_model('models/telugu_bpe')

    # Create tokenizer
    tokenizer = Tokenizer(vocab, trainer.merges)

    # Encode sample text
    encoded = tokenizer.encode(text[:1000])  # Test with first 1000 chars
    decoded = tokenizer.decode(encoded)

    # Calculate compression ratio
    ratio = calculate_compression_ratio(text[:1000], encoded)
    logger.info(f"Compression ratio: {ratio:.2f}")

    # Verify vocabulary size
    logger.info(f"Final vocabulary size: {len(vocab)}")

if __name__ == "__main__":
    main() 