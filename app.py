import gradio as gr
from src.tokenizer import Tokenizer
from src.bpe_trainer import BPETrainer
from src.utils import calculate_compression_ratio
import json
from pathlib import Path

def load_model(model_path: str = 'models/telugu_bpe'):
    """Load the trained BPE model."""
    model_path = Path(model_path)
    
    # Load vocabulary
    with open(model_path / 'vocab.json', 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    # Load merges
    with open(model_path / 'merges.txt', 'r', encoding='utf-8') as f:
        merges = [tuple(line.strip().split()) for line in f]
    
    return Tokenizer(vocab, merges)

def process_text(text: str):
    """Process input text and return tokenization results."""
    if not text.strip():
        return "Please enter some text.", ""
    
    try:
        tokenizer = load_model()
        
        # Tokenize the text
        encoded = tokenizer.encode(text)
        
        # Calculate compression ratio
        ratio = calculate_compression_ratio(text, encoded)
        
        # Get the actual tokens for display
        tokens = [tokenizer.vocab_reverse[idx] for idx in encoded]
        
        # Format the results
        token_display = " ".join(tokens)
        stats = f"Number of tokens: {len(encoded)}\nCompression ratio: {ratio:.2f}"
        
        return token_display, stats
    
    except Exception as e:
        return f"Error processing text: {str(e)}", ""

# Create the Gradio interface
with gr.Blocks(title="Telugu BPE Tokenizer") as demo:
    gr.Markdown("# Telugu BPE Tokenizer Demo")
    gr.Markdown("Enter Telugu text below to see how it gets tokenized using our BPE model.")
    
    with gr.Row():
        input_text = gr.Textbox(
            label="Input Text",
            placeholder="Enter Telugu text here...",
            lines=5
        )
    
    with gr.Row():
        submit_btn = gr.Button("Tokenize")
    
    with gr.Row():
        tokens_output = gr.Textbox(
            label="Tokens",
            lines=5,
            interactive=False
        )
        stats_output = gr.Textbox(
            label="Statistics",
            lines=2,
            interactive=False
        )
    
    submit_btn.click(
        fn=process_text,
        inputs=[input_text],
        outputs=[tokens_output, stats_output]
    )

demo.launch() 