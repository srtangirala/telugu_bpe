import gradio as gr
from src.tokenizer import Tokenizer
from src.bpe_trainer import BPETrainer
from src.utils import calculate_compression_ratio
import json
import pickle
from pathlib import Path
import random

def generate_distinct_colors(n):
    """Generate n visually distinct colors."""
    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEEAD",
        "#D4A5A5", "#9B59B6", "#3498DB", "#E67E22", "#1ABC9C",
        "#F1C40F", "#E74C3C", "#2ECC71", "#34495E", "#95A5A6"
    ]
    # If we need more colors than in our predefined list, generate random ones
    while len(colors) < n:
        colors.append(f"#{random.randint(0, 0xFFFFFF):06x}")
    return colors[:n]

def load_model(model_path: str = 'models/telugu_bpe'):
    """Load the trained BPE model."""
    model_path = Path(model_path)
    
    # Load vocabulary
    with open(model_path / 'vocab.json', 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    # Load merges from pkl file
    with open(model_path / 'merges.pkl', 'rb') as f:
        merges = pickle.load(f)
    
    return Tokenizer(vocab, merges)

def process_text(text: str):
    """Process input text and return tokenization results."""
    if not text.strip():
        return "Please enter some text.", ""
    
    try:
        tokenizer = load_model()
        encoded = tokenizer.encode(text)
        ratio = calculate_compression_ratio(text, encoded)
        
        # Create reverse vocabulary mapping
        vocab_reverse = {idx: token for token, idx in tokenizer.vocab.items()}
        
        # Generate distinct colors for tokens
        colors = generate_distinct_colors(len(tokenizer.vocab))
        color_map = {idx: color for idx, color in enumerate(colors)}
        
        # Format tokens with their IDs and colors
        tokens_with_ids = []
        for idx in encoded:
            token = vocab_reverse[idx]
            color = color_map[idx]
            tokens_with_ids.append(f'<span style="background-color: {color}; padding: 2px 4px; margin: 2px; border-radius: 3px;">{token} ({idx})</span>')
        
        # Join tokens with HTML formatting
        token_display = " ".join(tokens_with_ids)
        
        stats = (f"Number of tokens: {len(encoded)}\n"
                f"Compression ratio: {ratio:.2f}\n"
                f"Vocabulary size: {len(tokenizer.vocab)}")
        
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
    
    # Add examples section
    gr.Examples(
        examples=[
            "నమస్కారం! మీరు ఎలా ఉన్నారు?",
            "తెలుగు భాష చాలా అందమైన భాష.",
            "ప్రతి పూవు తన సొంత సౌందర్యాన్ని కలిగి ఉంటుంది.",
            "జీవితంలో ప్రతి రోజు కొత్త అవకాశం."
        ],
        inputs=input_text,
        label="Click on any example to load it"
    )
    
    with gr.Row():
        tokens_output = gr.HTML(
            label="Tokens",
        )
        stats_output = gr.Textbox(
            label="Statistics",
            lines=3,
            interactive=False
        )
    
    submit_btn.click(
        fn=process_text,
        inputs=[input_text],
        outputs=[tokens_output, stats_output]
    )

demo.launch() 