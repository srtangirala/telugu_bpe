---
title: Telugu BPE Tokenizer
emoji: 🔤
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# Telugu BPE Tokenizer

This is a demo of a BPE (Byte Pair Encoding) tokenizer trained on Telugu text. You can input any Telugu text and see how it gets tokenized, along with statistics about the tokenization.

## Features

- Tokenize Telugu text using BPE
- View individual tokens
- See compression ratio and token count
- Easy-to-use interface

## How to Use

1. Enter or paste Telugu text in the input box
2. Click "Tokenize"
3. View the resulting tokens and statistics 

## Training Logs

```
.venvsreddy@Sreekanths-MacBook-Pro 11_IndieBPE_2 % python3 train.py                                    
2025-01-17 08:58:50,152 - INFO - Successfully loaded text from data/telugu_text_2.txt
2025-01-17 08:58:50,152 - INFO - Loaded text with 108716 characters
2025-01-17 08:58:50,152 - INFO - Starting BPE training...
Training BPE: 100%|███████████████████████████████████████████████████████████████| 7810/7810 [1:02:22<00:00,  2.09it/s]
2025-01-17 10:01:12,721 - INFO - BPE training completed. Vocabulary size: 8000
2025-01-17 10:01:12,733 - INFO - Model saved to models/telugu_bpe
2025-01-17 10:01:12,734 - INFO - Compression ratio: 3.82
2025-01-17 10:01:12,734 - INFO - Final vocabulary size: 8000
.venvsreddy@Sreekanths-MacBook-Pro 11_IndieBPE_2 % 
```