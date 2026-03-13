#!/usr/bin/env python3
"""
Cirrus Training Script for Google Colab (GPU)

Usage in Google Colab:
1. Upload cirrus/ folder to Colab
2. Run: !python3 colab_train.py

OR copy-paste this into a Colab cell:
"""

import torch
import torch.nn as nn
from cirrus import CirrusModel, CirrusConfig
from transformers import AutoTokenizer
from datasets import load_dataset


def train_gpu(epochs=5, save_every=1000):
    print("=" * 50)
    print("Cirrus Training on GPU")
    print("=" * 50)

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: No GPU available!")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Model
    print("Loading model...")
    config = CirrusConfig.small()  # Use small for GPU
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    config.vocab_size = tokenizer.vocab_size
    model = CirrusModel(config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Dataset
    print("Loading C4 dataset...")
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    ds = ds.shuffle(buffer_size=10000)
    print("Ready!")

    # Training
    model.train()
    step = 0
    total_loss = 0

    while True:
        for batch in ds:
            text = batch.get("text", "")
            if len(text) < 50:
                continue

            tokens = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )["input_ids"][0]
            if tokens.shape[0] < 10:
                continue

            tokens = tokens.cuda()

            optimizer.zero_grad()
            logits, _, _, _ = model(tokens.unsqueeze(0))

            loss = nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, config.vocab_size),
                tokens[1:].reshape(-1),
                ignore_index=-1,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step += 1

            if step % 50 == 0:
                print(
                    f"Step {step} | loss: {loss.item():.4f} | avg: {total_loss / step:.4f}"
                )

            if step % save_every == 0:
                torch.save(model.state_dict(), f"cirrus_step{step}.pt")
                print(f"✓ Saved cirrus_step{step}.pt")

            if step >= 50000:  # 50k steps = good model
                break
        break

    # Final
    torch.save(model.state_dict(), "cirrus_final.pt")
    print("=" * 50)
    print(f"DONE! Saved cirrus_final.pt")
    print("=" * 50)


if __name__ == "__main__":
    train_gpu()
