#!/usr/bin/env python3
"""
Cirrus Training Script for Kaggle - Optimized

Usage in Kaggle:
1. Add accelerator: GPU
2. Run this script
"""

import sys

sys.path.insert(0, "/kaggle/working/cirrus-llm")

import torch
import torch.nn as nn
from cirrus import CirrusModel, CirrusConfig
from transformers import AutoTokenizer
from datasets import load_dataset


def train_gpu(save_every=1000, max_steps=50000, resume_from=None):
    print("=" * 50)
    print("Cirrus Training on Kaggle GPU")
    print("=" * 50)

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: No GPU!")
        return

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Model
    print("Loading model...")
    config = CirrusConfig.small()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    config.vocab_size = tokenizer.vocab_size
    model = CirrusModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    if resume_from:
        model.load_state_dict(torch.load(resume_from))
        print(f"Resumed from {resume_from}")

    # Dataset
    print("Loading C4 dataset...")
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    ds = ds.shuffle(buffer_size=10000)
    print("Ready!")

    # Training
    model.train()
    step = 0
    total_loss = 0

    while step < max_steps:
        for batch in ds:
            text = batch.get("text", "")
            if len(text) < 50:
                continue

            tokens = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )["input_ids"][0]
            if tokens.shape[0] < 10:
                continue

            tokens = tokens.to(device)

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

            if step % 10 == 0:
                print(f"Step {step}/{max_steps} | loss: {loss.item():.4f}")

            if step % save_every == 0:
                torch.save(model.state_dict(), f"/kaggle/working/cirrus_step{step}.pt")
                print(f"✓ Saved cirrus_step{step}.pt")

            if step >= max_steps:
                break

    # Final
    torch.save(model.state_dict(), "/kaggle/working/cirrus_final.pt")
    print("=" * 50)
    print(f"DONE! Saved /kaggle/working/cirrus_final.pt")
    print("=" * 50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument("--steps", type=int, default=50000, help="Max steps")
    args = parser.parse_args()

    train_gpu(resume_from=args.resume, max_steps=args.steps)
