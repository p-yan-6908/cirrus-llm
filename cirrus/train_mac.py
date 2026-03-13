#!/usr/bin/env python3
"""Efficient training for Mac."""

import argparse
import torch
import torch.nn as nn
from cirrus import CirrusModel, CirrusConfig
from transformers import AutoTokenizer
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    # Tiny model for Mac
    print("Loading model...")
    config = CirrusConfig.tiny()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    config.vocab_size = tokenizer.vocab_size
    model = CirrusModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    if args.resume:
        model.load_state_dict(torch.load(args.resume))
        print(f"Resumed from {args.resume}")

    # Load C4
    print("Loading C4 dataset...")
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    ds = ds.shuffle(buffer_size=10000)

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Training for {args.epochs} epochs...")

    model.train()
    step = 0
    epoch = 0
    total_loss = 0

    while epoch < args.epochs:
        for batch in ds:
            text = batch.get("text", "")
            if not text or len(text) < 50:
                continue

            # Tokenize (shorter for speed)
            tokens = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=128
            )["input_ids"][0]
            if tokens.shape[0] < 10:
                continue

            # Forward
            optimizer.zero_grad()
            logits, _, _, _ = model(tokens.unsqueeze(0))

            # Loss
            loss = nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, config.vocab_size),
                tokens[1:].reshape(-1),
                ignore_index=-1,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step += 1

            if step % 20 == 0:
                print(f"Epoch {epoch} step {step} loss={loss.item():.4f}")

            if step % args.save_every == 0:
                path = f"cirrus_tiny_step{step}.pt"
                torch.save(model.state_dict(), path)
                print(f"Saved {path}")

        epoch += 1
        print(f"Epoch {epoch} complete, avg loss: {total_loss / step:.4f}")

    # Final
    torch.save(model.state_dict(), "cirrus_tiny_final.pt")
    print("Saved cirrus_tiny_final.pt")


if __name__ == "__main__":
    main()
