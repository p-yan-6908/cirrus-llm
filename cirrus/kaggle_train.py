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

    # Auto-find latest checkpoint if none specified
    if not resume_from:
        import glob, re

        quick_checkpoints = sorted(
            glob.glob("/kaggle/working/cirrus_quick_*.pt"),
            key=lambda x: int(re.search(r"cirrus_quick_(\d+)", x).group(1))
            if re.search(r"cirrus_quick_(\d+)", x)
            else 0,
        )
        step_checkpoints = sorted(
            glob.glob("/kaggle/working/cirrus_step*.pt"),
            key=lambda x: int(re.search(r"cirrus_step(\d+)", x).group(1))
            if re.search(r"cirrus_step(\d+)", x)
            else 0,
        )
        all_checkpoints = quick_checkpoints + step_checkpoints
        if all_checkpoints:
            resume_from = all_checkpoints[-1]
            print(f"Auto-found checkpoint: {resume_from}")

    if resume_from:
        import re

        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint)
        match = re.search(r"cirrus_(?:quick_step|step)(\d+)", resume_from)
        start_step = int(match.group(1)) if match else 0
        print(f"Resumed from {resume_from} at step {start_step}")
    else:
        start_step = 0

    # Dataset
    print("Loading C4 dataset...")
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    ds = ds.shuffle(buffer_size=10000)
    print("Ready!")

    # Training
    model.train()
    step = start_step
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

            # Backup save every 50 steps, delete old one
            if step % 50 == 0:
                import glob, os

                for f in glob.glob("/kaggle/working/cirrus_quick_*.pt"):
                    try:
                        os.remove(f)
                    except:
                        pass
                torch.save(
                    model.state_dict(), f"/kaggle/working/cirrus_quick_{step}.pt"
                )

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
