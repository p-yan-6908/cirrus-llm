#!/usr/bin/env python3
"""
Cirrus Training Script for Kaggle - GPU Optimized

Usage in Kaggle:
1. Add accelerator: GPU
2. Run this script
"""

import sys

sys.path.insert(0, "/kaggle/working/cirrus-llm")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from cirrus import CirrusModel, CirrusConfig
from transformers import AutoTokenizer
from datasets import load_dataset


def train_gpu(save_every=1000, max_steps=50000, resume_from=None):
    print("=" * 50)
    print("Cirrus Training on Kaggle GPU - Optimized")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("ERROR: No GPU!")
        return

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Enable TF32 for faster computation
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("Loading model...")
    config = CirrusConfig.small()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    config.vocab_size = tokenizer.vocab_size

    model = CirrusModel(config).to(device)

    # Compile model for faster execution
    print("Compiling model (first run will be slow)...")
    model = torch.compile(model, mode="reduce-overhead")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Auto-find latest checkpoint
    if not resume_from:
        import glob, re

        quick = sorted(
            glob.glob("/kaggle/working/cirrus_quick_*.pt"),
            key=lambda x: int(re.search(r"cirrus_quick_(\d+)", x).group(1))
            if re.search(r"cirrus_quick_(\d+)", x)
            else 0,
        )
        steps = sorted(
            glob.glob("/kaggle/working/cirrus_step*.pt"),
            key=lambda x: int(re.search(r"cirrus_step(\d+)", x).group(1))
            if re.search(r"cirrus_step(\d+)", x)
            else 0,
        )
        all_ckpts = quick + steps
        if all_ckpts:
            resume_from = all_ckpts[-1]
            print(f"Auto-found: {resume_from}")

    if resume_from:
        import re

        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint)
        match = re.search(r"cirrus_(?:quick_)?step(\d+)", resume_from)
        start_step = int(match.group(1)) if match else 0
        print(f"Resumed from {resume_from} at step {start_step}")
    else:
        start_step = 0

    # Use DataLoader for batching
    print("Loading dataset...")
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    ds = ds.shuffle(buffer_size=10000)

    # Pre-tokenize for speed
    class TokenizedDataset:
        def __init__(self, raw_ds, tokenizer, max_len=512):
            self.raw_ds = raw_ds
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __iter__(self):
            for item in self.raw_ds:
                text = item.get("text", "")
                if len(text) < 50:
                    continue
                tokens = self.tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=self.max_len
                )["input_ids"][0]
                if tokens.shape[0] > 10:
                    yield tokens

    tokenized_ds = TokenizedDataset(ds, tokenizer)
    print("Ready!")

    model.train()
    step = start_step

    for tokens in tokenized_ds:
        if step >= max_steps:
            break

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

        step += 1

        if step % 10 == 0:
            print(f"Step {step}/{max_steps} | loss: {loss.item():.4f}")

        if step % 50 == 0:
            import glob as g, os

            for f in g.glob("/kaggle/working/cirrus_quick_*.pt")[:-2]:
                try:
                    os.remove(f)
                except:
                    pass
            torch.save(model.state_dict(), f"/kaggle/working/cirrus_quick_{step}.pt")
            torch.cuda.empty_cache()

        if step % save_every == 0:
            for f in g.glob("/kaggle/working/cirrus_step*.pt")[:-3]:
                try:
                    os.remove(f)
                except:
                    pass
            torch.save(model.state_dict(), f"/kaggle/working/cirrus_step{step}.pt")
            print(f"✓ Saved cirrus_step{step}.pt")

    torch.save(model.state_dict(), "/kaggle/working/cirrus_final.pt")
    print("=" * 50)
    print(f"DONE! Saved cirrus_final.pt")
    print("=" * 50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--steps", type=int, default=50000)
    args = parser.parse_args()
    train_gpu(resume_from=args.resume, max_steps=args.steps)
