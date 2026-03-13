#!/usr/bin/env python3
"""
Cirrus Training Script for Kaggle - Multi-GPU Optimized
"""

import sys

sys.path.insert(0, "/kaggle/working/cirrus-llm")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from cirrus import CirrusModel, CirrusConfig
from transformers import AutoTokenizer
from datasets import load_dataset


def train_gpu(save_every=1000, max_steps=50000, resume_from=None):
    print("=" * 50)
    print("Cirrus Training on Kaggle GPU")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("ERROR: No GPU!")
        return

    n_gpus = torch.cuda.device_count()
    print(f"GPUs: {n_gpus}")
    for i in range(n_gpus):
        print(f"  {i}: {torch.cuda.get_device_name(i)}")

    device = torch.device("cuda")

    print("Loading model...")
    config = CirrusConfig.small()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    config.vocab_size = tokenizer.vocab_size

    model = CirrusModel(config).to(device)

    # Wrap with DataParallel for multi-GPU
    if n_gpus > 1:
        print(f"Using {n_gpus} GPUs!")
        model = nn.DataParallel(model)

    # Skip torch.compile on slow GPUs
    USE_COMPILE = False

    if USE_COMPILE:
        try:
            compute_capability = torch.cuda.get_device_capability()
            if compute_capability[0] >= 7:
                print("Compiling model...")
                model = torch.compile(model, mode="reduce-overhead")
            else:
                print(f"GPU compute {compute_capability} doesn't support torch.compile")
        except:
            pass

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
        if n_gpus > 1:
            model.module.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        match = re.search(r"cirrus_(?:quick_)?step(\d+)", resume_from)
        start_step = int(match.group(1)) if match else 0
        print(f"Resumed from {resume_from} at step {start_step}")
    else:
        start_step = 0

    # Cleanup old saves but keep latest
    import glob as g, os

    old_files = sorted(
        g.glob("/kaggle/working/cirrus_*.pt"),
        key=lambda x: int(re.search(r"cirrus_(?:quick_)?step(\d+)", x).group(1))
        if re.search(r"cirrus_(?:quick_)?step(\d+)", x)
        else 0,
    )
    if old_files:
        for f in old_files[:-1]:
            try:
                os.remove(f)
            except:
                pass
        print(f"Kept only latest checkpoint")

    print("Loading dataset...")
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    ds = ds.shuffle(buffer_size=10000)

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

        # Keep only latest quick save
        if step % 50 == 0:
            for f in g.glob("/kaggle/working/cirrus_quick_*.pt"):
                try:
                    os.remove(f)
                except:
                    pass
            if n_gpus > 1:
                torch.save(
                    model.module.state_dict(), f"/kaggle/working/cirrus_quick_{step}.pt"
                )
            else:
                torch.save(
                    model.state_dict(), f"/kaggle/working/cirrus_quick_{step}.pt"
                )
            torch.cuda.empty_cache()

        # Keep only latest step save
        if step % save_every == 0:
            for f in g.glob("/kaggle/working/cirrus_step*.pt"):
                try:
                    os.remove(f)
                except:
                    pass
            if n_gpus > 1:
                torch.save(
                    model.module.state_dict(), f"/kaggle/working/cirrus_step{step}.pt"
                )
            else:
                torch.save(model.state_dict(), f"/kaggle/working/cirrus_step{step}.pt")
            print(f"✓ Saved cirrus_step{step}.pt")

    if n_gpus > 1:
        torch.save(model.module.state_dict(), "/kaggle/working/cirrus_final.pt")
    else:
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
