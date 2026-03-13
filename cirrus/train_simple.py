#!/usr/bin/env python3
"""Simple training script for Cirrus."""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from cirrus import CirrusModel, CirrusConfig


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, vocab_size, seq_len, size):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.randint(1, self.vocab_size, (self.seq_len,))


def collate(batch):
    max_len = max(x.shape[0] for x in batch)
    padded = []
    for x in batch:
        pad_len = max_len - x.shape[0]
        padded.append(nn.functional.pad(x, (0, pad_len)))
    return torch.stack(padded)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tiny")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=256)
    args = parser.parse_args()

    # Create model
    if args.model == "tiny":
        config = CirrusConfig.tiny()
    else:
        config = CirrusConfig.small()

    model = CirrusModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Create data
    dataset = SimpleDataset(config.vocab_size, args.seq, 100)
    loader = DataLoader(
        dataset, batch_size=args.batch, collate_fn=collate, shuffle=True
    )

    print(
        f"Model: {args.model}, params: {sum(p.numel() for p in model.parameters()):,}"
    )
    print(f"Training for {args.epochs} epochs, {len(loader)} batches each")

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for i, x in enumerate(loader):
            optimizer.zero_grad()
            logits, _, _, _ = model(x)
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, config.vocab_size),
                x[:, 1:].reshape(-1),
                ignore_index=0,
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 5 == 0:
                print(f"Epoch {epoch} batch {i}/{len(loader)} loss={loss.item():.4f}")
        print(f"Epoch {epoch} avg loss: {total_loss / len(loader):.4f}")

    print("Done!")


if __name__ == "__main__":
    train()
