#!/usr/bin/env python3
"""Train Cirrus on real text data."""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from cirrus import CirrusModel, CirrusConfig
from transformers import AutoTokenizer


class TextDataset(Dataset):
    def __init__(self, text, tokenizer, max_length):
        self.tokens = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        )["input_ids"][0]
        self.max_len = max_length

    def __len__(self):
        return max(0, self.max_len - 1)

    def __getitem__(self, idx):
        x = self.tokens[: idx + 1]
        y = self.tokens[idx + 1]
        return x, y


def collate_fn(batch):
    xs, ys = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    pad_x = torch.stack(
        [
            torch.cat([x, torch.zeros(max_len - x.shape[0], dtype=torch.long)])
            for x in xs
        ]
    )
    return pad_x, torch.tensor(ys)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tiny")
    parser.add_argument("--text", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    # Use tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Get text - from file or default sample
    if args.text:
        with open(args.text) as f:
            text = f.read()
    else:
        text = (
            """
        Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured, procedural, reflective, object-oriented and functional programming. It has a large and comprehensive standard library.

        Machine learning is a field of inquiry devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks. It is seen as a part of artificial intelligence. Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so.

        Neural networks, also known as artificial neural networks, are computing systems inspired by the biological neural networks. An ANN is based on a collection of connected units or nodes called artificial neurons, which loosely model the neurons in a biological brain. Each connection, like the synapses in a biological brain, can transmit a signal to other neurons.
        """
            * 100
        )  # Repeat for more data

    print(
        f"Training data: {len(text)} chars, {len(tokenizer(text)['input_ids'])} tokens"
    )

    # Create model
    config = CirrusConfig.tiny() if args.model == "tiny" else CirrusConfig.small()
    config.vocab_size = tokenizer.vocab_size
    model = CirrusModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Dataset
    dataset = TextDataset(text, tokenizer, args.max_len)
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn
    )

    print(
        f"Model: {args.model}, vocab: {config.vocab_size}, params: {sum(p.numel() for p in model.parameters()):,}"
    )
    print(f"Training: {args.epochs} epochs, {len(loader)} batches")

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for i, (x, y) in enumerate(loader):
            optimizer.zero_grad()
            logits, _, _, _ = model(x)
            loss = nn.functional.cross_entropy(logits[:, -1, :], y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 20 == 0:
                print(f"Epoch {epoch} batch {i}/{len(loader)} loss={loss.item():.4f}")
        print(f"Epoch {epoch} avg loss: {total_loss / len(loader):.4f}")

    # Save
    if args.save:
        torch.save(model.state_dict(), args.save)
        print(f"Saved to {args.save}")

    # Quick generation test
    model.eval()
    prompt = "Python is"
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    print(f"\nGeneration test:")
    print(f"Prompt: '{prompt}'")

    with torch.no_grad():
        for _ in range(30):
            logits, _, _, _ = model(ids)
            next_tok = logits[0, -1].argmax()
            ids = torch.cat([ids, next_tok.unsqueeze(0).unsqueeze(0)], dim=1)
            if next_tok == tokenizer.eos_token_id:
                break

    print(f"Generated: '{tokenizer.decode(ids[0])}'")


if __name__ == "__main__":
    train()
