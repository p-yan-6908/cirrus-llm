#!/usr/bin/env python3
"""Train Cirrus on real datasets."""

import argparse
import torch
import torch.nn as nn
from cirrus import CirrusModel, CirrusConfig


def train(args):
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader, IterableDataset

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    if args.model == "tiny":
        config = CirrusConfig.tiny()
    elif args.model == "small":
        config = CirrusConfig.small()
    else:
        config = CirrusConfig.base_10b()

    config.vocab_size = tokenizer.vocab_size
    model = CirrusModel(config)

    print(
        f"Model: {args.model}, vocab: {config.vocab_size}, params: {sum(p.numel() for p in model.parameters()):,}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Load dataset
    if args.dataset == "c4":
        from datasets import load_dataset

        print("Loading C4 dataset...")
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    elif args.dataset == "wikitext":
        from datasets import load_dataset

        print("Loading WikiText...")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    elif args.dataset == "python":
        from datasets import load_dataset

        print("Loading Python code...")
        ds = load_dataset("bigcode/the-stack", split="train", data_dir="data/python")
    elif args.dataset == "code":
        from datasets import load_dataset

        print("Loading code datasets...")
        ds = load_dataset("bigcode/the-stack", split="train")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Training
    model.train()

    if args.resume:
        model.load_state_dict(torch.load(args.resume))
        print(f"Resumed from {args.resume}")

    print(f"Training for {args.epochs} epochs...")

    epoch = 0
    step = 0
    total_loss = 0
    batch_tokens = 0

    while epoch < args.epochs:
        if hasattr(ds, "__iter__") and not hasattr(ds, "__getitem__"):
            # Iterable dataset (like streaming C4)
            iterator = iter(ds)
        else:
            iterator = iter(ds)

        for batch in iterator:
            # Get text
            if isinstance(batch, dict):
                if "text" in batch:
                    text = batch["text"]
                elif "content" in batch:
                    text = batch["content"]
                else:
                    continue
            else:
                continue

            if not text or len(text) < 10:
                continue

            # Tokenize
            tokens = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=args.max_len
            )
            input_ids = tokens["input_ids"][0]

            if input_ids.shape[0] < 10:
                continue

            # Forward
            optimizer.zero_grad()
            logits, _, _, _ = model(input_ids.unsqueeze(0))

            # Loss
            loss = nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, config.vocab_size),
                input_ids[1:].reshape(-1),
                ignore_index=-1,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_tokens += input_ids.shape[0]
            step += 1

            if step % args.log_every == 0:
                print(
                    f"Epoch {epoch} step {step} loss={loss.item():.4f} avg_loss={total_loss / args.log_every:.4f}"
                )
                total_loss = 0

            if step % args.save_every == 0:
                path = f"cirrus_{args.model}_step{step}.pt"
                torch.save(model.state_dict(), path)
                print(f"Saved {path}")

        epoch += 1
        print(f"Epoch {epoch} complete")

    # Final save
    torch.save(model.state_dict(), f"cirrus_{args.model}_final.pt")
    print(f"Saved cirrus_{args.model}_final.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="tiny", choices=["tiny", "small", "base_10b"]
    )
    parser.add_argument(
        "--dataset", default="c4", choices=["c4", "wikitext", "python", "code"]
    )
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--tiny", action="store_true", help="Use tiny model (faster)")
    args = parser.parse_args()

    if args.tiny:
        args.model = "tiny"

    train(args)
