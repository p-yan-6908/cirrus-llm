#!/usr/bin/env python3
"""Use trained Cirrus model for inference."""

import argparse
import torch
from cirrus import CirrusModel, CirrusConfig
from transformers import AutoTokenizer


def load_model(model_path=None, model_size="tiny"):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if model_size == "tiny":
        config = CirrusConfig.tiny()
    elif model_size == "small":
        config = CirrusConfig.small()
    else:
        config = CirrusConfig.small()

    config.vocab_size = tokenizer.vocab_size

    model = CirrusModel(config)

    if model_path:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"Loaded trained weights from {model_path}")
    else:
        print("Using untrained model (random weights)")

    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=1.0, top_k=None):
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _, _, _ = model(ids)

            # Apply temperature
            logits = logits[0, -1, :] / temperature

            # Apply top-k
            if top_k:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][-1]
                logits[indices_to_remove] = float("-inf")

            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1).item()

            ids = torch.cat([ids, torch.tensor([[next_tok]])], dim=1)

            if next_tok == tokenizer.eos_token_id:
                break

    return tokenizer.decode(ids[0])


def interactive(model, tokenizer):
    print("\n=== Cirrus Interactive Mode ===")
    print("Type 'quit' to exit\n")

    while True:
        prompt = input("> ")
        if prompt.lower() in ["quit", "exit", "q"]:
            break
        if not prompt.strip():
            continue

        output = generate(model, tokenizer, prompt, max_new_tokens=100)
        print(output)
        print()


def main():
    parser = argparse.ArgumentParser(description="Use trained Cirrus model")
    parser.add_argument(
        "--model", default="trained_model.pt", help="Path to model weights"
    )
    parser.add_argument("--size", default="tiny", choices=["tiny", "small"])
    parser.add_argument("--prompt", default=None, help="Prompt to generate from")
    parser.add_argument(
        "--tokens", type=int, default=100, help="Max tokens to generate"
    )
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.size)

    if args.prompt:
        output = generate(
            model, tokenizer, args.prompt, args.tokens, args.temp, args.top_k
        )
        print(output)
    else:
        interactive(model, tokenizer)


if __name__ == "__main__":
    main()
