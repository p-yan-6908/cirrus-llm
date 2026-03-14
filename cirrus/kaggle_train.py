#!/usr/bin/env python3
import sys

sys.path.insert(0, "/kaggle/working/cirrus-llm")

import torch
import torch.nn as nn
from cirrus import CirrusModel, CirrusConfig
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


def train_gpu(
    save_every=1000,
    max_steps=50000,
    resume_from=None,
    batch_size=2,
    grad_accum=2,
    compile_model=False,
    model_size="small",
    single_gpu=False,
):
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

    if single_gpu:
        n_gpus = 1
        print("Using SINGLE GPU mode (DataParallel broken)")
    elif multi_gpu:
        n_gpus = torch.cuda.device_count()
        print(f"Using {n_gpus} GPUs with DataParallel")
    else:
        n_gpus = 1
        print("Using SINGLE GPU mode")

    device = torch.device("cuda")

    print("Loading model...")
    if model_size == "tiny":
        config = CirrusConfig.tiny()
    elif model_size == "small":
        config = CirrusConfig.small()
    else:
        config = CirrusConfig.small()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    config.vocab_size = tokenizer.vocab_size

    model = CirrusModel(config)

    if compile_model:
        print("Compiling model with torch.compile (before DataParallel)...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile enabled!")
        except Exception as e:
            print(f"torch.compile failed: {e}, continuing without...")
            model = CirrusModel(config)

    if n_gpus > 1:
        print(
            f"Using {n_gpus} GPUs with DataParallel! Batch: {batch_size}, GradAccum: {grad_accum}"
        )
        model = nn.DataParallel(model, device_ids=list(range(n_gpus)))

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Effective batch: {batch_size} x {grad_accum} x {n_gpus} GPUs = {batch_size * grad_accum * n_gpus}"
    )

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
        checkpoint = torch.load(resume_from, map_location=device)
        if n_gpus > 1:
            model.module.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        import re

        match = re.search(r"cirrus_(?:quick_)?step(\d+)", resume_from)
        start_step = int(match.group(1)) if match else 0
        print(f"Resumed from {resume_from} at step {start_step}")
    else:
        start_step = 0

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
    batch_buffer = []
    grad_accum_counter = 0

    pbar = tqdm(total=max_steps, initial=step, desc="Training", unit="step")

    for item in tokenized_ds:
        if step >= max_steps:
            break

        batch_buffer.append(item)

        if len(batch_buffer) < batch_size:
            continue

        max_len = max(t.shape[0] for t in batch_buffer)
        padded = []
        for t in batch_buffer:
            if t.shape[0] < max_len:
                t = torch.cat(
                    [
                        t,
                        torch.full(
                            (max_len - t.shape[0],),
                            tokenizer.pad_token_id or 0,
                            dtype=t.dtype,
                        ),
                    ]
                )
            padded.append(t)
        batch = torch.stack(padded).to(device)
        batch_buffer = []

        if step % 100 == 0:
            print(f"Batch shape: {batch.shape} (should be [2, seq_len])")
            for i in range(n_gpus):
                mem_allocated = torch.cuda.memory_allocated(i) / 1e9
                mem_reserved = torch.cuda.memory_reserved(i) / 1e9
                print(
                    f"  GPU {i}: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved"
                )

        optimizer.zero_grad()
        logits, _, _, _ = model(batch)

        loss = nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, config.vocab_size),
            batch[:, 1:].reshape(-1),
            ignore_index=-1,
        )

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: NaN/Inf in loss at step {step}, skipping")
            batch_buffer = []
            continue

        loss = loss / grad_accum
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_accum_counter += 1

        if grad_accum_counter >= grad_accum:
            optimizer.step()
            grad_accum_counter = 0
            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}")

            torch.cuda.empty_cache()

            if step % 50 == 0:
                for f in g.glob("/kaggle/working/cirrus_quick_*.pt"):
                    try:
                        os.remove(f)
                    except:
                        pass
                if n_gpus > 1:
                    torch.save(
                        model.module.state_dict(),
                        f"/kaggle/working/cirrus_quick_{step}.pt",
                    )
                else:
                    torch.save(
                        model.state_dict(), f"/kaggle/working/cirrus_quick_{step}.pt"
                    )
                torch.cuda.empty_cache()

            if step % save_every == 0:
                for f in g.glob("/kaggle/working/cirrus_step*.pt"):
                    try:
                        os.remove(f)
                    except:
                        pass
                if n_gpus > 1:
                    torch.save(
                        model.module.state_dict(),
                        f"/kaggle/working/cirrus_step{step}.pt",
                    )
                else:
                    torch.save(
                        model.state_dict(), f"/kaggle/working/cirrus_step{step}.pt"
                    )
                pbar.write(f"✓ Saved cirrus_step{step}.pt")

    if batch_buffer:
        max_len = max(t.shape[0] for t in batch_buffer)
        padded = []
        for t in batch_buffer:
            if t.shape[0] < max_len:
                t = torch.cat(
                    [
                        t,
                        torch.full(
                            (max_len - t.shape[0],),
                            tokenizer.pad_token_id or 0,
                            dtype=t.dtype,
                        ),
                    ]
                )
            padded.append(t)
        batch = torch.stack(padded).to(device)
        optimizer.zero_grad()
        logits, _, _, _ = model(batch)
        loss = nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, config.vocab_size),
            batch[:, 1:].reshape(-1),
            ignore_index=-1,
        )
        loss = loss / grad_accum
        loss.backward()
        grad_accum_counter += 1

    if grad_accum_counter > 0:
        optimizer.step()
        step += 1
        pbar.update(1)
        pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}")

    pbar.close()

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
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--model", type=str, default="small", choices=["tiny", "small"])
    parser.add_argument("--single-gpu", action="store_true", default=True)
    parser.add_argument("--multi-gpu", action="store_true")
    args = parser.parse_args()
    train_gpu(
        resume_from=args.resume,
        max_steps=args.steps,
        batch_size=args.batch,
        grad_accum=args.grad_accum,
        compile_model=args.compile,
        model_size=args.model,
        single_gpu=not args.multi_gpu,
    )
