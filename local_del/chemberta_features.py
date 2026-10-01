"""Benchmark and generate mean-pooled ChemBERTa molecular embeddings."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "chemberta_77m_mlm"


def batches(values: list[str], size: int):
    for begin in range(0, len(values), size):
        yield begin, values[begin : begin + size]


def embed(smiles: list[str], batch_size: int, device: str) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True).eval().to(device)
    hidden = model.config.hidden_size
    result = np.empty((len(smiles), hidden), dtype=np.float16)
    with torch.inference_mode():
        for begin, chunk in batches(smiles, batch_size):
            tokens = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            states = model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            pooled = (states * mask).sum(1) / mask.sum(1).clamp_min(1)
            result[begin : begin + len(chunk)] = pooled.float().cpu().numpy().astype(np.float16)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["validation", "test"], default="validation")
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="0 embeds the complete dataset")
    parser.add_argument("--output")
    args = parser.parse_args()

    metadata = pd.read_parquet(ROOT / "features" / f"{args.dataset}_meta.parquet")
    smiles = metadata.canonical.fillna(metadata.SMILES).astype(str).tolist()
    if args.limit:
        # Evenly spaced molecules better represent the whole library than its head.
        index = np.linspace(0, len(smiles) - 1, min(args.limit, len(smiles)), dtype=int)
        smiles = [smiles[i] for i in index]
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")

    started = time.perf_counter()
    features = embed(smiles, args.batch_size, args.device)
    elapsed = time.perf_counter() - started
    report = {
        "dataset": args.dataset,
        "device": args.device,
        "batch_size": args.batch_size,
        "molecules": len(smiles),
        "shape": list(features.shape),
        "seconds": elapsed,
        "molecules_per_second": len(smiles) / elapsed,
        "finite": bool(np.isfinite(features).all()),
        "mean_norm": float(np.linalg.norm(features.astype(np.float32), axis=1).mean()),
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, features)
        report["output"] = str(output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
