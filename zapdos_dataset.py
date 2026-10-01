"""Turn a raw scakit seqdemo capture, or a directory of its seq_export recordings (.npz), into a Hugging Face dataset
with columns "audio" (the repetitions averaged, float32), "text" (the listing verbatim), "opcodes", "project" and
"num_samples". The capture's zarr.json files go into info.description.

    zapdos-dataset captures/capture_<...>/ my-dataset/
    zapdos-dataset seq_traces/ my-dataset/
"""
import json
from pathlib import Path

import click
import numpy as np
import zarr
from datasets import Dataset, DatasetInfo


def row(name, traces, text, keep_repetitions=False):
    audio = (traces if keep_repetitions else traces.mean(axis=0, dtype=np.float64)).astype(np.float32)
    opcodes = " ".join(line.split()[0] for line in text.splitlines() if line.strip())
    return {"audio": audio, "text": text, "opcodes": opcodes, "project": name, "num_samples": audio.shape[-1]}


def npz_rows(paths, keep_repetitions):
    for path in paths:
        with np.load(path) as npz:
            text = npz["sequence" if "sequence" in npz else "instructions"].item()  # older captures used "instructions"
            yield row(path.stem, npz["traces"], text, keep_repetitions)


def sequences(directory):
    """seq_id -> listing, from sequences.jsonl (one per line) or <seq_id>.s, as seqdemo stores them."""
    if (directory / "sequences.jsonl").exists():
        texts = [json.loads(line)["sequence"].strip() for line in (directory / "sequences.jsonl").open()]
        return lambda seq_id: texts[seq_id - 1]
    return lambda seq_id: (directory / f"{seq_id}.s").read_text().strip()


def capture_rows(root, text, keep_repetitions):
    traces, ids, chunk = root["traces"], root["seq_ids"][:].ravel(), root["traces"].chunks[0]
    bounds = [0, *(np.flatnonzero(np.diff(ids)) + 1).tolist(), len(ids)]  # a sequence's repetitions are consecutive
    start, block = 0, traces[:0]
    for lo, hi in zip(bounds, bounds[1:]):
        if hi > start + len(block):  # read whole chunks, so each is decompressed once
            start = lo - lo % chunk
            block = traces[start:max(hi, start + chunk)]
        yield row(f"trace_{ids[lo]}", block[lo - start:hi - start], text(int(ids[lo])), keep_repetitions)


def build(src, keep_repetitions=False, sequences_dir=None):
    src = Path(src).resolve()
    if (src / "zarr.json").exists():
        root = zarr.open_group(src, mode="r")
        text = sequences(Path(sequences_dir or root.attrs["sequences_dir"]))
        rows = lambda: capture_rows(root, text, keep_repetitions)
        metadata = {p.parent.name: json.loads(p.read_text()) for p in [src / "zarr.json", *sorted(src.glob("*/zarr.json"))]}
    else:
        paths = sorted(src.glob("*.npz"))
        rows = lambda: npz_rows(paths, keep_repetitions)
        metadata = {p.stem: json.loads(p.read_text()) for p in sorted(src.glob("metadata/*.json"))}
    return Dataset.from_generator(rows, info=DatasetInfo(description=json.dumps(metadata)))


@click.command()
@click.argument("src")
@click.argument("dst")
@click.option("--keep-repetitions", is_flag=True, help="Store every repetition instead of their mean.")
@click.option("--sequences", help="Directory with the capture's sequences (default: its sequences_dir).")
def main(src, dst, keep_repetitions, sequences):
    build(src, keep_repetitions, sequences).save_to_disk(dst)


if __name__ == "__main__":
    main()
