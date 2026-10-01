"""Turn a raw scakit seqdemo capture into a Hugging Face dataset with columns "audio" (the repetitions averaged,
float32), "text" (the listing verbatim), "opcodes", "project" and "num_samples".
The capture's zarr.json files go into info.description.

    zapdos-dataset captures/capture_<...>/ my-dataset/
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


def sequences(path):
    """seq_id -> listing, from a jsonl file (one sequence per line) or a directory of <seq_id>.s files."""
    if path.is_file():
        texts = [json.loads(line)["sequence"].strip() for line in path.open()]
        return lambda seq_id: texts[seq_id - 1]
    return lambda seq_id: (path / f"{seq_id}.s").read_text().strip()


def capture_rows(root, text, keep_repetitions):
    traces, ids, chunk = root["traces"], root["seq_ids"][:].ravel(), root["traces"].chunks[0]
    bounds = [0, *(np.flatnonzero(np.diff(ids)) + 1).tolist(), len(ids)]  # a sequence's repetitions are consecutive
    start, block = 0, traces[:0]
    for lo, hi in zip(bounds, bounds[1:]):
        if hi > start + len(block):  # read whole chunks, so each is decompressed once
            start = lo - lo % chunk
            block = traces[start:max(hi, start + chunk)]
        yield row(f"trace_{ids[lo]}", block[lo - start:hi - start], text(int(ids[lo])), keep_repetitions)


def build(src, keep_repetitions=False, sequences_path=None):
    src = Path(src).resolve()
    root = zarr.open_group(src, mode="r")
    if sequences_path is None:  # what the capture was recorded from
        sequences_path = Path(root.attrs["sequences_dir"])
        if root.attrs.get("sequences_format") == "jsonl":
            sequences_path /= "sequences.jsonl"
    text = sequences(Path(sequences_path))
    metadata = {p.parent.name: json.loads(p.read_text()) for p in [src / "zarr.json", *sorted(src.glob("*/zarr.json"))]}
    return Dataset.from_generator(lambda: capture_rows(root, text, keep_repetitions),
                                  info=DatasetInfo(description=json.dumps(metadata)))


@click.command()
@click.argument("src")
@click.argument("dst")
@click.option("--keep-repetitions", is_flag=True, help="Store every repetition instead of their mean.")
@click.option("--sequences", help="A sequences .jsonl file or a directory of .s files (default: what the capture was recorded from).")
def main(src, dst, keep_repetitions, sequences):
    build(src, keep_repetitions, sequences).save_to_disk(dst)


if __name__ == "__main__":
    main()
