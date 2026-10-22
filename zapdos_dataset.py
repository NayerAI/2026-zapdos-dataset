"""Turn a raw scakit seqdemo capture into a Hugging Face dataset with columns "audio" (the repetitions averaged,
float32), "text" (the listing verbatim), "opcodes", "project" and "num_samples", split into "train", "validation" and
"test".
The capture's zarr.json files go into info.description.

    zapdos-dataset captures/capture_<...>/ my-dataset/
    zapdos-dataset captures/capture_<...>/ my-dataset/ --validation 1000 --test 1000 --seed 1
"""
import json
import random
import uuid
from pathlib import Path

import click
import numpy as np
import zarr
from datasets import Dataset, DatasetDict, DatasetInfo


def split(dataset, validation, test, seed):
    """Shuffle the rows with seed into "train", "validation" and "test", each in capture order. A size at most 1 is a
    fraction of the rows, a larger one a row count. A row is a whole sequence or function, so none is on two sides."""
    wanted = lambda size: min(int(size * len(dataset) if size <= 1 else size), len(dataset))
    order = list(range(len(dataset)))
    random.Random(seed).shuffle(order)
    n_test, n_validation = wanted(test), wanted(validation)
    if n_test + n_validation >= len(dataset):
        raise click.ClickException(f"--validation and --test leave none of the {len(dataset)} rows to train on")
    sides = {"train": order[n_test + n_validation:], "validation": order[n_test:n_test + n_validation],
             "test": order[:n_test]}
    return DatasetDict({name: dataset.select(sorted(rows)) for name, rows in sides.items()})


def split_options(command):
    """The --validation, --test and --seed options of split."""
    command = click.option("--seed", default=42, show_default=True, help="Seed of the split's shuffle.")(command)
    command = click.option("--test", default=0.1, show_default=True,
                           help="Test rows: <=1 a fraction, >1 a count.")(command)
    return click.option("--validation", default=0.1, show_default=True,
                        help="Validation rows: <=1 a fraction, >1 a count.")(command)


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
                                  fingerprint=uuid.uuid4().hex,  # never reuse a cached build of an earlier capture
                                  info=DatasetInfo(description=json.dumps(metadata)))


@click.command()
@click.argument("src")
@click.argument("dst")
@click.option("--keep-repetitions", is_flag=True, help="Store every repetition instead of their mean.")
@click.option("--sequences", help="A sequences .jsonl file or a directory of .s files (default: what the capture was recorded from).")
@split_options
def main(src, dst, keep_repetitions, sequences, validation, test, seed):
    split(build(src, keep_repetitions, sequences), validation, test, seed).save_to_disk(dst)


if __name__ == "__main__":
    main()
