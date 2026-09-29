"""Turn a directory of scakit seq_export recordings (.npz) into a Hugging Face dataset with columns
"audio" (the repetitions averaged, float32), "text" (the listing verbatim), "opcodes", "project" and "num_samples".
The capture's metadata/*.json goes into info.description.

    zapdos-dataset seq_traces/ my-dataset/
"""
import json
from pathlib import Path

import click
import numpy as np
from datasets import Dataset, DatasetInfo


def row(path, keep_repetitions=False):
    with np.load(path) as npz:
        traces = npz["traces"]
        text = npz["sequence" if "sequence" in npz else "instructions"].item()  # older captures used "instructions"
    audio = (traces if keep_repetitions else traces.mean(axis=0, dtype=np.float64)).astype(np.float32)
    opcodes = " ".join(line.split()[0] for line in text.splitlines() if line.strip())
    return {"audio": audio, "text": text, "opcodes": opcodes, "project": path.stem, "num_samples": audio.shape[-1]}


def build(src, keep_repetitions=False):
    src = Path(src)
    paths = sorted(src.glob("*.npz"))
    metadata = {p.stem: json.loads(p.read_text()) for p in sorted(src.glob("metadata/*.json"))}
    return Dataset.from_generator(lambda: (row(p, keep_repetitions) for p in paths),
                                  info=DatasetInfo(description=json.dumps(metadata)))


@click.command()
@click.argument("src")
@click.argument("dst")
@click.option("--keep-repetitions", is_flag=True, help="Store every repetition instead of their mean.")
def main(src, dst, keep_repetitions):
    build(src, keep_repetitions).save_to_disk(dst)


if __name__ == "__main__":
    main()
