import json
import random
import re
import uuid
from pathlib import Path

import click
import numpy as np
import zarr
from capstone import CS_ARCH_ARM, CS_MODE_THUMB, Cs
from datasets import Dataset, DatasetDict, DatasetInfo


def number_of_rows(size, total):
    """A size at most 1 is a fraction of the rows, a larger one a row count."""
    if size <= 1:
        count = int(size * total)
    else:
        count = int(size)

    return min(count, total)


def split(dataset, validation, test, seed):
    """Shuffle the functions into "train", "validation" and "test", each kept in capture order."""
    total = len(dataset)
    n_validation = number_of_rows(validation, total)
    n_test = number_of_rows(test, total)

    if n_validation + n_test >= total:
        raise click.ClickException(f"--validation and --test leave none of the {total} rows to train on")

    order = list(range(total))
    random.Random(seed).shuffle(order)

    test_rows = order[:n_test]
    validation_rows = order[n_test:n_test + n_validation]
    train_rows = order[n_test + n_validation:]

    return DatasetDict({
        "train": dataset.select(sorted(train_rows)),
        "validation": dataset.select(sorted(validation_rows)),
        "test": dataset.select(sorted(test_rows)),
    })


def sources(path):
    """Function id -> C source, from a jsonl file (one {"source": ...} per line, in id order) or a directory of <id>.c."""
    if path.is_file():
        texts = [json.loads(line)["source"] for line in path.open()]

        def source(function_id):
            return texts[function_id - 1]
    else:
        def source(function_id):
            return (path / f"{function_id}.c").read_text()

    return source


def instructions(shard, i, start):
    """func_annotate's executed instructions of function i, and the samples they start at relative to start."""
    lo = shard["instr_offsets"][i]
    hi = shard["instr_offsets"][i + 1]
    opcodes = shard["instr_opcode"][lo:hi]
    sizes = shard["instr_size"][lo:hi]
    samples = shard["instr_sample"][lo:hi]

    disassembler = Cs(CS_ARCH_ARM, CS_MODE_THUMB)

    texts = []
    for opcode, size in zip(opcodes, sizes):
        code = int(opcode).to_bytes(4, "little")[:size]
        instruction = next(disassembler.disasm(code, 0))
        text = f"{instruction.mnemonic} {instruction.op_str}".strip()
        texts.append(re.sub(r"#-?(0x)?[0-9a-f]+", "#imm", text))

    timestamps = (samples - start).tolist()

    return texts, timestamps


def rows(root, source):
    """One row per function: the mean and standard deviation over its repetitions in volts, cropped to the function."""
    channel = root.attrs["scope"]["channels"][0]
    adc_min = root.attrs["adc_mincount"]
    adc_max = root.attrs["adc_maxcount"]
    volts_per_count = 2 * channel["range"] / (adc_max - adc_min)

    # The capture aligns every function's trace so that it starts at the same sample.
    start = round(root.attrs["alignment"]["rise_sample"])

    for s in range(root.attrs["num_shards"]):
        shard = root[f"shard_{s:02d}"]
        offsets = shard["func_offsets"][:]
        ends = shard["func_exit_sample"][:]

        for i in range(len(ends)):
            lo = int(offsets[i]) + start
            hi = int(offsets[i]) + int(ends[i])

            mean = shard["traces"][lo:hi].astype(np.float64)
            std = shard["traces_std"][lo:hi].astype(np.float64)

            trace = ((mean - adc_min) * volts_per_count - channel["range"] - channel["offset"]).astype(np.float32)
            std = (std * volts_per_count).astype(np.float32)

            function_id = s * root.attrs["shard_size"] + i + 1

            row = {
                "trace": trace,
                "std": std,
                "source": source(function_id),
                "project": f"func_{function_id}",
                "num_samples": len(trace),
            }

            if "instr_offsets" in shard:
                row["instructions"], row["timestamps"] = instructions(shard, i, start)
                row["cycles"] = int(shard["func_pulse_cycles"][i])

            yield row


def build(src, functions=None):
    root = zarr.open_group(src, mode="r")
    source = sources(Path(functions or root.attrs["func_dir"]))

    return Dataset.from_generator(
        lambda: rows(root, source),
        fingerprint=uuid.uuid4().hex,  # never reuse a cached build of an earlier capture
        info=DatasetInfo(description=json.dumps(dict(root.attrs))),
    )


@click.command()
@click.argument("src")
@click.argument("dst")
@click.option("--functions", help="A jsonl of the functions' sources or a directory of <id>.c (default: the capture's func_dir).")
@click.option("--validation", default=0.1, show_default=True, help="Validation rows: <=1 a fraction, >1 a count.")
@click.option("--test", default=0.1, show_default=True, help="Test rows: <=1 a fraction, >1 a count.")
@click.option("--seed", default=42, show_default=True, help="Seed of the split's shuffle.")
def main(src, dst, functions, validation, test, seed):
    dataset = build(src, functions)
    split(dataset, validation, test, seed).save_to_disk(dst)


if __name__ == "__main__":
    main()
