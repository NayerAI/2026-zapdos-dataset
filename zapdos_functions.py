import json
import math
import re
import uuid

import click
import numpy as np
import zarr
from capstone import CS_ARCH_ARM, CS_MODE_THUMB, Cs
from datasets import Dataset, DatasetDict, DatasetInfo
from sklearn.model_selection import GroupShuffleSplit


def number_of_projects(size, total):
    """A size below 1 is a fraction of the projects (rounded up, so a nonzero one takes at least one), a larger one a
    project count."""
    if size < 1:
        return math.ceil(size * total)

    return int(size)


def take(rows, projects, count, seed):
    """Split rows into those of count randomly chosen whole projects and the rest."""
    if count == 0:
        return np.array([], dtype=int), rows

    splitter = GroupShuffleSplit(n_splits=1, test_size=count, random_state=seed)
    rest, taken = next(splitter.split(rows, groups=projects[rows]))

    return rows[taken], rows[rest]


def split(dataset, validation, test, seed):
    """Split by project into "train", "validation" and "test", each kept in capture order."""
    projects = np.array(dataset["project"])
    total = len(set(projects))

    n_test = number_of_projects(test, total)
    n_validation = number_of_projects(validation, total)

    if n_test + n_validation >= total:
        raise click.ClickException(f"--validation ({n_validation}) and --test ({n_test}) leave none of the {total} "
                                   "projects to train on")

    test_rows, rest = take(np.arange(len(projects)), projects, n_test, seed)
    validation_rows, train_rows = take(rest, projects, n_validation, seed)

    splits = {"train": train_rows, "validation": validation_rows, "test": test_rows}
    for name, rows in splits.items():
        click.echo(f"{name}: {len(rows)} rows, {len(set(projects[rows]))} projects", err=True)

    return DatasetDict({name: dataset.select(sorted(rows.tolist())) for name, rows in splits.items()})


KEYS = ("digest", "project", "path", "name", "sample", "instructions")


def functions(path):
    """The functions' records, in id order: a jsonl of {"digest", "project", "path", "name", "sample", "instructions"}."""
    records = []
    with open(path) as file:
        for number, line in enumerate(file, 1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise click.ClickException(f"{path}:{number}: not JSON ({error})")

            missing = [key for key in KEYS if key not in record] if isinstance(record, dict) else list(KEYS)
            if missing:
                raise click.ClickException(f"{path}:{number}: missing {', '.join(missing)}")

            records.append(record)

    return records


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


def rows(root, records):
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
            if function_id > len(records):
                raise click.ClickException(f"the capture has function {function_id}, the jsonl only {len(records)}")
            record = records[function_id - 1]

            row = {
                "id": function_id,
                "trace": trace,
                "std": std,
                "source": record["sample"],
                "digest": record["digest"],
                "project": record["project"],
                "path": record["path"],
                "name": record["name"],
                "num_instructions": record["instructions"],
                "num_samples": len(trace),
            }

            if "instr_offsets" in shard:
                row["instructions"], row["timestamps"] = instructions(shard, i, start)
                row["cycles"] = int(shard["func_pulse_cycles"][i])

            yield row


def build(src, jsonl):
    root = zarr.open_group(src, mode="r")
    records = functions(jsonl)

    return Dataset.from_generator(
        lambda: rows(root, records),
        fingerprint=uuid.uuid4().hex,  # never reuse a cached build of an earlier capture
        info=DatasetInfo(description=json.dumps(dict(root.attrs))),
    )


@click.command()
@click.argument("src")
@click.argument("dst")
@click.option("--functions", required=True, type=click.Path(exists=True, dir_okay=False),
              help='The captured functions, one {"digest", "project", "path", "name", "sample", "instructions"} per line in id order.')
@click.option("--validation", default=0.1, show_default=True, help="Validation projects: <1 a fraction, >=1 a count.")
@click.option("--test", default=0.1, show_default=True, help="Test projects: <1 a fraction, >=1 a count.")
@click.option("--seed", default=42, show_default=True, help="Seed of the projects' shuffle.")
def main(src, dst, functions, validation, test, seed):
    dataset = build(src, functions)
    splits = split(dataset, validation, test, seed)
    splits.save_to_disk(dst, num_shards={name: 1 for name, rows in splits.items() if len(rows) == 0} or None)


if __name__ == "__main__":
    main()
