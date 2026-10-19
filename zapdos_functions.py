"""Turn a raw scakit function capture into a Hugging Face dataset, with columns "trace" (the repetitions averaged, in
volts, float32, cropped to the trigger pulse), "source" (the function's C code), "project" and "num_samples".
If scakit's func_annotate ran on the capture, also "instructions" (as executed on the device, immediates as #imm),
"timestamps" (the trace sample each starts at) and "cycles" (the pulse's length in device cycles).
With --std, also "std" (the per-sample standard deviation over the repetitions, float32), and with
--keep-repetitions "repetitions" (every repetition the mean was taken over, one row each; large).
The capture's attributes go into info.description.

    zapdos-functions captures/capture_<...>/ my-dataset/
"""
import json
import re
import uuid
from pathlib import Path

import click
import numpy as np
import zarr
from capstone import CS_ARCH_ARM, CS_MODE_THUMB, Cs
from datasets import Dataset, DatasetInfo


def sources(path):
    """function id -> C source, from a jsonl file (one {"source": ...} per line, in id order) or a directory of <id>.c."""
    if path.is_file():
        texts = [json.loads(line)["source"] for line in path.open()]
        return lambda fid: texts[fid - 1]
    return lambda fid: (path / f"{fid}.c").read_text()


def instructions(shard, i, start):
    """func_annotate's executed instructions of function i, and the samples they start at relative to start."""
    lo, hi = shard["instr_offsets"][i:i + 2]
    md = Cs(CS_ARCH_ARM, CS_MODE_THUMB)
    text = [next(md.disasm(int(op).to_bytes(4, "little")[:size], 0))
            for op, size in zip(shard["instr_opcode"][lo:hi], shard["instr_size"][lo:hi])]
    return ([re.sub(r"#-?(0x)?[0-9a-f]+", "#imm", f"{ins.mnemonic} {ins.op_str}".strip()) for ins in text],
            (shard["instr_sample"][lo:hi] - start).tolist())


def rows(root, source, std=False, keep_repetitions=False):
    channel = root.attrs["scope"]["channels"][0]
    lo, hi = root.attrs["adc_mincount"], root.attrs["adc_maxcount"]
    volts = lambda counts: ((counts - lo) / (hi - lo) - 0.5) * 2 * channel["range"] - channel["offset"]
    for s in range(root.attrs["num_shards"]):
        shard = root[f"shard_{s:02d}"]
        offsets, samples = shard["func_offsets"][:], shard["func_samples"][:]
        rise, fall, ok = shard["func_rise_sample"][:], shard["func_fall_sample"][:], shard["func_edge_flags"][:] == 0
        for i, n in enumerate(samples):
            if not ok[i].any():
                continue  # no repetition with both trigger edges
            traces = shard["traces"][offsets[i]:offsets[i + 1]].reshape(-1, n)
            start, end = round(np.median(rise[i][ok[i]])), round(np.median(fall[i][ok[i]]))
            reps = traces[ok[i][:len(traces)], start:end]
            trace = volts(reps.mean(axis=0, dtype=np.float64)).astype(np.float32)
            fid = s * root.attrs["shard_size"] + i + 1
            row = {"trace": trace, "source": source(fid), "project": f"func_{fid}", "num_samples": len(trace)}
            if std:
                row["std"] = (reps.std(axis=0, dtype=np.float64) * 2 * channel["range"] / (hi - lo)).astype(np.float32)
            if keep_repetitions:
                row["repetitions"] = volts(reps.astype(np.float32))
            if "instr_offsets" in shard:
                row["instructions"], row["timestamps"] = instructions(shard, i, start)
                row["cycles"] = int(shard["func_pulse_cycles"][i])
            yield row


def build(src, functions=None, std=False, keep_repetitions=False):
    root = zarr.open_group(src, mode="r")
    source = sources(Path(functions or root.attrs["func_dir"]))
    return Dataset.from_generator(lambda: rows(root, source, std, keep_repetitions),
                                  fingerprint=uuid.uuid4().hex,  # never reuse a cached build of an earlier capture
                                  info=DatasetInfo(description=json.dumps(dict(root.attrs))))


@click.command()
@click.argument("src")
@click.argument("dst")
@click.option("--std", is_flag=True, help="Also store the per-sample standard deviation over the repetitions.")
@click.option("--keep-repetitions", is_flag=True, help="Also store every repetition (one trace's size each).")
@click.option("--functions", help="A jsonl of the functions' sources or a directory of <id>.c (default: the capture's func_dir).")
def main(src, dst, std, keep_repetitions, functions):
    build(src, functions, std, keep_repetitions).save_to_disk(dst)


if __name__ == "__main__":
    main()
