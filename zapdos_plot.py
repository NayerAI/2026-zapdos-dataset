"""Plot the mean trace of one row of a dataset made by zapdos-dataset or zapdos-functions, picked by its "project"
(func_12, trace_12, or just 12). Optionally also every repetition, the standard deviation over them, and the
annotated instructions; an overlay the dataset has no data for is skipped with a warning.

    zapdos-plot my-dataset/ 12
    zapdos-plot my-dataset/ 12 --repetitions --std --instructions -o func_12.png
    zapdos-plot my-dataset/ func_12 --instructions --range 0:5000
"""
import re

import click
import matplotlib.pyplot as plt
import numpy as np
from datasets import DatasetDict, load_from_disk


def find(dataset, id):
    """The row whose project is id, or ends in _<id> for a bare number."""
    projects = list(dataset["project"])
    for i, project in enumerate(projects):
        if project == id or (id.isdigit() and re.fullmatch(rf"[^_]+_0*{int(id)}", project)):
            return dataset[i]
    raise click.BadParameter(f"no project {id!r} (e.g. {', '.join(projects[:3])}, ...)", param_hint="ID")


def signals(row):
    """The mean trace, the repetitions (one per row) if the dataset kept them, and the standard deviation over them if
    it kept that or the repetitions; None for what it lacks."""
    trace = np.asarray(row["trace"] if "trace" in row else row["audio"], dtype=np.float32)
    if trace.ndim == 2:  # zapdos-dataset --keep-repetitions stores them in place of the mean
        mean, repetitions = trace.mean(axis=0), trace
    else:
        mean, repetitions = trace, row.get("repetitions")
        repetitions = None if repetitions is None else np.asarray(repetitions, dtype=np.float32)
    if repetitions is not None and len(repetitions) > 1:
        return mean, repetitions, repetitions.std(axis=0)
    return mean, repetitions, None if row.get("std") is None else np.asarray(row["std"], dtype=np.float32)


def parse_range(ctx, param, value):
    """FROM:TO, either end optional, as (from, to) with None for a missing end."""
    if value is None:
        return None, None
    m = re.fullmatch(r"(-?\d*):(-?\d*)", value)
    if not m:
        raise click.BadParameter("expected FROM:TO, e.g. 1000:5000, 1000: or :5000")
    return tuple(int(v) if v else None for v in m.groups())


def skip(what, why):
    click.echo(f"zapdos-plot: no {what}: {why}", err=True)


@click.command()
@click.argument("dataset", type=click.Path(exists=True, file_okay=False))
@click.argument("id")
@click.option("--repetitions", "--all", "show_repetitions", is_flag=True, help="Every repetition the mean was taken over.")
@click.option("--std", "show_std", is_flag=True, help="Mean ± one standard deviation over the repetitions.")
@click.option("--instructions", "show_instructions", is_flag=True, help="Each annotated instruction at its start sample.")
@click.option("--range", "crop", callback=parse_range, metavar="FROM:TO",
              help="Only plot samples FROM up to TO; either end may be left out (1000:, :5000).")
@click.option("--split", default="train", show_default=True, help="Split to read from a DatasetDict.")
@click.option("-o", "--output", type=click.Path(dir_okay=False), help="Save the figure here instead of showing it.")
def main(dataset, id, show_repetitions, show_std, show_instructions, crop, split, output):
    dataset = load_from_disk(dataset)
    if isinstance(dataset, DatasetDict):
        dataset = dataset[split]
    row = find(dataset, id)
    mean, repetitions, std = signals(row)
    lo, hi, _ = slice(*crop).indices(len(mean))
    if hi <= lo:
        raise click.BadParameter(f"selects no samples of the {len(mean)}", param_hint="--range")
    mean, x = mean[lo:hi], np.arange(lo, hi)
    repetitions = None if repetitions is None else repetitions[:, lo:hi]
    std = None if std is None else std[lo:hi]

    fig, ax = plt.subplots(figsize=(16, 5))
    if show_repetitions:
        if repetitions is None:
            skip("repetitions", "the dataset only has their mean (zapdos-functions/zapdos-dataset --keep-repetitions)")
        else:
            ax.plot(x, repetitions.T, color="tab:gray", lw=0.4, alpha=min(1.0, 3 / len(repetitions)))
    if show_std:
        if std is None:
            skip("standard deviation", "the dataset has neither it nor the repetitions (zapdos-functions --std)")
        else:
            ax.fill_between(x, mean - std, mean + std, color="tab:blue", alpha=0.25, lw=0, label="± 1 std")
    ax.plot(x, mean, color="tab:blue", lw=0.8,
            label="mean" if repetitions is None else f"mean of {len(repetitions)} repetitions")
    if show_instructions:
        if not row.get("instructions"):
            skip("instructions", "the dataset has no \"instructions\" and \"timestamps\" (scakit's func_annotate "
                                 "or zapdos-alignment's annotate.py add them)")
        else:
            for t, ins in zip(row["timestamps"], row["instructions"]):
                if not lo <= t < hi:
                    continue
                ax.axvline(t, color="tab:red", lw=0.5, alpha=0.5)
                ax.text(t, 1, f" {ins}", transform=ax.get_xaxis_transform(), rotation=90, va="top", ha="right",
                        fontsize=6, color="tab:red", clip_on=True)

    title = row["project"] + (f", {row['cycles']} cycles" if row.get("cycles") else "")
    ax.set(title=title, xlabel="sample", ylabel="V" if "trace" in row else "", xlim=(lo, max(lo, hi - 1)))
    ax.legend(loc="lower right")
    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    main()
