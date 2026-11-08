import re

import click
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk


def find(dataset, id):
    """The row of function number id, or the first whose digest or name is id."""
    ids = list(dataset["id"])

    if id.isdigit():
        for i, number in enumerate(ids):
            if number == int(id):
                return dataset[i]
    else:
        for key in ("digest", "name"):
            for i, value in enumerate(list(dataset[key])):
                if value == id:
                    return dataset[i]

    raise click.BadParameter(f"no function {id!r} in this split (e.g. {', '.join(map(str, ids[:3]))}, ...)",
                             param_hint="ID")


def parse_range(ctx, param, value):
    """FROM:TO, either end optional, as (from, to) with None for a missing end."""
    if value is None:
        return None, None

    match = re.fullmatch(r"(-?\d*):(-?\d*)", value)
    if not match:
        raise click.BadParameter("expected FROM:TO, e.g. 1000:5000, 1000: or :5000")

    start, stop = match.groups()
    start = int(start) if start else None
    stop = int(stop) if stop else None

    return start, stop


@click.command()
@click.argument("dataset", type=click.Path(exists=True, file_okay=False))
@click.argument("id")
@click.option("--std", "show_std", is_flag=True, help="Mean ± one standard deviation over the repetitions.")
@click.option("--instructions", "show_instructions", is_flag=True, help="Each annotated instruction at its start sample.")
@click.option("--range", "crop", callback=parse_range, metavar="FROM:TO",
              help="Only plot samples FROM up to TO; either end may be left out (1000:, :5000).")
@click.option("--split", default="train", show_default=True, help="Split to read from.")
@click.option("-o", "--output", type=click.Path(dir_okay=False), help="Save the figure here instead of showing it.")
def main(dataset, id, show_std, show_instructions, crop, split, output):
    dataset = load_from_disk(dataset)[split]
    row = find(dataset, id)

    mean = np.asarray(row["trace"], dtype=np.float32)
    std = np.asarray(row["std"], dtype=np.float32)

    lo, hi, _ = slice(*crop).indices(len(mean))
    if hi <= lo:
        raise click.BadParameter(f"selects no samples of the {len(mean)}", param_hint="--range")

    x = np.arange(lo, hi)
    mean = mean[lo:hi]
    std = std[lo:hi]

    fig, ax = plt.subplots(figsize=(16, 5))

    if show_std:
        ax.fill_between(x, mean - std, mean + std, color="tab:blue", alpha=0.25, lw=0, label="± 1 std")

    ax.plot(x, mean, color="tab:blue", lw=0.8, label="mean")

    if show_instructions:
        if not row.get("instructions"):
            click.echo("zapdos-plot: the dataset has no instructions (scakit's func_annotate adds them)", err=True)
        else:
            for t, instruction in zip(row["timestamps"], row["instructions"]):
                if not lo <= t < hi:
                    continue

                ax.axvline(t, color="tab:red", lw=0.5, alpha=0.5)
                ax.text(t, 1, f" {instruction}", transform=ax.get_xaxis_transform(), rotation=90, va="top",
                        ha="right", fontsize=6, color="tab:red", clip_on=True)

    title = f"{row['id']}: {row['project']} {row['path']} {row['name']}"
    if row.get("cycles"):
        title += f", {row['cycles']} cycles"

    ax.set(title=title, xlabel="sample", ylabel="V", xlim=(lo, max(lo, hi - 1)))
    ax.legend(loc="lower right")
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    main()
