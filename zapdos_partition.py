"""Split a dataset made by zapdos-dataset, zapdos-functions or zapdos-split into "train", "validation" and "test", so
the training repo reads a fixed split instead of drawing its own. The split is by "group" (--group-by): all rows of a
group (a function's chunks after zapdos-split) go to the same side, so no function is trained on and evaluated on.

--validation and --test are a fraction of the rows when at most 1, else a number of rows. Groups are shuffled with
--seed and taken whole until a side has at least that many rows, test first, then validation; the rest is train.

    zapdos-partition my-dataset-split/ my-dataset-final/
    zapdos-partition my-dataset-split/ my-dataset-final/ --validation 0.1 --test 0.1 --seed 1
    zapdos-partition my-dataset/ my-dataset-final/ --group-by project  # zapdos-functions output, not chunked
"""
import random

import click
from datasets import Dataset, DatasetDict, load_from_disk


def rows_wanted(size, total):
    return min(int(size * total if size <= 1 else size), total)


def group_split(groups, sizes, seed=42):
    """Row indices per side: shuffled whole groups go to each side in turn until it has at least its size in rows;
    the rows left over make up the last list."""
    counts = {}
    for group in groups:
        counts[group] = counts.get(group, 0) + 1
    unique = sorted(counts)
    random.Random(seed).shuffle(unique)

    side_of, it = {}, iter(unique)
    for side, size in enumerate(sizes):
        count = 0
        while count < size and (group := next(it, None)) is not None:
            side_of[group], count = side, count + counts[group]
    sides = [[] for _ in range(len(sizes) + 1)]
    for i, group in enumerate(groups):
        sides[side_of.get(group, len(sizes))].append(i)
    return sides


@click.command()
@click.argument("src", type=click.Path(exists=True, file_okay=False))
@click.argument("dst")
@click.option("--validation", default=0.1, show_default=True, help="Validation rows: <=1 a fraction, >1 a count.")
@click.option("--test", default=0.1, show_default=True, help="Test rows: <=1 a fraction, >1 a count.")
@click.option("--seed", default=42, show_default=True, help="Seed of the group shuffle.")
@click.option("--group-by", default="group", show_default=True, help="Column whose rows stay on one side together.")
def main(src, dst, validation, test, seed, group_by):
    dataset = load_from_disk(src)
    if not isinstance(dataset, Dataset):
        raise click.ClickException(f"{src} is already split into {', '.join(dataset)}")
    if group_by not in dataset.column_names:
        raise click.ClickException(f"the dataset has no {group_by!r} column (it has {', '.join(dataset.column_names)})")
    groups = list(dataset[group_by])
    test_idx, validation_idx, train_idx = group_split(
        groups, [rows_wanted(test, len(groups)), rows_wanted(validation, len(groups))], seed)
    if not train_idx:
        raise click.ClickException("--validation and --test leave no rows to train on")
    splits = DatasetDict({"train": dataset.select(train_idx), "validation": dataset.select(validation_idx),
                          "test": dataset.select(test_idx)})
    for name, split in splits.items():
        click.echo(f"{name}: {len(split)} rows, {len(set(split[group_by]))} groups")
    splits.save_to_disk(dst)


if __name__ == "__main__":
    main()
