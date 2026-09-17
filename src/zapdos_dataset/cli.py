"""Command line front end: a directory of `seq_export` output -> a dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from zapdos_dataset.prepare import AGGREGATORS, DTYPES, build


def warn(message: str) -> None:
    print(f"[!] {message}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zapdos-dataset",
        description=(
            "Turn a directory of raw .npz recordings into a Hugging Face dataset, "
            "so the prepared dataset can be shipped instead of the raw traces."
        ),
    )
    parser.add_argument("raw_dir", type=Path, help="directory holding the .npz recordings")
    parser.add_argument("out_dir", type=Path, help="where to save the Hugging Face dataset")
    parser.add_argument(
        "--aggregator",
        choices=sorted(AGGREGATORS),
        default="mean",
        help="how to collapse the repeated recordings of a sequence (default: mean)",
    )
    parser.add_argument(
        "--keep-repetitions",
        action="store_true",
        help=(
            "store every repetition instead of aggregating them. Keeps the "
            "recording lossless at roughly the repetition count in size"
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPES),
        default="float32",
        help=(
            "sample dtype to store (default: float32, exact for int16 counts "
            "and their averages; use float64 for a float64 capture)"
        ),
    )
    parser.add_argument(
        "--trim-trailing-zeros",
        action="store_true",
        help="drop a run of exact zeros at the end of each trace (zero-padded captures only)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help=(
            "standardize each trace to zero mean and unit variance. Off by "
            "default: trace_mean and trace_std are stored either way, so the "
            "training side can do it without the dataset losing its units"
        ),
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=None,
        help=(
            "override the scope sample rate. Read from the capture metadata "
            "when it is present, so this is only needed for a capture exported "
            "without it"
        ),
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    dataset = build(
        args.raw_dir,
        args.out_dir,
        aggregator=args.aggregator,
        sample_rate_hz_override=args.sample_rate_hz,
        dtype=args.dtype,
        trim=args.trim_trailing_zeros,
        normalize=args.normalize,
        keep_repetitions=args.keep_repetitions,
        on_warning=warn,
    )
    print(f"[+] {len(dataset)} recordings -> {args.out_dir}")
    print(f"    {dataset.info.description}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
