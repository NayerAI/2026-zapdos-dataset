# zapdos-dataset

Turns a raw scakit seqdemo or function capture into a Hugging Face dataset to ship instead of the raw traces, split
into "train", "validation" and "test" (by default 10% each for validation and test; a size at most 1 is a fraction of
the rows, a larger one a row count). A row is a whole sequence or function, so none ends up on two sides, and
[zapdos-split](https://github.com/NayerAI/2026-zapdos-splitting) keeps each function's chunks on its side.

    pip install "git+https://github.com/NayerAI/2026-zapdos-dataset.git@main"
    zapdos-dataset captures/capture_<...>/ my-dataset/
    zapdos-functions captures/capture_<...>/ my-dataset/  # input for zapdos-alignment's annotate.py
    zapdos-functions captures/capture_<...>/ my-dataset/ --functions functions.jsonl  # sources as {"source": ...} per line
    zapdos-functions captures/capture_<...>/ my-dataset/ --std  # also the standard deviation over the repetitions
    zapdos-functions captures/capture_<...>/ my-dataset/ --validation 1000 --test 1000 --seed 1

Plot a row's mean trace (from "train" unless `--split` names another), optionally with every repetition (needs
`--keep-repetitions`), the standard deviation (needs `--std` or `--keep-repetitions`) and the annotated instructions:

    zapdos-plot my-dataset/ 12  # project func_12 (or trace_12)
    zapdos-plot my-dataset/ 12 --repetitions --std --instructions -o func_12.png
    zapdos-plot my-dataset/ 12 --instructions --range 1000:5000  # only samples 1000 to 4999

