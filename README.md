# zapdos-dataset

Turns a raw scakit seqdemo or function capture into a Hugging Face dataset to ship instead of the raw traces.

    pip install "git+https://github.com/NayerAI/2026-zapdos-dataset.git@main"
    zapdos-dataset captures/capture_<...>/ my-dataset/
    zapdos-functions captures/capture_<...>/ my-dataset/  # input for zapdos-alignment's annotate.py
    zapdos-functions captures/capture_<...>/ my-dataset/ --functions functions.jsonl  # sources as {"source": ...} per line
    zapdos-functions captures/capture_<...>/ my-dataset/ --std  # also the standard deviation over the repetitions

Plot a row's mean trace, optionally with every repetition (needs `--keep-repetitions`), the standard deviation (needs
`--std` or `--keep-repetitions`) and the annotated instructions:

    zapdos-plot my-dataset/ 12  # project func_12 (or trace_12)
    zapdos-plot my-dataset/ 12 --repetitions --std --instructions -o func_12.png
    zapdos-plot my-dataset/ 12 --instructions --range 1000:5000  # only samples 1000 to 4999

Split a dataset into "train", "validation" and "test" for training, by "group" so a function's chunks from
[zapdos-chunk](https://github.com/NayerAI/2026-zapdos-splitting) never end up on two sides. Run it last, on what
the training repo should read; sizes at most 1 are fractions of the rows, larger ones row counts:

    zapdos-split my-dataset-chunks/ my-dataset-final/                     # 10% validation, 10% test
    zapdos-split my-dataset-chunks/ my-dataset-final/ --validation 1000 --test 1000 --seed 42
    zapdos-split my-dataset/ my-dataset-final/ --group-by project        # unchunked: one row per function
