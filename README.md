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
