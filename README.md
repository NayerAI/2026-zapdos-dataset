# zapdos-dataset

Turns a scakit function capture, whose recordings are already aggregated into a mean and a standard deviation, into a
Hugging Face dataset split into "train", "validation" and "test" (by default 10% each for validation and test; a size
at most 1 is a fraction of the rows, a larger one a row count).

    pip install "git+https://github.com/NayerAI/2026-zapdos-dataset.git@main"
    zapdos-functions captures/capture_<...>/ my-dataset/
    zapdos-functions captures/capture_<...>/ my-dataset/ --functions functions.jsonl  # sources as {"source": ...} per line
    zapdos-functions captures/capture_<...>/ my-dataset/ --validation 1000 --test 1000 --seed 1

Plot a row's mean trace (from "train" unless `--split` names another), optionally with the standard deviation and the
annotated instructions:

    zapdos-plot my-dataset/ 12  # project func_12
    zapdos-plot my-dataset/ 12 --std --instructions -o func_12.png
    zapdos-plot my-dataset/ 12 --instructions --range 1000:5000  # only samples 1000 to 4999
