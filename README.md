# zapdos-dataset

Turns a scakit function capture, whose recordings are already aggregated into a mean and a standard deviation, into a
Hugging Face dataset split into "train", "validation" and "test" (by default 10% each for validation and test; a size
at most 1 is a fraction of the rows, a larger one a row count).

    pip install "git+https://github.com/NayerAI/2026-zapdos-dataset.git@main"
    zapdos-functions captures/capture_<...>/ my-dataset/ --functions functions.jsonl
    zapdos-functions captures/capture_<...>/ my-dataset/ --functions functions.jsonl --validation 1000 --test 1000 --seed 1

`--functions` is the jsonl of the captured functions, line n being function n:

    {"digest": "76e5...", "project": "android-vuln", "path": "CVE-2015-3636/.../InterfaceInit.c", "name": "bcm_usb_endpoint_dir_out", "sample": "#include <stddef.h>\n...", "instructions": 28}

Each row keeps the function's number as "id", its "sample" as "source", its "digest", "project", "path" and "name", and
its "instructions" count as "num_instructions".

Plot a row's mean trace (from "train" unless `--split` names another), optionally with the standard deviation and the
annotated instructions:

    zapdos-plot my-dataset/ 12  # function 12
    zapdos-plot my-dataset/ bcm_usb_endpoint_dir_out  # by name (or digest)
    zapdos-plot my-dataset/ 12 --std --instructions -o func_12.png
    zapdos-plot my-dataset/ 12 --instructions --range 1000:5000  # only samples 1000 to 4999
