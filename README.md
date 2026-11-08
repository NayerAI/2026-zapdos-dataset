# zapdos-dataset

Turns a scakit function capture, whose recordings are already aggregated into a mean and a standard deviation, into a
Hugging Face dataset split by project into "train", "validation" and "test", so all functions of a project land in the
same split (scikit-learn's `GroupShuffleSplit`, seeded with `--seed`). `--validation` and `--test` count projects, not
rows: below 1 is a fraction of the projects, rounded up so that each split gets at least one, 1 or more a project
count (by default 10% each).

    pip install "git+https://github.com/NayerAI/2026-zapdos-dataset.git@main"
    zapdos-functions captures/capture_<...>/ my-dataset/ --functions functions.jsonl
    zapdos-functions captures/capture_<...>/ my-dataset/ --functions functions.jsonl --validation 1000 --test 1000 --seed 1

`--functions` is the jsonl of the captured functions, line n being function n:

    {"digest": "76e5...", "project": "android-vuln", "path": "CVE-2015-3636/.../InterfaceInit.c", "name": "bcm_usb_endpoint_dir_out", "sample": "#include <stddef.h>\n...", "instructions": 28}

Each row keeps the function's number as "id", its "sample" as "source", its "digest", "project", "path" and "name", and
its "instructions" count as "num_instructions".

A capture that records each function at several averaging levels (its "checkpoints" attribute, e.g. `[100, 200, ...,
1000]`: the mean over the first 100, 200, ... repetitions) becomes one dataset per level, `<DST>-rec100`,
`<DST>-rec200`, ..., all with the same rows in "train", "validation" and "test". "repetitions" is the row's level.
"std" is the standard deviation over all repetitions in every level, since the capture records it only for the last
checkpoint. A capture with a single checkpoint is written to `<DST>` itself:

    zapdos-functions captures/functions-id4/ hf-functions-id4 --functions functions.jsonl  # hf-functions-id4-rec100, ...

Plot a row's mean trace (from "train" unless `--split` names another), optionally with the standard deviation and the
annotated instructions:

    zapdos-plot my-dataset/ 12  # function 12 (or hf-functions-id4-rec100/ 12)
    zapdos-plot my-dataset/ bcm_usb_endpoint_dir_out  # by name (or digest)
    zapdos-plot my-dataset/ 12 --std --instructions -o func_12.png
    zapdos-plot my-dataset/ 12 --instructions --range 1000:5000  # only samples 1000 to 4999
