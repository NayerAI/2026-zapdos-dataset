# zapdos-dataset

Turns a directory of scakit `seq_export` recordings, or a raw function capture, into a Hugging Face dataset to ship instead of the raw traces.

    pip install "git+https://github.com/NayerAI/2026-zapdos-dataset.git@main"
    zapdos-dataset captures/capture_<...>/ my-dataset/  # or seq_traces/
    zapdos-functions captures/capture_<...>/ my-dataset/  # input for zapdos-alignment's annotate.py
