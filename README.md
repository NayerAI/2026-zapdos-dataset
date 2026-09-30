# zapdos-dataset

Turns a directory of scakit `seq_export` recordings, or a raw funcdemo capture, into a Hugging Face dataset to ship instead of the raw traces.

    pip install "git+https://github.com/NayerAI/2026-zapdos-dataset.git@main"
    zapdos-dataset seq_traces/ my-dataset/
    zapdos-funcdemo captures/capture_<...>/ my-dataset/  # input for zapdos-alignment's annotate.py
