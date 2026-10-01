# zapdos-dataset

Turns a raw scakit seqdemo or function capture into a Hugging Face dataset to ship instead of the raw traces.

    pip install "git+https://github.com/NayerAI/2026-zapdos-dataset.git@main"
    zapdos-dataset captures/capture_<...>/ my-dataset/
    zapdos-functions captures/capture_<...>/ my-dataset/  # input for zapdos-alignment's annotate.py
