# zapdos-dataset

Turns a directory of raw side-channel recordings into a Hugging Face dataset,
so that the prepared dataset is what gets transmitted rather than the raw
traces.

Run it on the recording machine, right after `seq_export`, and send the
resulting directory on.

## Install

The package is installed straight from this repository:

```sh
pip install "git+https://github.com/NayerAI/2026-zapdos-dataset.git@main"
# or with uv
uv pip install "git+https://github.com/NayerAI/2026-zapdos-dataset.git@main"
```

Note the `git+` prefix and the quotes: `pip install https://github.com/...`
without it makes pip look for an archive rather than a git checkout, and the
`@branch` needs the quotes in most shells. Any branch or tag works in place of
`@main`.

It depends only on `datasets` and `numpy`, not on `scakit`, so it installs on a
recording machine that has no access to the scakit index.

## Use

One `.npz` file is one recorded sequence, as written by
`scakit_seqdemo.seq_export`:

| key                       | shape                        | meaning                         |
| ------------------------- | ---------------------------- | ------------------------------- |
| `traces`                  | `(n_repetitions, n_samples)` | the repeated recordings         |
| `sequence`                | scalar string                | the source listing, one instruction per line |

Captures exported before scakit-example renamed it call that second key
`instructions`; both names are read.

Alongside the recordings, `seq_export` writes `metadata/*.json`, the capture's
zarr attributes. Keep that directory next to the traces: it is where the scope
sample rate, channel range and offset come from.

```sh
seq_export captures/capture_<...>_seqdemo.elf/ -o seq_traces/
zapdos-dataset seq_traces/ my-dataset/
```

or from Python:

```python
from zapdos_dataset import build

build("seq_traces/", "my-dataset/")
```

Ship the resulting `my-dataset/` directory.

## Reading it back

```python
from zapdos_dataset import dataset_info, load

dataset = load("my-dataset/")
info = dataset_info(dataset)          # aggregator, sample rate, scope config
```

`load` is `datasets.load_from_disk` plus the dtype the dataset was built with.
Use it rather than `load_from_disk(...).with_format("numpy")`: the numpy
formatter in `datasets` is float32 by default whatever the stored feature says,
so a float64 dataset read that way is silently narrowed. Plain
`load_from_disk` without a format is exact — it hands back Python floats — but
it materializes every trace as a list of Python objects.

One more narrowing to know about, since it hides the first: comparing a numpy
float32 scalar against a Python float casts the float down to float32 first
(NEP 50), so a lost mantissa compares equal. Put `np.float64(...)` on both
sides when checking a trace against a reference.

## Output

One row per recording:

| column            | type     | content                                                               |
| ----------------- | -------- | --------------------------------------------------------------------- |
| `audio`           | float32  | the repetitions aggregated into one trace, in the units the scope recorded |
| `text`            | str      | the source listing verbatim, operands included                         |
| `opcodes`         | str      | the mnemonics of `text`, space separated — the usual training target   |
| `project`         | str      | the file stem, the grouping key for train/test splits                  |
| `num_samples`     | int      | `len(audio)`, for filtering by trace length without reading the traces |
| `num_repetitions` | int      | how many raw traces went into `audio`                                  |
| `trace_mean`      | float    | mean of `audio` before any normalization                               |
| `trace_std`       | float    | standard deviation of `audio` before any normalization                 |

The scope configuration, the aggregator, the sample rate and the choices made
below are stored as JSON in the dataset's `info.description`.

The output is model agnostic: no tokenizer is applied, so the same dataset
serves any model.

## What is and is not thrown away

The raw `.npz` files stay on the recording machine, so anything this tool drops
is dropped for good. It therefore defaults to keeping everything it reasonably
can:

- **Repetitions are aggregated.** This is the one lossy step taken by default,
  and it is the whole size win — a stock seqdemo capture records 25 traces per
  sequence. `--keep-repetitions` stores every trace instead.
- **The full listing is kept.** `text` is the listing verbatim; `opcodes` is
  derived from it. Reducing the listing to bare mnemonics at record time would
  throw the operands away permanently.
- **Traces stay in their recorded units.** `--normalize` standardizes them to
  zero mean and unit variance, but `trace_mean` and `trace_std` are stored
  either way, so the standardization is invertible and the training side can
  apply it itself for free.
- **Trailing zeros are kept.** `--trim-trailing-zeros` drops the run of exact
  zeros at the end of each trace. That only makes sense for recordings padded
  to a common length; a `seq_export` capture is the full scope window, where an
  exact zero is a real sample.
- **Samples are stored as float32.** That is exact for int16 ADC counts and for
  averages of them — 24 mantissa bits against 16 data bits plus at most ~5 bits
  of averaging gain over 25 repetitions — and halves what has to be
  transferred. `--dtype float64` keeps the full width of a float64 capture.
- **Averaging accumulates in float64.** `np.mean` over a float32 input
  otherwise sums in float32, which puts ~1e-6 of relative error back into the
  average the repetitions were recorded to remove.
- **The scope settings travel with the dataset.** The sample rate is read from
  the capture metadata rather than typed on the command line. The training side
  sets the frequency bins of its spectral front-end from it, so a dataset that
  arrives without it cannot be trained on until somebody remembers what the
  scope was set to.

Three things it can only warn about, because only a re-record fixes them:

- a capture with no `metadata/` directory, whose scope settings are lost;
- samples sitting at an ADC rail, which were clipped at capture time;
- recordings whose source listing is empty, which is what `seq_export` writes
  when it cannot resolve a `seq_id` back to its text.

All three go to stderr, and the build still completes.

## Coming from the old `dataset_np_to_hf.py`

The columns are not quite the same, so `prepare_sample` on the training side
needs two lines changed:

- `text` is now a plain string, not a one-element list, so `sample["text"][0]`
  becomes `sample["text"]`.
- the mnemonics are already in `sample["opcodes"]`, so the
  `[instr.split(" ")[0] for instr in text.split("\n")]` step can go. Keeping it
  would also keep its bug: splitting on a literal space leaves the mnemonic of
  a tab-separated line stuck to its first operand.

Traces arrive unnormalized, which is what `prepare_sample` already assumes.

## Options

```
zapdos-dataset RAW_DIR OUT_DIR
    --aggregator {mean,median}   how to collapse the repetitions (default: mean)
    --keep-repetitions           store every repetition instead of aggregating
    --dtype {float32,float64}    sample dtype to store (default: float32)
    --trim-trailing-zeros        drop a run of exact zeros at the end of each trace
    --normalize                  standardize each trace to zero mean, unit variance
    --sample-rate-hz HZ          override the rate read from the capture metadata
```

## Develop

```sh
uv sync
uv run pytest
```
