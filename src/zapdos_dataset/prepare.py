"""Raw side-channel recordings (.npz) -> a Hugging Face dataset.

One `.npz` file is one recorded sequence, as written by
`scakit_seqdemo.seq_export`. It holds:

    traces      (n_repetitions, n_samples) the repeated recordings of that
                sequence
    sequence    the sequence's source listing, one instruction per line
                (older captures call this key `instructions`; both are read)

Each file becomes one row of the output dataset:

    audio            the repetitions aggregated into a single trace, in the
                     units the scope recorded (volts, or raw ADC counts)
    text             the source listing verbatim -- the full instruction text,
                     operands included
    opcodes          the mnemonics of `text`, space separated, which is the
                     usual training target
    project          the file stem, used to keep recordings of the same
                     sequence on the same side of a train/test split
    num_samples      len(audio), so the training side can filter by trace
                     length without reading the traces themselves
    num_repetitions  how many raw traces went into `audio`
    trace_mean       mean of `audio` before any normalization
    trace_std        standard deviation of `audio` before any normalization

This module is deliberately conservative about what it discards. The dataset
it produces is what gets shipped in place of the raw recordings, so anything
dropped here is dropped for good: the raw `.npz` files stay on the recording
machine. Aggregating the repetitions is the one lossy step taken by default
(it is the whole size win, 25x on a stock seqdemo capture), and
`--keep-repetitions` turns even that off. Normalization and the reduction of
the listing to bare mnemonics are training-side choices, so they are off by
default and, where applied, recorded well enough to be undone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
from datasets import Dataset, DatasetInfo

from zapdos_dataset.metadata import (
    adc_rails,
    first_channel,
    read_capture_metadata,
    sample_rate_hz,
)

FORMAT_VERSION = 2

# The keys `seq_export` has used for the source listing. It was renamed from
# `instructions` to `sequence` in scakit-example ("seq instead of instr"), so
# a raw directory can hold either depending on when it was exported.
LISTING_KEYS = ("sequence", "instructions")

AGGREGATORS: Dict[str, Callable] = {
    "mean": np.mean,
    "median": np.median,
}

# Sample dtypes worth storing. float32 is exact for int16 ADC counts and for
# means of them (24 mantissa bits against 16 data bits plus at most ~5 bits of
# averaging gain over 25 repetitions), and halves what has to be transferred
# against float64. float64 is there for captures that were already float64.
DTYPES: Dict[str, Any] = {
    "float32": np.float32,
    "float64": np.float64,
}

# A sample within this fraction of the channel range of an ADC rail is treated
# as clipped -- about one LSB of an 8-bit ADC, the same tolerance
# `seq_clip_check` uses.
CLIP_TOLERANCE = 0.01


def listing_key(keys) -> str:
    """The key holding the source listing in an `.npz`, whichever name it used."""
    available = list(keys)
    for key in LISTING_KEYS:
        if key in available:
            return key
    raise KeyError(
        f"No source listing in the recording: expected one of {list(LISTING_KEYS)}, "
        f"found {sorted(available)}."
    )


def read_recording(path: Path) -> Tuple[np.ndarray, Any]:
    """Return `(traces, listing)` from one `.npz` recording."""
    with np.load(path, allow_pickle=False) as arr:
        if "traces" not in arr:
            raise KeyError(f"No 'traces' array in {path}: found {sorted(arr.files)}.")
        return arr["traces"], arr[listing_key(arr.files)]


def aggregate_traces(
    traces: np.ndarray, aggregator: str = "mean", dtype: str = "float32"
) -> np.ndarray:
    """Collapse the repeated recordings of one sequence into a single trace."""
    try:
        aggregator_fn = AGGREGATORS[aggregator]
    except KeyError:
        raise ValueError(
            f"Unknown aggregator '{aggregator}', expected one of {sorted(AGGREGATORS)}."
        ) from None
    # Accumulate in float64 and narrow once at the end. np.mean over a float32
    # input otherwise sums in float32, which costs ~1e-6 relative error over
    # the repetitions of a single sequence -- the averaging is the whole point
    # of recording them, so it should not be the step that adds noise back.
    # np.median takes no dtype argument, hence the widening cast up front
    # rather than a keyword.
    traces = np.atleast_2d(np.asarray(traces, dtype=np.float64))
    aggregated = aggregator_fn(traces, axis=0)
    return as_dtype(aggregated, dtype)


def as_dtype(trace: np.ndarray, dtype: str = "float32") -> np.ndarray:
    """Cast a trace to one of the storable sample dtypes."""
    try:
        return np.asarray(trace, dtype=DTYPES[dtype])
    except KeyError:
        raise ValueError(
            f"Unknown dtype '{dtype}', expected one of {sorted(DTYPES)}."
        ) from None


def trim_trailing_zeros(trace: np.ndarray) -> np.ndarray:
    """Drop the run of exact zeros at the end of a trace, if there is one.

    Only meaningful for recordings that were zero-padded to a common length.
    A `seq_export` capture is not padded -- every trace is the full scope
    window -- so this is off by default; on a voltage-scaled trace an exact
    zero is a real sample, not padding.
    """
    trace = np.asarray(trace)
    nonzero = np.nonzero(trace)[0]
    if nonzero.size:
        return trace[: nonzero[-1] + 1]
    return trace


def standardize(trace: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Standardize a trace to zero mean and unit variance."""
    trace = np.asarray(trace)
    scale = std if std > 0 else 1.0
    return ((trace - mean) / scale).astype(trace.dtype, copy=False)


def listing_text(listing) -> str:
    """The source listing of a recording as text.

    `seq_export` stores it as a 0-d numpy string array; older captures stored
    a 1-element array. Anything with more than one element would mean the
    listing had been split across elements, and silently keeping only the
    first would drop most of the sequence, so that is an error rather than a
    truncation.
    """
    listing = np.atleast_1d(listing)
    if listing.size != 1:
        raise ValueError(
            f"Expected a single source listing, got {listing.size} elements. "
            "Refusing to guess which one is the sequence."
        )
    value = listing.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode()
    return str(value)


def opcode_sequence(listing) -> str:
    """Reduce a source listing to its space-separated mnemonics.

    Splits on any whitespace, so a tab-separated listing keeps its mnemonic
    rather than yielding `"movs\\tr0,"`. Assembler directives (`.thumb`),
    bare labels (`loop:`) and whole-line comments contribute no instruction
    and are skipped.
    """
    text = listing if isinstance(listing, str) else listing_text(listing)
    opcodes = []
    for line in text.split("\n"):
        line = line.strip()
        for comment in ("//", "@", ";", "#"):
            if line.startswith(comment):
                line = ""
                break
        if not line or line.startswith(".") or line.endswith(":"):
            continue
        opcodes.append(line.split()[0])
    return " ".join(opcodes)


def _sort_key(path: Path):
    """Natural sort, so `trace_2` comes before `trace_10`."""
    stem = path.stem
    digits = "".join(ch for ch in reversed(stem) if ch.isdigit())[::-1]
    prefix = stem[: len(stem) - len(digits)] if digits else stem
    return (str(path.parent), prefix, int(digits) if digits else -1, stem)


def find_recordings(raw_dir: Path) -> List[Path]:
    """List the recordings under `raw_dir`, sorted so the row order is stable."""
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise NotADirectoryError(f"Directory does not exist: {raw_dir}")

    paths = sorted(
        (
            path
            for path in raw_dir.glob("**/*.npz")
            if path.is_file()
            and "metadata" not in path.name
            and "metadata" not in {part.lower() for part in path.relative_to(raw_dir).parts[:-1]}
        ),
        key=_sort_key,
    )
    if not paths:
        raise FileNotFoundError(f"No .npz recordings found under {raw_dir}")
    return paths


def iter_recordings(raw_dir: Path) -> Iterator[Tuple[Path, np.ndarray, Any]]:
    """Yield `(path, traces, listing)` for every recording under `raw_dir`."""
    for path in find_recordings(raw_dir):
        traces, listing = read_recording(path)
        yield path, traces, listing


def count_clipped(traces: np.ndarray, rails: Optional[tuple], range_peak: float) -> int:
    """Number of raw samples sitting at an ADC rail.

    A clipped sample is information the scope never captured, and no amount of
    later processing recovers it -- only a re-record at a wider range does.
    Counting it here is what lets the recording side find out while the target
    is still on the bench.
    """
    if rails is None or not range_peak:
        return 0
    low, high = rails
    tolerance = CLIP_TOLERANCE * abs(range_peak)
    traces = np.asarray(traces)
    return int(np.count_nonzero((traces <= low + tolerance) | (traces >= high - tolerance)))


def prepare_recording(
    path: Path,
    traces: np.ndarray,
    listing,
    aggregator: str = "mean",
    dtype: str = "float32",
    trim: bool = False,
    normalize: bool = False,
    keep_repetitions: bool = False,
) -> Dict:
    """Build the dataset row for a single recording."""
    traces = np.atleast_2d(np.asarray(traces))

    if keep_repetitions:
        audio = as_dtype(traces, dtype)
        reference = as_dtype(aggregate_traces(traces, aggregator, "float64"), "float64")
    else:
        audio = aggregate_traces(traces, aggregator, dtype)
        if trim:
            audio = trim_trailing_zeros(audio)
        reference = audio

    # Recorded before normalization so the shipped trace can always be put back
    # into the units the scope measured, whatever was done to it here.
    mean = float(np.mean(reference, dtype=np.float64))
    std = float(np.std(reference, dtype=np.float64))

    if normalize and not keep_repetitions:
        audio = standardize(audio, mean, std)

    text = listing_text(listing)
    return {
        "audio": audio,
        "text": text,
        "opcodes": opcode_sequence(text),
        "project": path.stem,
        "num_samples": int(audio.shape[-1]),
        "num_repetitions": int(traces.shape[0]),
        "trace_mean": mean,
        "trace_std": std,
    }


def build_dataset(
    raw_dir: Path,
    aggregator: str = "mean",
    sample_rate_hz_override: Optional[float] = None,
    dtype: str = "float32",
    trim: bool = False,
    normalize: bool = False,
    keep_repetitions: bool = False,
    on_warning: Optional[Callable[[str], None]] = None,
) -> Dataset:
    """Build the Hugging Face dataset for every recording under `raw_dir`."""

    # Resolved eagerly: `Dataset.from_generator` swallows whatever the
    # generator raises into a DatasetGenerationError, so a missing or empty
    # directory should be reported before it starts.
    paths = find_recordings(raw_dir)
    if aggregator not in AGGREGATORS:
        raise ValueError(
            f"Unknown aggregator '{aggregator}', expected one of {sorted(AGGREGATORS)}."
        )
    if dtype not in DTYPES:
        raise ValueError(f"Unknown dtype '{dtype}', expected one of {sorted(DTYPES)}.")

    warn = on_warning if on_warning is not None else (lambda message: None)

    capture_metadata = read_capture_metadata(raw_dir)
    rate = sample_rate_hz(capture_metadata)
    if sample_rate_hz_override is not None:
        if rate is not None and rate != sample_rate_hz_override:
            warn(
                f"--sample-rate-hz {sample_rate_hz_override:g} overrides the "
                f"{rate:g} Hz recorded with the capture."
            )
        rate, rate_source = sample_rate_hz_override, "--sample-rate-hz"
    elif rate is not None:
        rate_source = "capture metadata"
    else:
        rate_source = None
        warn(
            "No sample rate found: the capture metadata does not record one and "
            "--sample-rate-hz was not given. The training side needs it to set "
            "its spectral front-end, so the dataset is incomplete without it."
        )

    if not capture_metadata:
        warn(
            f"No capture metadata under {raw_dir}: the scope settings this was "
            "recorded with will not travel with the dataset. Export with a "
            "recent seq_export so metadata/ is written alongside the traces."
        )

    rails = adc_rails(capture_metadata)
    channel_range = first_channel(capture_metadata).get("range")
    if not isinstance(channel_range, (int, float)) or isinstance(channel_range, bool):
        channel_range = None

    clipped = {"samples": 0, "total": 0, "recordings": 0}
    empty_listings: List[str] = []

    def generator():
        for path in paths:
            traces, listing = read_recording(path)
            # seq_export writes an empty string when it cannot resolve a
            # seq_id's source text. Such a row trains the model to emit
            # nothing for a real trace, so it is worth naming rather than
            # shipping quietly.
            if not listing_text(listing).strip():
                empty_listings.append(path.stem)
            if channel_range:
                n_clipped = count_clipped(traces, rails, float(channel_range))
                if n_clipped:
                    clipped["samples"] += n_clipped
                    clipped["recordings"] += 1
                clipped["total"] += int(np.asarray(traces).size)
            yield prepare_recording(
                path,
                traces,
                listing,
                aggregator=aggregator,
                dtype=dtype,
                trim=trim,
                normalize=normalize,
                keep_repetitions=keep_repetitions,
            )

    description = {
        "format_version": FORMAT_VERSION,
        "aggregator": None if keep_repetitions else aggregator,
        "keep_repetitions": keep_repetitions,
        "dtype": dtype,
        "trimmed_trailing_zeros": trim and not keep_repetitions,
        "normalized": normalize and not keep_repetitions,
        "sample_rate_hz": rate,
        "sample_rate_source": rate_source,
        "capture_metadata": capture_metadata,
    }
    info = DatasetInfo(description=json.dumps(description))
    dataset = Dataset.from_generator(generator, info=info)

    if empty_listings:
        shown = ", ".join(empty_listings[:5])
        more = f" and {len(empty_listings) - 5} more" if len(empty_listings) > 5 else ""
        warn(
            f"{len(empty_listings)} recordings have an empty source listing "
            f"({shown}{more}). seq_export writes one when it cannot resolve the "
            "sequence text, so those rows carry a trace with no target -- check "
            "that the capture's sequences_dir is still where it was recorded."
        )
    if clipped["samples"]:
        share = 100.0 * clipped["samples"] / max(clipped["total"], 1)
        warn(
            f"{clipped['samples']} samples ({share:.3f}%) across "
            f"{clipped['recordings']} recordings sit at an ADC rail. Those "
            "samples were clipped at capture time and cannot be recovered from "
            "this dataset -- widen the channel range and re-record if that "
            "share is material."
        )
    return dataset


def build(
    raw_dir: Path,
    out_dir: Path,
    aggregator: str = "mean",
    sample_rate_hz_override: Optional[float] = None,
    dtype: str = "float32",
    trim: bool = False,
    normalize: bool = False,
    keep_repetitions: bool = False,
    on_warning: Optional[Callable[[str], None]] = None,
) -> Dataset:
    """Build the dataset and save it to `out_dir`."""
    dataset = build_dataset(
        raw_dir,
        aggregator=aggregator,
        sample_rate_hz_override=sample_rate_hz_override,
        dtype=dtype,
        trim=trim,
        normalize=normalize,
        keep_repetitions=keep_repetitions,
        on_warning=on_warning,
    )
    dataset.save_to_disk(Path(out_dir))
    return dataset


def dataset_info(dataset: Dataset) -> Dict[str, Any]:
    """The build settings recorded with a dataset, as a mapping.

    `save_to_disk` round-trips `DatasetInfo.description` and little else, so
    that is where the aggregator, the sample rate and the scope configuration
    are kept.
    """
    try:
        return json.loads(dataset.info.description)
    except (TypeError, ValueError):
        return {}


def load(path: Path) -> Dataset:
    """Load a dataset built by this package, in the dtype it was stored in.

    `datasets.load_from_disk(...).with_format("numpy")` formats float columns
    as float32 no matter what the stored feature says, so a float64 dataset
    read that way is silently narrowed. Reading the stored dtype back out of
    the dataset's own info and passing it to the formatter is what stops the
    trace being rounded on the way in.
    """
    from datasets import load_from_disk

    dataset = load_from_disk(str(path))
    dtype = DTYPES.get(dataset_info(dataset).get("dtype", "float32"), np.float32)
    return dataset.with_format("numpy", dtype=dtype)
