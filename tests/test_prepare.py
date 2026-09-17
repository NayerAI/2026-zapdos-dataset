import json

import numpy as np
import pytest
from datasets import load_from_disk

from zapdos_dataset import (
    aggregate_traces,
    build,
    build_dataset,
    count_clipped,
    dataset_info,
    find_recordings,
    listing_key,
    listing_text,
    load,
    opcode_sequence,
    prepare_recording,
    read_recording,
    standardize,
    trim_trailing_zeros,
)


# --- reading what seq_export actually writes --------------------------------


def test_reads_the_sequence_key_current_captures_use(tmp_path, write_recording):
    path = write_recording(tmp_path, "trace_1", [[1.0, 2.0]], "nop", key="sequence")
    traces, listing = read_recording(path)
    assert listing_text(listing) == "nop"
    assert traces.shape == (1, 2)


def test_reads_the_instructions_key_older_captures_use(tmp_path, write_recording):
    path = write_recording(tmp_path, "trace_1", [[1.0, 2.0]], "nop", key="instructions")
    assert listing_text(read_recording(path)[1]) == "nop"


def test_a_recording_with_neither_key_names_what_it_found(tmp_path):
    path = tmp_path / "trace_1.npz"
    np.savez(path, traces=np.zeros((1, 2)), asm=np.array("nop"))
    with pytest.raises(KeyError, match="asm"):
        read_recording(path)


def test_listing_key_prefers_the_current_name(tmp_path):
    assert listing_key(["traces", "sequence", "instructions"]) == "sequence"


def test_a_recording_without_traces_is_an_error(tmp_path):
    path = tmp_path / "trace_1.npz"
    np.savez(path, sequence=np.array("nop"))
    with pytest.raises(KeyError, match="traces"):
        read_recording(path)


# --- aggregation ------------------------------------------------------------


def test_aggregate_traces_defaults_to_a_float32_mean():
    aggregated = aggregate_traces(np.array([[0, 2], [2, 4]], dtype=np.int16))
    assert aggregated.dtype == np.float32
    assert aggregated.tolist() == [1.0, 3.0]


def test_aggregate_traces_median():
    traces = np.array([[1.0, 1.0], [2.0, 2.0], [9.0, 9.0]])
    assert aggregate_traces(traces, "median").tolist() == [2.0, 2.0]


def test_aggregate_traces_rejects_unknown_aggregator():
    with pytest.raises(ValueError):
        aggregate_traces(np.zeros((2, 2)), "geometric")


def test_aggregate_traces_rejects_unknown_dtype():
    with pytest.raises(ValueError):
        aggregate_traces(np.zeros((2, 2)), dtype="float16")


def test_aggregation_accumulates_in_float64():
    """A float32 accumulator loses ~1e-6 over many repetitions; float64 does not."""
    rng = np.random.default_rng(0)
    traces = (rng.standard_normal((5000, 4)).astype(np.float32) * 1e-3 + 1.0).astype(np.float32)
    exact = np.mean(traces.astype(np.float64), axis=0)
    naive = np.mean(traces, axis=0).astype(np.float64)

    ours = aggregate_traces(traces, dtype="float64").astype(np.float64)

    assert np.array_equal(ours, exact)
    assert np.max(np.abs(naive - exact)) > np.max(np.abs(ours - exact))


def test_float64_captures_can_be_kept_at_full_width():
    traces = np.array([[1.0 + 2.0**-40, 2.0]], dtype=np.float64)
    assert aggregate_traces(traces, dtype="float64")[0] == traces[0, 0]
    assert np.float64(aggregate_traces(traces, dtype="float32")[0]) != traces[0, 0]


def test_int16_counts_and_their_means_are_exact_in_float32():
    rng = np.random.default_rng(1)
    traces = rng.integers(-32768, 32768, size=(25, 512), dtype=np.int16)
    exact = np.mean(traces.astype(np.float64), axis=0)
    stored = aggregate_traces(traces, dtype="float32").astype(np.float64)
    assert np.max(np.abs(stored - exact)) < 0.01


# --- trace transforms -------------------------------------------------------


def test_trim_trailing_zeros():
    assert trim_trailing_zeros(np.array([1.0, 2.0, 0.0, 0.0])).tolist() == [1.0, 2.0]


def test_trim_keeps_interior_zeros():
    assert len(trim_trailing_zeros(np.array([1.0, 0.0, 3.0]))) == 3


def test_trim_survives_an_all_zero_trace():
    assert trim_trailing_zeros(np.zeros(4)).tolist() == [0.0, 0.0, 0.0, 0.0]


def test_standardize_is_invertible_from_the_stored_constants():
    trace = np.array([1.0, 2.0, 3.0, 8.0], dtype=np.float32)
    mean, std = float(trace.mean()), float(trace.std())
    restored = standardize(trace, mean, std) * std + mean
    assert restored == pytest.approx(trace, abs=1e-5)


def test_standardize_survives_a_constant_trace():
    assert standardize(np.array([2.0, 2.0]), 2.0, 0.0).tolist() == [0.0, 0.0]


# --- the listing ------------------------------------------------------------


def test_opcode_sequence_keeps_the_mnemonic_of_every_line():
    assert opcode_sequence("ldr r0, [r1]\nadd r0, r0, #1\nstr r0, [r1]") == "ldr add str"


def test_opcode_sequence_ignores_blank_lines():
    assert opcode_sequence("ldr r0, [r1]\n\nadd r0, #1\n") == "ldr add"


def test_opcode_sequence_splits_on_tabs_too():
    assert opcode_sequence("movs\tr0, #1\nadds\tr1, r2") == "movs adds"


def test_opcode_sequence_skips_directives_labels_and_comments():
    listing = ".syntax unified\nloop:\n// a comment\n@ another\nmovs r0, #1\nbne loop"
    assert opcode_sequence(listing) == "movs bne"


def test_opcode_sequence_accepts_the_zero_d_array_seq_export_writes():
    assert opcode_sequence(np.array("nop\nnop")) == "nop nop"


def test_a_multi_element_listing_is_refused_rather_than_truncated():
    with pytest.raises(ValueError, match="single source listing"):
        listing_text(np.array(["movs r0, #1", "adds r1, r2"]))


def test_listing_text_decodes_bytes():
    assert listing_text(np.array([b"nop"])) == "nop"


# --- the row ----------------------------------------------------------------


def test_prepare_recording_keeps_the_full_listing_and_the_opcodes(tmp_path, write_recording):
    path = write_recording(tmp_path, "trace_7", [[1.0, 2.0, 3.0]], "ldr r0, [r1]\nadd r2, #4")
    traces, listing = read_recording(path)

    row = prepare_recording(path, traces, listing)

    assert row["project"] == "trace_7"
    assert row["text"] == "ldr r0, [r1]\nadd r2, #4"
    assert row["opcodes"] == "ldr add"
    assert row["num_samples"] == 3
    assert row["num_repetitions"] == 1


def test_prepare_recording_stores_the_units_it_was_recorded_in(tmp_path, write_recording):
    traces = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
    path = write_recording(tmp_path, "trace_1", traces, "nop")

    row = prepare_recording(path, traces, np.array("nop"))

    assert np.asarray(row["audio"]).tolist() == [10.0, 20.0, 30.0]
    assert row["trace_mean"] == pytest.approx(20.0)
    assert row["trace_std"] == pytest.approx(np.std([10.0, 20.0, 30.0]))


def test_normalize_is_opt_in_and_stays_invertible(tmp_path, write_recording):
    traces = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
    path = write_recording(tmp_path, "trace_1", traces, "nop")

    row = prepare_recording(path, traces, np.array("nop"), normalize=True)
    audio = np.asarray(row["audio"])

    assert audio.mean() == pytest.approx(0.0, abs=1e-6)
    restored = audio * row["trace_std"] + row["trace_mean"]
    assert restored == pytest.approx([10.0, 20.0, 30.0], abs=1e-4)


def test_trailing_zeros_are_kept_unless_asked_for(tmp_path, write_recording):
    traces = np.array([[1.0, 2.0, 0.0, 0.0]])
    path = write_recording(tmp_path, "trace_1", traces, "nop")

    assert prepare_recording(path, traces, np.array("nop"))["num_samples"] == 4
    assert prepare_recording(path, traces, np.array("nop"), trim=True)["num_samples"] == 2


def test_keep_repetitions_stores_every_trace(tmp_path, write_recording):
    traces = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    path = write_recording(tmp_path, "trace_1", traces, "nop")

    row = prepare_recording(path, traces, np.array("nop"), keep_repetitions=True)

    assert np.asarray(row["audio"]).tolist() == traces.tolist()
    assert row["num_repetitions"] == 3
    assert row["num_samples"] == 2


def test_a_one_dimensional_recording_is_treated_as_one_repetition(tmp_path, write_recording):
    traces = np.array([1.0, 2.0, 3.0])
    path = write_recording(tmp_path, "trace_1", traces, "nop")
    row = prepare_recording(path, traces, np.array("nop"))
    assert row["num_repetitions"] == 1
    assert row["num_samples"] == 3


# --- discovery --------------------------------------------------------------


def test_recordings_are_found_in_subdirectories(tmp_path, write_recording):
    write_recording(tmp_path / "day-2", "trace_1", [[1.0]], "nop")
    assert [p.stem for p in find_recordings(tmp_path)] == ["trace_1"]


def test_recordings_sort_naturally(tmp_path, write_recording):
    for n in (1, 2, 10, 100):
        write_recording(tmp_path, f"trace_{n}", [[1.0]], "nop")
    assert [p.stem for p in find_recordings(tmp_path)] == [
        "trace_1", "trace_2", "trace_10", "trace_100"
    ]


def test_the_metadata_directory_is_not_mistaken_for_a_recording(
    tmp_path, write_recording, write_capture_metadata
):
    write_capture_metadata(tmp_path)
    write_recording(tmp_path, "trace_1", [[1.0]], "nop")
    write_recording(tmp_path / "metadata", "trace_2", [[1.0]], "nop")
    assert [p.stem for p in find_recordings(tmp_path)] == ["trace_1"]


def test_an_empty_directory_is_reported_before_the_build_starts(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_dataset(tmp_path)


def test_a_missing_directory_is_reported_before_the_build_starts(tmp_path):
    with pytest.raises(NotADirectoryError):
        build_dataset(tmp_path / "nope")


# --- clipping ---------------------------------------------------------------


def test_count_clipped_finds_samples_at_the_rails():
    traces = np.array([[0.0, 0.5, 1.0, -1.0]])
    assert count_clipped(traces, (-1.0, 1.0), 1.0) == 2


def test_count_clipped_is_a_no_op_without_rails():
    assert count_clipped(np.zeros((2, 2)), None, 1.0) == 0


# --- the whole build --------------------------------------------------------


def test_build_dataset_covers_every_recording(tmp_path, write_recording):
    write_recording(tmp_path, "trace_1", [[1.0, 2.0], [3.0, 4.0]], "ldr r0, [r1]")
    write_recording(tmp_path, "trace_2", [[5.0, 6.0]], "add r1, #1\nstr r2, [r3]")

    dataset = build_dataset(tmp_path)

    assert dataset["project"] == ["trace_1", "trace_2"]
    assert dataset["opcodes"] == ["ldr", "add str"]
    assert dataset["text"][1] == "add r1, #1\nstr r2, [r3]"
    assert dataset["num_repetitions"] == [2, 1]


def test_the_sample_rate_comes_from_the_capture_metadata(
    tmp_path, write_recording, write_capture_metadata
):
    write_capture_metadata(tmp_path, samplerate=2.5e9)
    write_recording(tmp_path, "trace_1", [[1.0, 2.0]], "nop")

    info = json.loads(build_dataset(tmp_path).info.description)

    assert info["sample_rate_hz"] == 2.5e9
    assert info["sample_rate_source"] == "capture metadata"
    assert info["capture_metadata"]["traces"]["scope"]["scopetype"] == "pico6"


def test_an_explicit_sample_rate_overrides_the_metadata(
    tmp_path, write_recording, write_capture_metadata
):
    write_capture_metadata(tmp_path, samplerate=2.5e9)
    write_recording(tmp_path, "trace_1", [[1.0, 2.0]], "nop")
    warnings = []

    info = json.loads(
        build_dataset(tmp_path, sample_rate_hz_override=1e9, on_warning=warnings.append)
        .info.description
    )

    assert info["sample_rate_hz"] == 1e9
    assert info["sample_rate_source"] == "--sample-rate-hz"
    assert any("overrides" in w for w in warnings)


def test_a_missing_sample_rate_is_warned_about(tmp_path, write_recording):
    write_recording(tmp_path, "trace_1", [[1.0, 2.0]], "nop")
    warnings = []

    info = json.loads(build_dataset(tmp_path, on_warning=warnings.append).info.description)

    assert info["sample_rate_hz"] is None
    assert any("No sample rate" in w for w in warnings)
    assert any("No capture metadata" in w for w in warnings)


def test_clipped_samples_are_warned_about(tmp_path, write_recording, write_capture_metadata):
    write_capture_metadata(tmp_path, range_peak=1.0, offset=0.0)
    write_recording(tmp_path, "trace_1", [[0.0, 0.1, 1.0]], "nop")
    warnings = []

    build_dataset(tmp_path, on_warning=warnings.append)

    assert any("ADC rail" in w for w in warnings)


def test_a_clean_capture_raises_no_clipping_warning(
    tmp_path, write_recording, write_capture_metadata
):
    write_capture_metadata(tmp_path, range_peak=1.0, offset=0.0)
    write_recording(tmp_path, "trace_1", [[0.0, 0.1, 0.2]], "nop")
    warnings = []

    build_dataset(tmp_path, on_warning=warnings.append)

    assert not any("ADC rail" in w for w in warnings)


def test_samples_survive_the_round_trip_to_disk_as_float32(
    tmp_path, write_recording, write_capture_metadata
):
    raw = tmp_path / "raw"
    write_capture_metadata(raw, samplerate=5e8)
    traces = np.array([[1.25, -3.5e-7, 1e-30], [1.25, -3.5e-7, 1e-30]], dtype=np.float32)
    write_recording(raw, "trace_1", traces, "ldr r0, [r1]\nnop")

    build(raw, tmp_path / "out")
    dataset = load_from_disk(str(tmp_path / "out"))

    assert dataset.features["audio"].feature.dtype == "float32"
    audio = np.asarray(dataset.with_format("numpy")[0]["audio"])
    assert audio.dtype == np.float32
    assert audio.tolist() == traces[0].tolist()
    assert dataset[0]["text"] == "ldr r0, [r1]\nnop"
    assert json.loads(dataset.info.description)["sample_rate_hz"] == 5e8


def test_float64_survives_the_round_trip_when_asked_for(tmp_path, write_recording):
    raw = tmp_path / "raw"
    traces = np.array([[1.0 + 2.0**-40, 2.0]], dtype=np.float64)
    write_recording(raw, "trace_1", traces, "nop")

    build(raw, tmp_path / "out", dtype="float64")

    assert load_from_disk(str(tmp_path / "out")).features["audio"].feature.dtype == "float64"
    assert np.float64(load(tmp_path / "out")[0]["audio"][0]) == traces[0, 0]


def test_load_does_not_let_the_numpy_formatter_narrow_the_trace(tmp_path, write_recording):
    """datasets' numpy formatter is float32 by default, whatever the feature says."""
    raw = tmp_path / "raw"
    value = 1.0 + 2.0**-40
    write_recording(raw, "trace_1", np.array([[value, 2.0]], dtype=np.float64), "nop")
    build(raw, tmp_path / "out", dtype="float64")

    narrowed = load_from_disk(str(tmp_path / "out")).with_format("numpy")[0]["audio"]
    kept = load(tmp_path / "out")[0]["audio"]

    # np.float64(...) on both sides: comparing a float32 scalar against a
    # Python float casts the float down first (NEP 50), which hides the loss.
    assert narrowed.dtype == np.float32
    assert np.float64(narrowed[0]) != np.float64(value)
    assert kept.dtype == np.float64
    assert np.float64(kept[0]) == np.float64(value)


def test_load_reads_a_float32_dataset_as_float32(tmp_path, write_recording):
    raw = tmp_path / "raw"
    write_recording(raw, "trace_1", np.array([[1.0, 2.0]], dtype=np.float32), "nop")
    build(raw, tmp_path / "out")
    assert load(tmp_path / "out")[0]["audio"].dtype == np.float32


def test_dataset_info_round_trips_the_build_settings(tmp_path, write_recording):
    raw = tmp_path / "raw"
    write_recording(raw, "trace_1", [[1.0, 2.0]], "nop")
    build(raw, tmp_path / "out", aggregator="median")
    assert dataset_info(load(tmp_path / "out"))["aggregator"] == "median"


def test_an_unresolved_sequence_is_warned_about(tmp_path, write_recording):
    """seq_export writes an empty listing when it cannot resolve the seq_id."""
    write_recording(tmp_path, "trace_1", [[1.0, 2.0]], "nop")
    write_recording(tmp_path, "trace_2", [[1.0, 2.0]], "")
    warnings = []

    build_dataset(tmp_path, on_warning=warnings.append)

    assert any("trace_2" in w and "empty source listing" in w for w in warnings)


def test_no_empty_listing_warning_when_every_sequence_resolved(tmp_path, write_recording):
    write_recording(tmp_path, "trace_1", [[1.0, 2.0]], "nop")
    warnings = []
    build_dataset(tmp_path, on_warning=warnings.append)
    assert not any("empty source listing" in w for w in warnings)
