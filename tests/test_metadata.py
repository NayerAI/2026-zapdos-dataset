import json

from zapdos_dataset import (
    adc_rails,
    find_metadata_files,
    first_channel,
    read_capture_metadata,
    sample_rate_hz,
    scope_config,
)


def test_reads_the_sample_rate_the_capture_was_taken_with(tmp_path, write_capture_metadata):
    write_capture_metadata(tmp_path, samplerate=2.5e9)
    assert sample_rate_hz(read_capture_metadata(tmp_path)) == 2.5e9


def test_keeps_the_whole_scope_block(tmp_path, write_capture_metadata):
    write_capture_metadata(tmp_path)
    scope = scope_config(read_capture_metadata(tmp_path))
    assert scope["scopetype"] == "pico6"
    assert first_channel(read_capture_metadata(tmp_path))["channel"] == "A"


def test_keeps_the_root_group_attributes_too(tmp_path, write_capture_metadata):
    write_capture_metadata(tmp_path)
    assert read_capture_metadata(tmp_path)["capture"]["traces_per_seq"] == 25


def test_rails_follow_range_and_offset(tmp_path, write_capture_metadata):
    write_capture_metadata(tmp_path, range_peak=0.05, offset=-6.13e-3)
    low, high = adc_rails(read_capture_metadata(tmp_path))
    assert low == -0.05 + 6.13e-3
    assert high == 0.05 + 6.13e-3


def test_no_rails_for_a_raw_scaled_capture(tmp_path, write_capture_metadata):
    write_capture_metadata(tmp_path, data_scaling="raw")
    assert adc_rails(read_capture_metadata(tmp_path)) is None


def test_a_capture_without_metadata_is_not_an_error(tmp_path):
    assert read_capture_metadata(tmp_path) == {}
    assert sample_rate_hz({}) is None
    assert adc_rails({}) is None


def test_malformed_metadata_is_skipped(tmp_path, write_capture_metadata):
    write_capture_metadata(tmp_path)
    (tmp_path / "metadata" / "broken.json").write_text("{not json")
    assert sample_rate_hz(read_capture_metadata(tmp_path)) == 500_000_000


def test_zarr_v2_style_attributes_are_accepted(tmp_path):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    (metadata_dir / "traces.json").write_text(
        json.dumps({"scope": {"timebase": {"samplerate": 1e8}}})
    )
    assert sample_rate_hz(read_capture_metadata(tmp_path)) == 1e8


def test_only_metadata_directories_are_read(tmp_path, write_capture_metadata):
    write_capture_metadata(tmp_path)
    (tmp_path / "notes.json").write_text(json.dumps({"scope": {"timebase": {"samplerate": 1}}}))
    assert len(find_metadata_files(tmp_path)) == 2
