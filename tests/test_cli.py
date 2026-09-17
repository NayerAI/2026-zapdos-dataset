import json

import numpy as np
from datasets import load_from_disk

from zapdos_dataset.cli import main


def test_cli_builds_a_dataset(tmp_path, capsys, write_recording, write_capture_metadata):
    raw = tmp_path / "raw"
    write_capture_metadata(raw, samplerate=1e9)
    write_recording(raw, "trace_1", [[1.0, 2.0], [3.0, 4.0]], "movs r0, #1")
    out = tmp_path / "out"

    assert main([str(raw), str(out)]) == 0

    dataset = load_from_disk(str(out))
    assert dataset["opcodes"] == ["movs"]
    assert np.asarray(dataset.with_format("numpy")[0]["audio"]).tolist() == [2.0, 3.0]
    assert json.loads(dataset.info.description)["sample_rate_hz"] == 1e9
    assert "1 recordings" in capsys.readouterr().out


def test_cli_warns_on_stderr_without_failing(tmp_path, capsys, write_recording):
    raw = tmp_path / "raw"
    write_recording(raw, "trace_1", [[1.0, 2.0]], "nop")

    assert main([str(raw), str(tmp_path / "out")]) == 0
    assert "No sample rate" in capsys.readouterr().err


def test_cli_passes_through_the_lossless_options(tmp_path, write_recording):
    raw = tmp_path / "raw"
    write_recording(raw, "trace_1", [[1.0, 2.0], [3.0, 4.0]], "nop")
    out = tmp_path / "out"

    main([str(raw), str(out), "--keep-repetitions", "--dtype", "float64"])

    dataset = load_from_disk(str(out))
    info = json.loads(dataset.info.description)
    assert info["keep_repetitions"] is True
    assert info["aggregator"] is None
    assert np.asarray(dataset.with_format("numpy")[0]["audio"]).tolist() == [[1.0, 2.0], [3.0, 4.0]]
