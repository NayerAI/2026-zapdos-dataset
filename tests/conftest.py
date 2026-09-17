import json

import numpy as np
import pytest


@pytest.fixture
def write_recording():
    """Write a `seq_export`-shaped .npz; `key` picks which listing key it uses."""

    def _write(directory, name, traces, listing, key="sequence"):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.npz"
        np.savez(path, traces=np.asarray(traces), **{key: np.array(listing)})
        return path

    return _write


@pytest.fixture
def write_capture_metadata():
    """Write a metadata/ sidecar the way `seq_export._export_zarr_metadata` does."""

    def _write(directory, samplerate=500_000_000, range_peak=1.0, offset=1.0e-3,
               data_scaling="voltage"):
        metadata_dir = directory / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        document = {
            "zarr_format": 3,
            "node_type": "array",
            "attributes": {
                "scope": {
                    "scopetype": "pico6",
                    "timebase": {"samplerate": samplerate},
                    "channels": [
                        {
                            "channel": "A",
                            "range": range_peak,
                            "offset": offset,
                            "coupling": "DC",
                            "data_scaling": data_scaling,
                        }
                    ],
                }
            },
        }
        (metadata_dir / "traces.json").write_text(json.dumps(document))
        (metadata_dir / "capture.json").write_text(
            json.dumps({"zarr_format": 3, "node_type": "group",
                        "attributes": {"traces_per_seq": 25, "num_sequences": 2}})
        )
        return metadata_dir

    return _write
