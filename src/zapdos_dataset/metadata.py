"""Read the capture metadata that `seq_export` writes next to the recordings.

`scakit_seqdemo.seq_export` copies every `zarr.json` of the capture into
`<out_dir>/metadata/<group-or-array-name>.json`. Those files carry the scope
configuration the capture was taken with -- sample rate, channel range and
offset, `data_scaling`, `traces_per_seq`. None of it is derivable from the
traces themselves, and once the raw directory stays behind on the recording
machine it cannot be recovered at all, so it is read here and shipped with the
dataset rather than left to be remembered and re-typed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# zarr v3 nests a node's user attributes under "attributes"; zarr v2 wrote them
# as the whole of a separate .zattrs file. Accept either shape.
_ATTRS_KEY = "attributes"


def _attributes(document: Any) -> Dict[str, Any]:
    if not isinstance(document, dict):
        return {}
    attrs = document.get(_ATTRS_KEY)
    if isinstance(attrs, dict):
        return attrs
    # A zarr v2 .zattrs payload is the attribute mapping itself. Everything
    # else in a v3 zarr.json is structural (zarr_format, node_type, shape,
    # codecs...) and never collides with a capture attribute, so passing the
    # document through is safe.
    return document


def find_metadata_files(raw_dir: Path) -> List[Path]:
    """The capture-metadata JSON files under `raw_dir`, sorted."""
    raw_dir = Path(raw_dir)
    return sorted(
        path
        for path in raw_dir.glob("**/*.json")
        if path.is_file() and "metadata" in {part.lower() for part in path.parts}
    )


def read_capture_metadata(raw_dir: Path) -> Dict[str, Any]:
    """Merge the capture's metadata files into one mapping, keyed by file stem.

    Unreadable or malformed files are skipped rather than raised on: the
    metadata is a bonus on top of the traces, and a capture that predates
    `_export_zarr_metadata` has none at all.
    """
    metadata: Dict[str, Any] = {}
    for path in find_metadata_files(raw_dir):
        try:
            document = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        attrs = _attributes(document)
        if attrs:
            metadata[path.stem] = attrs
    return metadata


def _find_key(node: Any, key: str) -> Optional[Any]:
    """Depth-first search for `key` anywhere in a nested mapping/sequence."""
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for value in node.values():
            found = _find_key(value, key)
            if found is not None:
                return found
    elif isinstance(node, (list, tuple)):
        for value in node:
            found = _find_key(value, key)
            if found is not None:
                return found
    return None


def scope_config(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """The `scope` block of the capture metadata, or `{}` if it is absent."""
    scope = _find_key(metadata, "scope")
    return scope if isinstance(scope, dict) else {}


def sample_rate_hz(metadata: Dict[str, Any]) -> Optional[float]:
    """The scope sample rate recorded with the capture, in Hz.

    The training side sets the frequency bins of its spectral front-end from
    this, so a dataset that arrives without it cannot be trained on until
    somebody remembers what the scope was set to.
    """
    for key in ("samplerate", "sample_rate", "sampling_rate", "sample_rate_hz"):
        value = _find_key(metadata, key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def first_channel(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """The first configured scope channel, or `{}` if there is none."""
    channels = scope_config(metadata).get("channels")
    if isinstance(channels, list) and channels and isinstance(channels[0], dict):
        return channels[0]
    return {}


def adc_rails(metadata: Dict[str, Any]) -> Optional[tuple]:
    """`(low, high)` volts the ADC clamps at, for a voltage-scaled capture.

    Mirrors `scakit_seqdemo.seq_clip_check`: the rails sit at
    `-range - offset` and `range - offset`. Returns None for a capture whose
    samples are raw ADC counts, because the stored metadata does not carry the
    ADC max count needed to place the rails in count space.
    """
    channel = first_channel(metadata)
    if channel.get("data_scaling", "voltage") != "voltage":
        return None
    range_peak = channel.get("range")
    offset = channel.get("offset")
    if not isinstance(range_peak, (int, float)) or not isinstance(offset, (int, float)):
        return None
    return (-float(range_peak) - float(offset), float(range_peak) - float(offset))
