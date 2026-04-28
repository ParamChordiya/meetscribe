"""Patch heavy optional dependencies before any meetscribe imports so tests
run without torch, sounddevice, faster-whisper, or resemblyzer installed."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


def _make_mock_module(name: str) -> MagicMock:
    mod = MagicMock()
    mod.__name__ = name
    mod.__spec__ = types.ModuleType(name)
    return mod


# Patch heavy deps before any project module is imported.
for _name in [
    "faster_whisper",
    "resemblyzer",
    "sounddevice",
    "torch",
    "webrtcvad",
    "sklearn",
    "sklearn.cluster",
]:
    if _name not in sys.modules:
        sys.modules[_name] = _make_mock_module(_name)

# sounddevice needs specific attributes used in devices.py
import sounddevice as _sd  # noqa: E402  (already mocked above)

_sd.query_devices.return_value = [
    {
        "name": "Built-in Microphone",
        "max_input_channels": 1,
        "max_output_channels": 0,
        "default_samplerate": 44100.0,
    },
    {
        "name": "BlackHole 2ch",
        "max_input_channels": 2,
        "max_output_channels": 2,
        "default_samplerate": 44100.0,
    },
    {
        "name": "Built-in Output",
        "max_input_channels": 0,
        "max_output_channels": 2,
        "default_samplerate": 44100.0,
    },
]
_sd.default.device = [0, 2]  # (input_index, output_index)
