from __future__ import annotations

from dataclasses import dataclass

import sounddevice as sd
from rich.console import Console
from rich.table import Table


@dataclass
class AudioDevice:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    is_default_input: bool = False


def list_input_devices() -> list[AudioDevice]:
    """Return all devices that have at least one input channel."""
    devices = sd.query_devices()
    default_idx = sd.default.device[0]
    result = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            result.append(
                AudioDevice(
                    index=i,
                    name=d["name"],
                    max_input_channels=d["max_input_channels"],
                    max_output_channels=d["max_output_channels"],
                    default_samplerate=d["default_samplerate"],
                    is_default_input=(i == default_idx),
                )
            )
    return result


def find_blackhole() -> AudioDevice | None:
    """Return the first BlackHole input device found, or None."""
    for d in list_input_devices():
        if "blackhole" in d.name.lower():
            return d
    return None


def find_device_by_name(name: str) -> AudioDevice | None:
    """Case-insensitive partial name match on input devices."""
    name_lower = name.lower()
    for d in list_input_devices():
        if name_lower in d.name.lower():
            return d
    return None


def get_default_input() -> AudioDevice:
    """Return the system default input device."""
    devices = sd.query_devices()
    default_idx = sd.default.device[0]
    d = devices[default_idx]
    return AudioDevice(
        index=default_idx,
        name=d["name"],
        max_input_channels=d["max_input_channels"],
        max_output_channels=d["max_output_channels"],
        default_samplerate=d["default_samplerate"],
        is_default_input=True,
    )


def print_devices(console: Console) -> None:
    """Print numbered list of input devices."""
    devices = list_input_devices()
    table = Table(title="Input Devices", show_header=True)
    table.add_column("#", style="cyan", width=4)
    table.add_column("Name")
    table.add_column("Ch", width=4)
    table.add_column("Rate", width=8)
    for i, d in enumerate(devices, 1):
        marker = " [green]*[/green]" if d.is_default_input else ""
        table.add_row(
            str(i),
            d.name + marker,
            str(d.max_input_channels),
            str(int(d.default_samplerate)),
        )
    console.print(table)
