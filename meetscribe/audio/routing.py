from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from meetscribe.audio.devices import find_blackhole
from meetscribe.config import Config

_BLACKHOLE_DRIVER_PATHS = [
    Path("/Library/Audio/Plug-Ins/HAL/BlackHole2ch.driver"),
    Path("/Library/Audio/Plug-Ins/HAL/BlackHole16ch.driver"),
    Path("/Library/Audio/Plug-Ins/HAL/BlackHole64ch.driver"),
]


def is_blackhole_installed() -> bool:
    """True if any BlackHole driver exists on disk or appears as a sounddevice device."""
    if any(p.exists() for p in _BLACKHOLE_DRIVER_PATHS):
        return True
    return find_blackhole() is not None


def get_install_instructions() -> str:
    return (
        "Install BlackHole (virtual audio loopback driver):\n\n"
        "  Option A — Homebrew (recommended):\n"
        "    brew install blackhole-2ch\n\n"
        "  Option B — Direct download:\n"
        "    https://existingcircuits.com/products/blackhole\n\n"
        "After installing, restart your computer or open Audio MIDI Setup once."
    )


def explain_routing_setup(console: Console) -> None:
    """Print step-by-step guide for setting up BlackHole + Multi-Output Device."""
    console.print(Panel(
        "[bold]Why BlackHole is needed[/bold]\n\n"
        "macOS routes audio to your output device (headphones or speakers) at the hardware level. "
        "There is no built-in way to intercept that signal in software.\n\n"
        "BlackHole creates a virtual audio device that acts as a software loopback: "
        "Teams sends audio to it in parallel with your real output, "
        "so meetscribe can capture remote participants [bold]regardless[/bold] of whether "
        "you are using headphones or speakers.",
        title="Audio Routing",
    ))

    console.print("\n[bold]Setup steps (one-time, ~2 minutes):[/bold]\n")
    steps = [
        "Install BlackHole 2ch  →  [cyan]brew install blackhole-2ch[/cyan]",
        "Open [bold]Audio MIDI Setup[/bold] (search with Spotlight)",
        "Click [bold]+[/bold] at the bottom left  →  [bold]Create Multi-Output Device[/bold]",
        "Check both [bold]BlackHole 2ch[/bold] AND your real output (headphones/speakers)",
        "Right-click the new device  →  [bold]Use This Device For Sound Output[/bold]",
        "Open Teams  →  Settings  →  Devices  →  set Speaker to the same Multi-Output Device",
        "meetscribe will now capture remote audio through the BlackHole input device",
    ]
    for i, step in enumerate(steps, 1):
        console.print(f"  [cyan]{i}.[/cyan] {step}")
    console.print()


def verify_audio_routing(config: Config, console: Console) -> bool:
    """Verify mic and system audio devices are accessible. Never raises."""
    ok = True

    mic_name = config.audio.mic_device or "system default"
    console.print(f"  Microphone:    [cyan]{mic_name}[/cyan]")

    bh = find_blackhole() if not config.audio.system_device else None
    system_name = config.audio.system_device or (bh.name if bh else None)

    if system_name:
        console.print(f"  System audio:  [cyan]{system_name}[/cyan]")
    else:
        console.print(
            "  System audio:  [yellow]BlackHole not found — remote participants will not be captured[/yellow]"
        )
        ok = False

    return ok
