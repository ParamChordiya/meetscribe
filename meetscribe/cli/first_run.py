from __future__ import annotations

import subprocess
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from meetscribe.audio.devices import find_blackhole, list_input_devices, print_devices
from meetscribe.audio.routing import (
    explain_routing_setup,
    get_install_instructions,
    is_blackhole_installed,
)
from meetscribe.cli.permissions import ensure_permissions
from meetscribe.config import Config
from meetscribe.llm.client import OllamaClient

_WHISPER_MODELS: list[tuple[str, str]] = [
    ("tiny",     "~40 MB  — fastest, lower accuracy"),
    ("base",     "~75 MB  — good balance (recommended)"),
    ("small",    "~245 MB — more accurate, slower"),
    ("medium",   "~770 MB — high accuracy, slow"),
    ("large-v3", "~1.5 GB — best quality, requires strong CPU"),
]


def run_first_run_wizard(console: Console) -> Config:
    """Full interactive first-run setup wizard. Returns a saved Config."""
    console.print(Panel(
        "[bold]Welcome to meetscribe[/bold]\n\n"
        "This one-time wizard sets up:\n"
        "  • macOS permissions (microphone, accessibility)\n"
        "  • Audio routing for capturing both sides of a call\n"
        "  • Your preferred microphone\n"
        "  • Where to save notes and transcripts\n"
        "  • Ollama language model\n"
        "  • Transcription model (Whisper)\n\n"
        "Settings are saved to [cyan]~/.config/meetscribe/config.yaml[/cyan].",
        style="green",
    ))

    config = Config()

    # ── Step 1: Permissions ───────────────────────────────────────────────────
    console.print("\n[bold]Step 1 of 6 — Permissions[/bold]")
    ensure_permissions(console)

    # ── Step 2: BlackHole audio routing ───────────────────────────────────────
    console.print("\n[bold]Step 2 of 6 — Audio Routing[/bold]")
    console.print(
        "To capture [bold]both sides[/bold] of a call (your mic [italic]and[/italic] "
        "remote participants), meetscribe uses [bold]BlackHole[/bold] — "
        "a free virtual audio driver that intercepts system audio at the software level.\n"
        "This works with headphones and speakers alike.\n"
    )

    if is_blackhole_installed():
        bh = find_blackhole()
        if bh:
            console.print(f"[green]✓ BlackHole detected:[/green] {bh.name}")
            config.audio.system_device = bh.name
        else:
            console.print(
                "[yellow]BlackHole driver found but device not visible yet.[/yellow]\n"
                "Try restarting Audio MIDI Setup or your computer."
            )
    else:
        console.print("[yellow]BlackHole is not installed.[/yellow]\n")
        explain_routing_setup(console)

        if Confirm.ask("Install BlackHole 2ch via Homebrew now?", default=True):
            console.print("[dim]Running: brew install blackhole-2ch...[/dim]")
            result = subprocess.run(["brew", "install", "blackhole-2ch"])
            if result.returncode == 0:
                console.print(
                    "[green]BlackHole installed.[/green] "
                    "If the device doesn't appear, restart Audio MIDI Setup.\n"
                    "Then follow the Multi-Output Device steps shown above."
                )
                bh = find_blackhole()
                if bh:
                    config.audio.system_device = bh.name
            else:
                console.print(
                    "[red]Installation failed.[/red] "
                    "You can install manually later:\n" + get_install_instructions()
                )
        else:
            console.print(
                "[dim]Skipping BlackHole. Only mic audio will be captured "
                "until you install and configure it.[/dim]"
            )

    # ── Step 3: Microphone ────────────────────────────────────────────────────
    console.print("\n[bold]Step 3 of 6 — Microphone[/bold]")
    print_devices(console)
    if Confirm.ask("Use the system default microphone?", default=True):
        console.print("[dim]Using system default.[/dim]")
    else:
        devices = list_input_devices()
        if devices:
            choices = [str(i) for i in range(1, len(devices) + 1)]
            idx = int(Prompt.ask("Microphone number", choices=choices)) - 1
            config.audio.mic_device = devices[idx].name
            console.print(f"[green]Using:[/green] {devices[idx].name}")

    # ── Step 4: Save location ─────────────────────────────────────────────────
    console.print("\n[bold]Step 4 of 6 — Save Location[/bold]")
    console.print(
        f"Meeting notes and transcripts will be saved as Markdown files.\n"
        f"Default: [cyan]{config.notes_dir}[/cyan]\n"
    )
    custom = Prompt.ask("Save path (press Enter for default)", default=str(config.notes_dir))
    config.notes_dir = Path(custom).expanduser().resolve()
    config.notes_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]✓ Saving to:[/green] {config.notes_dir}")

    # ── Step 5: Ollama ────────────────────────────────────────────────────────
    console.print("\n[bold]Step 5 of 6 — Language Model (Ollama)[/bold]")
    console.print(
        "Ollama runs language models locally to generate meeting notes, "
        "to-do lists, and summaries.\n"
    )
    ollama = OllamaClient(config)
    chosen = ollama.run_setup_wizard(console)
    if chosen:
        config.ollama_model = chosen

    # ── Step 6: Whisper model ─────────────────────────────────────────────────
    console.print("\n[bold]Step 6 of 6 — Transcription Model (Whisper)[/bold]")
    table = Table(show_header=True)
    table.add_column("#", style="cyan", width=4)
    table.add_column("Model")
    table.add_column("Description")
    for i, (name, desc) in enumerate(_WHISPER_MODELS, 1):
        suffix = " [green](recommended)[/green]" if name == "base" else ""
        table.add_row(str(i), name, desc + suffix)
    console.print(table)

    choices = [str(i) for i in range(1, len(_WHISPER_MODELS) + 1)]
    idx = int(Prompt.ask("Choose Whisper model", choices=choices, default="2")) - 1
    config.whisper_model = _WHISPER_MODELS[idx][0]

    console.print(
        "\n[bold]Speaker identification[/bold]\n"
        "meetscribe can label remote participants individually "
        "('Participant 1', 'Participant 2', etc.) using voice embeddings.\n"
        "Requires ~50 MB extra RAM. Adds small latency per chunk.\n"
    )
    config.speaker_diarization = Confirm.ask("Enable speaker identification?", default=True)

    # ── Summary + save ────────────────────────────────────────────────────────
    console.print("\n[bold]Configuration summary[/bold]")
    console.print(f"  Notes dir:    [cyan]{config.notes_dir}[/cyan]")
    console.print(f"  Whisper:      [cyan]{config.whisper_model}[/cyan]")
    console.print(f"  Ollama model: [cyan]{config.ollama_model}[/cyan]")
    console.print(f"  Mic:          [cyan]{config.audio.mic_device or 'system default'}[/cyan]")
    console.print(f"  System audio: [cyan]{config.audio.system_device or 'not configured'}[/cyan]")
    console.print(f"  Diarization:  [cyan]{'on' if config.speaker_diarization else 'off'}[/cyan]")

    if Confirm.ask("\nSave this configuration?", default=True):
        config.first_run_complete = True
        config.save()
        console.print("[green]✓ Configuration saved.[/green]")
    else:
        console.print("[yellow]Not saved. Re-run with --setup to configure again.[/yellow]")

    return config
