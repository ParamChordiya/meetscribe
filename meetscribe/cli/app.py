from __future__ import annotations

import sys
import threading
from datetime import datetime

import click
import numpy as np
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.spinner import Spinner
from rich.text import Text

from meetscribe.audio.capture import AudioCapture
from meetscribe.cli.first_run import run_first_run_wizard
from meetscribe.cli.permissions import ensure_permissions
from meetscribe.config import Config
from meetscribe.detection.teams import MeetingDetector
from meetscribe.llm.client import OllamaClient
from meetscribe.transcription.engine import TranscriptionEngine
from meetscribe.types import Utterance

console = Console()


class MeetScribeApp:
    def __init__(self, config: Config, debug: bool = False) -> None:
        self._config = config
        self._debug = debug
        self._utterances: list[Utterance] = []
        self._recording = False
        self._capture: AudioCapture | None = None
        self._engine: TranscriptionEngine | None = None
        self._detector: MeetingDetector | None = None
        self._ollama: OllamaClient | None = None
        self._meeting_start: datetime | None = None
        self._shutdown = threading.Event()
        self._meeting_ended = threading.Event()  # signals main thread to run post-meeting menu

    # ── setup ─────────────────────────────────────────────────────────────────

    def setup(self) -> None:
        console.print("[dim]Loading transcription model...[/dim]", end="")
        self._engine = TranscriptionEngine(self._config, debug=self._debug)
        self._engine.load()
        console.print(" [green]ready[/green]")

        self._ollama = OllamaClient(self._config)
        if not self._ollama.is_available():
            console.print(
                "[yellow]Ollama is not running — generation features will be unavailable.[/yellow]\n"
                "Start it with: [bold cyan]ollama serve[/bold cyan]"
            )

        self._detector = MeetingDetector(self._config)

    # ── run loop ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        console.print(Panel(
            "[bold]meetscribe[/bold] — watching for Teams meetings\n"
            f"Whisper [cyan]{self._config.whisper_model}[/cyan]  •  "
            f"Model [cyan]{self._config.ollama_model}[/cyan]  •  "
            f"Diarization [cyan]{'on' if self._config.speaker_diarization else 'off'}[/cyan]\n"
            "Commands: [bold]s[/bold] = start  [bold]e[/bold] = end  [bold]Ctrl+C[/bold] = quit",
            style="green",
        ))

        self._detector.start(
            on_start=self._on_meeting_start,
            on_end=self._on_meeting_end,
        )
        self._start_stdin_listener()

        try:
            while not self._shutdown.is_set():
                if self._meeting_ended.wait(timeout=1.0):
                    self._meeting_ended.clear()
                    self._process_ended_meeting()
        except KeyboardInterrupt:
            pass
        finally:
            self._cleanup()

    def _start_stdin_listener(self) -> None:
        """Background thread: reads single-char commands from stdin."""
        def _listen() -> None:
            while not self._shutdown.is_set():
                try:
                    cmd = input().strip().lower()
                except EOFError:
                    break
                if cmd == "s":
                    if not self._recording:
                        console.print("[dim]Manual start...[/dim]")
                        self._on_meeting_start()
                    else:
                        console.print("[dim]Already recording.[/dim]")
                elif cmd == "e":
                    if self._recording:
                        console.print("[dim]Manual end...[/dim]")
                        self._on_meeting_end()
                    else:
                        console.print("[dim]Not currently recording.[/dim]")

        threading.Thread(target=_listen, daemon=True).start()

    def start_manual(self) -> None:
        self._on_meeting_start()

    def stop_manual(self) -> None:
        self._on_meeting_end()
        self._process_ended_meeting()

    # ── meeting events ────────────────────────────────────────────────────────

    def _on_meeting_start(self) -> None:
        self._utterances = []
        self._recording = True
        self._meeting_start = datetime.now()

        self._capture = AudioCapture(self._config)
        self._engine.start_session()

        if not self._capture.has_system_audio:
            console.print(
                "\n[yellow]Warning: BlackHole not detected — recording mic only.[/yellow]\n"
                "Remote participants will not be transcribed.\n"
                "Run [bold]meetscribe --setup[/bold] to configure audio routing.\n"
            )

        self._capture.on_chunk(self._on_audio_chunk)
        self._capture.start()

        sys_info = (
            f" + [cyan]{self._capture.system_device_name}[/cyan]"
            if self._capture.has_system_audio
            else " (mic only)"
        )
        console.print(Panel(
            f"[bold green]Meeting started[/bold green] — "
            f"[cyan]{self._capture.mic_device_name}[/cyan]{sys_info}",
            style="green",
        ))

    def _on_meeting_end(self) -> None:
        """Called from the detector poll thread — do minimal work, signal main thread."""
        if not self._recording:
            return
        self._recording = False
        if self._capture:
            try:
                self._capture.stop()
            except Exception:
                pass
        self._meeting_ended.set()

    def _process_ended_meeting(self) -> None:
        """Called on the main thread after a meeting ends."""
        console.print(Panel("[bold yellow]Meeting ended — processing...[/bold yellow]"))

        if not self._utterances:
            console.print("[dim]No speech detected.[/dim]")
            return

        self._show_transcript()
        self._post_meeting_menu()

    def _on_audio_chunk(self, mic: np.ndarray, system: np.ndarray) -> None:
        try:
            new = self._engine.transcribe_chunk(mic, system)
        except Exception as e:
            console.print(f"[red dim]Transcription error: {e}[/red dim]")
            return

        self._utterances.extend(new)
        for u in new:
            h = int(u.timestamp // 3600)
            m = int((u.timestamp % 3600) // 60)
            s = int(u.timestamp % 60)
            ts = f"[dim][{h:02d}:{m:02d}:{s:02d}][/dim]"
            color = "cyan" if u.speaker == "You" else "green"
            console.print(f"{ts} [{color}]{u.speaker}[/{color}]: {u.text}")

    # ── post meeting ──────────────────────────────────────────────────────────

    def _show_transcript(self) -> None:
        lines = []
        for u in self._utterances:
            h = int(u.timestamp // 3600)
            m = int((u.timestamp % 3600) // 60)
            s = int(u.timestamp % 60)
            color = "cyan" if u.speaker == "You" else "green"
            lines.append(f"[dim][{h:02d}:{m:02d}:{s:02d}][/dim] [{color}]{u.speaker}[/{color}]: {u.text}")
        console.print(Panel(
            "\n".join(lines) if lines else "(empty)",
            title="[bold]Full Transcript[/bold]",
        ))

    def _post_meeting_menu(self) -> None:
        if not self._ollama or not self._ollama.is_available():
            console.print(
                "[yellow]Ollama is not running — skipping generation.[/yellow]\n"
                "Start Ollama and re-run meetscribe to generate notes from a saved transcript."
            )
            if self._utterances:
                self._save_results({})
            return

        console.print("\n[bold]Generate:[/bold]")
        console.print("  [cyan]1[/cyan]  Meeting notes")
        console.print("  [cyan]2[/cyan]  To-do tasks")
        console.print("  [cyan]3[/cyan]  Summary")
        console.print("  [cyan]4[/cyan]  All three")
        console.print("  [cyan]5[/cyan]  Skip generation")

        choice = Prompt.ask("Choice", choices=["1", "2", "3", "4", "5"], default="4")
        kinds_map = {
            "1": ["notes"],
            "2": ["todos"],
            "3": ["summary"],
            "4": ["notes", "todos", "summary"],
            "5": [],
        }
        kinds = kinds_map[choice]

        results: dict[str, str] = {}
        for kind in kinds:
            parts: list[str] = []
            title = f"[bold blue]{kind.title()}[/bold blue]"
            try:
                with Live(
                    Spinner("dots", text=f"[dim]Generating {kind}…[/dim]"),
                    console=console,
                    refresh_per_second=15,
                    transient=False,
                ) as live:
                    for token in self._ollama.generate_stream(self._utterances, kind):
                        parts.append(token)
                        live.update(Panel(Text("".join(parts)), title=title))
            except Exception as e:
                console.print(f"[red]Generation failed: {e}[/red]")
            if parts:
                console.print(Panel(Text("".join(parts)), title=title))
            results[kind] = "".join(parts)

        self._save_results(results)

    def _save_results(self, results: dict[str, str]) -> None:
        self._config.notes_dir.mkdir(parents=True, exist_ok=True)
        ts = (
            self._meeting_start.strftime("%Y-%m-%d_%H-%M")
            if self._meeting_start
            else "unknown"
        )
        outfile = self._config.notes_dir / f"{ts}_meeting.md"

        date_str = (
            self._meeting_start.strftime("%B %d, %Y %H:%M")
            if self._meeting_start
            else "Unknown date"
        )
        lines = [f"# Meeting — {date_str}", ""]

        if results.get("summary"):
            lines += ["## Summary", "", results["summary"], ""]
        if results.get("notes"):
            lines += ["## Meeting Notes", "", results["notes"], ""]
        if results.get("todos"):
            lines += ["## To-Do Tasks", "", results["todos"], ""]

        lines += ["## Transcript", ""]
        for u in self._utterances:
            h = int(u.timestamp // 3600)
            m = int((u.timestamp % 3600) // 60)
            s = int(u.timestamp % 60)
            lines.append(f"**[{h:02d}:{m:02d}:{s:02d}] {u.speaker}:** {u.text}  ")

        outfile.write_text("\n".join(lines))
        console.print(f"\n[green]Saved →[/green] {outfile}")

    # ── cleanup ───────────────────────────────────────────────────────────────

    def _cleanup(self) -> None:
        if self._capture and self._recording:
            try:
                self._capture.stop()
            except Exception:
                pass
        if self._detector:
            self._detector.stop()


# ── CLI entry point ───────────────────────────────────────────────────────────

@click.command()
@click.option("--setup", is_flag=True, help="Re-run the first-run setup wizard")
@click.option("--manual", is_flag=True, help="Record immediately without waiting for Teams")
@click.option("--model", default=None, help="Override Ollama model for this session")
@click.option("--whisper", default=None, help="Override Whisper model size for this session")
@click.option("--debug", is_flag=True, help="Print audio RMS and chunk info for troubleshooting")
def main(
    setup: bool,
    manual: bool,
    model: str | None,
    whisper: str | None,
    debug: bool,
) -> None:
    config = Config.load()

    if not config.first_run_complete or setup:
        config = run_first_run_wizard(console)

    if not ensure_permissions(console):
        console.print("[red]Required permissions not granted. Exiting.[/red]")
        sys.exit(1)

    if model:
        config.ollama_model = model
    if whisper:
        config.whisper_model = whisper

    app = MeetScribeApp(config, debug=debug)
    app.setup()

    if manual:
        console.print(Panel(
            "[bold]Manual mode[/bold]\n"
            "Press [bold]Enter[/bold] to start recording.",
            style="cyan",
        ))
        try:
            input()
        except KeyboardInterrupt:
            return
        app.start_manual()
        console.print("[dim]Recording… press [bold]Enter[/bold] to stop.[/dim]")
        try:
            input()
        except KeyboardInterrupt:
            pass
        app.stop_manual()
    else:
        app.run()
