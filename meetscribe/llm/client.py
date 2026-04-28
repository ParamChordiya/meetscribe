from __future__ import annotations

import json
import subprocess
import webbrowser
from typing import Iterator, Optional

import requests
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from meetscribe.config import Config
from meetscribe.llm.prompts import PROMPTS, format_transcript
from meetscribe.types import Utterance


class OllamaClient:
    OLLAMA_DOWNLOAD_URL = "https://ollama.com/download"

    RECOMMENDED_MODELS: list[tuple[str, str]] = [
        ("llama3.2", "3B — fast, 2 GB RAM, good quality (recommended)"),
        ("llama3.1:8b", "8B — more capable, 5 GB RAM"),
        ("mistral", "7B — fast, 4 GB RAM, strong reasoning"),
        ("gemma2:2b", "Google 2B — very fast, 1.5 GB RAM"),
        ("phi3:mini", "Microsoft 3.8B — efficient, 2 GB RAM"),
    ]

    def __init__(self, config: Config) -> None:
        self._config = config
        self._base = config.ollama_host.rstrip("/")

    # ── availability ──────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self._base}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def is_installed(self) -> bool:
        try:
            result = subprocess.run(
                ["which", "ollama"], capture_output=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def list_models(self) -> list[str]:
        try:
            r = requests.get(f"{self._base}/api/tags", timeout=5)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    # ── model management ──────────────────────────────────────────────────────

    def pull_model(self, model: str, console: Console) -> bool:
        console.print(f"[dim]Downloading [cyan]{model}[/cyan] — this may take a few minutes...[/dim]")
        try:
            with requests.post(
                f"{self._base}/api/pull",
                json={"name": model},
                stream=True,
                timeout=600,
            ) as resp:
                resp.raise_for_status()
                last_status = ""
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if data.get("error"):
                            console.print(f"[red]Error: {data['error']}[/red]")
                            return False
                        status = data.get("status", "")
                        if status and status != last_status:
                            console.print(f"  [dim]{status}[/dim]")
                            last_status = status
                    except json.JSONDecodeError:
                        pass
            console.print(f"[green]Model [cyan]{model}[/cyan] ready.[/green]")
            return True
        except Exception as e:
            console.print(f"[red]Failed to pull {model}: {e}[/red]")
            return False

    # ── setup wizard ──────────────────────────────────────────────────────────

    def run_setup_wizard(self, console: Console) -> Optional[str]:
        """Interactive Ollama setup wizard. Returns chosen model name or None if aborted."""
        console.print(Panel("[bold]Ollama Setup[/bold]", style="blue"))

        if not self.is_installed():
            console.print(
                "[yellow]Ollama is not installed.[/yellow]\n\n"
                f"Download it from: [bold]{self.OLLAMA_DOWNLOAD_URL}[/bold]\n"
                "After installing, run [bold cyan]ollama serve[/bold cyan] in a terminal, "
                "then re-run meetscribe."
            )
            if Confirm.ask("Open the download page in your browser?", default=True):
                webbrowser.open(self.OLLAMA_DOWNLOAD_URL)
            return None

        if not self.is_available():
            console.print(
                "[yellow]Ollama is installed but not running.[/yellow]\n\n"
                "Start it in a new terminal window:\n\n"
                "  [bold cyan]ollama serve[/bold cyan]\n"
            )
            input("Press Enter once Ollama is running...")
            if not self.is_available():
                console.print(
                    "[red]Ollama still not reachable. "
                    "Check that 'ollama serve' is running on port 11434.[/red]"
                )
                return None

        models = self.list_models()
        if not models:
            console.print("[yellow]No models downloaded yet.[/yellow]\n")
            chosen = self._pick_recommended(console)
            if chosen is None:
                return None
            if not self.pull_model(chosen, console):
                return None
            self._config.ollama_model = chosen
            return chosen

        return self.change_model(console)

    def change_model(self, console: Console) -> Optional[str]:
        """Let user choose from available models or pull a new one."""
        models = self.list_models()
        table = Table(title="Available Models", show_header=True)
        table.add_column("#", style="cyan", width=4)
        table.add_column("Model")
        for i, m in enumerate(models, 1):
            table.add_row(str(i), m)
        table.add_row(str(len(models) + 1), "[dim]Pull a different model[/dim]")
        console.print(table)

        choices = [str(i) for i in range(1, len(models) + 2)]
        choice = Prompt.ask("Choose model", choices=choices, default="1")
        idx = int(choice) - 1

        if idx < len(models):
            chosen = models[idx]
        else:
            chosen = self._pick_recommended(console)
            if chosen is None:
                return None
            if chosen not in models:
                if not self.pull_model(chosen, console):
                    return None

        self._config.ollama_model = chosen
        return chosen

    def _pick_recommended(self, console: Console) -> Optional[str]:
        table = Table(title="Recommended Models", show_header=True)
        table.add_column("#", style="cyan", width=4)
        table.add_column("Model")
        table.add_column("Description")
        for i, (name, desc) in enumerate(self.RECOMMENDED_MODELS, 1):
            table.add_row(str(i), name, desc)
        table.add_row(
            str(len(self.RECOMMENDED_MODELS) + 1),
            "custom",
            "Enter a model name manually",
        )
        console.print(table)

        choices = [str(i) for i in range(1, len(self.RECOMMENDED_MODELS) + 2)]
        choice = Prompt.ask("Choose model to download", choices=choices, default="1")
        idx = int(choice) - 1

        if idx < len(self.RECOMMENDED_MODELS):
            return self.RECOMMENDED_MODELS[idx][0]
        name = Prompt.ask("Model name (e.g. llama3.2:latest)").strip()
        return name or None

    # ── generation ────────────────────────────────────────────────────────────

    def generate_stream(self, utterances: list[Utterance], kind: str) -> Iterator[str]:
        """Stream generated text tokens for the given utterances and kind.

        kind: "notes" | "todos" | "summary"
        """
        if kind not in PROMPTS:
            raise ValueError(f"Unknown kind {kind!r}. Valid: {list(PROMPTS)}")

        transcript = format_transcript(utterances)
        prompt = PROMPTS[kind].format(transcript=transcript)

        payload = {
            "model": self._config.ollama_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }

        try:
            with requests.post(
                f"{self._base}/api/chat",
                json=payload,
                stream=True,
                timeout=300,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        pass
        except requests.exceptions.ConnectionError:
            yield "\n[Ollama connection lost. Is 'ollama serve' still running?]\n"
        except Exception as e:
            yield f"\n[Generation error: {e}]\n"
