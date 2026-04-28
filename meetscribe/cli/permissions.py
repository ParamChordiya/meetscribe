from __future__ import annotations

import subprocess
from typing import Optional

import sounddevice as sd
from rich.console import Console


def check_microphone_permission() -> bool:
    """Try opening a brief sounddevice InputStream to verify mic access."""
    try:
        stream = sd.InputStream(channels=1, samplerate=16000, dtype="float32")
        stream.start()
        stream.stop()
        stream.close()
        return True
    except Exception:
        return False


def check_accessibility_permission() -> bool:
    """Run a benign osascript to verify Accessibility access is granted."""
    try:
        result = subprocess.run(
            ["osascript", "-e", 'tell application "System Events" to return true'],
            capture_output=True,
            text=True,
            timeout=4,
        )
        return result.returncode == 0 and "true" in result.stdout.lower()
    except Exception:
        return False


def request_microphone_permission(console: Console) -> None:
    console.print(
        "\n[yellow]Microphone access required.[/yellow]\n"
        "Go to: [bold]System Settings → Privacy & Security → Microphone[/bold]\n"
        "Enable access for your terminal app (Terminal, iTerm2, etc.).\n"
    )
    _open_privacy_settings("microphone")


def request_accessibility_permission(console: Console) -> None:
    console.print(
        "\n[yellow]Accessibility access required.[/yellow]\n"
        "This lets meetscribe detect when you are in a Teams meeting.\n"
        "Go to: [bold]System Settings → Privacy & Security → Accessibility[/bold]\n"
        "Enable access for your terminal app.\n"
    )
    _open_privacy_settings("accessibility")


def _open_privacy_settings(section: str) -> None:
    try:
        subprocess.run(
            [
                "open",
                f"x-apple.systempreferences:com.apple.preference.security"
                f"?Privacy_{section.capitalize()}",
            ],
            check=False,
        )
    except Exception:
        pass


def check_all_permissions(console: Console) -> dict[str, bool]:
    """Check all permissions and print status. Returns {name: granted}."""
    results = {
        "microphone": check_microphone_permission(),
        "accessibility": check_accessibility_permission(),
    }
    for name, granted in results.items():
        icon = "[green]✓[/green]" if granted else "[red]✗[/red]"
        console.print(f"  {icon} {name.capitalize()}")
    return results


def ensure_permissions(console: Console) -> bool:
    """Check all permissions, guide user to grant any missing ones.
    Returns True only if microphone access is granted (accessibility is strongly recommended
    but meeting detection falls back to lsof without it).
    """
    console.print("\n[bold]Checking permissions...[/bold]")
    perms = check_all_permissions(console)

    if not perms["microphone"]:
        request_microphone_permission(console)
        input("\nPress Enter after granting microphone access...")
        if not check_microphone_permission():
            console.print("[red]Microphone access not granted. Cannot record audio.[/red]")
            return False

    if not perms["accessibility"]:
        request_accessibility_permission(console)
        input("\nPress Enter after granting accessibility access (or skip with Enter to continue without it)...")
        if not check_accessibility_permission():
            console.print(
                "[yellow]Accessibility access not confirmed. "
                "Auto-detection may be less reliable; --manual mode always works.[/yellow]"
            )

    return True
