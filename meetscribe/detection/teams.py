from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import Callable, Optional

import psutil

from meetscribe.config import Config

_TEAMS_PROCESS_NAMES = frozenset({
    "microsoft teams",
    "msteams",
    "com.microsoft.teams2",
    "teams",
})

# Only phrases that appear exclusively during an active call/meeting.
# Broad terms like "meeting", "mute", "participants" are intentionally excluded
# because they appear in Teams' regular navigation and calendar views.
_MEETING_WINDOW_KEYWORDS = frozenset({
    "call in progress",
    "video call",
    "audio call",
    "in a meeting",
    "joined the meeting",
    "you're in a call",
})

_TEAMS_APP_PATHS = [
    "/Applications/Microsoft Teams.app",
    "/Applications/Microsoft Teams (work or school).app",
    os.path.expanduser("~/Applications/Microsoft Teams.app"),
    os.path.expanduser("~/Applications/Microsoft Teams (work or school).app"),
]


class MeetingDetector:
    """Polls for an active Teams meeting using AppleScript window titles (primary)
    and CoreAudio file-handle inspection via lsof (fallback).
    """

    # Teams loads CoreAudio even when idle (notifications).
    # An active audio session (meeting) consistently shows a much higher count.
    _COREAUDIO_MEETING_THRESHOLD = 15

    def __init__(self, config: Config) -> None:
        self._config = config
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._in_meeting = False
        self._consecutive_hits = 0   # debounce: require 2 positive polls to start

    def start(
        self,
        on_start: Callable[[], None],
        on_end: Callable[[], None],
    ) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop,
            args=(on_start, on_end),
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=15)

    # ── detection ─────────────────────────────────────────────────────────────

    def _poll_loop(self, on_start: Callable, on_end: Callable) -> None:
        while self._running:
            detected = self._check_meeting()

            if detected:
                self._consecutive_hits += 1
            else:
                self._consecutive_hits = 0

            # Require 2 consecutive positive polls before declaring meeting start.
            # This prevents a single transient false positive from triggering recording.
            if self._consecutive_hits >= 2 and not self._in_meeting:
                self._in_meeting = True
                try:
                    on_start()
                except Exception:
                    pass
            elif not detected and self._in_meeting:
                self._in_meeting = False
                self._consecutive_hits = 0
                try:
                    on_end()
                except Exception:
                    pass

            time.sleep(self._config.poll_interval)

    def _check_meeting(self) -> bool:
        titles = self._get_teams_window_titles()
        if titles:
            joined = " ".join(t.lower() for t in titles)
            if any(kw in joined for kw in _MEETING_WINDOW_KEYWORDS):
                return True

        pids = self._get_teams_pids()
        if pids and self._teams_using_audio(pids):
            return True

        return False

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_teams_window_titles(self) -> list[str]:
        """Use osascript to retrieve Teams window titles (requires Accessibility permission)."""
        process_names = (
            '"MSTeams", "Microsoft Teams", '
            '"Microsoft Teams (work or school)", "com.microsoft.teams2"'
        )
        script = (
            f"tell application \"System Events\"\n"
            f"  set allTitles to {{}}\n"
            f"  repeat with pName in {{{process_names}}}\n"
            f"    if exists process pName then\n"
            f"      set allTitles to allTitles & (title of every window of process pName)\n"
            f"    end if\n"
            f"  end repeat\n"
            f"  return allTitles\n"
            f"end tell"
        )
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return [t.strip() for t in result.stdout.split(",") if t.strip()]
        except Exception:
            pass
        return []

    def _get_teams_pids(self) -> list[int]:
        pids = []
        for proc in psutil.process_iter(["name", "pid"]):
            try:
                name = proc.info["name"].lower()
                if any(n in name for n in _TEAMS_PROCESS_NAMES):
                    pids.append(proc.info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return pids

    def _teams_using_audio(self, pids: list[int]) -> bool:
        pid_str = ",".join(str(p) for p in pids)
        try:
            result = subprocess.run(
                f"lsof -p {pid_str} 2>/dev/null | grep -icE 'CoreAudio|coreaudio'",
                shell=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            count = int(result.stdout.strip() or "0")
            return count >= self._COREAUDIO_MEETING_THRESHOLD
        except Exception:
            return False

    @staticmethod
    def is_teams_installed() -> bool:
        return any(os.path.exists(p) for p in _TEAMS_APP_PATHS)
