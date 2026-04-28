from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Utterance:
    text: str
    speaker: str  # "You", "Participant 1", "Participant 2", etc.
    timestamp: float  # seconds from meeting start
    confidence: float = 1.0
