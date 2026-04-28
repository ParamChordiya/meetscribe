from __future__ import annotations

import time
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel  # type: ignore

from meetscribe.config import Config
from meetscribe.transcription.diarize import SpeakerDiarizer
from meetscribe.types import Utterance


class TranscriptionEngine:
    """Transcribes paired (mic, system) audio chunks into labeled Utterances.

    mic audio    →  speaker = "You"
    system audio →  speaker from SpeakerDiarizer ("Participant 1", etc.)
                    or "Participant" if diarization is disabled/unavailable
    """

    SILENCE_THRESHOLD = 0.001  # RMS below this → skip Whisper

    def __init__(self, config: Config) -> None:
        self._config = config
        self._model: Optional[WhisperModel] = None
        self._diarizer: Optional[SpeakerDiarizer] = None
        self._session_start: float = 0.0

    def load(self) -> None:
        """Load Whisper model and optionally the diarizer."""
        self._model = WhisperModel(
            self._config.whisper_model,
            device="cpu",
            compute_type="int8",
        )
        if self._config.speaker_diarization:
            self._diarizer = SpeakerDiarizer()
            try:
                self._diarizer.load()
            except Exception:
                self._diarizer = None

    def start_session(self) -> None:
        """Reset session timer and speaker profiles. Call when meeting starts."""
        self._session_start = time.time()
        if self._diarizer is not None:
            self._diarizer.reset()

    def transcribe_chunk(self, mic: np.ndarray, system: np.ndarray) -> list[Utterance]:
        """Transcribe a (mic, system) audio pair and return labeled Utterances."""
        if self._model is None:
            raise RuntimeError("Call load() before transcribe_chunk()")

        elapsed = time.time() - self._session_start
        utterances: list[Utterance] = []

        if self._has_speech(mic):
            for text, start, end in self._run_whisper(mic):
                utterances.append(
                    Utterance(
                        text=text,
                        speaker="You",
                        timestamp=elapsed + start,
                        confidence=1.0,
                    )
                )

        if self._has_speech(system):
            speaker = self._identify_speaker(system)
            for text, start, end in self._run_whisper(system):
                utterances.append(
                    Utterance(
                        text=text,
                        speaker=speaker,
                        timestamp=elapsed + start,
                        confidence=1.0,
                    )
                )

        utterances.sort(key=lambda u: u.timestamp)
        return utterances

    def _has_speech(self, audio: np.ndarray) -> bool:
        return float(np.sqrt(np.mean(audio ** 2))) > self.SILENCE_THRESHOLD

    def _run_whisper(self, audio: np.ndarray) -> list[tuple[str, float, float]]:
        segments, _ = self._model.transcribe(
            audio,
            beam_size=5,
            language="en",
            vad_filter=True,
        )
        return [
            (seg.text.strip(), seg.start, seg.end)
            for seg in segments
            if seg.text.strip()
        ]

    def _identify_speaker(self, audio: np.ndarray) -> str:
        if self._diarizer is None:
            return "Participant"
        return self._diarizer.identify(audio, self._config.audio.sample_rate)
