from __future__ import annotations

import time

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
        self._model: WhisperModel | None = None
        self._diarizer: SpeakerDiarizer | None = None
        self._session_start: float = 0.0

    def load(self) -> None:
        """Load Whisper model and optionally the diarizer."""
        self._model = self._load_whisper_model()
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

    def _load_whisper_model(self) -> WhisperModel:
        """Download and load the Whisper model, retrying with SSL verification
        disabled on corporate networks that use self-signed certificates."""
        kwargs = dict(device="cpu", compute_type="int8")
        try:
            return WhisperModel(self._config.whisper_model, **kwargs)
        except Exception as first_err:
            err_str = str(first_err)
            if any(kw in err_str for kw in ("CERTIFICATE_VERIFY_FAILED", "SSL", "ConnectError", "SSLError")):
                import os
                os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
                try:
                    return WhisperModel(self._config.whisper_model, **kwargs)
                except Exception:
                    pass
            raise RuntimeError(
                f"Failed to load Whisper model '{self._config.whisper_model}'.\n"
                "If you are on a corporate network with SSL inspection, run:\n"
                "  export HF_HUB_DISABLE_SSL_VERIFY=1\n"
                "then try again."
            ) from first_err

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
