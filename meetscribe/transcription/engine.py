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

    SILENCE_THRESHOLD = 0.0005  # RMS below this → skip Whisper (lowered from 0.001)

    def __init__(self, config: Config, debug: bool = False) -> None:
        self._config = config
        self._debug = debug
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

        mic_rms = float(np.sqrt(np.mean(mic ** 2)))
        sys_rms = float(np.sqrt(np.mean(system ** 2)))

        if self._debug:
            import sys as _sys
            _sys.stderr.write(
                f"[debug] chunk {len(mic)/self._config.audio.sample_rate:.1f}s  "
                f"mic_rms={mic_rms:.5f}  sys_rms={sys_rms:.5f}  "
                f"threshold={self.SILENCE_THRESHOLD}\n"
            )
            _sys.stderr.flush()
            if mic_rms == 0.0:
                _sys.stderr.write(
                    "[debug] WARNING: mic_rms is exactly 0 — the wrong input device may be\n"
                    "  selected. Run: python3 -c \"import sounddevice as sd; "
                    "print(sd.query_devices(sd.default.device[0]))\"\n"
                    "  to see which device is active, then set audio.mic_device in config.\n"
                )
                _sys.stderr.flush()

        if mic_rms > self.SILENCE_THRESHOLD:
            for text, start, end in self._run_whisper(mic):
                utterances.append(
                    Utterance(
                        text=text,
                        speaker="You",
                        timestamp=elapsed + start,
                        confidence=1.0,
                    )
                )
            if self._debug and not utterances:
                import sys as _sys
                _sys.stderr.write("[debug] mic passed RMS but Whisper returned no segments\n")
                _sys.stderr.flush()

        if sys_rms > self.SILENCE_THRESHOLD:
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
        # vad_filter omitted: we already gate on RMS above, and silero VAD
        # inside faster-whisper can aggressively drop real speech on quiet mics.
        segments, _ = self._model.transcribe(
            audio,
            beam_size=5,
            language="en",
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
