from __future__ import annotations

import threading
from typing import Optional

import numpy as np


class SpeakerDiarizer:
    """Identifies speakers in audio segments using voice embeddings (resemblyzer).

    Maintains per-session speaker profiles. On each new segment:
    - Embed the audio with VoiceEncoder
    - Compare against stored profiles via cosine similarity
    - If best match >= threshold  →  existing speaker (update running mean)
    - Otherwise                  →  new speaker ("Participant N")

    Call reset() at the start of each meeting.
    """

    MIN_SEGMENT_SAMPLES = 24000  # ~1.5 s at 16 kHz — minimum for a reliable embedding

    def __init__(self, similarity_threshold: float = 0.75) -> None:
        self._threshold = similarity_threshold
        self._encoder = None
        self._profiles: dict[str, np.ndarray] = {}
        self._profile_counts: dict[str, int] = {}
        self._speaker_count = 0
        self._lock = threading.Lock()

    def load(self) -> None:
        """Load the VoiceEncoder model (downloads ~17 MB on first use)."""
        from resemblyzer import VoiceEncoder  # type: ignore

        self._encoder = VoiceEncoder()

    def identify(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Return a consistent speaker label for the audio segment."""
        if self._encoder is None or len(audio) < self.MIN_SEGMENT_SAMPLES:
            return "Participant"

        try:
            from resemblyzer import preprocess_wav  # type: ignore

            wav = preprocess_wav(audio, source_sr=sample_rate)
            embedding: np.ndarray = self._encoder.embed_utterance(wav)
        except Exception:
            return "Participant"

        with self._lock:
            if not self._profiles:
                return self._new_speaker(embedding)

            best_name: Optional[str] = None
            best_sim = -1.0
            for name, profile in self._profiles.items():
                norm = np.linalg.norm(embedding) * np.linalg.norm(profile) + 1e-8
                sim = float(np.dot(embedding, profile) / norm)
                if sim > best_sim:
                    best_sim = sim
                    best_name = name

            if best_sim >= self._threshold and best_name is not None:
                n = self._profile_counts[best_name]
                self._profiles[best_name] = (self._profiles[best_name] * n + embedding) / (n + 1)
                self._profile_counts[best_name] = n + 1
                return best_name

            return self._new_speaker(embedding)

    def _new_speaker(self, embedding: np.ndarray) -> str:
        self._speaker_count += 1
        name = f"Participant {self._speaker_count}"
        self._profiles[name] = embedding.copy()
        self._profile_counts[name] = 1
        return name

    def reset(self) -> None:
        """Clear all speaker profiles. Call at meeting start."""
        with self._lock:
            self._profiles.clear()
            self._profile_counts.clear()
            self._speaker_count = 0
