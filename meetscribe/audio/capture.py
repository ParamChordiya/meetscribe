from __future__ import annotations

import threading
import time
from collections.abc import Callable
from math import gcd

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

from meetscribe.audio.devices import (
    AudioDevice,
    find_blackhole,
    find_device_by_name,
    get_default_input,
)
from meetscribe.config import Config

ChunkCallback = Callable[[np.ndarray, np.ndarray], None]


class AudioCapture:
    """Captures mic and system audio simultaneously as two synchronized streams.

    Mic (your voice) and BlackHole (remote participants via system audio) are
    opened as separate sounddevice InputStreams. Both accumulate independently;
    the chunk worker pairs them into (mic_chunk, system_chunk) tuples and fires
    registered callbacks every config.audio.chunk_seconds seconds.

    Works regardless of headphone use because BlackHole intercepts Teams audio
    at the software level, before it reaches the physical output device.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._sr = config.audio.sample_rate
        self._chunk_n = self._sr * config.audio.chunk_seconds

        self._callbacks: list[ChunkCallback] = []
        self._running = False

        self._full_mic: list[float] = []
        self._full_sys: list[float] = []
        self._mic_buf: list[float] = []
        self._sys_buf: list[float] = []
        self._lock = threading.Lock()

        self._mic_stream: sd.InputStream | None = None
        self._sys_stream: sd.InputStream | None = None
        self._worker: threading.Thread | None = None

        self._mic_device: AudioDevice = self._resolve_mic()
        self._sys_device: AudioDevice | None = self._resolve_system()

    def _resolve_mic(self) -> AudioDevice:
        if self._config.audio.mic_device:
            dev = find_device_by_name(self._config.audio.mic_device)
            if dev:
                return dev
        return get_default_input()

    def _resolve_system(self) -> AudioDevice | None:
        if self._config.audio.system_device:
            return find_device_by_name(self._config.audio.system_device)
        return find_blackhole()

    # ── stream callbacks ──────────────────────────────────────────────────────

    def _make_stream(
        self,
        device: AudioDevice,
        target_buf: list[float],
        full_buf: list[float],
        label: str = "device",
    ) -> sd.InputStream:
        native_sr = int(device.default_samplerate)
        target_sr = self._sr
        _zero_chunks: list[int] = [0]  # mutable counter for zero-audio detection

        def _cb(indata: np.ndarray, frames: int, time_info, status) -> None:
            if not self._running:
                return
            mono = indata[:, 0].astype(np.float32)
            if float(np.max(np.abs(mono))) == 0.0:
                _zero_chunks[0] += 1
                if _zero_chunks[0] == 10:  # ~1s of consecutive silence
                    import sys
                    sys.stderr.write(
                        f"\n[WARNING] {label!r} is delivering silence (all zeros).\n"
                        "  macOS does this when microphone permission is denied.\n"
                        "  Go to: System Settings → Privacy & Security → Microphone\n"
                        "  and enable access for your terminal app.\n\n"
                    )
                    sys.stderr.flush()
            else:
                _zero_chunks[0] = 0
            if native_sr != target_sr:
                g = gcd(native_sr, target_sr)
                mono = resample_poly(mono, target_sr // g, native_sr // g).astype(np.float32)
            samples = mono.tolist()
            with self._lock:
                target_buf.extend(samples)
                full_buf.extend(samples)

        sr_to_use = native_sr  # open at native rate; resampling happens in callback
        return sd.InputStream(
            device=device.index,
            channels=1,
            samplerate=sr_to_use,
            dtype="float32",
            callback=_cb,
            blocksize=1024,
        )

    # ── chunk worker ──────────────────────────────────────────────────────────

    def _chunk_worker(self) -> None:
        while self._running:
            with self._lock:
                mic_ready = len(self._mic_buf) >= self._chunk_n
                sys_ready = (self._sys_device is None) or (len(self._sys_buf) >= self._chunk_n)

            if mic_ready and sys_ready:
                with self._lock:
                    mic_chunk = np.array(self._mic_buf[: self._chunk_n], dtype=np.float32)
                    del self._mic_buf[: self._chunk_n]  # mutate in-place; keeps callback reference valid

                    if self._sys_device is not None:
                        sys_chunk = np.array(self._sys_buf[: self._chunk_n], dtype=np.float32)
                        del self._sys_buf[: self._chunk_n]
                    else:
                        sys_chunk = np.zeros(self._chunk_n, dtype=np.float32)

                for cb in self._callbacks:
                    try:
                        cb(mic_chunk, sys_chunk)
                    except Exception:
                        pass
            else:
                time.sleep(0.05)

        # flush remainder when stopping
        with self._lock:
            rem = len(self._mic_buf)
            if rem > 0:
                mic_chunk = np.array(self._mic_buf, dtype=np.float32)
                if self._sys_device is not None and self._sys_buf:
                    sys_chunk = np.array(self._sys_buf[:rem], dtype=np.float32)
                    pad = rem - len(sys_chunk)
                    if pad > 0:
                        sys_chunk = np.concatenate([sys_chunk, np.zeros(pad, dtype=np.float32)])
                else:
                    sys_chunk = np.zeros(rem, dtype=np.float32)
                self._mic_buf = []
                self._sys_buf = []
                for cb in self._callbacks:
                    try:
                        cb(mic_chunk, sys_chunk)
                    except Exception:
                        pass

    # ── public API ────────────────────────────────────────────────────────────

    def on_chunk(self, callback: ChunkCallback) -> None:
        """Register a callback invoked with (mic_chunk, system_chunk) every chunk_seconds."""
        self._callbacks.append(callback)

    def start(self) -> None:
        self._running = True
        self._full_mic = []
        self._full_sys = []
        self._mic_buf = []
        self._sys_buf = []

        self._mic_stream = self._make_stream(self._mic_device, self._mic_buf, self._full_mic, label=self._mic_device.name)
        self._mic_stream.start()

        if self._sys_device is not None:
            self._sys_stream = self._make_stream(self._sys_device, self._sys_buf, self._full_sys, label=self._sys_device.name)
            self._sys_stream.start()

        self._worker = threading.Thread(target=self._chunk_worker, daemon=True)
        self._worker.start()

    def stop(self) -> tuple[np.ndarray, np.ndarray]:
        """Stop capture. Returns (full_mic_audio, full_system_audio) as float32 arrays."""
        self._running = False

        for stream in (self._mic_stream, self._sys_stream):
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass

        if self._worker:
            self._worker.join(timeout=15)

        return (
            np.array(self._full_mic, dtype=np.float32),
            np.array(self._full_sys, dtype=np.float32),
        )

    @property
    def has_system_audio(self) -> bool:
        return self._sys_device is not None

    @property
    def mic_device_name(self) -> str:
        return self._mic_device.name

    @property
    def system_device_name(self) -> str | None:
        return self._sys_device.name if self._sys_device else None
