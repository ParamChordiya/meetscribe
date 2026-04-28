from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import ClassVar, Optional

import yaml


@dataclass
class AudioSettings:
    mic_device: Optional[str] = None      # None = system default
    system_device: Optional[str] = None   # BlackHole device name; None = auto-detect
    sample_rate: int = 16000
    chunk_seconds: int = 20               # seconds per transcription chunk


@dataclass
class Config:
    CONFIG_PATH: ClassVar[Path] = Path.home() / ".config" / "meetscribe" / "config.yaml"

    notes_dir: Path = field(default_factory=lambda: Path.home() / "Documents" / "MeetScribe")
    whisper_model: str = "base"           # tiny | base | small | medium | large-v3
    ollama_model: str = "llama3.2"
    ollama_host: str = "http://localhost:11434"
    speaker_diarization: bool = True
    audio: AudioSettings = field(default_factory=AudioSettings)
    save_audio: bool = False
    save_transcript: bool = True
    poll_interval: int = 5                # seconds between Teams checks
    first_run_complete: bool = False

    @classmethod
    def load(cls) -> Config:
        if not cls.CONFIG_PATH.exists():
            return cls()
        with open(cls.CONFIG_PATH) as f:
            data = yaml.safe_load(f) or {}
        audio_data = data.pop("audio", {})
        audio_fields = {
            k: v for k, v in audio_data.items()
            if k in AudioSettings.__dataclass_fields__
        }
        audio = AudioSettings(**audio_fields)
        if "notes_dir" in data:
            data["notes_dir"] = Path(data["notes_dir"])
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(audio=audio, **valid)

    def save(self) -> None:
        self.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["notes_dir"] = str(self.notes_dir)
        with open(self.CONFIG_PATH, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
