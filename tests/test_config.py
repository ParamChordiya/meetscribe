from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from meetscribe.config import AudioSettings, Config


@pytest.fixture
def tmp_config(tmp_path, monkeypatch):
    """Redirect Config.CONFIG_PATH to a temp directory."""
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(Config, "CONFIG_PATH", config_path)
    return config_path


class TestConfigDefaults:
    def test_default_notes_dir(self):
        c = Config()
        assert c.notes_dir == Path.home() / "Documents" / "MeetScribe"

    def test_default_whisper_model(self):
        assert Config().whisper_model == "base"

    def test_default_ollama_model(self):
        assert Config().ollama_model == "llama3.2"

    def test_default_ollama_host(self):
        assert Config().ollama_host == "http://localhost:11434"

    def test_default_speaker_diarization_on(self):
        assert Config().speaker_diarization is True

    def test_default_first_run_not_complete(self):
        assert Config().first_run_complete is False

    def test_default_poll_interval(self):
        assert Config().poll_interval == 5

    def test_audio_defaults(self):
        a = Config().audio
        assert a.mic_device is None
        assert a.system_device is None
        assert a.sample_rate == 16000
        assert a.chunk_seconds == 20


class TestConfigLoadSave:
    def test_load_returns_defaults_when_no_file(self, tmp_config):
        c = Config.load()
        assert c.whisper_model == "base"

    def test_save_creates_file(self, tmp_config):
        c = Config()
        c.save()
        assert tmp_config.exists()

    def test_roundtrip(self, tmp_config):
        original = Config(
            whisper_model="small",
            ollama_model="mistral",
            speaker_diarization=False,
            first_run_complete=True,
            poll_interval=10,
        )
        original.save()
        loaded = Config.load()
        assert loaded.whisper_model == "small"
        assert loaded.ollama_model == "mistral"
        assert loaded.speaker_diarization is False
        assert loaded.first_run_complete is True
        assert loaded.poll_interval == 10

    def test_notes_dir_saved_as_string_loaded_as_path(self, tmp_config):
        c = Config(notes_dir=Path("/tmp/meeting_notes"))
        c.save()
        raw = yaml.safe_load(tmp_config.read_text())
        assert isinstance(raw["notes_dir"], str)
        loaded = Config.load()
        assert isinstance(loaded.notes_dir, Path)
        assert loaded.notes_dir == Path("/tmp/meeting_notes")

    def test_audio_settings_roundtrip(self, tmp_config):
        c = Config()
        c.audio.mic_device = "MacBook Pro Microphone"
        c.audio.system_device = "BlackHole 2ch"
        c.save()
        loaded = Config.load()
        assert loaded.audio.mic_device == "MacBook Pro Microphone"
        assert loaded.audio.system_device == "BlackHole 2ch"

    def test_load_ignores_unknown_keys(self, tmp_config):
        tmp_config.write_text("whisper_model: tiny\nunknown_key: ignored\n")
        c = Config.load()
        assert c.whisper_model == "tiny"

    def test_load_fills_missing_keys_with_defaults(self, tmp_config):
        tmp_config.write_text("whisper_model: medium\n")
        c = Config.load()
        assert c.whisper_model == "medium"
        assert c.ollama_model == "llama3.2"  # default filled in


class TestAudioSettings:
    def test_defaults(self):
        a = AudioSettings()
        assert a.mic_device is None
        assert a.system_device is None
        assert a.sample_rate == 16000
        assert a.chunk_seconds == 20

    def test_custom_values(self):
        a = AudioSettings(mic_device="Mic", system_device="BH", sample_rate=8000, chunk_seconds=10)
        assert a.mic_device == "Mic"
        assert a.sample_rate == 8000
