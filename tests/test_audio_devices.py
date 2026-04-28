from __future__ import annotations

from meetscribe.audio.devices import (
    find_blackhole,
    find_device_by_name,
    get_default_input,
    list_input_devices,
)


class TestListInputDevices:
    def test_excludes_output_only_devices(self):
        devices = list_input_devices()
        names = [d.name for d in devices]
        assert "Built-in Output" not in names

    def test_includes_mic_and_blackhole(self):
        devices = list_input_devices()
        names = [d.name for d in devices]
        assert "Built-in Microphone" in names
        assert "BlackHole 2ch" in names

    def test_returns_audio_device_objects(self):
        devices = list_input_devices()
        assert len(devices) >= 1
        d = devices[0]
        assert hasattr(d, "index")
        assert hasattr(d, "name")
        assert hasattr(d, "max_input_channels")
        assert d.max_input_channels > 0

    def test_default_device_flagged(self):
        devices = list_input_devices()
        defaults = [d for d in devices if d.is_default_input]
        assert len(defaults) == 1


class TestFindBlackhole:
    def test_finds_blackhole_device(self):
        bh = find_blackhole()
        assert bh is not None
        assert "BlackHole" in bh.name

    def test_returns_none_when_absent(self, monkeypatch):
        import sounddevice as sd
        monkeypatch.setattr(sd, "query_devices", lambda: [
            {
                "name": "Built-in Microphone",
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 44100.0,
            }
        ])
        monkeypatch.setattr(sd, "default", type("D", (), {"device": [0, 0]})())
        assert find_blackhole() is None


class TestFindDeviceByName:
    def test_partial_match_case_insensitive(self):
        d = find_device_by_name("blackhole")
        assert d is not None
        assert "BlackHole" in d.name

    def test_partial_match_uppercase(self):
        d = find_device_by_name("BUILT-IN")
        assert d is not None

    def test_returns_none_for_unknown(self):
        assert find_device_by_name("nonexistent_device_xyz") is None


class TestGetDefaultInput:
    def test_returns_device_at_default_index(self):
        d = get_default_input()
        assert d.index == 0
        assert d.is_default_input is True
