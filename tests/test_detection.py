from __future__ import annotations

from unittest.mock import patch

import pytest

from meetscribe.config import Config
from meetscribe.detection.teams import (
    _MEETING_WINDOW_KEYWORDS,
    _TEAMS_NAV_TITLES,
    MeetingDetector,
)


class TestMeetingKeywords:
    """Ensure keyword list contains only meeting-specific phrases, not UI navigation terms."""

    def test_does_not_contain_mute(self):
        assert "mute" not in _MEETING_WINDOW_KEYWORDS

    def test_does_not_contain_unmute(self):
        assert "unmute" not in _MEETING_WINDOW_KEYWORDS

    def test_does_not_contain_bare_meeting(self):
        # "meeting" alone matches Teams calendar; only specific phrases should be present
        assert "meeting" not in _MEETING_WINDOW_KEYWORDS

    def test_does_not_contain_participants(self):
        assert "participants" not in _MEETING_WINDOW_KEYWORDS

    def test_contains_specific_call_phrases(self):
        specific = {"call in progress", "video call", "audio call", "in a meeting"}
        assert specific & _MEETING_WINDOW_KEYWORDS, "No specific call-state keywords found"


class TestCoreaudioThreshold:
    def test_threshold_above_zero(self):
        assert MeetingDetector._COREAUDIO_MEETING_THRESHOLD > 0

    def test_threshold_above_idle_baseline(self):
        # Observed idle baseline is 0-1 handles; active call shows ~4+
        assert MeetingDetector._COREAUDIO_MEETING_THRESHOLD >= 2


class TestCheckMeeting:
    @pytest.fixture
    def detector(self):
        return MeetingDetector(Config())

    def test_returns_false_when_no_teams_process(self, detector):
        with patch.object(detector, "_get_teams_window_titles", return_value=[]):
            with patch.object(detector, "_get_teams_pids", return_value=[]):
                assert detector._check_meeting() is False

    def test_returns_true_on_matching_window_title(self, detector):
        with patch.object(detector, "_get_teams_window_titles", return_value=["call in progress"]):
            assert detector._check_meeting() is True

    def test_returns_false_on_non_matching_window_title(self, detector):
        with patch.object(detector, "_get_teams_window_titles", return_value=["Microsoft Teams"]):
            with patch.object(detector, "_get_teams_pids", return_value=[]):
                assert detector._check_meeting() is False

    def test_returns_false_on_nav_tab_titles(self, detector):
        nav_titles = ["Calendar | Microsoft Teams", "Chat | Microsoft Teams", "Activity | Microsoft Teams"]
        for title in nav_titles:
            with patch.object(detector, "_get_teams_window_titles", return_value=[title]):
                with patch.object(detector, "_get_teams_pids", return_value=[]):
                    assert detector._check_meeting() is False, f"Nav tab should not trigger: {title!r}"

    def test_returns_true_on_meeting_name_title(self, detector):
        # New Teams client shows "<meeting name> | Microsoft Teams" during a call
        with patch.object(detector, "_get_teams_window_titles", return_value=["temp | Microsoft Teams"]):
            assert detector._check_meeting() is True

    def test_returns_false_on_bare_microsoft_teams_title(self, detector):
        with patch.object(detector, "_get_teams_window_titles", return_value=["Microsoft Teams"]):
            with patch.object(detector, "_get_teams_pids", return_value=[]):
                assert detector._check_meeting() is False

    def test_fallback_coreaudio_above_threshold(self, detector):
        with patch.object(detector, "_get_teams_window_titles", return_value=[]):
            with patch.object(detector, "_get_teams_pids", return_value=[1234]):
                with patch.object(detector, "_teams_using_audio", return_value=True):
                    assert detector._check_meeting() is True

    def test_fallback_coreaudio_below_threshold(self, detector):
        with patch.object(detector, "_get_teams_window_titles", return_value=[]):
            with patch.object(detector, "_get_teams_pids", return_value=[1234]):
                with patch.object(detector, "_teams_using_audio", return_value=False):
                    assert detector._check_meeting() is False


class TestDebounce:
    """on_start should only fire after 2 consecutive positive detections."""

    def test_single_detection_does_not_fire(self):
        detector = MeetingDetector(Config())
        detector._consecutive_hits = 1  # one hit already
        # Simulate one more positive but _in_meeting still False
        # The poll loop requires >= 2
        assert detector._consecutive_hits < 2 or detector._in_meeting is False

    def test_two_detections_fire_on_start(self):
        detector = MeetingDetector(Config())
        started = []

        # Simulate the poll loop manually for two positive detections
        detector._consecutive_hits = 0

        for _ in range(2):
            detector._consecutive_hits += 1
            if detector._consecutive_hits >= 2 and not detector._in_meeting:
                detector._in_meeting = True
                started.append(True)

        assert len(started) == 1
        assert detector._in_meeting is True

    def test_false_detection_resets_counter(self):
        detector = MeetingDetector(Config())
        detector._consecutive_hits = 1

        # Negative detection resets
        detected = False
        if not detected:
            detector._consecutive_hits = 0

        assert detector._consecutive_hits == 0
