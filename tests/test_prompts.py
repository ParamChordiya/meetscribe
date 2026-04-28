from __future__ import annotations

from meetscribe.llm.prompts import PROMPTS, format_transcript
from meetscribe.types import Utterance


class TestFormatTranscript:
    def test_empty_list(self):
        assert format_transcript([]) == ""

    def test_single_utterance(self):
        u = Utterance(text="Hello everyone", speaker="You", timestamp=0.0)
        result = format_transcript([u])
        assert "[00:00:00] You: Hello everyone" == result

    def test_timestamp_formatting(self):
        u = Utterance(text="Hi", speaker="Participant 1", timestamp=3661.0)  # 1h 1m 1s
        result = format_transcript([u])
        assert "[01:01:01] Participant 1: Hi" == result

    def test_multiple_utterances_newline_separated(self):
        utterances = [
            Utterance(text="Hi", speaker="You", timestamp=0.0),
            Utterance(text="Hello", speaker="Participant 1", timestamp=5.0),
        ]
        result = format_transcript(utterances)
        lines = result.split("\n")
        assert len(lines) == 2
        assert "You: Hi" in lines[0]
        assert "Participant 1: Hello" in lines[1]

    def test_speaker_labels_preserved(self):
        utterances = [
            Utterance(text="A", speaker="You", timestamp=0.0),
            Utterance(text="B", speaker="Participant 1", timestamp=1.0),
            Utterance(text="C", speaker="Participant 2", timestamp=2.0),
        ]
        result = format_transcript(utterances)
        assert "You:" in result
        assert "Participant 1:" in result
        assert "Participant 2:" in result

    def test_timestamp_minutes_and_seconds(self):
        u = Utterance(text="X", speaker="You", timestamp=90.0)  # 1m 30s
        result = format_transcript([u])
        assert "[00:01:30]" in result


class TestPromptTemplates:
    def test_all_kinds_present(self):
        for kind in ("notes", "todos", "summary"):
            assert kind in PROMPTS

    def test_each_prompt_has_transcript_placeholder(self):
        for kind, template in PROMPTS.items():
            assert "{transcript}" in template, f"Prompt '{kind}' missing {{transcript}} placeholder"

    def test_prompts_are_non_empty_strings(self):
        for kind, template in PROMPTS.items():
            assert isinstance(template, str)
            assert len(template) > 50, f"Prompt '{kind}' seems too short"
