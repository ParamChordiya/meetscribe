from meetscribe.types import Utterance


def test_utterance_required_fields():
    u = Utterance(text="Hello", speaker="You", timestamp=0.0)
    assert u.text == "Hello"
    assert u.speaker == "You"
    assert u.timestamp == 0.0


def test_utterance_default_confidence():
    u = Utterance(text="Hi", speaker="Participant 1", timestamp=5.5)
    assert u.confidence == 1.0


def test_utterance_custom_confidence():
    u = Utterance(text="Hi", speaker="You", timestamp=1.0, confidence=0.8)
    assert u.confidence == 0.8


def test_utterance_timestamp_zero():
    u = Utterance(text="Start", speaker="You", timestamp=0.0)
    assert u.timestamp == 0.0
