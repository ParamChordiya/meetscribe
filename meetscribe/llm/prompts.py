from __future__ import annotations

from meetscribe.types import Utterance

MEETING_NOTES_PROMPT = """\
You are a professional note-taker. Generate comprehensive meeting notes from the transcript below.

Structure:
## Summary
2-3 sentence overview of the meeting.

## Discussion Topics
For each major topic: a heading and bullet points capturing key points.

## Decisions Made
Bullet list of concrete decisions (skip if none).

## Open Questions
Items raised but not resolved (skip if none).

Keep it factual and concise. Use markdown.

Transcript:
{transcript}"""

TODOS_PROMPT = """\
Extract all action items from this meeting transcript.

For each action item output:
- **Task**: what needs to be done (specific, actionable)
- **Owner**: who is responsible (use name if mentioned, otherwise "Unassigned")
- **Due**: deadline or timeframe (use "Not specified" if not mentioned)

Numbered list. Only include concrete tasks — ignore vague discussion.

Transcript:
{transcript}"""

SUMMARY_PROMPT = """\
Write a concise 3-5 sentence executive summary of this meeting.
Cover: the meeting's purpose, the key topics discussed, important decisions, and critical next steps.

Transcript:
{transcript}"""

PROMPTS: dict[str, str] = {
    "notes": MEETING_NOTES_PROMPT,
    "todos": TODOS_PROMPT,
    "summary": SUMMARY_PROMPT,
}


def format_transcript(utterances: list[Utterance]) -> str:
    """Format Utterance list as timestamped readable text."""
    lines = []
    for u in utterances:
        h = int(u.timestamp // 3600)
        m = int((u.timestamp % 3600) // 60)
        s = int(u.timestamp % 60)
        lines.append(f"[{h:02d}:{m:02d}:{s:02d}] {u.speaker}: {u.text}")
    return "\n".join(lines)
