# meetscribe

**Local meeting transcription — both sides, speaker identification, automatic notes. Nothing leaves your machine.**

meetscribe runs in your terminal and automatically detects when you join a Microsoft Teams meeting. It captures audio from both you and remote participants, transcribes everything in real time using [Whisper](https://github.com/openai/whisper), identifies individual speakers, and after the meeting generates notes, action items, and a summary using a local language model via [Ollama](https://ollama.com). No cloud. No subscriptions. No data leaves your machine.

---

## Features

- **Auto-detection** — watches for an active Teams meeting and starts recording automatically
- **Both sides captured** — records your microphone and remote participants simultaneously
- **Works with headphones** — audio routing via BlackHole intercepts Teams audio at the software level, so headphones vs. speakers makes no difference
- **Speaker identification** — labels utterances as "You", "Participant 1", "Participant 2", etc. using voice embeddings
- **Real-time transcript** — see what is being said as the meeting progresses
- **Post-meeting generation** — choose to generate meeting notes, a to-do list, a summary, or all three
- **Saves as Markdown** — output is saved to a configurable folder as dated `.md` files
- **Fully local** — Whisper and Ollama both run on your machine; no API keys required
- **Manual mode** — skip Teams detection and start recording immediately with `--manual`

---

## Requirements

| Requirement | Notes |
|---|---|
| macOS 12+ | Uses macOS-native audio and AppleScript APIs |
| Python 3.11+ | |
| [Ollama](https://ollama.com) | Local language model server |
| [BlackHole 2ch](https://existingcircuits.com/products/blackhole) | Virtual audio driver for capturing system audio |

---

## Why BlackHole?

macOS routes audio to your output device (headphones or speakers) at the hardware level. There is no built-in way to intercept what is playing to your headphones in software.

**BlackHole** solves this by creating a virtual audio device that acts as a software loopback. You set up a *Multi-Output Device* in macOS Audio MIDI Setup that sends audio to both BlackHole and your real output simultaneously. Teams audio then appears at the BlackHole input device, which meetscribe captures — regardless of whether you are using headphones or speakers.

**One-time setup (~2 minutes):**

1. Install BlackHole: `brew install blackhole-2ch`
2. Open **Audio MIDI Setup** (find via Spotlight)
3. Click **+** at the bottom left → **Create Multi-Output Device**
4. Check both **BlackHole 2ch** and your real output device (headphones or speakers)
5. Right-click the new device → **Use This Device For Sound Output**
6. In Teams → Settings → Devices → set **Speaker** to the same Multi-Output Device
7. meetscribe will now capture remote audio from the BlackHole input device

The first-run wizard walks through all of these steps.

---

## Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/meetscribe.git
cd meetscribe

# Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Install BlackHole (if not already installed)
brew install blackhole-2ch
```

---

## First Run

```bash
python main.py
```

On the first launch, an interactive wizard guides you through:

1. macOS permissions (microphone, accessibility)
2. BlackHole installation and Multi-Output Device setup
3. Microphone selection
4. Notes save location
5. Ollama installation and model selection
6. Whisper model size selection
7. Speaker identification toggle

Settings are saved to `~/.config/meetscribe/config.yaml` and the wizard only runs once.

---

## Usage

```bash
# Auto-detect Teams meetings (default)
python main.py

# Start recording immediately without waiting for Teams
python main.py --manual

# Re-run the setup wizard
python main.py --setup

# Override the language model for this session
python main.py --model mistral

# Override the Whisper model for this session
python main.py --whisper small
```

meetscribe prints a live transcript as the meeting progresses. When the meeting ends (or you press Enter in manual mode), it shows a menu:

```
Generate:
  1  Meeting notes
  2  To-do tasks
  3  Summary
  4  All three
  5  Skip generation
```

The result is printed to the terminal and saved as a Markdown file in your configured notes folder.

---

## Configuration

Settings live at `~/.config/meetscribe/config.yaml`:

```yaml
notes_dir: ~/Documents/MeetScribe
whisper_model: base          # tiny | base | small | medium | large-v3
ollama_model: llama3.2
ollama_host: http://localhost:11434
speaker_diarization: true
audio:
  mic_device: null           # null = system default
  system_device: null        # null = auto-detect BlackHole
  sample_rate: 16000
  chunk_seconds: 20          # transcription interval
save_audio: false
save_transcript: true
poll_interval: 5             # seconds between Teams checks
```

---

## Privacy

- All audio processing happens on your machine via Whisper (CPU inference)
- Language model generation uses Ollama running locally
- No audio, transcripts, or generated text are sent anywhere
- Saved files go to your configured `notes_dir`; nothing is uploaded

---

## Supported Models

**Whisper (transcription):**

| Model | RAM | Speed |
|---|---|---|
| tiny | ~40 MB | Fastest |
| base | ~75 MB | Good balance (default) |
| small | ~245 MB | More accurate |
| medium | ~770 MB | High accuracy |
| large-v3 | ~1.5 GB | Best quality |

**Ollama (generation):** any model available via `ollama list`. Recommended: `llama3.2` (fast, good quality) or `mistral`.

---

## License

MIT — see [LICENSE](LICENSE).
