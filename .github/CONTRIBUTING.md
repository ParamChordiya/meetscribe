# Contributing to meetscribe

Thanks for your interest in contributing. This document covers how to get set up, the conventions we follow, and the process for submitting changes.

---

## Development setup

```bash
git clone https://github.com/ParamChordiya/meetscribe.git
cd meetscribe

python3 -m venv .venv
source .venv/bin/activate

# Install with dev extras
pip install -e ".[dev]"
```

### System requirements

- macOS 12+
- Python 3.11+
- [BlackHole 2ch](https://existingcircuits.com/products/blackhole) for audio routing tests that need hardware
- [Ollama](https://ollama.com) for integration tests

---

## Running tests

```bash
# All tests
pytest

# With coverage
pytest --cov=meetscribe --cov-report=term-missing

# A specific file
pytest tests/test_config.py -v
```

Tests mock all heavy dependencies (Whisper, resemblyzer, sounddevice, torch) so they run without GPU or audio hardware. CI runs on ubuntu-latest for this reason.

---

## Code style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# Check
ruff check .

# Fix auto-fixable issues
ruff check . --fix
```

The CI will fail if `ruff check .` reports any errors.

**Conventions:**
- Python 3.11+, full type hints on all functions
- No comments explaining *what* code does — only *why* when non-obvious
- No docstrings on private methods
- Prefer explicit over implicit

---

## Submitting a pull request

1. Fork the repository and create a branch from `main`
2. Make your changes with tests where applicable
3. Run `ruff check .` and `pytest` locally — both must pass
4. Open a PR against `main` with a clear description
5. Fill in the PR template

Branch naming suggestions:
- `fix/description-of-fix`
- `feat/description-of-feature`
- `docs/what-changed`

---

## Reporting bugs

Use the [bug report template](https://github.com/ParamChordiya/meetscribe/issues/new?template=bug_report.yml). Include the full error output and your environment details.

## Suggesting features

Use the [feature request template](https://github.com/ParamChordiya/meetscribe/issues/new?template=feature_request.yml). For larger ideas, open a [discussion](https://github.com/ParamChordiya/meetscribe/discussions) first.

---

## Project structure

```
meetscribe/
├── meetscribe/
│   ├── audio/          # Dual-channel capture (mic + BlackHole)
│   ├── transcription/  # Whisper engine + speaker diarization
│   ├── detection/      # Teams meeting detector
│   ├── llm/            # Ollama client + prompt templates
│   └── cli/            # Terminal app + first-run wizard
├── tests/              # pytest test suite
└── .github/            # Workflows, templates, community files
```

---

## Licence

By contributing you agree that your contributions will be licensed under the [MIT Licence](../LICENSE).
