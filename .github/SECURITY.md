# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| Latest (`main`) | Yes |

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, use [GitHub's private security advisory feature](https://github.com/ParamChordiya/meetscribe/security/advisories/new).

Include:
- A description of the vulnerability and its potential impact
- Steps to reproduce
- Any suggested fixes if you have them

You can expect an acknowledgement within 48 hours and a resolution timeline within 7 days for confirmed issues.

## Scope

meetscribe runs entirely locally. There are no servers, no authentication flows, and no network endpoints beyond:
- Downloading Whisper models from HuggingFace (on first run)
- Communicating with a locally-running Ollama server

Reports related to the local Ollama server or HuggingFace infrastructure are out of scope.
