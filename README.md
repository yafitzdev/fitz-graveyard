# fitz-graveyard

[![PyPI version](https://img.shields.io/pypi/v/fitz-graveyard.svg)](https://pypi.org/project/fitz-graveyard/)
[![Python versions](https://img.shields.io/pypi/pyversions/fitz-graveyard.svg)](https://pypi.org/project/fitz-graveyard/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/yafitzdev/fitz-graveyard/workflows/tests/badge.svg)](https://github.com/yafitzdev/fitz-graveyard/actions)

MCP server for overnight AI architectural planning using local LLMs.

## Features

- Local-first planning with Ollama (Qwen Coder Next 80B/32B)
- Five-stage pipeline: context, architecture, design, roadmap, risk
- SQLite job queue with crash recovery and checkpoint resume
- KRAG-powered codebase context via fitz-ai integration
- Per-section confidence scoring with optional API review
- Cross-platform: Windows, macOS, Linux

## Prerequisites

- Python 3.10+
- Ollama installed and running ([https://ollama.com](https://ollama.com))
- Qwen model pulled:
  ```bash
  ollama pull qwen2.5-coder:32b
  ```

## Installation

```bash
pip install fitz-graveyard
```

For API review feature:

```bash
pip install "fitz-graveyard[api-review]"
```

## Usage - MCP Server (Claude Code)

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "fitz-planner": {
      "command": "fitz-graveyard"
    }
  }
}
```

Restart Claude Desktop, then use planning tools directly in conversations.

## Available Tools

| Tool | Description |
|------|-------------|
| `create_plan` | Queue a new planning job |
| `check_status` | Check job progress |
| `get_plan` | Retrieve completed plan |
| `list_plans` | List all planning jobs |
| `retry_job` | Retry a failed job |
| `confirm_review` | Approve API review after seeing cost |
| `cancel_review` | Skip API review, finalize plan |

## Configuration

Default configuration location:

- **Windows**: `%APPDATA%\fitz-graveyard\config.yaml`
- **macOS**: `~/Library/Application Support/fitz-graveyard/config.yaml`
- **Linux**: `~/.config/fitz-graveyard/config.yaml`

Key settings (YAML):

```yaml
llm:
  provider: ollama
  model: qwen2.5-coder:32b
  base_url: http://localhost:11434

planning:
  confidence_threshold: 0.75
  max_retries: 3
  checkpoint_interval: 60

api_review:
  provider: anthropic  # Optional: for confidence scoring
  model: claude-opus-4
  enabled: false
```

## Development

```bash
git clone https://github.com/yafitzdev/fitz-graveyard.git
cd fitz-graveyard
pip install -e ".[dev]"
pytest
```

## Architecture

```
fitz_graveyard/
├── server.py           # MCP server entry point
├── engine/             # Planning orchestration
│   ├── stages/         # Five-stage pipeline
│   └── checkpoints/    # Crash recovery
├── queue/              # SQLite job queue
├── context/            # fitz-ai KRAG integration
├── review/             # Optional API review
└── config/             # Configuration management
```

## License

[MIT](LICENSE)
