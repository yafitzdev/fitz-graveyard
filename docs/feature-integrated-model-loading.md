# Feature: Integrated Model Loading

**Status:** Proposed — waiting for medium-scope quality improvements to be validated first.

## User Experience

```
$ fitz-graveyard plan "build an openai chat llm plugin"

Available models:
  1. qwen3-coder-30b-q8.gguf (32.1 GB)  [local]
  2. qwen3.5-35b-a3b-q6.gguf (18.4 GB)  [local]
  3. qwen/qwen3-coder-30b               [lm_studio — running]

Select model [1]: 1
Loading qwen3-coder-30b-q8.gguf... ready (14.2s)
Planning...
```

No separate LM Studio or Ollama window. One command, pick a number, plan runs.

## How It Works

1. **Discovery** — CLI scans two sources:
   - Local GGUF files in a configurable directory (default: `~/.fitz-graveyard/models/`)
   - Running LM Studio / Ollama servers (existing health check + model listing)

2. **Selection** — Numbered menu, user picks one. Default is last-used model (stored in config).

3. **Loading** — Depends on source:
   - **Local GGUF**: Spawn `llama-server --model <path> --port <free-port> --n-gpu-layers 99` as subprocess. Wait for health check. Connect with existing `LMStudioClient` (already OpenAI-compatible).
   - **Running server**: Connect directly with existing client code. No loading needed.

4. **Lifecycle** — llama-server subprocess lives for the duration of the plan. Killed on completion or Ctrl+C. PID stored in config dir for cleanup.

## Why llama-server (not in-process llama-cpp-python)

- Existing `LMStudioClient` works unchanged — OpenAI-compatible API, zero new LLM client code
- Best inference speed (pure C++, no Python GIL)
- Process isolation — model crash doesn't kill the CLI
- Same binary works on Windows/Linux/Mac

## Config

```yaml
# New fields in config.yaml
models:
  search_dirs:
    - "~/.fitz-graveyard/models/"    # default
    - "D:/models/"                    # user can add more
  last_used: "qwen3-coder-30b-q8.gguf"
  llama_server_path: null             # auto-detect from PATH, or explicit path
  default_gpu_layers: 99              # -ngl flag, 99 = all layers on GPU
  default_port: 0                     # 0 = find free port automatically
```

## Open Questions

- **Model downloading**: Should `fitz-graveyard download <hf-repo>` pull GGUFs from HuggingFace? Or keep it simple — user downloads manually, points config at the directory? Start simple (scan dir), add download later if needed.
- **Multiple GPU configs**: Should the menu show estimated VRAM per model? Would need to parse GGUF metadata for parameter count + quantization level.
- **Persistent server mode**: Should `fitz-graveyard serve-model <path>` keep llama-server running across multiple plans? Avoids cold start on every `plan` command. Trade-off: more complexity, daemon management.
- **Ollama integration**: Ollama also manages models well. Could offer `fitz-graveyard pull <model>` that delegates to `ollama pull`. But that adds Ollama as a dependency for this path.

## Affected Files (estimated)

| File | Change |
|------|--------|
| `fitz_graveyard/cli.py` | Model selection prompt before `plan` command |
| `fitz_graveyard/config/schema.py` | New `ModelsConfig` section |
| `fitz_graveyard/llm/discovery.py` | **New** — scan dirs for GGUFs, query running servers |
| `fitz_graveyard/llm/llama_server.py` | **New** — subprocess management (spawn, health wait, kill) |
| `fitz_graveyard/background/worker.py` | Accept dynamically-created client from CLI |
| Tests | Discovery, subprocess lifecycle, model selection |

## Dependencies

- `llama-server` binary must be installed (from llama.cpp releases or `pip install llama-cpp-python[server]`)
- No new Python dependencies for the core path (subprocess + existing httpx/openai)

## Priority

Low-medium. Current setup (LM Studio running separately) works. This is a UX improvement that removes the "start LM Studio, load model, then run CLI" friction. Implement after validating that the diagnostics and prompt quality changes are producing better plans.
