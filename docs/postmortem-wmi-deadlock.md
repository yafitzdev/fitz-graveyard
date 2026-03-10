# Post-Mortem: WMI Deadlock & Import Hang (2026-03-10)

## Incident

`fitz-graveyard plan` hung immediately after showing the TUI. No error, no output, infinite hang.

## Root Causes (two separate bugs)

### Bug 1: `platform._wmi_query()` deadlock

On this Windows system (Microsoft Store Python 3.12), the WMI service is broken/slow.
`platform._wmi_query()` blocks indefinitely when called.

The `openai` SDK calls `platform.machine()` from a background thread (in `_base_client.py`
`get_platform()`) to build User-Agent headers. This triggers `platform.uname()` →
`platform._wmi_query()` → deadlock. The main thread can't proceed because the background
thread holds resources.

### Bug 2: `ollama` package blocks on import

The `ollama` Python package creates a **default client instance** at module level
(`ollama/__init__.py` line 44). This client constructor calls `platform.machine()` →
same WMI deadlock as Bug 1. Even without the WMI issue, it also hangs when the Ollama
server isn't running because it tries to connect during initialization.

## What Was Changed (7 files)

| # | File | Change | Verdict |
|---|------|--------|---------|
| 1 | `__init__.py` | Monkey-patch `platform._wmi_query` to raise `OSError` | **Proper fix** (1 line) |
| 2 | `llm/factory.py` | `OllamaClient` import moved to `TYPE_CHECKING` + lazy runtime import | **Proper fix** |
| 3 | `llm/retry.py` | `from ollama import ResponseError` made lazy inside function | **Proper fix** |
| 4 | `llm/__init__.py` | Emptied (was re-exporting `OllamaClient`) | **Proper fix** |
| 5 | `background/worker.py` | `OllamaClient` → `TYPE_CHECKING`, isinstance → hasattr | **Proper fix** |
| 6 | `background/lifecycle.py` | `OllamaClient` → `TYPE_CHECKING` | **Proper fix** |

## Assessment

**All 6 changes are proper fixes.**

The WMI monkey-patch (`platform._wmi_query = raise OSError`) is clean because CPython's
`platform.py` already has `except OSError` fallbacks everywhere `_wmi_query` is called —
falling back to registry lookups and env vars. The output is correct: `platform.platform()`
returns `Windows-11-10.0.26200-SP0`, `platform.processor()` returns the full CPU string.
One line in `fitz_graveyard/__init__.py` kills all WMI deadlocks at the source.

The lazy import pattern (`TYPE_CHECKING` + `from __future__ import annotations`) is
standard Python. These changes correctly ensure that `import ollama` only happens when
`provider=ollama` is configured. This is how it should have been from the start —
importing an optional provider's SDK unconditionally is a design bug regardless of the
WMI issue.

## Why it looked like "so much stuff"

The `import ollama` chain was deeply embedded in the import graph through **four separate
paths**, each discovered one at a time:

```
cli.py → tools/ → factory.py → client.py → import ollama          (path 1)
cli.py → tools/ → factory.py → __init__.py → client.py            (path 2, via re-export)
cli.py → background/ → lifecycle.py → client.py → import ollama   (path 3)
cli.py → background/ → worker.py → client.py → import ollama      (path 4)
```

Each path had to be individually traced via thread dumps because the hang produced no
error message — just silence. The WMI workaround also had two subtle bugs that required
iterative fixing:

1. `uname_result()` doesn't accept keyword arguments (Python 3.12) — discovered via TypeError
2. `uname_result()` takes 5 args, not 6 (`processor` is a lazy `@cached_property`) — discovered via TypeError
3. `not getattr(platform, '_uname_cache', None)` evaluates truthiness on `uname_result`,
   which iterates all fields including the lazy `processor` property → triggers `_wmi_query()`
   again — discovered via thread dump showing deadlock in the guard condition itself

## What should have been done differently

### 1. Start with `_wmi_query` monkey-patch, not `_uname_cache` pre-population

The initial approach of pre-populating `_uname_cache` was whack-a-mole. It prevented
`platform.uname()` from calling WMI, but `platform.platform()` has additional WMI entry
points (the lazy `processor` property, `win32_ver()` via `_win32_ver()`). Each one
required a separate fix. Three iterations of the workaround failed before landing on the
correct approach: patch `_wmi_query` at the source.

CPython's `platform.py` already has `except OSError` fallbacks at every `_wmi_query` call
site, falling back to registry and env vars. Making `_wmi_query` raise `OSError` is not
a hack — it's using the existing error handling designed for exactly this scenario.

### 2. Provider isolation from day one

All provider-specific imports (`ollama`, `openai`) should have been lazy from the start.
A provider you're not using should never be imported. This is a design principle, not just
a workaround.

## Action Items

- [ ] Consider fixing WMI on the system itself (`winmgmt /resetrepository` or similar)
- [ ] Add integration test: import the full CLI with no LLM servers running, assert it doesn't hang
