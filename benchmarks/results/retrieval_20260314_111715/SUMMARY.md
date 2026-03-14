# Retrieval Benchmark (3 runs)

## Timing
- Min: 13.0s
- Max: 13.4s
- Avg: 13.2s

## File Counts
- Total files: 30-30
- Scan hits: 12-12

## Scan Hit Frequency (across 3 runs)
| File | Hits | % |
|------|------|---|
| fitz_ai/llm/providers/base.py | 3/3 | 100% |
| fitz_ai/llm/providers/openai.py | 3/3 | 100% |
| fitz_ai/llm/providers/anthropic.py | 3/3 | 100% |
| fitz_ai/llm/providers/cohere.py | 3/3 | 100% |
| fitz_ai/llm/providers/ollama.py | 3/3 | 100% |
| fitz_ai/core/query.py | 3/3 | 100% |
| fitz_ai/core/engine.py | 3/3 | 100% |
| fitz_ai/governance/governor.py | 3/3 | 100% |
| fitz_ai/services/fitz_service.py | 3/3 | 100% |
| fitz_ai/api/routes/query.py | 3/3 | 100% |
| fitz_ai/api/models/schemas.py | 3/3 | 100% |
| fitz_ai/logging/logger.py | 3/3 | 100% |

## All Selected Files Frequency
| File | Hits | % | Signal |
|------|------|---|--------|
| fitz_ai/llm/providers/base.py | 3/3 | 100% | scan |
| fitz_ai/llm/providers/openai.py | 3/3 | 100% | scan |
| fitz_ai/llm/providers/anthropic.py | 3/3 | 100% | scan |
| fitz_ai/llm/providers/cohere.py | 3/3 | 100% | scan |
| fitz_ai/llm/providers/ollama.py | 3/3 | 100% | scan |
| fitz_ai/core/query.py | 3/3 | 100% | scan |
| fitz_ai/core/engine.py | 3/3 | 100% | scan |
| fitz_ai/governance/governor.py | 3/3 | 100% | scan |
| fitz_ai/services/fitz_service.py | 3/3 | 100% | scan |
| fitz_ai/api/routes/query.py | 3/3 | 100% | scan |
| fitz_ai/api/models/schemas.py | 3/3 | 100% | scan |
| fitz_ai/logging/logger.py | 3/3 | 100% | scan |
| fitz_ai/llm/auth/__init__.py | 3/3 | 100% | import |
| fitz_ai/llm/auth/httpx_auth.py | 3/3 | 100% | import |
| fitz_ai/core/answer_mode.py | 3/3 | 100% | import |
| fitz_ai/governance/constraints/base.py | 3/3 | 100% | import |
| fitz_ai/cli/context.py | 3/3 | 100% | import |
| fitz_ai/vector_db/registry.py | 3/3 | 100% | import |
| fitz_ai/runtime/__init__.py | 3/3 | 100% | import |
| fitz_ai/core/paths/__init__.py | 3/3 | 100% | import |
| fitz_ai/retrieval/rewriter/types.py | 3/3 | 100% | import |
| fitz_ai/core/__init__.py | 3/3 | 100% | import |
| fitz_ai/llm/__init__.py | 3/3 | 100% | import |
| fitz_ai/api/dependencies.py | 3/3 | 100% | import |
| fitz_ai/api/error_handlers.py | 3/3 | 100% | import |
| fitz_ai/utils/logging.py | 3/3 | 100% | import |
| fitz_ai/engines/fitz_krag/engine.py | 3/3 | 100% | ? |
| fitz_ai/engines/fitz_krag/ingestion/pipeline.py | 3/3 | 100% | ? |
| fitz_ai/engines/fitz_krag/progressive/worker.py | 3/3 | 100% | ? |
| fitz_ai/ingestion/diff/executor.py | 3/3 | 100% | ? |

## Critical File Discovery
| File | Found | % |
|------|-------|---|
| fitz_ai/engines/fitz_krag/engine.py | 3/3 | 100% |
| fitz_ai/core/answer.py | 0/3 | 0% |
| fitz_ai/engines/fitz_krag/query_analyzer.py | 0/3 | 0% |
| fitz_ai/retrieval/detection/registry.py | 0/3 | 0% |