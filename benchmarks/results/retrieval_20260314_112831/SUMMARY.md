# Retrieval Benchmark (3 runs)

## Timing
- Min: 11.4s
- Max: 12.6s
- Avg: 11.8s

## File Counts
- Total files: 30-30
- Scan hits: 11-11

## Scan Hit Frequency (across 3 runs)
| File | Hits | % |
|------|------|---|
| fitz_ai/llm/providers/base.py | 3/3 | 100% |
| fitz_ai/llm/providers/openai.py | 3/3 | 100% |
| fitz_ai/llm/providers/anthropic.py | 3/3 | 100% |
| fitz_ai/llm/providers/cohere.py | 3/3 | 100% |
| fitz_ai/llm/providers/ollama.py | 3/3 | 100% |
| fitz_ai/core/instrumentation.py | 3/3 | 100% |
| fitz_ai/logging/logger.py | 3/3 | 100% |
| fitz_ai/logging/tags.py | 3/3 | 100% |
| fitz_ai/engines/fitz_krag/engine.py | 3/3 | 100% |
| fitz_ai/sdk/fitz.py | 3/3 | 100% |
| fitz_ai/api/routes/query.py | 3/3 | 100% |

## All Selected Files Frequency
| File | Hits | % | Signal |
|------|------|---|--------|
| fitz_ai/llm/providers/base.py | 3/3 | 100% | scan |
| fitz_ai/llm/providers/openai.py | 3/3 | 100% | scan |
| fitz_ai/llm/providers/anthropic.py | 3/3 | 100% | scan |
| fitz_ai/llm/providers/cohere.py | 3/3 | 100% | scan |
| fitz_ai/llm/providers/ollama.py | 3/3 | 100% | scan |
| fitz_ai/core/instrumentation.py | 3/3 | 100% | scan |
| fitz_ai/logging/logger.py | 3/3 | 100% | scan |
| fitz_ai/logging/tags.py | 3/3 | 100% | scan |
| fitz_ai/engines/fitz_krag/engine.py | 3/3 | 100% | scan |
| fitz_ai/sdk/fitz.py | 3/3 | 100% | scan |
| fitz_ai/api/routes/query.py | 3/3 | 100% | scan |
| fitz_ai/llm/auth/__init__.py | 3/3 | 100% | import |
| fitz_ai/llm/auth/httpx_auth.py | 3/3 | 100% | import |
| fitz_ai/utils/logging.py | 3/3 | 100% | import |
| fitz_ai/engines/fitz_krag/progressive/worker.py | 3/3 | 100% | import |
| fitz_ai/engines/fitz_krag/ingestion/import_graph_store.py | 3/3 | 100% | import |
| fitz_ai/logging/__init__.py | 3/3 | 100% | import |
| fitz_ai/engines/fitz_krag/ingestion/schema.py | 3/3 | 100% | import |
| fitz_ai/cloud/client.py | 3/3 | 100% | import |
| fitz_ai/engines/fitz_krag/config/schema.py | 3/3 | 100% | import |
| fitz_ai/retrieval/vocabulary/matcher.py | 3/3 | 100% | import |
| fitz_ai/engines/fitz_krag/retrieval/table_handler.py | 3/3 | 100% | import |
| fitz_ai/storage/postgres.py | 3/3 | 100% | import |
| fitz_ai/engines/fitz_krag/progressive/builder.py | 3/3 | 100% | import |
| fitz_ai/engines/fitz_krag/ingestion/strategies/java.py | 3/3 | 100% | import |
| fitz_ai/llm/factory.py | 3/3 | 100% | import |
| fitz_ai/tabular/store/postgres.py | 3/3 | 100% | import |
| fitz_ai/engines/fitz_krag/query_analyzer.py | 3/3 | 100% | import |
| fitz_ai/retrieval/entity_graph/store.py | 3/3 | 100% | import |
| fitz_ai/governance/constraints/feature_extractor.py | 3/3 | 100% | import |

## Critical File Discovery
| File | Found | % |
|------|-------|---|
| fitz_ai/engines/fitz_krag/engine.py | 3/3 | 100% |
| fitz_ai/core/answer.py | 0/3 | 0% |
| fitz_ai/engines/fitz_krag/query_analyzer.py | 3/3 | 100% |
| fitz_ai/retrieval/detection/registry.py | 0/3 | 0% |