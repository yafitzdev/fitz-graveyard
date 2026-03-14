# Retrieval Benchmark (3 runs)

## Timing
- Min: 11.5s
- Max: 13.1s
- Avg: 12.2s

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
| fitz_ai/engines/fitz_krag/ingestion/pipeline.py | 3/3 | 100% | ? |
| fitz_ai/ingestion/diff/executor.py | 3/3 | 100% | ? |
| fitz_ai/ingestion/enrichment/pipeline.py | 3/3 | 100% | ? |
| tests/integration/cloud_fixtures.py | 3/3 | 100% | ? |
| tools/governance/eval_pipeline.py | 3/3 | 100% | ? |
| tools/governance/extract_features.py | 3/3 | 100% | ? |
| fitz_ai/cli/commands/eval.py | 3/3 | 100% | ? |
| fitz_ai/llm/auth/composite.py | 3/3 | 100% | ? |
| fitz_ai/llm/auth/m2m.py | 3/3 | 100% | ? |
| fitz_ai/llm/auth/certificates.py | 3/3 | 100% | ? |
| fitz_ai/llm/auth/api_key.py | 3/3 | 100% | ? |
| fitz_ai/llm/auth/base.py | 3/3 | 100% | ? |
| fitz_ai/llm/auth/token_provider.py | 3/3 | 100% | ? |
| fitz_ai/sdk/__init__.py | 3/3 | 100% | ? |
| fitz_ai/core/engine.py | 3/3 | 100% | ? |
| fitz_ai/core/utils.py | 3/3 | 100% | ? |
| fitz_ai/core/constraints.py | 3/3 | 100% | ? |
| fitz_ai/core/answer.py | 3/3 | 100% | ? |
| fitz_ai/core/query.py | 3/3 | 100% | ? |

## Critical File Discovery
| File | Found | % |
|------|-------|---|
| fitz_ai/engines/fitz_krag/engine.py | 3/3 | 100% |
| fitz_ai/core/answer.py | 3/3 | 100% |
| fitz_ai/engines/fitz_krag/query_analyzer.py | 0/3 | 0% |
| fitz_ai/retrieval/detection/registry.py | 0/3 | 0% |