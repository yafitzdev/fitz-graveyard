# Retrieval Benchmark (10 runs)

## Timing
- Min: 11.9s
- Max: 27.7s
- Avg: 13.8s

## File Counts
- Total files: 30-30
- Scan hits: 11-11

## Scan Hit Frequency (across 10 runs)
| File | Hits | % |
|------|------|---|
| fitz_ai/llm/providers/base.py | 10/10 | 100% |
| fitz_ai/llm/providers/openai.py | 10/10 | 100% |
| fitz_ai/llm/providers/anthropic.py | 10/10 | 100% |
| fitz_ai/llm/providers/cohere.py | 10/10 | 100% |
| fitz_ai/llm/providers/ollama.py | 10/10 | 100% |
| fitz_ai/core/instrumentation.py | 10/10 | 100% |
| fitz_ai/logging/logger.py | 10/10 | 100% |
| fitz_ai/logging/tags.py | 10/10 | 100% |
| fitz_ai/engines/fitz_krag/engine.py | 10/10 | 100% |
| fitz_ai/sdk/fitz.py | 10/10 | 100% |
| fitz_ai/api/routes/query.py | 10/10 | 100% |

## All Selected Files Frequency
| File | Hits | % | Signal |
|------|------|---|--------|
| fitz_ai/llm/providers/base.py | 10/10 | 100% | scan |
| fitz_ai/llm/providers/openai.py | 10/10 | 100% | scan |
| fitz_ai/llm/providers/anthropic.py | 10/10 | 100% | scan |
| fitz_ai/llm/providers/cohere.py | 10/10 | 100% | scan |
| fitz_ai/llm/providers/ollama.py | 10/10 | 100% | scan |
| fitz_ai/core/instrumentation.py | 10/10 | 100% | scan |
| fitz_ai/logging/logger.py | 10/10 | 100% | scan |
| fitz_ai/logging/tags.py | 10/10 | 100% | scan |
| fitz_ai/engines/fitz_krag/engine.py | 10/10 | 100% | scan |
| fitz_ai/sdk/fitz.py | 10/10 | 100% | scan |
| fitz_ai/api/routes/query.py | 10/10 | 100% | scan |
| fitz_ai/engines/fitz_krag/ingestion/pipeline.py | 10/10 | 100% | ? |
| fitz_ai/ingestion/diff/executor.py | 10/10 | 100% | ? |
| fitz_ai/ingestion/enrichment/pipeline.py | 10/10 | 100% | ? |
| tests/integration/cloud_fixtures.py | 10/10 | 100% | ? |
| tools/governance/eval_pipeline.py | 10/10 | 100% | ? |
| tools/governance/extract_features.py | 10/10 | 100% | ? |
| fitz_ai/cli/commands/eval.py | 10/10 | 100% | ? |
| fitz_ai/llm/auth/token_provider.py | 10/10 | 100% | ? |
| fitz_ai/llm/auth/base.py | 10/10 | 100% | ? |
| fitz_ai/llm/auth/certificates.py | 10/10 | 100% | ? |
| fitz_ai/llm/auth/m2m.py | 10/10 | 100% | ? |
| fitz_ai/llm/auth/api_key.py | 10/10 | 100% | ? |
| fitz_ai/llm/auth/composite.py | 10/10 | 100% | ? |
| fitz_ai/core/paths/workspace.py | 10/10 | 100% | ? |
| fitz_ai/core/answer.py | 10/10 | 100% | ? |
| fitz_ai/core/constraints.py | 10/10 | 100% | ? |
| fitz_ai/core/utils.py | 10/10 | 100% | ? |
| fitz_ai/core/provenance.py | 10/10 | 100% | ? |
| fitz_ai/core/chunk.py | 10/10 | 100% | ? |

## Critical File Discovery
| File | Found | % |
|------|-------|---|
| fitz_ai/engines/fitz_krag/engine.py | 10/10 | 100% |
| fitz_ai/core/answer.py | 10/10 | 100% |
| fitz_ai/engines/fitz_krag/query_analyzer.py | 0/10 | 0% |
| fitz_ai/retrieval/detection/registry.py | 0/10 | 0% |