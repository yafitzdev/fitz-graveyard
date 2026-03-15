# Reasoning Benchmark (5 runs)

## Timing
- Min: 360s
- Max: 684s
- Avg: 476s

## Success: 5/5

## Architecture Decisions
| Run | Recommended | Time | Size |
|-----|-------------|------|------|
| 1 | Direct Provider Implementation | 684.0s | 65570B |
| 2 | Protocol Extension with Optional Methods | 428.0s | 96914B |
| 3 | Wrapper Injection Approach | 521.5s | 50962B |
| 4 | Instrumentation-Based Tracking | 359.9s | 49977B |
| 5 | Extend Provider Protocols with Structured Responses | 387.1s | 76432B |

## Decision Frequency
| Approach | Count | % |
|----------|-------|---|
| Direct Provider Implementation | 1 | 20% |
| Protocol Extension with Optional Methods | 1 | 20% |
| Wrapper Injection Approach | 1 | 20% |
| Instrumentation-Based Tracking | 1 | 20% |
| Extend Provider Protocols with Structured Responses | 1 | 20% |

## Avg Stage Timings
| Stage | Avg | Min | Max |
|-------|-----|-----|-----|
| agent_gathering | 7s | 6s | 9s |
| architecture_design | 313s | 243s | 501s |
| coherence_check | 4s | 2s | 5s |
| context | 51s | 41s | 62s |
| implementation_check | 3s | 3s | 4s |
| roadmap_risk | 98s | 42s | 201s |