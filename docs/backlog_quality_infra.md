# Backlog: Infrastructure-Based Quality Improvements

Solutions 2 and 3 from the quality infrastructure plan. Solution 1 (reverse-import
caller expansion) is already implemented.

---

## Solution 2: Deterministic Verification Commands

### Problem

`ensure_concrete_verification` uses an LLM call to generate test/verification commands
from deliverable paths. The LLM often hallucinates commands or invents flags that don't
exist. This is a deterministic problem — given a file path, the verification command is
predictable.

### Design

Replace the LLM call with pure Python generation:

1. Parse deliverable paths from the roadmap output
2. Match against known test patterns:
   - `tests/**/*.py` -> `pytest <path>`
   - `*.py` with a known test runner -> `python -m pytest`
   - `*.js/*.ts` -> `npm test` / `jest <path>`
   - Config files -> `python -c "import yaml; yaml.safe_load(open('<path>'))"` (schema validation)
3. Generate `pytest`/`python -c`/shell commands deterministically
4. No LLM needed — 100% reliable output

### Files to Change

- `fitz_graveyard/planning/pipeline/validators.py` — replace LLM call in
  `ensure_concrete_verification` with deterministic command builder
- New helper: `_build_verification_command(path: str) -> str`
- Tests in `tests/unit/test_validators.py`

---

## Solution 3: Structured Risk Grounding

### Problem

`ensure_grounded_risks` uses substring matching to check if risks reference real
codebase artifacts. This is fragile — a risk mentioning "the authentication system"
passes if any file has "auth" in its path, even if the risk is completely conceptual.

### Design

Replace substring matching with entity-set validation:

1. Build a known-entities set from `prior_outputs`:
   - Component names from architecture output (e.g., "GovernanceEngine", "WebhookDispatcher")
   - File paths from deliverables (e.g., "fitz_ai/governance/engine.py")
   - Module names from the structural index
   - API/class names from design output
2. For each risk, check if it references at least one known entity (exact match or
   close match against the entity set)
3. Risks that are purely conceptual with no codebase anchor are flagged for replacement
4. The validator either strips ungrounded risks or regenerates them with entity context

### Files to Change

- `fitz_graveyard/planning/pipeline/validators.py` — replace `ensure_grounded_risks`
  with entity-set approach
- New helper: `_build_known_entities(prior_outputs: dict) -> set[str]`
- New helper: `_risk_references_entity(risk_text: str, entities: set[str]) -> bool`
- Tests in `tests/unit/test_validators.py`
