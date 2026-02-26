# fitz_graveyard/planning/pipeline/output.py
"""
Plan renderer for converting PipelineResult to structured markdown.

Converts pipeline outputs into a formatted markdown file with metadata frontmatter.
"""

from datetime import datetime
from typing import Any

from fitz_graveyard.planning.schemas.plan_output import PlanOutput


class PlanRenderer:
    """
    Converts PlanOutput to structured markdown format.

    Format:
        ---
        generated_at: ISO timestamp
        git_sha: commit hash
        overall_quality_score: float
        section_scores:
          context: float
          architecture: float
          ...
        ---

        # Project: {project_name}

        ## Context
        ...

        ## Architecture
        ...

        ## Design
        ...

        ## Roadmap
        ...

        ## Risk Analysis
        ...
    """

    def render(self, plan: PlanOutput, head_advanced: bool = False) -> str:
        """
        Render PlanOutput to markdown string.

        Args:
            plan: Validated PlanOutput with all stage outputs
            head_advanced: If True, HEAD changed during generation

        Returns:
            Formatted markdown string
        """
        sections = []

        # Frontmatter
        sections.append(self._render_frontmatter(plan))

        # Freshness warning
        if head_advanced:
            sections.append("> **WARNING**: Repository HEAD advanced during plan generation.")
            sections.append("> This plan may not reflect the latest codebase state.")
            sections.append("")

        # Title
        project_name = plan.context.project_description.split("\n")[0][:80]
        sections.append(f"# Project: {project_name}")
        sections.append("")

        # Context
        sections.append("## Context")
        sections.append("")
        sections.append(f"**Description:**\n{plan.context.project_description}")
        sections.append("")
        if plan.context.key_requirements:
            sections.append("**Requirements:**")
            for req in plan.context.key_requirements:
                sections.append(f"- {req}")
            sections.append("")
        if plan.context.constraints:
            sections.append("**Constraints:**")
            for constraint in plan.context.constraints:
                sections.append(f"- {constraint}")
            sections.append("")
        if plan.context.stakeholders:
            sections.append("**Stakeholders:**")
            for stakeholder in plan.context.stakeholders:
                sections.append(f"- {stakeholder}")
            sections.append("")

        if plan.context.existing_files:
            sections.append("**Existing Files:**")
            for f in plan.context.existing_files:
                sections.append(f"- {f}")
            sections.append("")

        if plan.context.needed_artifacts:
            sections.append("**Expected Deliverables:**")
            for a in plan.context.needed_artifacts:
                sections.append(f"- {a}")
            sections.append("")

        if plan.context.assumptions:
            sections.append("**Assumptions (verify these):**")
            for a in plan.context.assumptions:
                sections.append(f"- **{a.assumption}** [{a.confidence}] — if wrong: {a.impact}")
            sections.append("")

        # Architecture
        sections.append("## Architecture")
        sections.append("")
        sections.append("### Explored Approaches")
        for i, approach in enumerate(plan.architecture.approaches, start=1):
            sections.append(f"**{i}. {approach.name}**")
            sections.append(f"- Description: {approach.description}")
            sections.append(
                f"- Pros: {', '.join(approach.pros) if approach.pros else 'None listed'}"
            )
            sections.append(
                f"- Cons: {', '.join(approach.cons) if approach.cons else 'None listed'}"
            )
            sections.append("")

        sections.append(f"### Recommended: {plan.architecture.recommended}")
        sections.append(f"\n{plan.architecture.reasoning}")
        sections.append("")

        if plan.architecture.scope_statement:
            sections.append(f"*{plan.architecture.scope_statement}*")
            sections.append("")

        # Design
        sections.append("## Design")
        sections.append("")
        if plan.design.adrs:
            sections.append("### Architectural Decision Records")
            for adr in plan.design.adrs:
                sections.append(f"**ADR: {adr.title}**")
                sections.append(f"- Decision: {adr.decision}")
                sections.append(f"- Rationale: {adr.rationale}")
                sections.append(
                    f"- Alternatives: {', '.join(adr.alternatives_considered) if adr.alternatives_considered else 'None listed'}"
                )
                sections.append("")

        if plan.design.components:
            sections.append("### Components")
            for comp in plan.design.components:
                sections.append(f"**{comp.name}**")
                sections.append(f"- Purpose: {comp.purpose}")
                sections.append(f"- Responsibilities: {', '.join(comp.responsibilities) if comp.responsibilities else 'None listed'}")
                sections.append(
                    f"- Interfaces: {', '.join(comp.interfaces) if comp.interfaces else 'None listed'}"
                )
                sections.append("")

        if plan.design.data_model:
            sections.append("### Data Model")
            for entity, attrs in plan.design.data_model.items():
                sections.append(f"**{entity}**")
                for attr in attrs:
                    sections.append(f"- {attr}")
                sections.append("")

        if plan.design.artifacts:
            sections.append("### Artifacts")
            for artifact in plan.design.artifacts:
                sections.append(f"**`{artifact.filename}`**")
                if artifact.purpose:
                    sections.append(f"*{artifact.purpose}*")
                sections.append("")
                # Detect language from extension for syntax highlighting
                ext = artifact.filename.rsplit(".", 1)[-1] if "." in artifact.filename else ""
                lang_map = {"yaml": "yaml", "yml": "yaml", "sql": "sql", "py": "python",
                            "json": "json", "toml": "toml", "sh": "bash", "dockerfile": "dockerfile"}
                lang = lang_map.get(ext.lower(), "")
                sections.append(f"```{lang}")
                sections.append(artifact.content)
                sections.append("```")
                sections.append("")

        # Roadmap
        sections.append("## Roadmap")
        sections.append("")
        for i, phase in enumerate(plan.roadmap.phases, start=1):
            sections.append(f"### Phase {phase.number}: {phase.name}")
            sections.append(f"**Objective:** {phase.objective}")
            sections.append("")
            if phase.deliverables:
                sections.append("**Deliverables:**")
                for deliverable in phase.deliverables:
                    sections.append(f"- {deliverable}")
                sections.append("")
            if phase.estimated_effort:
                sections.append(f"**Effort:** {phase.estimated_effort}")
                sections.append("")
            if phase.dependencies:
                sections.append(f"**Dependencies:** Phases {', '.join(str(d) for d in phase.dependencies)}")
                sections.append("")
            if phase.verification_command:
                sections.append("**Verify:**")
                sections.append(f"```\n{phase.verification_command}\n```")
                sections.append("")

        if plan.roadmap.critical_path:
            sections.append("### Critical Path")
            for item in plan.roadmap.critical_path:
                sections.append(f"- {item}")
            sections.append("")

        # Risk Analysis
        sections.append("## Risk Analysis")
        sections.append("")
        for risk in plan.risk.risks:
            sections.append(f"### {risk.category.title()} Risk")
            sections.append(f"**Description:** {risk.description}")
            sections.append(f"**Impact:** {risk.impact} | **Likelihood:** {risk.likelihood}")
            sections.append(f"**Mitigation:** {risk.mitigation}")
            if risk.contingency:
                sections.append(f"**Contingency:** {risk.contingency}")
            if risk.verification:
                sections.append("**Verify:**")
                sections.append(f"```\n{risk.verification}\n```")
            if risk.affected_phases:
                sections.append(f"**Affected Phases:** {', '.join(str(p) for p in risk.affected_phases)}")
            sections.append("")

        # API Review (if requested)
        if plan.api_review_requested:
            sections.append("## API Review")
            sections.append("")

            # Check if review was performed (has cost data)
            if plan.api_review_cost and plan.api_review_cost.get("sections_reviewed", 0) > 0:
                # Review was performed - show cost summary
                cost = plan.api_review_cost
                sections.append("### Cost Summary")
                sections.append(f"- Sections reviewed: {cost.get('sections_reviewed', 0)}")
                sections.append(f"- Input tokens: {cost.get('actual_input_tokens', 0):,}")
                sections.append(f"- Output tokens: {cost.get('actual_output_tokens', 0):,}")
                sections.append(f"- Cost: ${cost.get('actual_cost_usd', 0.0):.4f} USD (€{cost.get('actual_cost_eur', 0.0):.4f} EUR)")

                # Show model from estimate if available
                if cost.get('estimate') and cost['estimate'].get('model'):
                    sections.append(f"- Model: {cost['estimate']['model']}")
                sections.append("")

                # Show per-section feedback
                if plan.api_review_feedback:
                    sections.append("### Section Feedback")
                    sections.append("")
                    for section_name, feedback in plan.api_review_feedback.items():
                        # Get confidence score for section if available
                        score = plan.section_scores.get(section_name, 0.0)
                        sections.append(f"#### {section_name.title()} (score: {score:.2f})")
                        sections.append(feedback)
                        sections.append("")
            elif plan.api_review_cost and plan.api_review_cost.get("sections_reviewed", 0) == 0:
                # API review requested but no sections flagged
                sections.append("All sections above confidence threshold. No API review required.")
                sections.append("Estimated cost: $0.00 USD (€0.00 EUR)")
                sections.append("")
            else:
                # API review requested but not yet performed (shouldn't happen in final output)
                sections.append("API review pending.")
                sections.append("")

        return "\n".join(sections)

    def _render_frontmatter(self, plan: PlanOutput) -> str:
        """Render YAML frontmatter with metadata."""
        lines = ["---"]
        lines.append(f'generated_at: "{plan.generated_at.isoformat()}"')
        lines.append(f'git_sha: "{plan.git_sha}"')
        lines.append(f"overall_quality_score: {plan.overall_quality_score}")

        if plan.section_scores:
            lines.append("section_scores:")
            for section, score in plan.section_scores.items():
                lines.append(f"  {section}: {score}")

        # API Review metadata
        lines.append(f"api_review_requested: {str(plan.api_review_requested).lower()}")
        if plan.api_review_cost:
            cost_usd = plan.api_review_cost.get("actual_cost_usd", 0.0)
            cost_eur = plan.api_review_cost.get("actual_cost_eur", 0.0)
            lines.append(f"api_review_cost_usd: {cost_usd:.4f}")
            lines.append(f"api_review_cost_eur: {cost_eur:.4f}")

        lines.append("---")
        lines.append("")
        return "\n".join(lines)
