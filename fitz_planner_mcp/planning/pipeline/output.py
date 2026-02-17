# fitz_planner_mcp/planning/pipeline/output.py
"""
Plan renderer for converting PipelineResult to structured markdown.

Converts pipeline outputs into a formatted markdown file with metadata frontmatter.
"""

from datetime import datetime
from typing import Any

from fitz_planner_mcp.planning.schemas.plan_output import PlanOutput


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

    def render(self, plan: PlanOutput) -> str:
        """
        Render PlanOutput to markdown string.

        Args:
            plan: Validated PlanOutput with all stage outputs

        Returns:
            Formatted markdown string
        """
        sections = []

        # Frontmatter
        sections.append(self._render_frontmatter(plan))

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
        sections.append(f"\n{plan.architecture.rationale}")
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
                    f"- Alternatives: {', '.join(adr.alternatives) if adr.alternatives else 'None listed'}"
                )
                sections.append("")

        if plan.design.components:
            sections.append("### Components")
            for comp in plan.design.components:
                sections.append(f"**{comp.name}**")
                sections.append(f"- Responsibility: {comp.responsibility}")
                sections.append(
                    f"- Interfaces: {', '.join(comp.interfaces) if comp.interfaces else 'None listed'}"
                )
                sections.append("")

        if plan.design.data_model:
            sections.append("### Data Model")
            sections.append(plan.design.data_model)
            sections.append("")

        # Roadmap
        sections.append("## Roadmap")
        sections.append("")
        for i, phase in enumerate(plan.roadmap.phases, start=1):
            sections.append(f"### Phase {i}: {phase.name}")
            sections.append(f"**Goal:** {phase.goal}")
            sections.append("")
            if phase.tasks:
                sections.append("**Tasks:**")
                for task in phase.tasks:
                    sections.append(f"- {task}")
                sections.append("")
            if phase.dependencies:
                sections.append(f"**Dependencies:** {', '.join(phase.dependencies)}")
                sections.append("")
            sections.append(f"**Duration:** {phase.duration}")
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
            sections.append(f"### {risk.title}")
            sections.append(f"**Description:** {risk.description}")
            sections.append(f"**Impact:** {risk.impact} | **Likelihood:** {risk.likelihood}")
            sections.append(f"**Mitigation:** {risk.mitigation}")
            if risk.contingency:
                sections.append(f"**Contingency:** {risk.contingency}")
            if risk.affected_phases:
                sections.append(f"**Affected Phases:** {', '.join(risk.affected_phases)}")
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

        lines.append("---")
        lines.append("")
        return "\n".join(lines)
