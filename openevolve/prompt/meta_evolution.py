"""
Meta-evolution of prompt templates for OpenEvolve.

Inspired by the Darwin Gödel Machine paper, this module enables OpenEvolve
to evolve its own prompts based on empirical success rates.
"""

import logging
import random
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """An evolvable prompt template with success tracking."""

    id: str
    system_template: str
    user_template: str
    # Success tracking
    uses: int = 0
    successes: int = 0  # Number of times mutation was accepted
    improvements: int = 0  # Number of times mutation improved fitness
    total_fitness_delta: float = 0.0  # Sum of fitness changes
    # Lineage
    parent_id: Optional[str] = None
    generation: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Fraction of uses that resulted in accepted mutations."""
        return self.successes / self.uses if self.uses > 0 else 0.0

    @property
    def improvement_rate(self) -> float:
        """Fraction of uses that resulted in fitness improvement."""
        return self.improvements / self.uses if self.uses > 0 else 0.0

    @property
    def avg_fitness_delta(self) -> float:
        """Average fitness change per use."""
        return self.total_fitness_delta / self.uses if self.uses > 0 else 0.0

    def compute_score(
        self,
        weight_success: float = 0.3,
        weight_improvement: float = 0.4,
        weight_fitness_delta: float = 0.3,
        min_uses: int = 5,
        neutral_prior: float = 0.5,
    ) -> float:
        """
        Compute score for template quality with configurable weights.

        Args:
            weight_success: Weight for success rate (mutations accepted)
            weight_improvement: Weight for improvement rate (fitness increased)
            weight_fitness_delta: Weight for avg fitness delta magnitude
            min_uses: Minimum uses before score is calculated
            neutral_prior: Score returned when uses < min_uses

        Returns:
            Combined score between 0 and 1
        """
        if self.uses < min_uses:
            return neutral_prior
        # Weighted combination
        return (
            weight_success * self.success_rate
            + weight_improvement * self.improvement_rate
            + weight_fitness_delta * min(1.0, self.avg_fitness_delta + 0.5)
        )

    @property
    def score(self) -> float:
        """
        Combined score for template quality using default weights.
        For configurable weights, use compute_score() method.
        """
        return self.compute_score()

    def record_use(
        self,
        accepted: bool,
        fitness_delta: float = 0.0,
    ) -> None:
        """Record the outcome of using this template."""
        self.uses += 1
        if accepted:
            self.successes += 1
        if fitness_delta > 0:
            self.improvements += 1
        self.total_fitness_delta += fitness_delta

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "system_template": self.system_template,
            "user_template": self.user_template,
            "uses": self.uses,
            "successes": self.successes,
            "improvements": self.improvements,
            "total_fitness_delta": self.total_fitness_delta,
            "parent_id": self.parent_id,
            "generation": self.generation,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTemplate":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            system_template=data["system_template"],
            user_template=data["user_template"],
            uses=data.get("uses", 0),
            successes=data.get("successes", 0),
            improvements=data.get("improvements", 0),
            total_fitness_delta=data.get("total_fitness_delta", 0.0),
            parent_id=data.get("parent_id"),
            generation=data.get("generation", 0),
            metadata=data.get("metadata", {}),
        )


class PromptArchive:
    """
    Archive of evolvable prompt templates.

    Maintains a population of templates, tracks their success rates,
    and supports sampling and evolution.
    """

    def __init__(
        self,
        max_size: int = 20,
        min_uses_for_evolution: int = 10,
        elite_fraction: float = 0.3,
        exploration_rate: float = 0.2,
        # Scoring weights
        score_weight_success: float = 0.3,
        score_weight_improvement: float = 0.4,
        score_weight_fitness_delta: float = 0.3,
        score_min_uses: int = 5,
        score_neutral_prior: float = 0.5,
    ):
        """
        Initialize the prompt archive.

        Args:
            max_size: Maximum number of templates to keep
            min_uses_for_evolution: Minimum uses before a template can be evolved
            elite_fraction: Fraction of top templates to preserve
            exploration_rate: Probability of sampling a random/new template
            score_weight_success: Weight for success rate in scoring
            score_weight_improvement: Weight for improvement rate in scoring
            score_weight_fitness_delta: Weight for fitness delta in scoring
            score_min_uses: Minimum uses before calculating score
            score_neutral_prior: Score for templates with insufficient uses
        """
        self.max_size = max_size
        self.min_uses_for_evolution = min_uses_for_evolution
        self.elite_fraction = elite_fraction
        self.exploration_rate = exploration_rate

        # Scoring configuration
        self.score_weight_success = score_weight_success
        self.score_weight_improvement = score_weight_improvement
        self.score_weight_fitness_delta = score_weight_fitness_delta
        self.score_min_uses = score_min_uses
        self.score_neutral_prior = score_neutral_prior

        self.templates: Dict[str, PromptTemplate] = {}
        self.default_template_id: Optional[str] = None

    def get_template_score(self, template: PromptTemplate) -> float:
        """Get the score for a template using configured weights."""
        return template.compute_score(
            weight_success=self.score_weight_success,
            weight_improvement=self.score_weight_improvement,
            weight_fitness_delta=self.score_weight_fitness_delta,
            min_uses=self.score_min_uses,
            neutral_prior=self.score_neutral_prior,
        )

    def add_template(
        self,
        system_template: str,
        user_template: str,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_default: bool = False,
    ) -> PromptTemplate:
        """Add a new template to the archive."""
        template_id = str(uuid.uuid4())[:8]

        # Determine generation
        generation = 0
        if parent_id and parent_id in self.templates:
            generation = self.templates[parent_id].generation + 1

        template = PromptTemplate(
            id=template_id,
            system_template=system_template,
            user_template=user_template,
            parent_id=parent_id,
            generation=generation,
            metadata=metadata or {},
        )

        self.templates[template_id] = template

        # Set as default if first template or explicitly requested
        if self.default_template_id is None or is_default:
            self.default_template_id = template_id

        # Prune if over capacity
        self._prune_if_needed()

        logger.info(
            f"Added prompt template {template_id} (generation {generation}, "
            f"archive size: {len(self.templates)})"
        )

        return template

    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """Get a template by ID."""
        return self.templates.get(template_id)

    def sample_template(self) -> PromptTemplate:
        """
        Sample a template for use.

        Uses a mix of exploitation (high-scoring templates) and
        exploration (less-used or random templates).
        """
        if not self.templates:
            raise ValueError("No templates in archive")

        # Exploration: occasionally pick a random template
        if random.random() < self.exploration_rate:
            template = random.choice(list(self.templates.values()))
            logger.debug(f"Sampled template {template.id} (exploration)")
            return template

        # Exploitation: prefer high-scoring templates
        # Weight by score, with bonus for less-used templates
        templates = list(self.templates.values())
        weights = []
        for t in templates:
            # Score-based weight with exploration bonus for under-used templates
            exploration_bonus = max(0, 1.0 - t.uses / 20) * 0.3
            weights.append(self.get_template_score(t) + exploration_bonus)

        # Normalize weights
        total = sum(weights)
        if total == 0:
            template = random.choice(templates)
        else:
            weights = [w / total for w in weights]
            template = random.choices(templates, weights=weights, k=1)[0]

        logger.debug(
            f"Sampled template {template.id} (score={self.get_template_score(template):.3f}, "
            f"uses={template.uses})"
        )
        return template

    def record_outcome(
        self,
        template_id: str,
        accepted: bool,
        fitness_delta: float = 0.0,
    ) -> None:
        """Record the outcome of using a template."""
        if template_id not in self.templates:
            logger.warning(f"Template {template_id} not found in archive")
            return

        self.templates[template_id].record_use(accepted, fitness_delta)
        logger.debug(
            f"Template {template_id}: accepted={accepted}, "
            f"fitness_delta={fitness_delta:.4f}, "
            f"new_score={self.get_template_score(self.templates[template_id]):.3f}"
        )

    def get_templates_for_evolution(self) -> List[PromptTemplate]:
        """Get templates that are ready for evolution (enough uses)."""
        return [t for t in self.templates.values() if t.uses >= self.min_uses_for_evolution]

    def get_top_templates(self, n: int = 5) -> List[PromptTemplate]:
        """Get the top N templates by score."""
        sorted_templates = sorted(
            self.templates.values(),
            key=lambda t: self.get_template_score(t),
            reverse=True,
        )
        return sorted_templates[:n]

    def get_statistics(self) -> Dict[str, Any]:
        """Get archive statistics."""
        if not self.templates:
            return {"size": 0}

        templates = list(self.templates.values())
        total_uses = sum(t.uses for t in templates)
        total_successes = sum(t.successes for t in templates)

        return {
            "size": len(templates),
            "total_uses": total_uses,
            "total_successes": total_successes,
            "overall_success_rate": (total_successes / total_uses if total_uses > 0 else 0),
            "max_generation": max(t.generation for t in templates),
            "avg_score": sum(self.get_template_score(t) for t in templates) / len(templates),
            "top_template_id": self.get_top_templates(1)[0].id if templates else None,
        }

    def _prune_if_needed(self) -> None:
        """Remove lowest-scoring templates if over capacity."""
        if len(self.templates) <= self.max_size:
            return

        # Keep elite templates
        num_elite = max(1, int(self.max_size * self.elite_fraction))
        sorted_templates = sorted(
            self.templates.values(),
            key=lambda t: self.get_template_score(t),
            reverse=True,
        )

        # Templates to keep: elite + default
        elite_ids = {t.id for t in sorted_templates[:num_elite]}

        # Also keep default template
        if self.default_template_id:
            elite_ids.add(self.default_template_id)

        # Remove lowest scoring non-elite templates
        to_remove = []
        for t in reversed(sorted_templates):
            if t.id not in elite_ids and len(self.templates) - len(to_remove) > self.max_size:
                to_remove.append(t.id)

        for tid in to_remove:
            del self.templates[tid]
            logger.debug(f"Pruned template {tid} from archive")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize archive to dictionary."""
        return {
            "max_size": self.max_size,
            "min_uses_for_evolution": self.min_uses_for_evolution,
            "elite_fraction": self.elite_fraction,
            "exploration_rate": self.exploration_rate,
            "default_template_id": self.default_template_id,
            "templates": {tid: t.to_dict() for tid, t in self.templates.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptArchive":
        """Deserialize archive from dictionary."""
        archive = cls(
            max_size=data.get("max_size", 20),
            min_uses_for_evolution=data.get("min_uses_for_evolution", 10),
            elite_fraction=data.get("elite_fraction", 0.3),
            exploration_rate=data.get("exploration_rate", 0.2),
        )
        archive.default_template_id = data.get("default_template_id")

        for tid, tdata in data.get("templates", {}).items():
            archive.templates[tid] = PromptTemplate.from_dict(tdata)

        return archive


# Prompt for evolving prompts (meta!)
PROMPT_EVOLUTION_SYSTEM = """You are an expert at crafting prompts for code evolution systems.
Your task is to improve prompts that guide an LLM to generate better code mutations.

A good evolution prompt should:
1. Clearly explain the task and expected output format
2. Provide useful context without overwhelming detail
3. Encourage creative yet targeted improvements
4. Guide the LLM to explain its reasoning
"""

PROMPT_EVOLUTION_USER = """# Current Prompt Performance

The following prompt template has been used {uses} times:
- Success rate (mutations accepted): {success_rate:.1%}
- Improvement rate (fitness increased): {improvement_rate:.1%}
- Average fitness change: {avg_fitness_delta:+.4f}

## Current System Template
```
{system_template}
```

## Current User Template
```
{user_template}
```

## Top Performing Templates for Reference

{top_templates_section}

# Task

Create an improved version of this prompt that will lead to better mutation success rates.

Focus on:
1. Clearer instructions for the type of changes to make
2. Better guidance on analyzing the current program
3. More effective use of the evolution history
4. Encouraging both exploitation (improving what works) and exploration (trying new approaches)

Provide your improved templates in the following format:

<system_template>
Your improved system template here
</system_template>

<user_template>
Your improved user template here
</user_template>

Explain your changes briefly after the templates.
"""


def evolve_prompt(
    template: PromptTemplate,
    top_templates: List[PromptTemplate],
    llm_generate_fn: Callable[[str, str], str],
    score_fn: Optional[Callable[[PromptTemplate], float]] = None,
) -> Optional[Tuple[str, str]]:
    """
    Evolve a prompt template using an LLM.

    Args:
        template: The template to evolve
        top_templates: Top performing templates for reference
        llm_generate_fn: Function to call LLM (takes system, user, returns str)
        score_fn: Optional function to compute template scores (defaults to template.score)

    Returns:
        Tuple of (new_system_template, new_user_template) or None if evolution failed
    """
    # Use provided score function or fall back to default
    get_score = score_fn if score_fn is not None else (lambda t: t.score)

    # Format top templates section
    top_section = ""
    for i, t in enumerate(top_templates[:3]):
        if t.id == template.id:
            continue
        top_section += f"""### Template {i + 1} (score: {get_score(t):.3f}, success: {t.success_rate:.1%})
System (truncated): {t.system_template[:200]}...
User (truncated): {t.user_template[:300]}...

"""

    user_prompt = PROMPT_EVOLUTION_USER.format(
        uses=template.uses,
        success_rate=template.success_rate,
        improvement_rate=template.improvement_rate,
        avg_fitness_delta=template.avg_fitness_delta,
        system_template=template.system_template,
        user_template=template.user_template,
        top_templates_section=top_section or "No other templates available yet.",
    )

    try:
        response = llm_generate_fn(PROMPT_EVOLUTION_SYSTEM, user_prompt)

        # Parse response
        new_system = _extract_between_tags(response, "system_template")
        new_user = _extract_between_tags(response, "user_template")

        if new_system and new_user:
            logger.info(f"Successfully evolved template {template.id}")
            return new_system, new_user
        else:
            logger.warning("Failed to parse evolved template from response")
            return None

    except Exception as e:
        logger.error(f"Error evolving template: {e}")
        return None


def _extract_between_tags(text: str, tag: str) -> Optional[str]:
    """Extract content between XML-style tags."""
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None
