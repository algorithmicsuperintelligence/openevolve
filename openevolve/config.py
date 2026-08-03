"""
Configuration handling for OpenEvolve
"""

import os
import re
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import dacite
import yaml

if TYPE_CHECKING:
    from openevolve.llm.base import LLMInterface


_ENV_VAR_PATTERN = re.compile(r"^\$\{([^}]+)\}$")  # ${VAR}


def _resolve_env_var(value: Optional[str]) -> Optional[str]:
    """
    Resolve ${VAR} environment variable reference in a string value.
    In current implementation pattern must match the entire string (e.g., "${OPENAI_API_KEY}"),
    not embedded within other text.

    Args:
        value: The string value that may contain ${VAR} syntax

    Returns:
        The resolved value with environment variable expanded, or original value if no match

    Raises:
        ValueError: If the environment variable is referenced but not set
    """
    if value is None:
        return None

    match = _ENV_VAR_PATTERN.match(value)
    if not match:
        return value

    var_name = match.group(1)
    env_value = os.environ.get(var_name)
    if env_value is None:
        raise ValueError(f"Environment variable {var_name} is not set")
    return env_value


@dataclass
class LLMModelConfig:
    """Configuration for a single LLM model"""

    # API configuration
    api_base: str = None
    api_key: Optional[str] = None
    name: str = None

    # LLM provider: "openai" (default), "claude_code" (Claude Code CLI)
    provider: Optional[str] = None

    # Custom LLM client
    init_client: Optional[Callable] = None

    # Weight for model in ensemble
    weight: float = 1.0

    # Generation parameters
    system_message: Optional[str] = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int = None

    # Request parameters
    timeout: int = None
    retries: int = None
    retry_delay: int = None

    # Reproducibility
    random_seed: Optional[int] = None

    # Reasoning parameters
    reasoning_effort: Optional[str] = None

    # Claude Code CLI budget per call (USD)
    max_budget_usd: Optional[float] = None

    # Manual mode (human-in-the-loop)
    manual_mode: Optional[bool] = None
    _manual_queue_dir: Optional[str] = None

    def __post_init__(self):
        """Post-initialization to resolve ${VAR} env var references in api_key"""
        self.api_key = _resolve_env_var(self.api_key)


@dataclass
class LLMConfig(LLMModelConfig):
    """Configuration for LLM models"""

    # API configuration
    api_base: str = "https://api.openai.com/v1"

    # Generation parameters
    system_message: Optional[str] = "system_message"
    temperature: float | None = 0.7
    top_p: float | None = None
    max_tokens: int = 4096

    # Request parameters
    timeout: int = 60
    retries: int = 3
    retry_delay: int = 5

    # n-model configuration for evolution LLM ensemble
    models: List[LLMModelConfig] = field(default_factory=list)

    # n-model configuration for evaluator LLM ensemble
    evaluator_models: List[LLMModelConfig] = field(default_factory=lambda: [])

    # Backwardes compatibility with primary_model(_weight) options
    primary_model: str = None
    primary_model_weight: float = None
    secondary_model: str = None
    secondary_model_weight: float = None

    # Reasoning parameters (inherited from LLMModelConfig but can be overridden)
    reasoning_effort: Optional[str] = None

    # Manual mode switch
    manual_mode: bool = False

    def __post_init__(self):
        """Post-initialization to set up model configurations"""
        super().__post_init__()  # Resolve ${VAR} in api_key at LLMConfig level

        # Handle backward compatibility for primary_model(_weight) and secondary_model(_weight).
        if self.primary_model:
            # Create primary model
            primary_model = LLMModelConfig(
                name=self.primary_model, weight=self.primary_model_weight or 1.0
            )
            self.models.append(primary_model)

        if self.secondary_model:
            # Create secondary model (only if weight > 0)
            if self.secondary_model_weight is None or self.secondary_model_weight > 0:
                secondary_model = LLMModelConfig(
                    name=self.secondary_model,
                    weight=(
                        self.secondary_model_weight
                        if self.secondary_model_weight is not None
                        else 0.2
                    ),
                )
                self.models.append(secondary_model)

        # Only validate if this looks like a user config (has some model info)
        # Don't validate during internal/default initialization
        if (
            self.primary_model
            or self.secondary_model
            or self.primary_model_weight
            or self.secondary_model_weight
        ) and not self.models:
            raise ValueError(
                "No LLM models configured. Please specify 'models' array or "
                "'primary_model' in your configuration."
            )

        # If no evaluator models are defined, use the same models as for evolution
        if not self.evaluator_models:
            self.evaluator_models = self.models.copy()

        # Update models with shared configuration values
        shared_config = {
            # `provider` must be shared: without it a top-level `llm.provider`
            # (e.g. "claude_code") never reaches the per-model configs, every model
            # keeps provider=None, and LLMEnsemble silently routes them all to the
            # OpenAI backend. Propagation uses overwrite=False, so an explicit
            # per-model provider still wins.
            "provider": self.provider,
            "api_base": self.api_base,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "retries": self.retries,
            "retry_delay": self.retry_delay,
            "random_seed": self.random_seed,
            "reasoning_effort": self.reasoning_effort,
            "manual_mode": self.manual_mode,
        }
        self.update_model_params(shared_config)

    def update_model_params(self, args: Dict[str, Any], overwrite: bool = False) -> None:
        """Update model parameters for all models"""
        for model in self.models + self.evaluator_models:
            for key, value in args.items():
                if overwrite or getattr(model, key, None) is None:
                    setattr(model, key, value)

    def rebuild_models(self) -> None:
        """Rebuild the models list after primary_model/secondary_model field changes"""
        # Clear existing models lists
        self.models = []
        self.evaluator_models = []

        # Re-run model generation logic from __post_init__
        if self.primary_model:
            # Create primary model
            primary_model = LLMModelConfig(
                name=self.primary_model, weight=self.primary_model_weight or 1.0
            )
            self.models.append(primary_model)

        if self.secondary_model:
            # Create secondary model (only if weight > 0)
            if self.secondary_model_weight is None or self.secondary_model_weight > 0:
                secondary_model = LLMModelConfig(
                    name=self.secondary_model,
                    weight=(
                        self.secondary_model_weight
                        if self.secondary_model_weight is not None
                        else 0.2
                    ),
                )
                self.models.append(secondary_model)

        # If no evaluator models are defined, use the same models as for evolution
        if not self.evaluator_models:
            self.evaluator_models = self.models.copy()

        # Update models with shared configuration values
        shared_config = {
            # See the note in __post_init__: `provider` has to be propagated here too,
            # or rebuilding the models drops a top-level provider back to None.
            "provider": self.provider,
            "api_base": self.api_base,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "retries": self.retries,
            "retry_delay": self.retry_delay,
            "random_seed": self.random_seed,
            "reasoning_effort": self.reasoning_effort,
        }
        self.update_model_params(shared_config)


@dataclass
class PromptConfig:
    """Configuration for prompt generation"""

    template_dir: Optional[str] = None
    system_message: str = "system_message"
    evaluator_system_message: str = "evaluator_system_message"

    # Large-codebase mode: represent programs in prompts via compact changes descriptions
    programs_as_changes_description: bool = False
    system_message_changes_description: Optional[str] = None
    initial_changes_description: str = ""

    # Number of examples to include in the prompt
    num_top_programs: int = 3
    num_diverse_programs: int = 2

    # Template stochasticity
    use_template_stochasticity: bool = True
    template_variations: Dict[str, List[str]] = field(default_factory=dict)

    # Meta-prompting
    # Note: meta-prompting features not implemented
    use_meta_prompting: bool = False
    meta_prompt_weight: float = 0.1

    # Artifact rendering
    include_artifacts: bool = True
    max_artifact_bytes: int = 20 * 1024  # 20KB in prompt
    artifact_security_filter: bool = True
    # Optional ordered subset of evaluator artifacts rendered in prompts.
    # Stored artifacts remain complete.
    artifact_include_names: Optional[List[str]] = None
    # Optional evaluator artifact collected across the complete live archive
    # and rendered as one compact deterministic prompt artifact.
    archive_context_artifact: Optional[str] = None
    archive_context_max_items: int = 64
    # Optional evaluator-produced neighborhood. The artifact must be schema
    # version 1 JSON with an identity_artifact name and ordered options carrying
    # identity fields. The controller removes identities already retained in the
    # live archive and renders the remainder as proposal-options.json.
    proposal_neighborhood_artifact: Optional[str] = None
    proposal_options_max_items: int = 32

    # Feature extraction and program labeling
    suggest_simplification_after_chars: Optional[int] = (
        500  # Suggest simplifying if program exceeds this many characters
    )
    include_changes_under_chars: Optional[int] = (
        100  # Include change descriptions in features if under this length
    )
    concise_implementation_max_lines: Optional[int] = (
        10  # Label as "concise" if program has this many lines or fewer
    )
    comprehensive_implementation_min_lines: Optional[int] = (
        50  # Label as "comprehensive" if program has this many lines or more
    )

    # Diff summary formatting for "Previous Attempts" section
    diff_summary_max_line_len: int = 100  # Truncate lines longer than this
    diff_summary_max_lines: int = 30  # Max lines per SEARCH/REPLACE block

    # Backward compatibility - deprecated
    code_length_threshold: Optional[int] = (
        None  # Deprecated: use suggest_simplification_after_chars
    )


@dataclass
class ControllerSchedulerConfig:
    """Optional observed-result island allocator for bounded live comparisons."""

    enabled: bool = False
    score_metric: str = "combined_score"
    # Optional evaluator artifact used to count distinct measured phenotypes.
    # When absent, source-code hashes preserve the legacy definition.
    diversity_artifact: Optional[str] = None
    exploitation_weight: float = 0.0
    underexplored_weight: float = 0.0
    validity_weight: float = 0.0
    diversity_weight: float = 0.0
    token_efficiency_weight: float = 0.0
    rejection_penalty: float = 0.0
    minimum_calls: int = 1
    # Optional post-warmup quality band. When set, allocate only among islands
    # whose retained best score is within this absolute distance of the global
    # retained leader. None preserves unrestricted legacy allocation.
    leader_score_band: Optional[float] = None
    # Optional post-warmup parent band within the selected island. When set,
    # parent sampling is limited to retained programs whose score is within
    # this absolute distance of that island's retained leader.
    parent_score_band: Optional[float] = None
    # Keep this many islands outside the active worker frontier so a completed
    # call leaves the allocator a real choice. This does not reduce concurrency
    # when num_islands > parallel_evaluations.
    reserve_islands: int = 1
    # Optional lower in-flight limit after the balanced warmup proposal count
    # has been submitted. This trades some throughput for fresher archive
    # context during quality-focused adaptive search.
    adaptive_parallelism: Optional[int] = None


@dataclass
class DatabaseConfig:
    """Configuration for the program database"""

    # General settings
    db_path: Optional[str] = None  # Path to store database on disk
    in_memory: bool = True

    # Prompt and response logging to programs/<id>.json
    log_prompts: bool = True

    # Evolutionary parameters
    population_size: int = 1000
    archive_size: int = 100
    num_islands: int = 5

    # Selection parameters
    elite_selection_ratio: float = 0.1
    exploration_ratio: float = 0.2
    exploitation_ratio: float = 0.7
    controller_scheduler: ControllerSchedulerConfig = field(
        default_factory=ControllerSchedulerConfig
    )
    # Note: diversity_metric fixed to "edit_distance"
    diversity_metric: str = "edit_distance"  # Options: "edit_distance", "feature_based"

    # Feature map dimensions for MAP-Elites
    # Default to complexity and diversity for better exploration
    # CRITICAL: For custom dimensions, evaluators must return RAW VALUES, not bin indices
    # Built-in: "complexity", "diversity", "score" (always available)
    # Custom: Any metric from your evaluator (must be continuous values)
    feature_dimensions: List[str] = field(
        default_factory=lambda: ["complexity", "diversity"],
        metadata={
            "help": "List of feature dimensions for MAP-Elites grid. "
            "Built-in dimensions: 'complexity', 'diversity', 'score'. "
            "Custom dimensions: Must match metric names from evaluator. "
            "IMPORTANT: Evaluators must return raw continuous values for custom dimensions, "
            "NOT pre-computed bin indices. OpenEvolve handles all scaling and binning internally."
        },
    )
    feature_bins: Union[int, Dict[str, int]] = 10  # Can be int (all dims) or dict (per-dim)
    diversity_reference_size: int = 20  # Size of reference set for diversity calculation

    # Migration parameters for island-based evolution
    migration_interval: int = 50  # Migrate every N generations
    migration_rate: float = 0.1  # Fraction of population to migrate

    # Random seed for reproducible sampling
    random_seed: Optional[int] = 42

    # Artifact storage
    artifacts_base_path: Optional[str] = None  # Defaults to db_path/artifacts
    artifact_size_threshold: int = 32 * 1024  # 32KB threshold
    cleanup_old_artifacts: bool = True
    artifact_retention_days: int = 30
    max_snapshot_artifacts: Optional[int] = (
        100  # Max artifacts in worker snapshots (None=unlimited)
    )

    novelty_llm: Optional["LLMInterface"] = None
    embedding_model: Optional[str] = None
    similarity_threshold: float = 0.99


@dataclass
class EvaluatorConfig:
    """Configuration for program evaluation"""

    # General settings
    timeout: int = 300  # Maximum evaluation time in seconds
    max_retries: int = 3

    # Resource limits for evaluation
    # Note: resource limits not implemented
    memory_limit_mb: Optional[int] = None
    cpu_limit: Optional[float] = None

    # Evaluation strategies
    cascade_evaluation: bool = True
    cascade_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.75, 0.9])

    # Parallel evaluation
    parallel_evaluations: int = 1
    # Note: distributed evaluation not implemented
    distributed: bool = False

    # LLM-based feedback
    use_llm_feedback: bool = False
    llm_feedback_weight: float = 0.1

    # Artifact handling
    enable_artifacts: bool = True
    max_artifact_storage: int = 100 * 1024 * 1024  # 100MB per program


@dataclass
class EvolutionTraceConfig:
    """Configuration for evolution trace logging"""

    enabled: bool = False
    format: str = "jsonl"  # Options: "jsonl", "json", "hdf5"
    include_code: bool = False
    include_prompts: bool = True
    output_path: Optional[str] = None
    buffer_size: int = 10
    compress: bool = False


@dataclass
class Config:
    """Master configuration for OpenEvolve"""

    # General settings
    max_iterations: int = 10000
    # Optional per-run proposal-model budgets. Calls are reserved before
    # submission; provider-token totals come from completed unique receipts.
    max_llm_calls: Optional[int] = None
    max_total_provider_tokens: Optional[int] = None
    checkpoint_interval: int = 100
    log_level: str = "INFO"
    log_dir: Optional[str] = None
    random_seed: Optional[int] = 42
    language: str = None
    file_suffix: str = ".py"

    # Component configurations
    llm: LLMConfig = field(default_factory=LLMConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    evolution_trace: EvolutionTraceConfig = field(default_factory=EvolutionTraceConfig)

    # Evolution settings
    diff_based_evolution: bool = True
    max_code_length: int = 10000
    diff_pattern: str = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
    strict_diff_application: bool = False
    enforce_evolve_blocks: bool = False
    max_diff_blocks: int = 32
    # Optional evaluator artifact whose exact value identifies the candidate's
    # behaviorally meaningful contents. When configured, a child duplicating
    # any program already known to the live archive is rejected.
    program_identity_artifact: Optional[str] = None
    # Optional second evaluator identity for exact measured behavior. This is
    # checked only after evaluation, so structurally different programs that
    # produce an already archived phenotype do not bloat parent selection.
    phenotype_identity_artifact: Optional[str] = None

    # Early stopping settings
    early_stopping_patience: Optional[int] = None
    convergence_threshold: float = 0.001
    early_stopping_metric: str = "combined_score"

    # Parallel controller settings
    max_tasks_per_child: Optional[int] = None

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from a YAML file"""
        config_path = Path(path).resolve()
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        config = cls.from_dict(config_dict)

        # Resolve template_dir relative to config file location
        if config.prompt.template_dir:
            template_path = Path(config.prompt.template_dir)
            if not template_path.is_absolute():
                config.prompt.template_dir = str((config_path.parent / template_path).resolve())

        return config

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        if "diff_pattern" in config_dict:
            try:
                re.compile(config_dict["diff_pattern"])
            except re.error as e:
                raise ValueError(f"Invalid regex pattern in diff_pattern: {e}")

        # Remove None values for temperature and top_p to avoid dacite type errors;
        # alternatively, pass check_types=False to dacite.from_dict, but that can hide other issues
        if "llm" in config_dict:
            if "temperature" in config_dict["llm"] and config_dict["llm"]["temperature"] is None:
                del config_dict["llm"]["temperature"]
            if "top_p" in config_dict["llm"] and config_dict["llm"]["top_p"] is None:
                del config_dict["llm"]["top_p"]

        config: Config = dacite.from_dict(
            data_class=cls,
            data=config_dict,
            config=dacite.Config(
                cast=[List, Union],
                forward_references={"LLMInterface": Any},
            ),
        )

        if config.database.random_seed is None and config.random_seed is not None:
            config.database.random_seed = config.random_seed

        config.validate()
        return config

    def validate(self) -> None:
        """Validate combinations used by both YAML and programmatic callers."""
        for field_name in ("max_llm_calls", "max_total_provider_tokens"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{field_name} must be a positive integer or None")
        try:
            re.compile(self.diff_pattern)
        except re.error as error:
            raise ValueError(f"Invalid regex pattern in diff_pattern: {error}") from error
        if self.prompt.programs_as_changes_description and not self.diff_based_evolution:
            raise ValueError(
                "prompt.programs_as_changes_description=true requires diff_based_evolution=true "
                "(full rewrites cannot reliably update code and changes_description together)"
            )
        if self.enforce_evolve_blocks and not self.strict_diff_application:
            raise ValueError(
                "enforce_evolve_blocks=true requires strict_diff_application=true"
            )
        if self.enforce_evolve_blocks and not self.diff_based_evolution:
            raise ValueError(
                "enforce_evolve_blocks=true requires diff_based_evolution=true"
            )
        if self.max_diff_blocks < 1:
            raise ValueError("max_diff_blocks must be at least 1")
        if self.prompt.archive_context_artifact is not None and (
            not isinstance(self.prompt.archive_context_artifact, str)
            or not self.prompt.archive_context_artifact.strip()
        ):
            raise ValueError(
                "prompt.archive_context_artifact must be a non-empty string or None"
            )
        if self.prompt.proposal_neighborhood_artifact is not None and (
            not isinstance(self.prompt.proposal_neighborhood_artifact, str)
            or not self.prompt.proposal_neighborhood_artifact.strip()
        ):
            raise ValueError(
                "prompt.proposal_neighborhood_artifact must be a non-empty "
                "string or None"
            )
        artifact_names = self.prompt.artifact_include_names
        if artifact_names is not None:
            if not isinstance(artifact_names, list) or any(
                not isinstance(name, str) or not name.strip()
                for name in artifact_names
            ):
                raise ValueError(
                    "prompt.artifact_include_names must be a list of non-empty strings or None"
                )
            if len(set(artifact_names)) != len(artifact_names):
                raise ValueError(
                    "prompt.artifact_include_names must not contain duplicates"
                )
            if (
                self.prompt.archive_context_artifact is not None
                and "archive-context.json" not in artifact_names
            ):
                raise ValueError(
                    "prompt.artifact_include_names must include archive-context.json "
                    "when archive_context_artifact is configured"
                )
            if (
                self.prompt.proposal_neighborhood_artifact is not None
                and "proposal-options.json" not in artifact_names
            ):
                raise ValueError(
                    "prompt.artifact_include_names must include "
                    "proposal-options.json when proposal_neighborhood_artifact "
                    "is configured"
                )
        if (
            isinstance(self.prompt.archive_context_max_items, bool)
            or not isinstance(self.prompt.archive_context_max_items, int)
            or self.prompt.archive_context_max_items < 1
        ):
            raise ValueError(
                "prompt.archive_context_max_items must be a positive integer"
            )
        if (
            isinstance(self.prompt.proposal_options_max_items, bool)
            or not isinstance(self.prompt.proposal_options_max_items, int)
            or self.prompt.proposal_options_max_items < 1
        ):
            raise ValueError(
                "prompt.proposal_options_max_items must be a positive integer"
            )
        for field_name in (
            "program_identity_artifact",
            "phenotype_identity_artifact",
        ):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{field_name} must be a non-empty string or None")
        scheduler = self.database.controller_scheduler
        if not isinstance(scheduler.enabled, bool):
            raise ValueError("database.controller_scheduler.enabled must be boolean")
        if not isinstance(scheduler.score_metric, str) or not scheduler.score_metric.strip():
            raise ValueError(
                "database.controller_scheduler.score_metric "
                "must be a non-empty string"
            )
        if scheduler.diversity_artifact is not None and (
            not isinstance(scheduler.diversity_artifact, str)
            or not scheduler.diversity_artifact.strip()
        ):
            raise ValueError(
                "database.controller_scheduler.diversity_artifact "
                "must be a non-empty string or None"
            )
        for field_name in (
            "exploitation_weight",
            "underexplored_weight",
            "validity_weight",
            "diversity_weight",
            "token_efficiency_weight",
            "rejection_penalty",
        ):
            value = getattr(scheduler, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(
                    f"database.controller_scheduler.{field_name} "
                    "must be finite and nonnegative"
                )
        for field_name in ("leader_score_band", "parent_score_band"):
            value = getattr(scheduler, field_name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(
                    f"database.controller_scheduler.{field_name} "
                    "must be finite and nonnegative when provided"
                )
        if (
            isinstance(scheduler.minimum_calls, bool)
            or not isinstance(scheduler.minimum_calls, int)
            or scheduler.minimum_calls < 1
        ):
            raise ValueError(
                "database.controller_scheduler.minimum_calls "
                "must be a positive integer"
            )
        if (
            isinstance(scheduler.reserve_islands, bool)
            or not isinstance(scheduler.reserve_islands, int)
            or scheduler.reserve_islands < 0
            or (
                scheduler.enabled
                and scheduler.reserve_islands >= self.database.num_islands
            )
        ):
            raise ValueError(
                "database.controller_scheduler.reserve_islands must be an integer "
                "in [0, database.num_islands) when the scheduler is enabled"
            )
        if scheduler.adaptive_parallelism is not None and (
            isinstance(scheduler.adaptive_parallelism, bool)
            or not isinstance(scheduler.adaptive_parallelism, int)
            or scheduler.adaptive_parallelism < 1
            or scheduler.adaptive_parallelism
            > self.evaluator.parallel_evaluations
            or (
                scheduler.enabled
                and scheduler.adaptive_parallelism
                > self.database.num_islands - scheduler.reserve_islands
            )
        ):
            raise ValueError(
                "database.controller_scheduler.adaptive_parallelism must be a "
                "positive integer no greater than the evaluator worker count or "
                "selectable island count"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file"""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from a YAML file or use defaults"""
    if config_path and os.path.exists(config_path):
        config = Config.from_yaml(config_path)
    else:
        config = Config()

        # Use environment variables if available
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

        config.llm.update_model_params({"api_key": api_key, "api_base": api_base})

    # Make the system message available to the individual models, in case it is not provided from the prompt sampler
    config.llm.update_model_params({"system_message": config.prompt.system_message})

    return config
