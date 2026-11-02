# Crafting Effective System Messages

**System messages are the secret to successful evolution.** They guide the LLM's understanding of your domain, constraints, and optimization goals. A well-crafted system message can be the difference between random mutations and targeted improvements.

### Why System Messages Matter

The system message in your config.yaml is arguably the most important component for evolution success:

- **Domain Expertise**: Provides LLM with specific knowledge about your problem space
- **Constraint Awareness**: Defines what can and cannot be changed during evolution
- **Optimization Focus**: Guides the LLM toward meaningful improvements
- **Error Prevention**: Helps avoid common pitfalls and compilation errors

### The Iterative Creation Process

Based on successful OpenEvolve implementations, system messages are best created through iteration:

### Step-by-Step Process

**Phase 1: Initial Draft**

1. Start with a basic system message describing your goal
2. Run 20-50 evolution iterations to observe behavior
3. Note where the system gets "stuck" or makes poor choices

**Phase 2: Refinement**

4. Add specific guidance based on observed issues
5. Include domain-specific terminology and concepts
6. Define clear constraints and optimization targets
7. Run another batch of iterations

**Phase 3: Specialization**

8. Add detailed examples of good vs bad approaches
9. Include specific library/framework guidance
10. Add error avoidance patterns you've observed
11. Fine-tune based on artifact feedback

**Phase 4: Optimization**

12. Consider using OpenEvolve itself to optimize your prompt
13. Measure improvements using combined score metrics

### Examples by Complexity

#### **Simple: General Optimization**
```yaml
prompt:
  system_message: |
    You are an expert programmer specializing in optimization algorithms.
    Your task is to improve a function minimization algorithm to find the
    global minimum reliably, escaping local minima that might trap simple algorithms.
```

#### **Intermediate: Domain-Specific Guidance**
```yaml
prompt:
  system_message: |
    You are an expert prompt engineer. Your task is to revise prompts for LLMs.

    Your improvements should:
    * Clarify vague instructions and eliminate ambiguity
    * Strengthen alignment between prompt and desired task outcome
    * Improve robustness against edge cases
    * Include formatting instructions and examples where helpful
    * Avoid unnecessary verbosity

    Return only the improved prompt text without explanations.
```

#### **Advanced: Hardware-Specific Optimization**
```yaml
prompt:
  system_message: |
    You are an expert Metal GPU programmer specializing in custom attention
    kernels for Apple Silicon.

    # TARGET: Optimize Metal Kernel for Grouped Query Attention (GQA)
    # HARDWARE: Apple M-series GPUs with unified memory architecture
    # GOAL: 5-15% performance improvement

    # OPTIMIZATION OPPORTUNITIES:
    **1. Memory Access Pattern Optimization:**
    - Coalesced access patterns for Apple Silicon
    - Vectorized loading using SIMD
    - Pre-compute frequently used indices

    **2. Algorithm Fusion:**
    - Combine max finding with score computation
    - Reduce number of passes through data

    # CONSTRAINTS - CRITICAL SAFETY RULES:
    **MUST NOT CHANGE:**
    - Kernel function signature
    - Template parameter names or types
    - Overall algorithm correctness

    **ALLOWED TO OPTIMIZE:**
    - Memory access patterns and indexing
    - Computation order and efficiency
    - Vectorization and SIMD utilization
    - Apple Silicon specific optimizations
```

### Best Practices

### Prompt Engineering Patterns

**Structure Your Message:** Start with role definition → Define task/context → List optimization opportunities → Set constraints → Success criteria

**Use Specific Examples:**
```yaml
# Good: "Focus on reducing memory allocations. Example: Replace `new Vector()` with pre-allocated arrays."
# Avoid: "Make the code faster"
```

**Include Domain Knowledge:**
```yaml
# Good: "For GPU kernels: 1) Memory coalescing 2) Occupancy 3) Shared memory usage"
# Avoid: "Optimize the algorithm"
```

**Set Clear Boundaries:**
```yaml
system_message: |
  MUST NOT CHANGE: - Function signatures - Algorithm correctness - External API
  ALLOWED: - Internal implementation - Data structures - Performance optimizations
```

### Advanced Techniques

**Artifact-Driven Iteration:** Enable artifacts in config → Include common error patterns in system message → Add guidance based on stderr/warning patterns

**Multi-Phase Evolution:** Start broad ("Explore different algorithmic approaches"), then focus ("Given successful simulated annealing, focus on parameter tuning")

**Template Stochasticity:** See the [Configuration section](configuration.md) for complete template variation examples.

### Meta-Evolution: Using OpenEvolve to Optimize Prompts

**You can use OpenEvolve to evolve your system messages themselves!** This powerful technique lets you optimize prompts for better LLM performance automatically.

See the [LLM Prompt Optimization example](../examples/llm_prompt_optimization/) for a complete implementation, including the HotpotQA case study with +23% accuracy improvement.

### Common Pitfalls to Avoid

- **Too Vague**: "Make the code better" → Specify exactly what "better" means
- **Too Restrictive**: Over-constraining can prevent useful optimizations
- **Missing Context**: Include relevant domain knowledge and terminology
- **No Examples**: Concrete examples guide LLM better than abstract descriptions
- **Ignoring Artifacts**: Don't refine prompts based on error feedback
