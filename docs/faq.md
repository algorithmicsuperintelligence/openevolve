# FAQ

### How much does it cost to run?

See [Cost Estimation](llm-providers.md#cost-estimation) for detailed pricing information and cost-saving tips.

### 🆚 How does this compare to manual optimization?

| Aspect | Manual | OpenEvolve |
|--------|--------|------------|
| **Initial Learning** | Weeks to understand domain | Minutes to start |
| **Solution Quality** | Depends on expertise | Consistently explores novel approaches |
| **Time Investment** | Days-weeks per optimization | Hours for complete evolution |
| **Reproducibility** | Hard to replicate exact process | Perfect reproduction with seeds |
| **Scaling** | Doesn't scale beyond human capacity | Parallel evolution across islands |

**OpenEvolve shines** when you need to explore large solution spaces or optimize for multiple objectives simultaneously.

### Can I use my own LLM?

**Yes!** OpenEvolve supports any OpenAI-compatible API:

- **Commercial**: OpenAI, Google, Cohere
- **Local**: Ollama, vLLM, LM Studio, text-generation-webui
- **Advanced**: OptiLLM for routing and test-time compute

Just set the `api_base` in your config to point to your endpoint.

### What if evolution gets stuck?

**Built-in mechanisms prevent stagnation:**

- **Island migration**: Fresh genes from other populations
- **Temperature control**: Exploration vs exploitation balance
- **Diversity maintenance**: MAP-Elites prevents convergence
- **Artifact feedback**: Error messages guide improvements
- **Template stochasticity**: Randomized prompts break patterns

**Manual interventions:**
- Increase `num_diverse_programs` for more exploration
- Add custom feature dimensions to diversify search
- Use template variations to randomize prompts
- Adjust migration intervals for more cross-pollination

### How do I measure success?

**Multiple success metrics:**

1. **Primary Metric**: Your evaluator's `combined_score` or metric average
2. **Convergence**: Best score improvement over time
3. **Diversity**: MAP-Elites grid coverage
4. **Efficiency**: Iterations to reach target performance
5. **Robustness**: Performance across different test cases

**Use the visualizer** to track all metrics in real-time and identify when evolution has converged.
