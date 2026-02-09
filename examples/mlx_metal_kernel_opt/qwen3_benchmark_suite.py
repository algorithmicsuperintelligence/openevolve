"""
Comprehensive Benchmark Suite for Qwen3-0.6B Metal Kernel Optimization
======================================================================

This benchmark suite tests various scenarios to establish baseline performance
and validate evolved Metal kernel optimizations. Tests the custom Metal kernel
discovered by OpenEvolve against MLX's standard attention implementation.

Target Model: mlx-community/Qwen3-0.6B-bf16
Target Hardware: Apple M4 24GB
Optimization: Custom Metal kernel for GQA attention (16 query heads : 8 KV heads)
Baseline: mx.fast.scaled_dot_product_attention
"""

import time
import json
import subprocess
import tempfile
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class BenchmarkResult:
    """Single benchmark result"""

    name: str
    prompt_tokens: int
    generated_tokens: int
    prefill_tokens_per_sec: float
    decode_tokens_per_sec: float
    total_tokens_per_sec: float
    peak_memory_gb: float
    total_time_sec: float
    prompt: str
    generated_text: str


@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""

    name: str
    prompt: str
    max_tokens: int
    description: str


class Qwen3BenchmarkSuite:
    """Comprehensive benchmark suite for Qwen3-0.6B Metal kernel optimization"""

    def __init__(
        self,
        model_path: str = "mlx-community/Qwen3-0.6B-bf16",
        hook_program_path: Optional[str] = None,
    ):
        self.model_path = model_path
        # When set, benchmarks will run via `mlx_lm_generate_with_hook.py` so that
        # the attention monkey-patch is applied inside the subprocess.
        self.hook_program_path = hook_program_path
        self.results: List[BenchmarkResult] = []

    def create_benchmark_configs(self) -> List[BenchmarkConfig]:
        """Create comprehensive benchmark configurations"""

        configs = []

        # 1. Context Length Variations
        configs.extend(
            [
                BenchmarkConfig(
                    name="short_context_quick",
                    prompt="Brief answer: What is artificial intelligence?",
                    max_tokens=50,
                    description="Short context, quick response - chat scenario",
                ),
                BenchmarkConfig(
                    name="medium_context_analysis",
                    prompt=self._create_medium_context_prompt(),
                    max_tokens=200,
                    description="Medium context, analytical response",
                ),
                BenchmarkConfig(
                    name="long_context_detailed",
                    prompt=self._create_long_context_prompt(),
                    max_tokens=500,
                    description="Long context, detailed analysis",
                ),
                BenchmarkConfig(
                    name="very_long_context_comprehensive",
                    prompt=self._create_very_long_context_prompt(),
                    max_tokens=1000,
                    description="Very long context, comprehensive response",
                ),
            ]
        )

        # 2. Generation Length Patterns
        configs.extend(
            [
                BenchmarkConfig(
                    name="micro_generation",
                    prompt="Complete this sentence: The future of AI is",
                    max_tokens=10,
                    description="Micro generation - attention prefill dominated",
                ),
                BenchmarkConfig(
                    name="short_generation",
                    prompt="Explain in one paragraph: What makes transformers effective?",
                    max_tokens=100,
                    description="Short generation - balanced prefill/decode",
                ),
                BenchmarkConfig(
                    name="long_generation",
                    prompt="Write a detailed technical explanation of how neural networks learn:",
                    max_tokens=1000,
                    description="Long generation - decode performance critical",
                ),
                BenchmarkConfig(
                    name="very_long_generation",
                    prompt="Write a comprehensive guide to machine learning for beginners:",
                    max_tokens=2000,
                    description="Very long generation - sustained decode performance",
                ),
                BenchmarkConfig(
                    name="ultra_long_generation",
                    prompt="The future of AI is",
                    max_tokens=5000,
                    description="Ultra long generation - memory scaling test",
                ),
            ]
        )

        # 3. Different Use Case Patterns
        configs.extend(
            [
                BenchmarkConfig(
                    name="code_generation",
                    prompt="""Write a Python function to implement binary search:

def binary_search(arr, target):
    \"\"\"
    Implement binary search algorithm
    Args:
        arr: sorted array
        target: element to find
    Returns:
        index of target or -1 if not found
    \"\"\"
""",
                    max_tokens=300,
                    description="Code generation - structured output patterns",
                ),
                BenchmarkConfig(
                    name="step_by_step_reasoning",
                    prompt="""Solve this step by step:

A train travels from City A to City B at 80 mph. The distance is 240 miles. 
If it leaves at 2:00 PM, what time will it arrive? Show your work.""",
                    max_tokens=400,
                    description="Step-by-step reasoning - logical sequence patterns",
                ),
                BenchmarkConfig(
                    name="creative_writing",
                    prompt="""Write a short story about a robot who discovers emotions for the first time. 
Include dialogue and describe the robot's internal experience as it learns about feelings like 
joy, sadness, and wonder. Make it engaging and thoughtful.""",
                    max_tokens=800,
                    description="Creative writing - diverse vocabulary and narrative",
                ),
                BenchmarkConfig(
                    name="technical_documentation",
                    prompt="""Create comprehensive documentation for a REST API with the following endpoints:
- GET /users - List all users
- POST /users - Create new user  
- GET /users/{id} - Get specific user
- PUT /users/{id} - Update user
- DELETE /users/{id} - Delete user

Include request/response examples, error codes, and authentication details.""",
                    max_tokens=1200,
                    description="Technical documentation - structured information",
                ),
                BenchmarkConfig(
                    name="conversational_assistant",
                    prompt="""You are a helpful AI assistant. A user asks:

"I'm planning a trip to Japan for 2 weeks. I've never been there before. I like 
history, food, and nature. I have a moderate budget. Can you help me plan an 
itinerary with recommendations for cities to visit, things to do, and travel tips?"

Provide a detailed, helpful response:""",
                    max_tokens=1500,
                    description="Conversational assistant - helpful response patterns",
                ),
            ]
        )

        # 4. Memory Pressure Scenarios
        configs.extend(
            [
                BenchmarkConfig(
                    name="progressive_context_building",
                    prompt=self._create_progressive_context_prompt(),
                    max_tokens=600,
                    description="Progressive context building - KV cache growth",
                ),
                BenchmarkConfig(
                    name="repetitive_pattern_generation",
                    prompt="Generate a list of 100 creative product names for a tech startup, with explanations:",
                    max_tokens=2000,
                    description="Repetitive patterns - memory efficiency test",
                ),
            ]
        )

        # 5. Extended Long Generation Tests (for sustained decode performance)
        configs.extend(
            [
                BenchmarkConfig(
                    name="extreme_long_generation",
                    prompt="Write a complete tutorial on deep learning from basics to advanced topics, including mathematical foundations, architectures, training techniques, and real-world applications:",
                    max_tokens=8000,
                    description="Extreme long generation - maximum decode performance test",
                ),
                BenchmarkConfig(
                    name="sustained_dialogue_generation",
                    prompt="Create a detailed dialogue between an AI researcher and a software engineer discussing the future of artificial intelligence, covering topics like AGI, safety, ethics, and technological implications. Make it engaging and informative:",
                    max_tokens=6000,
                    description="Sustained dialogue - consistent long-form generation",
                ),
                BenchmarkConfig(
                    name="comprehensive_analysis_generation",
                    prompt="Analyze the evolution of computer programming languages from assembly to modern high-level languages. Discuss paradigms, performance considerations, developer productivity, and future trends:",
                    max_tokens=7000,
                    description="Comprehensive analysis - complex reasoning with long output",
                ),
                BenchmarkConfig(
                    name="maximum_context_stress_test",
                    prompt=self._create_maximum_context_prompt(),
                    max_tokens=10000,
                    description="Maximum context stress test - ultimate performance challenge",
                ),
            ]
        )

        return configs

    def _create_medium_context_prompt(self) -> str:
        """Create medium-length context prompt"""
        return """Context: Machine learning has revolutionized many industries in recent years. 
From healthcare diagnosis to autonomous vehicles, AI systems are becoming increasingly 
sophisticated. However, challenges remain in areas like interpretability, fairness, 
and robustness. Recent advances in transformer architectures have shown remarkable 
capabilities in natural language processing, while computer vision has benefited 
from innovations in convolutional neural networks and attention mechanisms.

Question: Based on this context, analyze the current state of AI development and 
predict the most important research directions for the next 5 years. Consider both 
technical advances and societal implications."""

    def _create_long_context_prompt(self) -> str:
        """Create long context prompt"""
        return """Research Paper Summary:

Title: "Advances in Large Language Models: Architecture, Training, and Applications"

Abstract: This paper reviews recent developments in large language models (LLMs), 
focusing on architectural innovations, training methodologies, and real-world applications. 
We examine the evolution from early transformer models to current state-of-the-art systems, 
analyzing key improvements in efficiency, capability, and safety.

Introduction: The field of natural language processing has undergone a paradigm shift 
with the introduction of transformer-based architectures. Starting with the original 
Transformer paper in 2017, we have witnessed exponential growth in model size and 
capability. From GPT-1's 117M parameters to models with hundreds of billions of parameters, 
the scaling trend has consistently led to emergent capabilities.

Architecture Evolution: Modern LLMs incorporate several key innovations:
1. Attention mechanisms have evolved from basic dot-product attention to more efficient 
variants like sparse attention, local attention, and grouped query attention (GQA).
2. Position encoding schemes have advanced from sinusoidal embeddings to learnable 
position encodings and rotary position embeddings (RoPE).
3. Normalization techniques have shifted from post-norm to pre-norm configurations, 
with RMSNorm becoming preferred over LayerNorm for efficiency.
4. Activation functions have evolved from ReLU to GELU to SwiGLU for better performance.

Training Methodologies: The training of LLMs involves several sophisticated techniques:
- Pre-training on diverse text corpora using next-token prediction
- Instruction tuning to align models with human preferences
- Reinforcement learning from human feedback (RLHF)
- Constitutional AI for improved safety and alignment

Question: Given this comprehensive background, provide a detailed analysis of how 
these architectural and training advances specifically impact inference efficiency 
on mobile and edge devices. Consider memory requirements, computational complexity, 
and potential optimization strategies."""

    def _create_very_long_context_prompt(self) -> str:
        """Create very long context prompt to test KV cache scaling"""
        base_context = self._create_long_context_prompt()

        extended_context = (
            base_context
            + """

Detailed Technical Analysis:

Model Architecture Deep Dive:
The transformer architecture consists of an encoder-decoder structure, though many 
modern LLMs use decoder-only architectures. The core components include:

1. Multi-Head Attention Mechanism:
   - Allows the model to focus on different parts of the input simultaneously
   - Scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
   - Multiple attention heads capture different types of relationships
   - Grouped Query Attention (GQA) reduces memory requirements by sharing key-value pairs

2. Feed-Forward Networks:
   - Two linear transformations with a non-linear activation in between
   - Typically 4x the hidden dimension for the intermediate layer
   - SwiGLU activation: SwiGLU(x) = Swish(xW_1) ⊙ (xW_2)
   - Crucial for the model's capacity to learn complex patterns

3. Layer Normalization:
   - RMSNorm: RMSNorm(x) = x / RMS(x) * g, where RMS(x) = √(1/n Σx_i²)
   - Applied before each sub-layer (pre-norm) for training stability
   - Critical for deep network training convergence

4. Position Encodings:
   - Rotary Position Embedding (RoPE) rotates query and key vectors
   - Enables length generalization beyond training context
   - More efficient than absolute position encodings

Training Optimization Techniques:
- Gradient accumulation for effective large batch training
- Mixed precision training using bfloat16 for memory efficiency
- Gradient clipping to prevent exploding gradients
- Learning rate scheduling with warmup and decay
- Data parallelism and model parallelism for distributed training

Hardware Considerations:
Modern LLM training requires specialized hardware:
- GPUs with high memory bandwidth (A100, H100)
- Tensor cores optimized for mixed precision operations
- High-speed interconnects for multi-GPU training
- Efficient memory hierarchies for large model parameters

Inference Optimization Strategies:
- KV caching to avoid recomputing attention weights
- Quantization techniques (INT8, INT4) to reduce memory footprint
- Pruning methods to remove redundant parameters
- Distillation to create smaller, faster models
- Speculative decoding for improved throughput

Now, considering all this technical detail and the specific challenges of deploying 
large language models on resource-constrained devices, provide a comprehensive 
analysis of optimization strategies specifically for Apple Silicon devices, 
considering unified memory architecture, Metal Performance Shaders, and the 
specific computational characteristics of M-series chips."""
        )

        return extended_context

    def _create_progressive_context_prompt(self) -> str:
        """Create prompt that builds context progressively"""
        return """Chapter 1: The Beginning

In the early days of artificial intelligence, researchers dreamed of creating 
machines that could think and reason like humans. The field began in the 1950s 
with pioneers like Alan Turing, who proposed the famous Turing Test as a measure 
of machine intelligence.

Chapter 2: Early Developments  

The 1960s and 1970s saw the development of expert systems and symbolic AI. 
Researchers focused on rule-based systems that could encode human knowledge 
in formal logical structures. However, these systems were brittle and couldn't 
handle uncertainty or learning.

Chapter 3: The Neural Network Revolution

The 1980s brought renewed interest in neural networks, inspired by biological 
neurons. Backpropagation was rediscovered, enabling the training of multi-layer 
networks. This marked the beginning of connectionist AI approaches.

Chapter 4: Machine Learning Boom

The 1990s and 2000s saw machine learning become dominant. Support vector machines, 
random forests, and ensemble methods proved effective for many practical problems. 
The internet provided vast amounts of data to train these systems.

Chapter 5: Deep Learning Era

The 2010s marked the deep learning revolution. Convolutional neural networks 
revolutionized computer vision, recurrent networks advanced natural language 
processing, and deep reinforcement learning achieved superhuman performance 
in games like Go and Chess.

Now, continue this historical narrative by writing Chapter 6, focusing on the 
transformer era and large language models. Discuss the key innovations, 
breakthrough applications, and current challenges in the field."""

    def _create_maximum_context_prompt(self) -> str:
        """Create maximum length context prompt for stress testing"""
        base_context = self._create_very_long_context_prompt()

        extended_context = (
            base_context
            + """

Further Technical Deep Dive:

Advanced Optimization Techniques:
Modern LLM optimization goes beyond basic training approaches. Key areas include:

1. Memory Optimization:
   - Gradient checkpointing to trade compute for memory
   - Model parallelism across multiple devices
   - ZeRO optimizer states for distributed training
   - Mixed precision training with automatic loss scaling
   - Activation recomputation strategies

2. Computational Efficiency:
   - Flash Attention for memory-efficient attention computation
   - Gradient accumulation for effective large batch sizes
   - Dynamic loss scaling for stable mixed precision training
   - Automatic mixed precision (AMP) for optimal performance
   - Custom CUDA kernels for specific operations

3. Distributed Training Strategies:
   - Data parallelism with all-reduce communication
   - Model parallelism for very large models
   - Pipeline parallelism for sequential processing
   - 3D parallelism combining all approaches
   - Efficient communication backends (NCCL, Gloo)

4. Apple Silicon Specific Optimizations:
   - Unified memory architecture advantages
   - Metal Performance Shaders (MPS) acceleration
   - Neural Engine utilization for specific operations
   - Memory bandwidth optimization for M-series chips
   - Custom MLX primitives for Apple hardware

Inference Optimization Deep Dive:
Optimizing LLM inference requires different strategies than training:

1. Model Compression:
   - Quantization to 8-bit or 4-bit precision
   - Pruning redundant parameters
   - Knowledge distillation to smaller models
   - Low-rank approximations
   - Sparsity-aware inference engines

2. Runtime Optimization:
   - KV cache management for autoregressive generation
   - Batch processing for multiple requests
   - Dynamic batching for variable sequence lengths
   - Speculative decoding for faster generation
   - Continuous batching for improved throughput

3. Hardware-Specific Optimization:
   - GPU kernel fusion for reduced memory transfers
   - CPU optimization with vectorized operations
   - Mobile optimization for edge deployment
   - FPGA acceleration for specific use cases
   - Neuromorphic computing for ultra-low power

4. Serving Infrastructure:
   - Model serving frameworks (TensorRT, TorchServe)
   - Load balancing across multiple instances
   - Auto-scaling based on demand
   - Caching strategies for common requests
   - Request prioritization and queuing

Emerging Paradigms:
The field continues to evolve with new approaches:

1. Architecture Innovations:
   - Mixture of Experts (MoE) for conditional computation
   - State Space Models for long sequence modeling
   - Retrieval-augmented generation (RAG) systems
   - Multi-modal models combining text, vision, and audio
   - Constitutional AI for aligned behavior

2. Training Innovations:
   - Reinforcement Learning from Human Feedback (RLHF)
   - Constitutional AI training approaches
   - Curriculum learning for improved convergence
   - Meta-learning for few-shot adaptation
   - Continual learning to avoid catastrophic forgetting

3. Evaluation and Safety:
   - Comprehensive benchmark suites
   - Adversarial testing for robustness
   - Bias detection and mitigation
   - Interpretability and explainability
   - Safety alignment techniques

Real-World Deployment Challenges:
Deploying LLMs in production involves numerous considerations:

1. Scalability:
   - Handling millions of concurrent users
   - Geographic distribution for low latency
   - Cost optimization for sustainable operations
   - Resource allocation and scheduling
   - Auto-scaling based on demand patterns

2. Reliability:
   - Fault tolerance and error recovery
   - Monitoring and alerting systems
   - A/B testing for model updates
   - Gradual rollouts for risk mitigation
   - Backup systems for high availability

3. Security and Privacy:
   - Data protection and encryption
   - Secure model serving environments
   - Privacy-preserving inference techniques
   - Audit trails and compliance
   - Protection against adversarial attacks

Future Directions:
The field continues to advance rapidly with several promising directions:

1. Efficiency Improvements:
   - Novel architectures with better scaling properties
   - More efficient training algorithms
   - Better hardware-software co-design
   - Energy-efficient computing approaches
   - Sustainable AI development practices

2. Capability Enhancement:
   - Improved reasoning and planning abilities
   - Better multi-modal understanding
   - Enhanced code generation capabilities
   - Scientific discovery applications
   - Creative and artistic applications

3. Democratization:
   - Open-source model development
   - Accessible training and inference tools
   - Educational resources and tutorials
   - Community-driven improvements
   - Ethical AI development practices

Given this comprehensive overview of the current state and future directions of large language model optimization, provide a detailed analysis of how these various optimization techniques specifically apply to Apple Silicon hardware, particularly focusing on the M4 chip architecture, unified memory advantages, and how developers can best leverage these capabilities for maximum performance in LLM inference workloads."""
        )

        return extended_context

    def run_single_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run a single benchmark configuration with proper warmup"""
        print(f"\n{'='*60}")
        print(f"Running: {config.name}")
        print(f"Description: {config.description}")
        print(f"Max tokens: {config.max_tokens}")
        print(f"{'='*60}")

        # Performance measurement parameters
        WARMUP_RUNS = 2  # Warmup runs to eliminate cold start effects
        MEASUREMENT_RUNS = 3  # Multiple measurement runs for reliability

        # Create temporary prompt file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(config.prompt)
            prompt_file = f.name

        try:
            # Build command
            if self.hook_program_path:
                wrapper_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "mlx_lm_generate_with_hook.py",
                )
                cmd = [
                    sys.executable,
                    "-W", "ignore::RuntimeWarning",  # Suppress harmless import warnings
                    wrapper_path,
                    "--hook-program",
                    self.hook_program_path,
                    "--model",
                    self.model_path,
                    "--prompt",
                    config.prompt,
                    "--max-tokens",
                    str(config.max_tokens),
                ]
            else:
                cmd = [
                    sys.executable,
                    "-W", "ignore::RuntimeWarning",  # Suppress harmless import warnings
                    "-m",
                    "mlx_lm.generate",
                    "--model",
                    self.model_path,
                    "--prompt",
                    config.prompt,
                    "--max-tokens",
                    str(config.max_tokens),
                ]

            # Clear MLX cache before starting
            print(f"🧹 Clearing MLX cache...")
            mx.clear_cache()

            # Warmup runs - don't measure these
            print(f"🔥 Running {WARMUP_RUNS} warmup runs to eliminate cold start effects...")
            for i in range(WARMUP_RUNS):
                try:
                    print(f"   Warmup run {i+1}/{WARMUP_RUNS}...")
                    warmup_result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    if warmup_result.returncode != 0:
                        # Filter out harmless warnings from stderr
                        stderr_clean = "\n".join(
                            line for line in warmup_result.stderr.split("\n")
                            if "RuntimeWarning" not in line and line.strip()
                        )
                        if stderr_clean:
                            print(f"   ⚠️  Warmup run {i+1} failed (code {warmup_result.returncode}): {stderr_clean[:200]}...")
                        else:
                            print(f"   ⚠️  Warmup run {i+1} failed (code {warmup_result.returncode})")
                    else:
                        print(f"   ✅ Warmup run {i+1} completed")

                    # Clear cache and add small delay between runs to reduce GPU contention
                    mx.clear_cache()
                    time.sleep(0.5)

                except subprocess.TimeoutExpired:
                    print(f"   ⏰ Warmup run {i+1} timed out")
                except Exception as e:
                    print(f"   ❌ Warmup run {i+1} error: {e}")

            print(f"📊 Running {MEASUREMENT_RUNS} measurement runs...")

            # Measurement runs
            successful_results = []
            for run_idx in range(MEASUREMENT_RUNS):
                try:
                    print(f"   Measurement run {run_idx+1}/{MEASUREMENT_RUNS}...")

                    # Clear cache before each measurement run for consistency
                    mx.clear_cache()
                    initial_memory = mx.get_active_memory()

                    # Run benchmark
                    start_time = time.perf_counter()
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    end_time = time.perf_counter()

                    if result.returncode != 0:
                        # Filter out harmless warnings from stderr
                        stderr_clean = "\n".join(
                            line for line in result.stderr.split("\n")
                            if "RuntimeWarning" not in line and line.strip()
                        )
                        if stderr_clean:
                            print(f"   ❌ Measurement run {run_idx+1} failed (code {result.returncode}): {stderr_clean[:200]}...")
                        else:
                            print(f"   ❌ Measurement run {run_idx+1} failed (code {result.returncode})")
                        time.sleep(0.5)  # Small delay before retry
                        continue

                    # Parse output
                    parsed_result = self._parse_benchmark_output(
                        result.stdout, config, end_time - start_time
                    )

                    if parsed_result:
                        successful_results.append(parsed_result)
                        print(
                            f"   ✅ Run {run_idx+1}: {parsed_result.decode_tokens_per_sec:.1f} tokens/sec"
                        )
                    else:
                        print(f"   ❌ Run {run_idx+1}: Failed to parse output")

                except subprocess.TimeoutExpired:
                    print(f"   ⏰ Measurement run {run_idx+1} timed out")
                except Exception as e:
                    print(f"   ❌ Measurement run {run_idx+1} error: {e}")

            # Require at least 2 successful runs for reliable results
            if len(successful_results) < 2:
                print(
                    f"❌ Only {len(successful_results)}/{MEASUREMENT_RUNS} measurement runs succeeded"
                )
                print(f"❌ Need at least 2 successful runs for reliable results")
                raise RuntimeError(
                    f"Insufficient successful runs: {len(successful_results)}/{MEASUREMENT_RUNS}"
                )

            # Calculate statistics from multiple runs
            decode_speeds = [r.decode_tokens_per_sec for r in successful_results]
            prefill_speeds = [r.prefill_tokens_per_sec for r in successful_results]
            memories = [r.peak_memory_gb for r in successful_results]
            times = [r.total_time_sec for r in successful_results]

            # Use median for more robust results (less sensitive to outliers)
            final_result = BenchmarkResult(
                name=config.name,
                prompt_tokens=int(np.median([r.prompt_tokens for r in successful_results])),
                generated_tokens=int(np.median([r.generated_tokens for r in successful_results])),
                prefill_tokens_per_sec=float(np.median(prefill_speeds)),
                decode_tokens_per_sec=float(np.median(decode_speeds)),
                total_tokens_per_sec=float(
                    np.median([r.total_tokens_per_sec for r in successful_results])
                ),
                peak_memory_gb=float(np.median(memories)),
                total_time_sec=float(np.median(times)),
                prompt=config.prompt[:200] + "..." if len(config.prompt) > 200 else config.prompt,
                generated_text=successful_results[0].generated_text,  # Use first result's text
            )

            # Print final results with statistics
            print(f"\n📈 Final Results (median of {len(successful_results)} runs):")
            print(f"  Prompt tokens: {final_result.prompt_tokens}")
            print(f"  Generated tokens: {final_result.generated_tokens}")
            print(f"  Prefill speed: {final_result.prefill_tokens_per_sec:.2f} tokens/sec")
            print(
                f"  Decode speed: {final_result.decode_tokens_per_sec:.2f} tokens/sec (σ={np.std(decode_speeds):.2f})"
            )
            print(f"  Overall speed: {final_result.total_tokens_per_sec:.2f} tokens/sec")
            print(f"  Peak memory: {final_result.peak_memory_gb:.3f} GB")
            print(f"  Total time: {final_result.total_time_sec:.2f} seconds")

            if len(decode_speeds) > 1:
                print(
                    f"  Performance consistency: {np.std(decode_speeds)/np.mean(decode_speeds)*100:.1f}% CV"
                )

            return final_result

        finally:
            # Clean up
            if os.path.exists(prompt_file):
                os.unlink(prompt_file)

    def _parse_benchmark_output(
        self, stdout: str, config: BenchmarkConfig, total_time: float
    ) -> Optional[BenchmarkResult]:
        """Parse mlx-lm output to extract performance metrics"""
        output_lines = stdout.strip().split("\n")

        # Find the generated text (between ========== markers)
        generated_text = ""
        in_generation = False
        prompt_tokens = 0
        generation_tokens = 0
        prompt_speed = 0.0
        generation_speed = 0.0
        peak_memory_str = ""

        for line in output_lines:
            if line.strip() == "==========":
                in_generation = not in_generation
            elif in_generation:
                generated_text += line + "\n"
            elif "Prompt:" in line and "tokens-per-sec" in line:
                # Parse: "Prompt: 13 tokens, 310.367 tokens-per-sec"
                parts = line.split(",")
                prompt_tokens = int(parts[0].split(":")[1].strip().split()[0])
                prompt_speed = float(parts[1].strip().split()[0])
            elif "Generation:" in line and "tokens-per-sec" in line:
                # Parse: "Generation: 468 tokens, 69.860 tokens-per-sec"
                parts = line.split(",")
                generation_tokens = int(parts[0].split(":")[1].strip().split()[0])
                generation_speed = float(parts[1].strip().split()[0])
            elif "Peak memory:" in line:
                peak_memory_str = line.split(":")[1].strip()

        # Parse peak memory
        peak_memory_gb = 0.0
        if peak_memory_str:
            if "GB" in peak_memory_str:
                peak_memory_gb = float(peak_memory_str.replace("GB", "").strip())
            elif "MB" in peak_memory_str:
                peak_memory_gb = float(peak_memory_str.replace("MB", "").strip()) / 1024

        # Validate we got meaningful results
        if generation_tokens == 0 or generation_speed == 0:
            return None

        # Calculate overall tokens per second
        total_tokens_per_sec = generation_tokens / total_time if total_time > 0 else 0

        return BenchmarkResult(
            name=config.name,
            prompt_tokens=prompt_tokens,
            generated_tokens=generation_tokens,
            prefill_tokens_per_sec=prompt_speed,
            decode_tokens_per_sec=generation_speed,
            total_tokens_per_sec=total_tokens_per_sec,
            peak_memory_gb=peak_memory_gb,
            total_time_sec=total_time,
            prompt=config.prompt[:200] + "..." if len(config.prompt) > 200 else config.prompt,
            generated_text=(
                generated_text.strip()[:200] + "..."
                if len(generated_text.strip()) > 200
                else generated_text.strip()
            ),
        )

    def run_full_benchmark_suite(self) -> Dict:
        """Run the complete benchmark suite"""
        print(f"\n{'='*80}")
        print(f"Qwen3-0.6B Comprehensive Benchmark Suite")
        print(f"Model: {self.model_path}")
        print(f"Hardware: Apple M4 24GB")
        print(f"Target: Custom Metal kernel optimization validation")
        print(f"{'='*80}")

        configs = self.create_benchmark_configs()
        results = []

        for i, config in enumerate(configs, 1):
            print(f"\n[{i}/{len(configs)}] Starting benchmark: {config.name}")
            try:
                result = self.run_single_benchmark(config)
                results.append(result)
                self.results.append(result)
            except Exception as e:
                print(f"Failed to run benchmark {config.name}: {e}")
                continue

        # Generate summary
        summary = self.generate_summary(results)
        self.save_results(results, summary)

        return {"results": [self._result_to_dict(r) for r in results], "summary": summary}

    def generate_summary(self, results: List[BenchmarkResult]) -> Dict:
        """Generate benchmark summary statistics"""
        if not results:
            return {}

        # Overall statistics
        decode_speeds = [r.decode_tokens_per_sec for r in results if r.decode_tokens_per_sec > 0]
        prefill_speeds = [r.prefill_tokens_per_sec for r in results if r.prefill_tokens_per_sec > 0]
        memories = [r.peak_memory_gb for r in results if r.peak_memory_gb > 0]

        summary = {
            "total_benchmarks": len(results),
            "avg_decode_speed": np.mean(decode_speeds) if decode_speeds else 0,
            "min_decode_speed": np.min(decode_speeds) if decode_speeds else 0,
            "max_decode_speed": np.max(decode_speeds) if decode_speeds else 0,
            "avg_prefill_speed": np.mean(prefill_speeds) if prefill_speeds else 0,
            "min_prefill_speed": np.min(prefill_speeds) if prefill_speeds else 0,
            "max_prefill_speed": np.max(prefill_speeds) if prefill_speeds else 0,
            "avg_memory_usage": np.mean(memories) if memories else 0,
            "max_memory_usage": np.max(memories) if memories else 0,
            "min_memory_usage": np.min(memories) if memories else 0,
        }

        # Category analysis
        categories = {
            "context_length": [r for r in results if "context" in r.name],
            "generation_length": [r for r in results if "generation" in r.name],
            "use_cases": [
                r
                for r in results
                if any(
                    x in r.name
                    for x in ["code", "reasoning", "creative", "technical", "conversational"]
                )
            ],
            "memory_pressure": [
                r for r in results if any(x in r.name for x in ["progressive", "repetitive"])
            ],
        }

        for category, cat_results in categories.items():
            if cat_results:
                cat_decode_speeds = [
                    r.decode_tokens_per_sec for r in cat_results if r.decode_tokens_per_sec > 0
                ]
                summary[f"{category}_avg_decode_speed"] = (
                    np.mean(cat_decode_speeds) if cat_decode_speeds else 0
                )
                summary[f"{category}_count"] = len(cat_results)

        return summary

    def save_results(self, results: List[BenchmarkResult], summary: Dict):
        """Save benchmark results to files"""
        timestamp = int(time.time())

        # Save detailed results
        detailed_results = {
            "timestamp": timestamp,
            "model": self.model_path,
            "hardware": "Apple M4 24GB",
            "optimization": "Custom Metal kernel for GQA attention",
            "mlx_version": mx.__version__,
            "results": [self._result_to_dict(r) for r in results],
            "summary": summary,
        }

        with open(f"qwen3_benchmark_results_{timestamp}.json", "w") as f:
            json.dump(detailed_results, f, indent=2)

        # Save CSV for easy analysis
        import csv

        with open(f"qwen3_benchmark_results_{timestamp}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "name",
                    "description",
                    "prompt_tokens",
                    "generated_tokens",
                    "prefill_tokens_per_sec",
                    "decode_tokens_per_sec",
                    "total_tokens_per_sec",
                    "peak_memory_gb",
                    "total_time_sec",
                ]
            )

            configs = self.create_benchmark_configs()
            config_dict = {c.name: c for c in configs}

            for result in results:
                config = config_dict.get(result.name)
                writer.writerow(
                    [
                        result.name,
                        config.description if config else "",
                        result.prompt_tokens,
                        result.generated_tokens,
                        result.prefill_tokens_per_sec,
                        result.decode_tokens_per_sec,
                        result.total_tokens_per_sec,
                        result.peak_memory_gb,
                        result.total_time_sec,
                    ]
                )

        print(f"\n{'='*60}")
        print(f"Results saved to:")
        print(f"  - qwen3_benchmark_results_{timestamp}.json")
        print(f"  - qwen3_benchmark_results_{timestamp}.csv")
        print(f"{'='*60}")

    def _result_to_dict(self, result: BenchmarkResult) -> Dict:
        """Convert BenchmarkResult to dictionary"""
        return {
            "name": result.name,
            "prompt_tokens": result.prompt_tokens,
            "generated_tokens": result.generated_tokens,
            "prefill_tokens_per_sec": result.prefill_tokens_per_sec,
            "decode_tokens_per_sec": result.decode_tokens_per_sec,
            "total_tokens_per_sec": result.total_tokens_per_sec,
            "peak_memory_gb": result.peak_memory_gb,
            "total_time_sec": result.total_time_sec,
            "prompt": result.prompt,
            "generated_text": result.generated_text,
        }

    def print_summary_table(self):
        """Print a summary table of all results"""
        if not self.results:
            print("No benchmark results available")
            return

        print(f"\n{'='*120}")
        print(f"{'Benchmark Summary':^120}")
        print(f"{'='*120}")
        print(
            f"{'Name':<25} {'Tokens':<8} {'Prefill':<10} {'Decode':<10} {'Overall':<10} {'Memory':<8} {'Time':<8}"
        )
        print(f"{'='*120}")

        for result in self.results:
            print(
                f"{result.name:<25} "
                f"{result.generated_tokens:<8} "
                f"{result.prefill_tokens_per_sec:<10.1f} "
                f"{result.decode_tokens_per_sec:<10.1f} "
                f"{result.total_tokens_per_sec:<10.1f} "
                f"{result.peak_memory_gb:<8.2f} "
                f"{result.total_time_sec:<8.1f}"
            )

        print(f"{'='*120}")

        # Summary statistics
        decode_speeds = [
            r.decode_tokens_per_sec for r in self.results if r.decode_tokens_per_sec > 0
        ]
        if decode_speeds:
            print(f"Average decode speed: {np.mean(decode_speeds):.1f} tokens/sec")
            print(f"Best decode speed: {np.max(decode_speeds):.1f} tokens/sec")
            print(f"Worst decode speed: {np.min(decode_speeds):.1f} tokens/sec")


def main():
    """Run the complete benchmark suite"""
    print("Running Qwen3-0.6B Comprehensive Benchmark Suite")
    print("Ensure mlx-lm is installed: pip install mlx-lm")
    print("Target: Validate custom Metal kernel optimization performance")

    benchmark_suite = Qwen3BenchmarkSuite()
    results = benchmark_suite.run_full_benchmark_suite()
    benchmark_suite.print_summary_table()

    print(f"\n{'='*80}")
    print("Benchmark Suite Complete!")
    print("These results will serve as baseline for Metal kernel optimization validation.")
    print("Target: Improve decode speed by 10%+ through evolved custom Metal kernels")
    print(f"{'='*80}")

    return results


if __name__ == "__main__":
    main()
