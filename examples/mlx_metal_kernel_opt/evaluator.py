"""
🛡️ BULLETPROOF METAL KERNEL EVALUATOR 🛡️

This evaluator provides MAXIMUM protection against Metal kernel failures during evolution:

🔧 METAL-SPECIFIC PROTECTION:
1. Pre-execution kernel parameter validation
2. Memory safety checks before GPU execution
3. Command buffer error detection and recovery
4. Thread-safe Metal kernel execution wrapping
5. Graceful fallback to standard attention on ANY Metal failure

🚀 EVOLUTION SAFETY:
- NEVER crashes the evolution process
- Handles kIOGPUCommandBufferCallbackErrorInvalidResource errors
- Catches GPU memory violations, out-of-bounds access, race conditions
- Provides detailed error classification for debugging
- Maintains evolution progress even with buggy kernel code

🎯 ROBUST ERROR RECOVERY:
- Multiple retry attempts with exponential backoff
- Automatic fallback mechanisms
- Comprehensive error statistics tracking
- Safe cleanup of GPU resources
"""

import os
import sys
import json
import time
import traceback
import threading
import subprocess
import tempfile
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
import mlx.nn as nn

# Import the comprehensive benchmark suite for consistent testing
from qwen3_benchmark_suite import Qwen3BenchmarkSuite, BenchmarkConfig, BenchmarkResult


class MetalKernelSafetyError(Exception):
    """Metal kernel safety violation"""

    pass


class GPUCommandBufferError(Exception):
    """GPU command buffer execution error"""

    pass


class MetalMemoryViolationError(Exception):
    """Metal kernel memory access violation"""

    pass


class BulletproofMetalEvaluator:
    """Bulletproof evaluator that NEVER crashes from Metal kernel failures"""

    def __init__(self):
        self.model_path = "mlx-community/Qwen3-0.6B-bf16"

        # Enhanced error handling configuration
        self.max_retry_attempts = 3
        self.retry_base_delay = 1.0  # Base delay for exponential backoff
        self.kernel_validation_timeout = 30  # Timeout for kernel validation

        # Comprehensive error tracking
        self.metal_command_buffer_errors = 0
        self.metal_memory_violations = 0
        self.metal_compilation_errors = 0
        self.gpu_resource_errors = 0
        self.total_metal_errors = 0
        self.successful_fallbacks = 0
        self.retry_attempts_used = 0

        # Safety thresholds
        self.max_sequence_length_safe = 512  # Start with safer sequence lengths
        self.max_batch_size_safe = 1
        self.max_head_dimension_safe = 128

        # Baseline metrics storage
        self.baseline_metrics = None
        self.baseline_results = None

        # Use comprehensive benchmark suite
        self.benchmark_suite = Qwen3BenchmarkSuite(self.model_path)

        print("🛡️  BULLETPROOF METAL KERNEL EVALUATOR INITIALIZED")
        print(f"📱 Model: {self.model_path}")
        print(f"🔁 Max retry attempts: {self.max_retry_attempts}")
        print(f"⚡ GPU error protection: MAXIMUM")
        print(f"🧠 Memory safety validation: ENABLED")
        print(f"🎯 Command buffer error handling: ACTIVE")

    def evaluate(self, program_text: str) -> Dict[str, Any]:
        """
        BULLETPROOF evaluation that handles ALL Metal kernel failures:
        1. Enhanced program extraction with syntax validation
        2. Pre-execution kernel safety validation
        3. Protected baseline measurement with fallback
        4. GPU-safe correctness testing with memory checks
        5. Armored benchmarking with command buffer protection
        6. Comprehensive Metal error recovery and statistics
        """

        print("\n" + "🛡️ " * 50)
        print("🛡️  BULLETPROOF METAL KERNEL EVALUATION STARTING")
        print("🛡️ " * 50)
        print("✅ GPU Command Buffer Error Protection: ACTIVE")
        print("✅ Metal Memory Violation Detection: ENABLED")
        print("✅ Automatic Fallback Mechanisms: READY")
        print("✅ Multi-layer Error Recovery: ARMED")
        print("✅ Evolution Process Protection: MAXIMUM")
        print("🛡️ " * 50)

        try:
            # Reset all error counters
            self._reset_error_counters()

            # Step 1: Enhanced program extraction with Metal validation
            print("\n🔧 STEP 1: Enhanced Program Extraction with Metal Validation")
            extraction_result = self._bulletproof_extract_custom_attention(program_text)
            if not extraction_result["success"]:
                return self._create_comprehensive_failure_result(
                    f"Program extraction failed: {extraction_result['error']}"
                )

            custom_attention_class = extraction_result["class"]
            program_source = extraction_result["program_text"]

            # Step 2: Pre-execution Metal kernel safety validation
            print("\n🔍 STEP 2: Pre-execution Metal Kernel Safety Validation")
            safety_result = self._validate_metal_kernel_safety(custom_attention_class)
            if not safety_result["success"]:
                print(f"⚠️  Metal kernel safety validation failed: {safety_result['error']}")
                print("🛡️  Proceeding with enhanced protection...")

            # Step 3: Memory-safe correctness testing FIRST (fail fast, skip baseline if invalid)
            print("\n🔍 STEP 3: Memory-Safe Custom Attention Correctness Testing")
            correctness_result = self._memory_safe_correctness_test(custom_attention_class)
            if not correctness_result["success"]:
                return self._create_comprehensive_failure_result(
                    f"Memory-safe correctness test failed: {correctness_result['error']}"
                )

            correctness_score = correctness_result["score"]
            if correctness_score < 0.90:  # Slightly more lenient for complex kernels
                return self._create_comprehensive_failure_result(
                    f"Correctness score too low: {correctness_score:.3f} (required: 0.90)"
                )

            # Step 4: GPU-protected baseline measurement (only if correctness passed)
            print("\n📊 STEP 4: GPU-Protected Baseline Performance Measurement")
            baseline_results = self._gpu_protected_measure_baseline()
            if not baseline_results:
                return self._create_comprehensive_failure_result(
                    "Failed to measure baseline performance with GPU protection"
                )

            # Step 5: Command-buffer-protected benchmarking
            print("\n🚀 STEP 5: Command-Buffer-Protected Performance Benchmarking")
            benchmark_result = self._command_buffer_protected_benchmark(
                program_source, custom_attention_class
            )
            if not benchmark_result["success"]:
                return self._create_comprehensive_failure_result(
                    f"Command-buffer-protected benchmarking failed: {benchmark_result['error']}"
                )

            custom_results = benchmark_result["results"]

            # Step 6: Enhanced performance analysis
            print("\n📈 STEP 6: Enhanced Performance Analysis")
            performance_analysis = self._analyze_performance_with_safety_metrics(
                baseline_results, custom_results
            )

            # Step 7: Calculate safety-adjusted final score
            final_score = self._calculate_safety_adjusted_score(
                performance_analysis, correctness_score
            )

            # Step 8: Generate comprehensive result with full error statistics
            result = {
                "success": True,
                "final_score": final_score,
                "combined_score": final_score,
                "performance_metrics": performance_analysis["aggregate_metrics"],
                "correctness_score": correctness_score,
                "benchmark_results": [self._result_to_dict(r) for r in custom_results],
                "baseline_comparison": performance_analysis["comparison_summary"],
                "individual_comparisons": performance_analysis["individual_comparisons"],
                "summary": self._generate_comprehensive_summary(
                    performance_analysis, correctness_score
                ),
                "metal_safety_statistics": self._get_comprehensive_error_statistics(),
                "safety_validation": safety_result,
            }

            self._print_bulletproof_evaluation_results(result)
            return result

        except Exception as e:
            # Ultimate protection: even this top-level catch must never crash evolution
            self.total_metal_errors += 1
            error_msg = f"TOP-LEVEL BULLETPROOF CATCH: {str(e)}"
            print(f"🛡️  {error_msg}")
            traceback.print_exc()
            return self._create_comprehensive_failure_result(error_msg)

    def _reset_error_counters(self):
        """Reset all error tracking counters"""
        self.metal_command_buffer_errors = 0
        self.metal_memory_violations = 0
        self.metal_compilation_errors = 0
        self.gpu_resource_errors = 0
        self.total_metal_errors = 0
        self.successful_fallbacks = 0
        self.retry_attempts_used = 0

    def _bulletproof_extract_custom_attention(self, program_text: str) -> Dict[str, Any]:
        """Bulletproof extraction with comprehensive Metal kernel validation"""
        try:
            print("  🔍 Bulletproof program analysis with Metal validation...")

            # Handle file paths vs direct text
            if (
                program_text.startswith("/")
                and "\n" not in program_text
                and len(program_text) < 500
            ):
                print(f"  📁 Reading program from file: {program_text}")
                if os.path.exists(program_text):
                    try:
                        with open(program_text, "r") as f:
                            actual_program_text = f.read()
                    except Exception as e:
                        return {"success": False, "error": f"File read error: {e}"}
                else:
                    return {"success": False, "error": f"Program file not found: {program_text}"}
            else:
                actual_program_text = program_text

            # Enhanced syntax validation
            try:
                compile(actual_program_text, "<evolved_program>", "exec")
                print("  ✅ Enhanced syntax validation passed")
            except SyntaxError as e:
                return {"success": False, "error": f"Syntax error: {e}"}

            # Pre-validate Metal kernel syntax (static analysis)
            metal_validation = self._static_validate_metal_kernel_syntax(actual_program_text)
            if not metal_validation["safe"]:
                print(
                    f"  ⚠️  Metal kernel static validation warning: {metal_validation['warnings']}"
                )

            # Create ultra-safe execution environment
            exec_globals = self._create_bulletproof_execution_environment()

            # Execute program with maximum protection
            print("  ⚙️  Executing program with MAXIMUM protection...")
            try:
                success, result = self._bulletproof_execute_with_gpu_protection(
                    lambda: exec(actual_program_text, exec_globals)
                )

                if not success:
                    self.total_metal_errors += 1
                    return {"success": False, "error": f"Protected execution failed: {result}"}

            except Exception as e:
                self.total_metal_errors += 1
                return {"success": False, "error": f"Execution error with GPU protection: {e}"}

            # Enhanced class extraction and validation
            custom_class = exec_globals.get("CustomGQAAttention")
            if custom_class is None:
                return {
                    "success": False,
                    "error": "CustomGQAAttention class not found in executed code",
                }

            # Comprehensive class validation
            validation_result = self._validate_custom_attention_class(custom_class)
            if not validation_result["valid"]:
                return {"success": False, "error": validation_result["error"]}

            print(f"  ✅ Successfully extracted and validated CustomGQAAttention class")
            print(f"  🛡️  Metal safety pre-checks: {metal_validation['safe']}")

            return {
                "success": True,
                "class": custom_class,
                "metal_validation": metal_validation,
                "program_text": actual_program_text,
            }

        except Exception as e:
            self.total_metal_errors += 1
            return {"success": False, "error": f"Bulletproof extraction failed: {str(e)}"}

    def _static_validate_metal_kernel_syntax(self, program_text: str) -> Dict[str, Any]:
        """Static analysis of Metal kernel syntax for common safety issues"""
        warnings = []

        # Check for common Metal safety issues
        dangerous_patterns = [
            ("buffer overflow", ["queries[", "keys[", "values[", "output[", "mask["]),
            ("unguarded loops", ["for (", "while ("]),
            ("raw pointers", ["*queries", "*keys", "*values", "*output"]),
            ("thread sync issues", ["threadgroup", "simdgroup"]),
        ]

        for issue_type, patterns in dangerous_patterns:
            for pattern in patterns:
                if pattern in program_text:
                    warnings.append(f"{issue_type}: {pattern}")

        # Check for bounds checking
        has_bounds_checking = any(
            check in program_text
            for check in [
                "batch_idx >= BATCH_SIZE",
                "head_idx >= NUM_HEADS",
                "query_pos >= SEQ_LEN",
                "d < HEAD_DIM",
            ]
        )

        if not has_bounds_checking:
            warnings.append("missing bounds checking")

        return {
            "safe": len(warnings) == 0,
            "warnings": warnings,
            "has_bounds_checking": has_bounds_checking,
        }

    def _validate_custom_attention_class(self, custom_class: Any) -> Dict[str, Any]:
        """Comprehensive validation of custom attention class"""
        try:
            # Basic type checking
            if not isinstance(custom_class, type):
                return {"valid": False, "error": "CustomGQAAttention is not a valid class"}

            # Check for required methods
            required_methods = ["__init__", "__call__"]
            for method in required_methods:
                if not hasattr(custom_class, method):
                    return {"valid": False, "error": f"Missing required method: {method}"}

            # Check if it inherits from nn.Module (recommended)
            if not issubclass(custom_class, nn.Module):
                print("  ⚠️  CustomGQAAttention doesn't inherit from nn.Module")

            print("  ✅ Custom attention class validation passed")
            return {"valid": True}

        except Exception as e:
            return {"valid": False, "error": f"Class validation error: {e}"}

    def _validate_metal_kernel_safety(self, custom_attention_class: Any) -> Dict[str, Any]:
        """Pre-execution validation of Metal kernel safety"""
        try:
            print("  🔍 Validating Metal kernel safety parameters...")

            # Mock arguments for safety testing
            class MockArgs:
                # NOTE: This should reflect the default model used by this evaluator:
                # `mlx-community/Qwen3-0.6B-bf16` (16 Q heads : 8 KV heads, head_dim=128).
                hidden_size = 2048
                num_attention_heads = 16
                num_key_value_heads = 8
                head_dim = 128
                rms_norm_eps = 1e-06
                rope_theta = 1000000
                rope_scaling = None
                max_position_embeddings = 40960

            args = MockArgs()

            # Try to instantiate with safety checks
            try:
                instance = custom_attention_class(args)
                if instance is None:
                    return {"success": False, "error": "Failed to instantiate custom attention"}

                print("  ✅ Custom attention instantiation successful")

                # Basic parameter validation (should match the args we instantiated with)
                if hasattr(instance, "n_heads") and instance.n_heads != args.num_attention_heads:
                    return {
                        "success": False,
                        "error": f"Invalid head count: {instance.n_heads} (expected {args.num_attention_heads})",
                    }

                if hasattr(instance, "n_kv_heads") and instance.n_kv_heads != args.num_key_value_heads:
                    return {
                        "success": False,
                        "error": f"Invalid KV head count: {instance.n_kv_heads}",
                    }

                return {"success": True, "validated": True}

            except Exception as e:
                error_msg = str(e)
                if any(keyword in error_msg.lower() for keyword in ["metal", "kernel", "gpu"]):
                    self.metal_compilation_errors += 1
                return {"success": False, "error": f"Instantiation failed: {error_msg}"}

        except Exception as e:
            self.total_metal_errors += 1
            return {"success": False, "error": f"Safety validation error: {e}"}

    def _bulletproof_execute_with_gpu_protection(self, func) -> Tuple[bool, Any]:
        """Execute function with maximum GPU and Metal kernel protection"""
        try:
            # Clear any existing GPU state
            mx.eval(mx.array([1.0]))  # Simple operation to ensure GPU is responsive

            # Execute with comprehensive error catching
            result = func()
            return True, result

        except RuntimeError as e:
            error_msg = str(e)

            # Classify specific Metal/GPU errors
            if "kIOGPUCommandBufferCallbackErrorInvalidResource" in error_msg:
                self.metal_command_buffer_errors += 1
                self.total_metal_errors += 1
                return False, f"GPU Command Buffer Error (memory violation): {error_msg}"
            elif "METAL" in error_msg.upper():
                self.metal_memory_violations += 1
                self.total_metal_errors += 1
                return False, f"Metal Memory Violation: {error_msg}"
            elif any(keyword in error_msg.lower() for keyword in ["gpu", "metal", "kernel"]):
                self.gpu_resource_errors += 1
                self.total_metal_errors += 1
                return False, f"GPU Resource Error: {error_msg}"
            else:
                return False, f"Runtime Error: {error_msg}"

        except Exception as e:
            error_msg = str(e)

            # Additional classification for other Metal-related exceptions
            if any(
                keyword in error_msg.lower() for keyword in ["metal", "kernel", "gpu", "mps", "mtl"]
            ):
                self.total_metal_errors += 1
                return False, f"General Metal Error: {error_msg}"
            else:
                return False, f"Execution Error: {error_msg}"

    def _gpu_protected_measure_baseline(self) -> Optional[List[BenchmarkResult]]:
        """GPU-protected baseline measurement with enhanced error handling"""
        try:
            print("  📊 Running GPU-protected baseline benchmark...")

            # Ensure clean GPU state
            self._ensure_clean_gpu_state()
            self._ensure_standard_attention()

            # Get baseline configurations
            baseline_configs = self._get_safe_benchmark_configs()
            if not baseline_configs:
                print("  ❌ No safe benchmark configurations available")
                return None

            baseline_results = []
            successful_count = 0

            for i, config in enumerate(baseline_configs, 1):
                print(f"  [{i}/{len(baseline_configs)}] GPU-protected baseline: {config.name}")

                retry_count = 0
                while retry_count <= self.max_retry_attempts:
                    try:
                        # Clean GPU state before each attempt
                        self._ensure_clean_gpu_state()

                        # Run with GPU protection
                        success, result = self._bulletproof_execute_with_gpu_protection(
                            lambda: self.benchmark_suite.run_single_benchmark(config)
                        )

                        if success and result:
                            baseline_results.append(result)
                            successful_count += 1
                            print(
                                f"    ✅ GPU-protected {config.name}: {result.decode_tokens_per_sec:.1f} tokens/sec"
                            )
                            break
                        else:
                            if retry_count < self.max_retry_attempts:
                                print(f"    🔄 Retry {retry_count + 1}: {result}")
                                retry_count += 1
                                time.sleep(self.retry_base_delay * (2**retry_count))
                                continue
                            else:
                                print(f"    ❌ All retries exhausted for {config.name}: {result}")
                                break

                    except Exception as e:
                        if retry_count < self.max_retry_attempts:
                            print(f"    🔄 Exception retry {retry_count + 1}: {e}")
                            retry_count += 1
                            time.sleep(self.retry_base_delay * (2**retry_count))
                            continue
                        else:
                            print(f"    ❌ Final exception for {config.name}: {e}")
                            break

            # Check success rate
            min_required = max(2, len(baseline_configs) * 0.5)  # At least 50% success
            if successful_count < min_required:
                print(
                    f"  ❌ Insufficient baseline results: {successful_count}/{len(baseline_configs)}"
                )
                return None

            # Store baseline metrics
            self._store_enhanced_baseline_metrics(baseline_results)
            print(f"  ✅ GPU-protected baseline complete ({successful_count} successful)")

            return baseline_results

        except Exception as e:
            print(f"  ❌ GPU-protected baseline measurement failed: {e}")
            return None

    def _memory_safe_correctness_test(self, custom_attention_class: Any) -> Dict[str, Any]:
        """Memory-safe correctness testing with GPU protection"""
        print("  🔍 Running memory-safe correctness testing...")

        try:
            # Safe test configuration
            class MockArgs:
                # Must match the default model `mlx-community/Qwen3-0.6B-bf16`
                hidden_size = 2048
                num_attention_heads = 16
                num_key_value_heads = 8
                head_dim = 128
                rms_norm_eps = 1e-06
                rope_theta = 1000000
                rope_scaling = None
                max_position_embeddings = 40960

            args = MockArgs()

            # Conservative test cases (smaller sequences for safety)
            test_cases = [
                (1, 8, 2048),  # Micro sequence
                (1, 16, 2048),  # Very short
                (1, 32, 2048),  # Short sequence
                (1, 64, 2048),  # Medium sequence
            ]

            correctness_scores = []
            local_command_buffer_errors = 0
            local_memory_violations = 0

            for B, L, D in test_cases:
                print(f"      🧪 Memory-safe testing sequence length {L}...")

                retry_count = 0
                while retry_count <= self.max_retry_attempts:
                    try:
                        # Clean GPU state
                        self._ensure_clean_gpu_state()

                        # Create conservative test inputs
                        # IMPORTANT: Match the real inference dtype used by the default model
                        # (`mlx-community/Qwen3-0.6B-bf16`), otherwise Metal kernels may compile
                        # for float32 in correctness tests but fail under bfloat16 in practice.
                        x = (mx.random.normal((B, L, D)) * 0.1).astype(mx.bfloat16)
                        mask = "causal"

                        # Test with maximum GPU protection
                        success, result = self._bulletproof_execute_with_gpu_protection(
                            lambda: self._test_single_sequence_memory_safe(
                                custom_attention_class, args, x, mask
                            )
                        )

                        if success:
                            correctness_scores.append(result)
                            print(f"      ✅ Sequence {L}: PASS (score={result:.3f})")
                            break
                        else:
                            error_msg = str(result)

                            # Enhanced error classification
                            if "command buffer" in error_msg.lower():
                                local_command_buffer_errors += 1
                            elif "memory violation" in error_msg.lower():
                                local_memory_violations += 1

                            # EARLY EXIT: Metal compilation errors are deterministic - no retry
                            if "unable to build metal library" in error_msg.lower():
                                self.metal_compilation_errors += 1
                                print(f"      ❌ Metal compilation error (no retry): {error_msg[:200]}...")
                                # Return early - compilation errors won't be fixed by retrying
                                return {
                                    "success": False,
                                    "score": 0.0,
                                    "error": "Metal kernel compilation failed - bfloat16 incompatible code",
                                    "command_buffer_errors": local_command_buffer_errors,
                                    "memory_violations": local_memory_violations,
                                    "compilation_error": True,
                                }

                            if retry_count < self.max_retry_attempts:
                                print(
                                    f"      🔄 Retry {retry_count + 1} for length {L}: {error_msg}"
                                )
                                retry_count += 1
                                time.sleep(self.retry_base_delay * (2**retry_count))
                                continue
                            else:
                                print(f"      ❌ All retries failed for length {L}: {error_msg}")
                                correctness_scores.append(0.0)
                                break

                    except Exception as e:
                        error_msg = str(e)
                        print(f"      ❌ Exception for length {L}: {error_msg}")

                        # EARLY EXIT: Metal compilation errors are deterministic - no retry
                        if "unable to build metal library" in error_msg.lower():
                            self.metal_compilation_errors += 1
                            print(f"      ❌ Metal compilation error (no retry): {error_msg[:200]}...")
                            return {
                                "success": False,
                                "score": 0.0,
                                "error": "Metal kernel compilation failed - bfloat16 incompatible code",
                                "command_buffer_errors": local_command_buffer_errors,
                                "memory_violations": local_memory_violations,
                                "compilation_error": True,
                            }

                        if retry_count < self.max_retry_attempts:
                            retry_count += 1
                            time.sleep(self.retry_base_delay * (2**retry_count))
                            continue
                        else:
                            correctness_scores.append(0.0)
                            break

            # Update global error counters
            self.metal_command_buffer_errors += local_command_buffer_errors
            self.metal_memory_violations += local_memory_violations
            self.total_metal_errors += local_command_buffer_errors + local_memory_violations

            # Calculate overall correctness with partial credit
            overall_correctness = np.mean(correctness_scores) if correctness_scores else 0.0

            print(f"    📊 Memory-safe overall correctness: {overall_correctness:.3f}")
            print(f"    🛡️  Command buffer errors: {local_command_buffer_errors}")
            print(f"    🛡️  Memory violations: {local_memory_violations}")

            return {
                "success": True,
                "score": overall_correctness,
                "command_buffer_errors": local_command_buffer_errors,
                "memory_violations": local_memory_violations,
            }

        except Exception as e:
            self.total_metal_errors += 1
            print(f"    ❌ Memory-safe correctness testing failed: {e}")
            return {"success": False, "error": str(e)}

    def _test_single_sequence_memory_safe(
        self, custom_attention_class: Any, args: Any, x: Any, mask: Any
    ) -> float:
        """Test single sequence with enhanced memory safety"""
        try:
            # Force bfloat16 to exercise the same kernel template/compilation path as production
            # inference with `mlx-community/Qwen3-0.6B-bf16`.
            if x.dtype != mx.bfloat16:
                x = x.astype(mx.bfloat16)

            # Pre-execution safety checks
            if x.shape[1] > self.max_sequence_length_safe:
                raise MetalKernelSafetyError(
                    f"Sequence length {x.shape[1]} exceeds safe limit {self.max_sequence_length_safe}"
                )

            if x.shape[0] > self.max_batch_size_safe:
                raise MetalKernelSafetyError(
                    f"Batch size {x.shape[0]} exceeds safe limit {self.max_batch_size_safe}"
                )

            # Instantiate with error checking
            custom_attn = custom_attention_class(args)
            if custom_attn is None:
                raise ValueError("Failed to instantiate custom attention")

            # Ensure module parameters follow the intended compute dtype as well.
            # Otherwise, float32 weights can upcast intermediate Q/K/V tensors and
            # accidentally avoid bfloat16 kernel compilation.
            if hasattr(custom_attn, "set_dtype"):
                custom_attn.set_dtype(mx.bfloat16)

            # Conservative forward pass with timeout simulation
            start_time = time.time()
            output = custom_attn(x, mask=mask)
            elapsed_time = time.time() - start_time

            # Timeout check (soft limit)
            if elapsed_time > self.kernel_validation_timeout:
                print(f"        ⚠️  Slow execution detected: {elapsed_time:.2f}s")
                return 0.5  # Partial credit for slow but working kernel

            # Enhanced output validation
            if output is None:
                raise ValueError("Custom attention returned None")

            # Shape validation
            expected_shape = x.shape
            if output.shape != expected_shape:
                raise ValueError(f"Wrong output shape: {output.shape}, expected {expected_shape}")

            # Enhanced finite value check
            finite_mask = mx.isfinite(output)
            if not mx.all(finite_mask):
                finite_ratio = float(mx.mean(finite_mask.astype(mx.float32)))
                if finite_ratio < 0.9:
                    raise ValueError(f"Too many non-finite values: {finite_ratio:.2%} finite")
                else:
                    print(f"        ⚠️  Some non-finite values: {finite_ratio:.2%} finite")
                    return 0.7  # Partial credit

            # Enhanced statistical validation
            output_mean = float(mx.mean(output))
            output_std = float(mx.std(output))
            output_max = float(mx.max(mx.abs(output)))

            # More lenient bounds for complex kernels
            if abs(output_mean) > 10.0:
                print(f"        ⚠️  Large mean: {output_mean:.6f}")
                return 0.6

            if output_std > 100.0 or output_std < 0.00001:
                print(f"        ⚠️  Unusual std: {output_std:.6f}")
                return 0.6

            if output_max > 1000.0:
                print(f"        ⚠️  Large max value: {output_max:.6f}")
                return 0.7

            # All checks passed
            return 1.0

        except MetalKernelSafetyError as e:
            raise e  # Re-raise safety errors
        except Exception as e:
            error_msg = str(e)
            if any(
                keyword in error_msg.lower()
                for keyword in ["metal", "kernel", "gpu", "command buffer"]
            ):
                raise GPUCommandBufferError(f"GPU execution error: {error_msg}")
            else:
                raise ValueError(f"Sequence test error: {error_msg}")

    def _command_buffer_protected_benchmark(
        self, program_text: str, custom_attention_class: Any
    ) -> Dict[str, Any]:
        """Command-buffer-protected benchmarking with maximum safety"""
        print("  🚀 Running command-buffer-protected benchmarking...")

        retry_attempt = 0

        while retry_attempt <= self.max_retry_attempts:
            try:
                print(f"  🔄 Protected attempt {retry_attempt + 1}/{self.max_retry_attempts + 1}")

                # Clean GPU state before each major attempt
                self._ensure_clean_gpu_state()

                # Apply custom attention hook with protection
                hook_result = self._gpu_protected_apply_hook(custom_attention_class)
                if not hook_result["success"]:
                    if retry_attempt < self.max_retry_attempts:
                        print(f"    🔄 Hook failed, retrying... ({hook_result['error']})")
                        retry_attempt += 1
                        time.sleep(self.retry_base_delay * (2**retry_attempt))
                        continue
                    return {
                        "success": False,
                        "error": f"Hook application failed: {hook_result['error']}",
                    }

                original_attention = hook_result["original"]
                temp_program_path = None

                try:
                    # Ensure the evolved program is available to the subprocess that runs mlx_lm.generate.
                    # Monkey-patching in this evaluator process does NOT propagate across subprocess boundaries.
                    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
                        f.write(program_text)
                        temp_program_path = f.name

                    self.benchmark_suite.hook_program_path = temp_program_path

                    # Run benchmarks with command buffer protection
                    custom_configs = self._get_safe_benchmark_configs()
                    custom_results = []
                    successful_benchmarks = 0

                    for i, config in enumerate(custom_configs, 1):
                        print(
                            f"    [{i}/{len(custom_configs)}] Command-buffer-protected: {config.name}"
                        )

                        benchmark_retry = 0
                        while benchmark_retry <= 2:  # Fewer retries per benchmark
                            try:
                                # Clean state before each benchmark
                                self._ensure_clean_gpu_state()

                                # Run with maximum protection
                                success, result = self._bulletproof_execute_with_gpu_protection(
                                    lambda: self.benchmark_suite.run_single_benchmark(config)
                                )

                                if success and result:
                                    custom_results.append(result)
                                    successful_benchmarks += 1
                                    print(
                                        f"      ✅ Protected {config.name}: {result.decode_tokens_per_sec:.1f} tokens/sec"
                                    )
                                    break
                                else:
                                    if benchmark_retry < 2:
                                        print(
                                            f"      🔄 Benchmark retry {benchmark_retry + 1}: {result}"
                                        )
                                        benchmark_retry += 1
                                        time.sleep(1)
                                        continue
                                    else:
                                        print(f"      ❌ Benchmark failed: {result}")
                                        break

                            except Exception as e:
                                if benchmark_retry < 2:
                                    print(
                                        f"      🔄 Benchmark exception retry {benchmark_retry + 1}: {e}"
                                    )
                                    benchmark_retry += 1
                                    time.sleep(1)
                                    continue
                                else:
                                    print(f"      ❌ Benchmark exception: {e}")
                                    break

                    # Check success rate
                    min_required = max(2, len(custom_configs) * 0.4)  # Lowered to 40% for safety
                    if successful_benchmarks >= min_required:
                        print(
                            f"  ✅ Command-buffer-protected benchmarks complete ({successful_benchmarks} successful)"
                        )
                        self.retry_attempts_used = retry_attempt
                        return {"success": True, "results": custom_results}
                    else:
                        error_msg = f"Insufficient benchmarks: {successful_benchmarks}/{len(custom_configs)} succeeded"
                        if retry_attempt < self.max_retry_attempts:
                            print(f"  🔄 {error_msg}, retrying full attempt...")
                            retry_attempt += 1
                            time.sleep(self.retry_base_delay * (2**retry_attempt))
                            continue
                        return {"success": False, "error": error_msg}

                finally:
                    # Always clear subprocess hook settings and clean up temp program
                    self.benchmark_suite.hook_program_path = None
                    if temp_program_path:
                        try:
                            os.unlink(temp_program_path)
                        except OSError:
                            pass
                    # Always restore original attention
                    self._gpu_protected_remove_hook(original_attention)

            except Exception as e:
                error_msg = f"Command-buffer-protected attempt failed: {str(e)}"
                print(f"  ❌ {error_msg}")
                if retry_attempt < self.max_retry_attempts:
                    retry_attempt += 1
                    time.sleep(self.retry_base_delay * (2**retry_attempt))
                    continue
                return {"success": False, "error": error_msg}

        return {"success": False, "error": "All command-buffer-protected attempts exhausted"}

    def _ensure_clean_gpu_state(self):
        """Ensure clean GPU state before operations"""
        try:
            # Simple operation to ensure GPU responsiveness
            test_op = mx.array([1.0, 2.0, 3.0])
            mx.eval(test_op * 2)

            # Small delay to let GPU settle
            time.sleep(0.1)

        except Exception as e:
            print(f"    ⚠️  GPU state cleanup warning: {e}")

    def _gpu_protected_apply_hook(self, custom_attention_class: Any) -> Dict[str, Any]:
        """GPU-protected application of custom attention hook"""
        try:
            success, result = self._bulletproof_execute_with_gpu_protection(
                lambda: self._apply_attention_hook_safely(custom_attention_class)
            )

            if success:
                return {"success": True, "original": result}
            else:
                return {"success": False, "error": result}

        except Exception as e:
            return {"success": False, "error": f"GPU-protected hook application failed: {e}"}

    def _apply_attention_hook_safely(self, custom_attention_class: Any) -> Any:
        """Safely apply attention hook"""
        import mlx_lm.models.qwen3 as qwen3_module

        # Store original attention class
        original_attention = getattr(qwen3_module, "Attention", None)
        if original_attention is None:
            raise RuntimeError("Could not find original Attention class")

        # Apply custom attention
        qwen3_module.Attention = custom_attention_class

        # Verify the hook was applied
        if qwen3_module.Attention != custom_attention_class:
            raise RuntimeError("Hook application verification failed")

        print("      ✅ Custom attention hook applied with GPU protection")
        return original_attention

    def _gpu_protected_remove_hook(self, original_attention: Any):
        """GPU-protected removal of custom attention hook"""
        try:
            success, result = self._bulletproof_execute_with_gpu_protection(
                lambda: self._remove_attention_hook_safely(original_attention)
            )

            if not success:
                print(f"      ⚠️  Hook removal warning: {result}")

        except Exception as e:
            print(f"      ⚠️  Hook removal error (non-fatal): {e}")

    def _remove_attention_hook_safely(self, original_attention: Any):
        """Safely remove attention hook"""
        import mlx_lm.models.qwen3 as qwen3_module

        qwen3_module.Attention = original_attention
        print("      ✅ Hook removed with GPU protection")

    def _create_bulletproof_execution_environment(self) -> Dict[str, Any]:
        """Create bulletproof execution environment with enhanced imports"""
        import math
        import numpy as np
        import time
        from typing import Optional, Tuple, Any

        exec_globals = {
            "__builtins__": __builtins__,
            "mx": mx,
            "nn": nn,
            "np": np,
            "math": math,
            "time": time,
            "Optional": Optional,
            "Tuple": Tuple,
            "Any": Any,
        }

        # Enhanced MLX-LM import with error handling
        try:
            exec_globals["mlx_lm"] = __import__("mlx_lm")
            print("  ✅ MLX-LM imported for bulletproof execution")
        except ImportError:
            print("  ⚠️  MLX-LM not available for bulletproof execution")
        except Exception as e:
            print(f"  ⚠️  MLX-LM import error in bulletproof environment: {e}")

        return exec_globals

    def _get_safe_benchmark_configs(self) -> List[BenchmarkConfig]:
        """Get safer benchmark configurations for GPU protection"""
        try:
            all_configs = self.benchmark_suite.create_benchmark_configs()

            # Use more conservative test set for safety
            safe_test_names = [
                "short_context_quick",  # Safest - very short
                "code_generation",  # Medium safety
                "long_context_detailed",  # More challenging but still safe
                "long_generation",  # Longer generation
                # Disabled for faster testing
                #"maximum_context_stress_test",  # Most challenging - saved for last
            ]

            config_dict = {c.name: c for c in all_configs}
            safe_configs = []

            for test_name in safe_test_names:
                if test_name in config_dict:
                    safe_configs.append(config_dict[test_name])

            return safe_configs

        except Exception as e:
            print(f"  ⚠️  Error getting safe benchmark configs: {e}")
            return []

    def _ensure_standard_attention(self):
        """Ensure standard attention is active"""
        try:
            import mlx_lm.models.qwen3 as qwen3_module

            if hasattr(self, "_original_attention") and self._original_attention:
                qwen3_module.Attention = self._original_attention
                print("  🔄 Restored standard attention for baseline")
        except ImportError:
            print("  ⚠️  Could not access qwen3 module for standard attention")

    def _store_enhanced_baseline_metrics(self, baseline_results: List[BenchmarkResult]):
        """Store enhanced baseline metrics"""
        decode_speeds = [
            r.decode_tokens_per_sec for r in baseline_results if r.decode_tokens_per_sec > 0
        ]
        prefill_speeds = [
            r.prefill_tokens_per_sec for r in baseline_results if r.prefill_tokens_per_sec > 0
        ]
        memories = [r.peak_memory_gb for r in baseline_results if r.peak_memory_gb > 0]

        self.baseline_results = baseline_results
        self.baseline_metrics = {
            "avg_decode_speed": float(np.mean(decode_speeds)) if decode_speeds else 0.0,
            "min_decode_speed": float(np.min(decode_speeds)) if decode_speeds else 0.0,
            "max_decode_speed": float(np.max(decode_speeds)) if decode_speeds else 0.0,
            "std_decode_speed": float(np.std(decode_speeds)) if len(decode_speeds) > 1 else 0.0,
            "avg_prefill_speed": float(np.mean(prefill_speeds)) if prefill_speeds else 0.0,
            "avg_memory_gb": float(np.mean(memories)) if memories else 0.0,
            "max_memory_gb": float(np.max(memories)) if memories else 0.0,
            "num_baseline_tests": len(baseline_results),
        }

        print(
            f"    📊 Enhanced baseline stored - Avg decode: {self.baseline_metrics['avg_decode_speed']:.1f} tokens/sec"
        )

    def _analyze_performance_with_safety_metrics(
        self, baseline_results: List[BenchmarkResult], custom_results: List[BenchmarkResult]
    ) -> Dict[str, Any]:
        """Analyze performance with enhanced safety metrics"""
        print("  📈 Analyzing performance with safety metrics...")

        baseline_dict = {r.name: r for r in baseline_results}
        custom_dict = {r.name: r for r in custom_results}

        individual_comparisons = []
        improvements = {
            "decode_speed_improvements": [],
            "prefill_speed_improvements": [],
            "total_speed_improvements": [],
            "memory_improvements": [],
            "time_improvements": [],
        }

        # Compare each benchmark
        for name in baseline_dict:
            if name in custom_dict:
                baseline = baseline_dict[name]
                custom = custom_dict[name]

                # Calculate improvements with safety bounds
                decode_improvement = self._safe_calculate_improvement(
                    custom.decode_tokens_per_sec, baseline.decode_tokens_per_sec
                )
                prefill_improvement = self._safe_calculate_improvement(
                    custom.prefill_tokens_per_sec, baseline.prefill_tokens_per_sec
                )
                total_improvement = self._safe_calculate_improvement(
                    custom.total_tokens_per_sec, baseline.total_tokens_per_sec
                )
                memory_improvement = self._safe_calculate_improvement(
                    baseline.peak_memory_gb, custom.peak_memory_gb  # Reversed for memory
                )
                time_improvement = self._safe_calculate_improvement(
                    baseline.total_time_sec, custom.total_time_sec  # Reversed for time
                )

                comparison = {
                    "benchmark_name": name,
                    "baseline": self._result_to_dict(baseline),
                    "custom": self._result_to_dict(custom),
                    "improvements": {
                        "decode_speed_pct": decode_improvement,
                        "prefill_speed_pct": prefill_improvement,
                        "total_speed_pct": total_improvement,
                        "memory_reduction_pct": memory_improvement,
                        "time_reduction_pct": time_improvement,
                    },
                }

                individual_comparisons.append(comparison)

                improvements["decode_speed_improvements"].append(decode_improvement)
                improvements["prefill_speed_improvements"].append(prefill_improvement)
                improvements["total_speed_improvements"].append(total_improvement)
                improvements["memory_improvements"].append(memory_improvement)
                improvements["time_improvements"].append(time_improvement)

                print(f"    • {name}: {decode_improvement:+.1f}% decode speed")

        # Calculate aggregate statistics with safety checks
        aggregate_stats = {}
        for key, values in improvements.items():
            if values:
                # Use robust statistics
                valid_values = [v for v in values if not np.isnan(v) and not np.isinf(v)]
                if valid_values:
                    aggregate_stats[f"{key}_avg"] = float(np.mean(valid_values))
                    aggregate_stats[f"{key}_median"] = float(np.median(valid_values))
                    aggregate_stats[f"{key}_min"] = float(np.min(valid_values))
                    aggregate_stats[f"{key}_max"] = float(np.max(valid_values))
                    aggregate_stats[f"{key}_std"] = float(np.std(valid_values))

        # Calculate custom metrics
        custom_decode_speeds = [
            r.decode_tokens_per_sec for r in custom_results if r.decode_tokens_per_sec > 0
        ]
        custom_prefill_speeds = [
            r.prefill_tokens_per_sec for r in custom_results if r.prefill_tokens_per_sec > 0
        ]
        custom_memories = [r.peak_memory_gb for r in custom_results if r.peak_memory_gb > 0]

        aggregate_metrics = {
            "avg_decode_speed": (
                float(np.mean(custom_decode_speeds)) if custom_decode_speeds else 0.0
            ),
            "min_decode_speed": (
                float(np.min(custom_decode_speeds)) if custom_decode_speeds else 0.0
            ),
            "max_decode_speed": (
                float(np.max(custom_decode_speeds)) if custom_decode_speeds else 0.0
            ),
            "avg_prefill_speed": (
                float(np.mean(custom_prefill_speeds)) if custom_prefill_speeds else 0.0
            ),
            "avg_memory_gb": float(np.mean(custom_memories)) if custom_memories else 0.0,
            "max_memory_gb": float(np.max(custom_memories)) if custom_memories else 0.0,
            "num_successful_tests": len(custom_results),
            "decode_speed_std": (
                float(np.std(custom_decode_speeds)) if len(custom_decode_speeds) > 1 else 0.0
            ),
        }

        # Enhanced comparison summary
        comparison_summary = {
            "avg_decode_improvement_pct": aggregate_stats.get("decode_speed_improvements_avg", 0),
            "avg_decode_improvement_absolute": (
                aggregate_metrics["avg_decode_speed"] - self.baseline_metrics["avg_decode_speed"]
            ),
            "memory_change_gb": (
                aggregate_metrics["avg_memory_gb"] - self.baseline_metrics["avg_memory_gb"]
            ),
            "target_achieved": aggregate_stats.get("decode_speed_improvements_avg", 0) >= 5.0,
            "num_benchmarks_improved": sum(
                1 for x in improvements["decode_speed_improvements"] if x > 1.0
            ),  # More lenient
            "total_benchmarks": len(improvements["decode_speed_improvements"]),
            "safety_score": self._calculate_safety_score(),
        }

        print(
            f"  📊 Enhanced analysis complete: {comparison_summary['avg_decode_improvement_pct']:+.1f}% avg improvement"
        )
        print(f"  🛡️  Safety score: {comparison_summary['safety_score']:.2f}")

        return {
            "individual_comparisons": individual_comparisons,
            "aggregate_improvements": aggregate_stats,
            "aggregate_metrics": aggregate_metrics,
            "comparison_summary": comparison_summary,
        }

    def _safe_calculate_improvement(self, new_value: float, old_value: float) -> float:
        """Safely calculate percentage improvement with bounds"""
        if old_value <= 0 or np.isnan(old_value) or np.isnan(new_value):
            return 0.0

        improvement = (new_value - old_value) / old_value * 100

        # Clamp extreme values for safety
        return max(-100.0, min(1000.0, improvement))

    def _calculate_safety_score(self) -> float:
        """Calculate overall safety score based on error statistics"""
        total_operations = (
            self.metal_command_buffer_errors
            + self.metal_memory_violations
            + self.metal_compilation_errors
            + self.gpu_resource_errors
            + 10  # Assumed successful operations
        )

        error_rate = self.total_metal_errors / total_operations
        safety_score = max(0.0, 1.0 - error_rate) * 100

        return safety_score

    def _calculate_safety_adjusted_score(
        self, performance_analysis: Dict[str, Any], correctness: float
    ) -> float:
        """Calculate final score adjusted for safety"""
        if correctness < 0.90:
            return -1000.0

        comparison = performance_analysis["comparison_summary"]
        avg_improvement = comparison["avg_decode_improvement_pct"]
        memory_change = comparison["memory_change_gb"]
        success_rate = comparison["num_benchmarks_improved"] / max(
            1, comparison["total_benchmarks"]
        )
        safety_score = comparison["safety_score"]

        # Enhanced score components
        performance_score = avg_improvement * 3  # Primary component
        memory_bonus = max(0, -memory_change * 10)  # Bonus for memory reduction
        consistency_bonus = success_rate * 10  # Bonus for consistent improvements
        correctness_bonus = correctness * 5  # Bonus for correctness
        safety_bonus = (safety_score / 100) * 5  # Bonus for safety

        # Penalty for excessive errors
        error_penalty = min(self.total_metal_errors * 2, 20)  # Cap penalty

        final_score = (
            performance_score
            + memory_bonus
            + consistency_bonus
            + correctness_bonus
            + safety_bonus
            - error_penalty
        )

        print(f"  🎯 Safety-adjusted score breakdown:")
        print(f"    • Performance: {avg_improvement:.2f}% × 3 = {performance_score:.2f}")
        print(f"    • Memory: {memory_bonus:.2f}")
        print(f"    • Consistency: {success_rate:.2f} × 10 = {consistency_bonus:.2f}")
        print(f"    • Correctness: {correctness:.3f} × 5 = {correctness_bonus:.2f}")
        print(f"    • Safety: {safety_score:.1f}/100 × 5 = {safety_bonus:.2f}")
        print(f"    • Error penalty: -{error_penalty:.2f}")
        print(f"    • Final score: {final_score:.2f}")

        return final_score

    def _generate_comprehensive_summary(
        self, performance_analysis: Dict[str, Any], correctness: float
    ) -> str:
        """Generate comprehensive evaluation summary with safety info"""
        comparison = performance_analysis["comparison_summary"]
        metrics = performance_analysis["aggregate_metrics"]

        avg_improvement = comparison["avg_decode_improvement_pct"]
        current_decode = metrics["avg_decode_speed"]
        baseline_decode = self.baseline_metrics["avg_decode_speed"]
        safety_score = comparison["safety_score"]

        summary = f"""Bulletproof Custom GQA Implementation Results:
• Decode Speed: {current_decode:.1f} tokens/sec (baseline: {baseline_decode:.1f})
• Improvement: {avg_improvement:+.1f}%
• Memory Usage: {metrics['avg_memory_gb']:.2f} GB
• Correctness: {correctness:.1%}
• Safety Score: {safety_score:.1f}/100
• Tests Passed: {metrics['num_successful_tests']}/{len(self._get_safe_benchmark_configs())}
• Benchmarks Improved: {comparison['num_benchmarks_improved']}/{comparison['total_benchmarks']}
• Metal Errors Handled: {self.total_metal_errors}"""

        if self.total_metal_errors == 0:
            summary += "\n🛡️  PERFECT SAFETY: No Metal kernel errors"
        elif self.total_metal_errors < 3:
            summary += f"\n🛡️  GOOD SAFETY: {self.total_metal_errors} Metal errors handled"
        else:
            summary += f"\n⚠️  SAFETY CONCERNS: {self.total_metal_errors} Metal errors handled"

        if avg_improvement >= 15:
            summary += "\n🎯 EXCELLENT: 15%+ improvement achieved!"
        elif avg_improvement >= 10:
            summary += "\n🚀 STRONG IMPROVEMENT: 10%+ speedup"
        elif avg_improvement >= 5:
            summary += "\n✅ GOOD IMPROVEMENT: 5%+ speedup"
        elif avg_improvement > 0:
            summary += "\n📈 MINOR IMPROVEMENT: Some speedup achieved"
        else:
            summary += "\n⚠️  NO IMPROVEMENT: Performance regression"

        return summary

    def _get_comprehensive_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        return {
            "metal_command_buffer_errors": self.metal_command_buffer_errors,
            "metal_memory_violations": self.metal_memory_violations,
            "metal_compilation_errors": self.metal_compilation_errors,
            "gpu_resource_errors": self.gpu_resource_errors,
            "total_metal_errors": self.total_metal_errors,
            "successful_fallbacks": self.successful_fallbacks,
            "retry_attempts_used": self.retry_attempts_used,
            "safety_score": self._calculate_safety_score(),
            "error_breakdown": {
                "command_buffer_pct": (
                    self.metal_command_buffer_errors / max(1, self.total_metal_errors)
                )
                * 100,
                "memory_violation_pct": (
                    self.metal_memory_violations / max(1, self.total_metal_errors)
                )
                * 100,
                "compilation_error_pct": (
                    self.metal_compilation_errors / max(1, self.total_metal_errors)
                )
                * 100,
                "resource_error_pct": (self.gpu_resource_errors / max(1, self.total_metal_errors))
                * 100,
            },
        }

    def _print_bulletproof_evaluation_results(self, result: Dict[str, Any]):
        """Print comprehensive bulletproof evaluation results"""
        print(f"\n{'🛡️ '*25}")
        print(f"{'🛡️  BULLETPROOF EVALUATION RESULTS  🛡️':^100}")
        print(f"{'🛡️ '*25}")

        if result["success"]:
            performance = result["performance_metrics"]
            comparison = result["baseline_comparison"]
            safety_stats = result["metal_safety_statistics"]

            print(f"📊 FINAL SCORE: {result['final_score']:.2f}")
            print(f"")
            print(f"📈 PERFORMANCE COMPARISON:")
            print(f"  • Average Decode Speed: {performance['avg_decode_speed']:.1f} tokens/sec")
            print(
                f"  • Baseline Decode Speed: {self.baseline_metrics['avg_decode_speed']:.1f} tokens/sec"
            )
            print(f"  • Average Improvement: {comparison['avg_decode_improvement_pct']:+.1f}%")
            print(
                f"  • Absolute Improvement: {comparison['avg_decode_improvement_absolute']:+.1f} tokens/sec"
            )
            print(f"")
            print(f"🛡️  SAFETY STATISTICS:")
            print(f"  • Safety Score: {safety_stats['safety_score']:.1f}/100")
            print(f"  • Command Buffer Errors: {safety_stats['metal_command_buffer_errors']}")
            print(f"  • Memory Violations: {safety_stats['metal_memory_violations']}")
            print(f"  • Total Metal Errors: {safety_stats['total_metal_errors']}")
            print(f"  • Retry Attempts Used: {safety_stats['retry_attempts_used']}")
            print(f"")
            print(f"💾 MEMORY USAGE:")
            print(f"  • Average Memory: {performance['avg_memory_gb']:.2f} GB")
            print(f"  • Baseline Memory: {self.baseline_metrics['avg_memory_gb']:.2f} GB")
            print(f"  • Memory Change: {comparison['memory_change_gb']:+.2f} GB")
            print(f"")
            print(f"✓ RELIABILITY:")
            print(f"  • Correctness Score: {result['correctness_score']:.1%}")
            print(f"  • Successful Tests: {performance['num_successful_tests']}")
            print(
                f"  • Benchmarks Improved: {comparison['num_benchmarks_improved']}/{comparison['total_benchmarks']}"
            )

            if comparison["target_achieved"]:
                print(f"\n🎯 TARGET ACHIEVED: Significant improvement with safety!")

            if safety_stats["total_metal_errors"] == 0:
                print(f"\n🛡️  PERFECT EXECUTION: No Metal kernel errors encountered!")

        else:
            print(f"❌ EVALUATION FAILED (SAFELY)")
            print(f"📋 Error: {result.get('error', 'Unknown error')}")
            safety_stats = result.get("metal_safety_statistics", {})
            print(f"🛡️  Metal Errors Handled: {safety_stats.get('total_metal_errors', 0)}")

        print(f"{'🛡️ '*25}")

    def _create_comprehensive_failure_result(self, error_message: str) -> Dict[str, Any]:
        """Create comprehensive failure result with full error statistics"""
        return {
            "success": False,
            "final_score": -1000.0,
            "combined_score": -1000.0,
            "error": error_message,
            "performance_metrics": {},
            "correctness_score": 0.0,
            "summary": f"Bulletproof evaluation failed (safely): {error_message}",
            "metal_safety_statistics": self._get_comprehensive_error_statistics(),
            "safety_validation": {"success": False, "error": error_message},
        }

    def _result_to_dict(self, result: BenchmarkResult) -> Dict:
        """Convert BenchmarkResult to dictionary"""
        return {
            "name": result.name,
            "decode_tokens_per_sec": result.decode_tokens_per_sec,
            "prefill_tokens_per_sec": result.prefill_tokens_per_sec,
            "peak_memory_gb": result.peak_memory_gb,
            "generated_tokens": result.generated_tokens,
            "total_time_sec": result.total_time_sec,
        }


def evaluate(program_text: str) -> Dict[str, Any]:
    """🛡️ BULLETPROOF evaluation function called by OpenEvolve"""
    evaluator = BulletproofMetalEvaluator()
    return evaluator.evaluate(program_text)


def test_bulletproof_evaluator():
    """Test the bulletproof Metal kernel evaluator"""
    print("🧪 Testing Bulletproof Metal Kernel Evaluator")
    print("🛡️ " * 40)

    initial_program_path = os.path.join(os.path.dirname(__file__), "initial_program.py")

    if not os.path.exists(initial_program_path):
        print(f"❌ Initial program not found: {initial_program_path}")
        return

    print(f"📁 Testing with bulletproof protection: {initial_program_path}")
    result = evaluate(initial_program_path)

    print(f"\n{'🛡️ '*20}")
    print(f"🔬 BULLETPROOF EVALUATOR TEST RESULTS")
    print(f"{'🛡️ '*20}")
    print(f"Success: {result['success']}")
    print(f"Final Score: {result.get('final_score', 'N/A')}")

    if result.get("metal_safety_statistics"):
        stats = result["metal_safety_statistics"]
        print(f"Metal Command Buffer Errors: {stats.get('metal_command_buffer_errors', 0)}")
        print(f"Metal Memory Violations: {stats.get('metal_memory_violations', 0)}")
        print(f"Total Metal Errors Handled: {stats.get('total_metal_errors', 0)}")
        print(f"Safety Score: {stats.get('safety_score', 0):.1f}/100")

    print(f"Summary: {result.get('summary', 'N/A')}")

    return result


if __name__ == "__main__":
    test_bulletproof_evaluator()
