# Portrait background blur — evolving a hot function behind a hard quality gate

Optimise the per-frame background blur of a video pipeline. The score is the
**speedup**, but fidelity is a **hard gate**: a candidate that is fast and wrong
scores exactly zero.

This is the "optimise a real hot function" pattern. The interesting part is not the
LLM — it is the **evaluator**. Getting the gate right *is* the job.

```
speedup = baseline_time / candidate_time

reject if mean SSIM         < 0.98    (overall fidelity)
       or worst-frame SSIM  < 0.95    (no bad frames)
       or worst-REGION SSIM < 0.90    (no bad regions)
otherwise score = speedup
```

Everything is deterministic and self-contained: the "video" is synthesised from a
fixed seed, so there is no webcam, no GPU, no ML model and no dataset to download.
`numpy` is the only dependency.

## The task

For each frame, blur the background and composite the sharp person back on top,
using a supplied segmentation mask:

```python
blurred = gaussian_blur(frame, sigma)
out     = mask * frame + (1 - mask) * blurred
```

The seed (`initial_program.py`) is **correct but slow**: it convolves with the full
2D Gaussian directly — O(k²) passes per frame. It is the honest implementation you
would write first, and it is the thing to beat.

## Run it

```bash
export OPENAI_API_KEY=sk-or-...   # your OpenRouter key

python openevolve-run.py \
  examples/background_blur/initial_program.py \
  examples/background_blur/evaluator.py \
  --config examples/background_blur/config.yaml \
  --iterations 100
```

`config.yaml` uses OpenRouter with a cheap model (`google/gemini-2.5-flash-lite`); the
key is read from `OPENAI_API_KEY`.
Any OpenAI-compatible endpoint works — point `api_base` at a local optillm server to
run with no hosted API and no key at all.

## Results

100 iterations. Timings are best-of-N on an idle machine (see the timing note below —
this matters more than you would think).

| | ms/frame | speedup | mean SSIM | worst-region |
|---|---:|---:|---:|---:|
| seed — naive O(k²) convolution | 33.4 | 1.0x | 1.0000 | 1.0000 |
| expert — hand-written separable + fp32 | 1.46 | 22.8x | 1.0000 | 1.0000 |
| **evolved (OpenEvolve)** | **0.54** | **62x** | 0.9915 | 0.9723 |

**Read those two speedup columns carefully.** 62x is against a baseline that was
*written to be bad on purpose*, so it flatters the result. The number that actually
means something is the comparison against a competent implementation:

> **The evolved solution is 2.6x faster than a hand-written separable+fp32 blur** —
> and it gets there by spending part of the fidelity budget the gate allows
> (SSIM 0.9915 vs the expert's exact 1.0000). It is a real win, not a free one.

28 of the 100 candidates were **rejected outright by the quality gate**. The gate is
not decoration.

### What it discovered

The winning program stacks five distinct optimisations, one of which (blurring at
reduced resolution) is the same class of trick reported for this problem elsewhere:

```python
# 1. separable Gaussian: two 1D passes instead of k^2
# 2. blur at HALF resolution, then upsample  (quarter the pixels)
downsample  = sigma >= 1.0
sigma_eff   = sigma / 2.0                      # rescale sigma for the small domain
low         = frames_arr[:, ::2, ::2, :]
blurred_low = _separable_blur_batch(low, kernel)
blurred     = np.repeat(np.repeat(blurred_low, 2, axis=1), 2, axis=2)

# 3. float32 rather than float64 (halves memory traffic)
# 4. BATCH the whole sequence into (N, H, W, C) and blur it in one vectorised pass
frames_arr  = np.stack(frames).astype(np.float32)

# 5. lerp composite: one multiply fewer than m*f + (1-m)*b
composited  = blurred + masks_arr[..., None] * (frames_arr - blurred)
```

Nobody told it to batch across frames. It is search, not magic — but it is real search.

## Attack your own evaluator first

Before spending a single token, prove the cheats lose:

```bash
python examples/background_blur/test_gaming.py
```

Each test is a candidate that is **fast but wrong**, and must score 0:

| cheat | why it is tempting |
|---|---|
| return the input untouched | infinitely fast |
| blur everything, ignore the mask | skips compositing |
| under-blur with a tiny kernel | far cheaper than the real sigma |
| blur frame 0's background, reuse it forever | ~50x faster |

...plus one honest optimisation (separable convolution) that **must** be accepted.

### The gate we would have shipped was broken

The first version of this evaluator used the obvious gate — mean SSIM ≥ 0.98 and
worst-frame SSIM ≥ 0.95. The **stale-background** cheat sailed straight through and
scored **47x**:

| candidate | mean | worst-frame | **worst-region** |
|---|---:|---:|---:|
| separable (exact) | 1.0000 | 1.0000 | 1.0000 |
| **CHEAT: stale background** | 0.9871 | **0.9806** | **0.7445** |
| half-res blur (legitimate) | 0.9915 | 0.9912 | 0.9723 |
| 3× box blur (legitimate) | 0.9949 | 0.9944 | 0.9901 |

Reusing one blurred background leaves a **person-shaped ghost** where the person used
to be. A human sees it instantly — but it damages a small patch, and whole-image SSIM
averages it away. Both frame-level gates pass.

The fix is to grade the **worst 16×16 block**, not the whole frame. The ghost region
scores 0.74 while genuine approximations stay above 0.97, so `worst_region ≥ 0.90`
separates them cleanly.

**The lesson generalises:** an aggregate metric hides localised damage. If your quality
bar is an average, the search will find the thing your average cannot see. Write the
cheats yourself and make sure they lose.

## Timing as fitness is its own trap

`parallel_evaluations: 1` is necessary — concurrent evaluations contend for CPU and
corrupt the measurement — but it is **not sufficient**.

The first version of this evaluator measured the baseline **once**, cached it, and
compared every later candidate against it. That one measurement happened while the
machine was busy, so the baseline was inflated (95.6 ms/frame vs a true 33.4) — and
therefore *every speedup reported during the run was inflated too*. The winner was
reported at **82x**; it is really **62x**.

The fix (`_benchmark()` in `evaluator.py`):

- measure baseline and candidate **interleaved, in the same call**, so slow drift
  cancels instead of accumulating into the ratio;
- **warm up** before timing;
- take the **minimum** of N runs — background load can only ever *add* time, so the
  minimum is the best estimate of the true cost, for both sides.

If your fitness is a wall-clock number, re-measure your final result on an idle
machine before you believe it.

## Why OpenEvolve fits this problem

**Cascade evaluation — don't benchmark garbage.** Timing is the expensive part, so it
is the last thing we do:

```
stage 1  cheap smoke test  (2 frames: shape, finite, did you blur at all?)
stage 2  the quality gate  (all frames: SSIM vs the reference)
stage 3  the benchmark     (only for candidates that earned it)
```

**Artifacts — tell the model *why* it failed.** A scalar score says "0". OpenEvolve's
artifact side-channel hands the next prompt the actual reason:

> QUALITY GATE FAILED: mean SSIM 0.9871 (needs >= 0.98), worst-frame SSIM 0.9806 on
> frame 6, worst-REGION SSIM 0.7445 (needs >= 0.90). ... do not reuse a stale
> background (it leaves a person-shaped ghost that wrecks one region while barely
> moving the frame average).

**MAP-Elites — keep rival strategies alive.** The grid is `(complexity, ssim)`, so
"exact and fast" (separable, SSIM 1.0) and "approximate and faster" (half-res, SSIM
0.99) both survive instead of the population collapsing onto whichever appeared first.

## Files

| file | what it is |
|---|---|
| `initial_program.py` | the seed — correct, slow, `EVOLVE-BLOCK` markers |
| `evaluator.py` | 3-stage cascade, the hard gate, interleaved timing, artifacts |
| `fixtures.py` | deterministic scene, trusted reference, SSIM + worst-region SSIM |
| `test_gaming.py` | adversarial tests — the cheats must lose |
| `config.yaml` | OpenRouter endpoint, MAP-Elites grid, `parallel_evaluations: 1` |

## Adapting this to your own hot function

1. Wrap the function in `EVOLVE-BLOCK-START` / `EVOLVE-BLOCK-END`.
2. Write a **reference** implementation the evaluator owns (never handed to the
   candidate) and a **fidelity metric** with a hard threshold.
3. Write the cheats and prove they score zero. Then go looking for the *localised*
   cheat your aggregate metric cannot see.
4. Put the expensive measurement in the last cascade stage, and measure it fairly.
5. Return artifacts explaining every rejection.
6. Compare against a **competent** implementation, not just your slow seed — that is
   the only number that means anything.
