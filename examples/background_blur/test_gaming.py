"""
Adversarial tests for the evaluator - the part people skip.

The article that inspired this example makes the point well: a lazy candidate that
skips the real work can score beautifully if your scorer lets it. So before you
spend a single LLM token, attack your own evaluator and prove the cheats lose.

Each test below is a candidate that is FAST but WRONG. All of them must score 0.
Then we prove a genuinely better implementation does win.

    python examples/background_blur/test_gaming.py
"""

import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

import evaluator as ev  # noqa: E402

HEADER = "import numpy as np\n"


def _write(code):
    f = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    f.write(HEADER + code)
    f.close()
    return f.name


def _score(code):
    path = _write(code)
    try:
        return ev.evaluate(path)
    finally:
        os.unlink(path)


# --------------------------------------------------------------------------
# Cheats - all of these are fast, and all of them must be REJECTED
# --------------------------------------------------------------------------
CHEAT_IDENTITY = """
def process_sequence(frames, masks, sigma):
    # "Optimised": do nothing at all. Blazing fast.
    return [f.copy() for f in frames]
"""

CHEAT_NO_MASK = """
def process_sequence(frames, masks, sigma):
    # Blur everything, including the person - skips the compositing work.
    out = []
    for f in frames:
        b = f.copy()
        for _ in range(3):
            b = (b + np.roll(b, 1, 0) + np.roll(b, -1, 0) + np.roll(b, 1, 1) + np.roll(b, -1, 1)) / 5.0
        out.append(b.astype(np.float32))
    return out
"""

CHEAT_UNDER_BLUR = """
def process_sequence(frames, masks, sigma):
    # Use a much smaller kernel than asked for: far cheaper, visibly wrong.
    out = []
    for f, m in zip(frames, masks):
        b = f.copy()
        b = (b + np.roll(b, 1, 0) + np.roll(b, -1, 0)) / 3.0
        mm = m[:, :, None]
        out.append((mm * f + (1 - mm) * b).astype(np.float32))
    return out
"""

CHEAT_STALE_BACKGROUND = """
_cache = {}
def process_sequence(frames, masks, sigma):
    # Blur frame 0's background once and reuse it for every frame. The background
    # actually drifts, so this must fail the worst-frame gate.
    from math import ceil
    r = int(ceil(3.0 * sigma))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2)); k /= k.sum()

    def blur(img):
        p = np.pad(img, ((0, 0), (r, r), (0, 0)), mode="reflect")
        t = np.zeros_like(img)
        for i, w in enumerate(k):
            t += w * p[:, i:i + img.shape[1], :]
        p = np.pad(t, ((r, r), (0, 0), (0, 0)), mode="reflect")
        o = np.zeros_like(t)
        for i, w in enumerate(k):
            o += w * p[i:i + img.shape[0], :, :]
        return o

    stale = blur(frames[0].astype(np.float64))
    out = []
    for f, m in zip(frames, masks):
        mm = m[:, :, None].astype(np.float64)
        out.append((mm * f + (1 - mm) * stale).astype(np.float32))
    return out
"""

# --------------------------------------------------------------------------
# A legitimate optimisation - must be ACCEPTED and be much faster
# (separable convolution: 2k passes instead of k^2)
# --------------------------------------------------------------------------
HONEST_SEPARABLE = """
def process_sequence(frames, masks, sigma):
    r = int(np.ceil(3.0 * sigma))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2)); k /= k.sum()

    def blur(img):
        p = np.pad(img, ((0, 0), (r, r), (0, 0)), mode="reflect")
        t = np.zeros_like(img)
        for i, w in enumerate(k):
            t += w * p[:, i:i + img.shape[1], :]
        p = np.pad(t, ((r, r), (0, 0), (0, 0)), mode="reflect")
        o = np.zeros_like(t)
        for i, w in enumerate(k):
            o += w * p[i:i + img.shape[0], :, :]
        return o

    out = []
    for f, m in zip(frames, masks):
        img = f.astype(np.float64)
        b = blur(img)
        mm = m[:, :, None].astype(np.float64)
        out.append((mm * img + (1 - mm) * b).astype(np.float32))
    return out
"""


class TestEvaluatorCannotBeGamed(unittest.TestCase):
    def _assert_rejected(self, code, label):
        res = _score(code)
        score = res.metrics.get("combined_score", 0.0)
        self.assertEqual(score, 0.0, f"{label} should be REJECTED but scored {score}")
        # A rejection must always explain itself.
        self.assertIn("failure_reason", res.artifacts, f"{label}: no failure_reason artifact")
        print(f"  [rejected] {label}: {res.artifacts['failure_reason'][:90]}")

    def test_identity_is_rejected(self):
        self._assert_rejected(CHEAT_IDENTITY, "do-nothing (returns input)")

    def test_blurring_the_person_is_rejected(self):
        self._assert_rejected(CHEAT_NO_MASK, "blurs everything (ignores mask)")

    def test_under_blur_is_rejected(self):
        self._assert_rejected(CHEAT_UNDER_BLUR, "under-blurs (kernel far too small)")

    def test_stale_background_is_rejected(self):
        self._assert_rejected(CHEAT_STALE_BACKGROUND, "reuses frame 0's background")

    def test_honest_optimisation_wins(self):
        res = _score(HONEST_SEPARABLE)
        speedup = res.metrics.get("combined_score", 0.0)
        ssim = res.metrics.get("ssim", 0.0)
        self.assertGreater(speedup, 1.0, "separable blur should beat the naive baseline")
        self.assertGreaterEqual(ssim, ev.MEAN_SSIM_GATE, "and it should pass the quality gate")
        print(f"  [accepted] separable convolution: {speedup:.2f}x speedup, SSIM {ssim:.4f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
