"""
Deterministic fixtures, trusted reference implementation, and SSIM.

Everything here is used ONLY by the evaluator - the evolved program never sees it.
That matters: the golden output is never handed to the candidate, so it cannot be
copied, only reproduced.

No external data, no webcam, no ML model: the "video" is synthesised from a fixed
seed so the benchmark is byte-for-byte reproducible on any machine.
"""

import numpy as np

# Scene / benchmark constants
HEIGHT = 180
WIDTH = 240
N_FRAMES = 8
SIGMA = 4.0  # background blur strength


# --------------------------------------------------------------------------
# Trusted reference (evaluator-side only)
# --------------------------------------------------------------------------
def gaussian_kernel_1d(sigma: float) -> np.ndarray:
    """Truncated (3 sigma) normalised 1D Gaussian, float64."""
    radius = int(np.ceil(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x**2) / (2.0 * sigma**2))
    return k / k.sum()


def reference_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Exact separable Gaussian blur in float64 with reflect padding.

    This defines the ground truth. The seed program computes the same thing the
    slow way (direct 2D convolution), so the seed scores SSIM ~= 1.0 and speedup
    1.0x - it is correct, just slow.
    """
    k = gaussian_kernel_1d(sigma)
    r = len(k) // 2
    img = image.astype(np.float64)

    # Horizontal pass
    padded = np.pad(img, ((0, 0), (r, r), (0, 0)), mode="reflect")
    tmp = np.zeros_like(img)
    for i, w in enumerate(k):
        tmp += w * padded[:, i : i + img.shape[1], :]

    # Vertical pass
    padded = np.pad(tmp, ((r, r), (0, 0), (0, 0)), mode="reflect")
    out = np.zeros_like(img)
    for i, w in enumerate(k):
        out += w * padded[i : i + img.shape[0], :, :]

    return out


def reference_process(frames, masks, sigma: float):
    """Ground-truth portrait blur: sharp person over a blurred background."""
    out = []
    for frame, mask in zip(frames, masks):
        blurred = reference_blur(frame, sigma)
        m = mask[:, :, None].astype(np.float64)
        out.append((m * frame.astype(np.float64) + (1.0 - m) * blurred).astype(np.float32))
    return out


def naive_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """The BASELINE to beat: direct 2D convolution, O(k^2) passes per frame.

    This mirrors the seed program. The evaluator keeps its own copy so the
    baseline timing never depends on (and cannot be tampered with by) the
    candidate under test.
    """
    k1 = gaussian_kernel_1d(sigma)
    k2 = np.outer(k1, k1)
    r = k2.shape[0] // 2
    img = image.astype(np.float64)
    padded = np.pad(img, ((r, r), (r, r), (0, 0)), mode="reflect")
    out = np.zeros_like(img)
    h, w = img.shape[0], img.shape[1]
    for dy in range(k2.shape[0]):
        for dx in range(k2.shape[1]):
            out += k2[dy, dx] * padded[dy : dy + h, dx : dx + w, :]
    return out


def naive_process(frames, masks, sigma: float):
    """Baseline pipeline (what the seed program does)."""
    out = []
    for frame, mask in zip(frames, masks):
        blurred = naive_blur(frame, sigma)
        m = mask[:, :, None].astype(np.float64)
        out.append((m * frame.astype(np.float64) + (1.0 - m) * blurred).astype(np.float32))
    return out


# --------------------------------------------------------------------------
# Synthetic "webcam" sequence
# --------------------------------------------------------------------------
def _background(h: int, w: int, t: float, rng: np.random.Generator) -> np.ndarray:
    """A textured background with fine detail (so blur is actually measurable)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Gradient + high-frequency texture: detail here is what blurring destroys,
    # which is what makes SSIM sensitive to under-blurring.
    base = 0.35 + 0.25 * (xx / w) + 0.15 * (yy / h)
    texture = 0.10 * np.sin(xx / 3.0) * np.cos(yy / 4.0)
    stripes = 0.06 * np.sin((xx + yy) / 2.5)

    img = np.stack(
        [
            base + texture + stripes,
            base * 0.9 + 0.8 * texture,
            base * 0.8 + stripes,
        ],
        axis=-1,
    )

    # A couple of hard-edged objects (blur must smear these)
    img[20:60, 30:80] += 0.20
    img[100:150, 160:220] -= 0.15

    # Slow global lighting drift across the sequence: the background genuinely
    # changes per frame, so "blur once and reuse for every frame" is NOT valid.
    img = img * (1.0 + 0.05 * np.sin(t))

    img += rng.normal(0.0, 0.01, img.shape)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _person_mask(h: int, w: int, cx: float, cy: float) -> np.ndarray:
    """Soft-edged head+torso silhouette (values in [0,1], 1 = person)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    def soft_ellipse(ex, ey, rx, ry):
        d = ((xx - ex) / rx) ** 2 + ((yy - ey) / ry) ** 2
        # Soft (anti-aliased) edge rather than a hard step
        return np.clip(1.5 * (1.0 - d), 0.0, 1.0)

    head = soft_ellipse(cx, cy - 30, 22, 26)
    torso = soft_ellipse(cx, cy + 45, 40, 40)
    return np.clip(head + torso, 0.0, 1.0).astype(np.float32)


def make_sequence(n_frames: int = N_FRAMES, h: int = HEIGHT, w: int = WIDTH):
    """Deterministic frames + masks. The person moves; the background drifts."""
    rng = np.random.default_rng(1234)
    frames, masks = [], []
    for i in range(n_frames):
        t = 2.0 * np.pi * i / max(1, n_frames)
        cx = w * 0.5 + 25.0 * np.sin(t)  # person walks side to side
        cy = h * 0.5 + 6.0 * np.cos(t)

        bg = _background(h, w, t, rng)
        mask = _person_mask(h, w, cx, cy)

        # Person appearance: distinctly different from the background
        person = np.stack(
            [
                np.full((h, w), 0.85, np.float32),
                np.full((h, w), 0.55, np.float32),
                np.full((h, w), 0.45, np.float32),
            ],
            axis=-1,
        )
        m = mask[:, :, None]
        frame = (m * person + (1.0 - m) * bg).astype(np.float32)

        frames.append(frame)
        masks.append(mask)
    return frames, masks


# --------------------------------------------------------------------------
# SSIM (numpy, no scipy/skimage dependency)
# --------------------------------------------------------------------------
def _blur64(img2d: np.ndarray, k: np.ndarray) -> np.ndarray:
    r = len(k) // 2
    p = np.pad(img2d, ((0, 0), (r, r)), mode="reflect")
    tmp = np.zeros_like(img2d)
    for i, w in enumerate(k):
        tmp += w * p[:, i : i + img2d.shape[1]]
    p = np.pad(tmp, ((r, r), (0, 0)), mode="reflect")
    out = np.zeros_like(img2d)
    for i, w in enumerate(k):
        out += w * p[i : i + img2d.shape[0], :]
    return out


def ssim_map(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> np.ndarray:
    """Per-pixel SSIM map, averaged over colour channels. Shape (H, W)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    k = gaussian_kernel_1d(1.5)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    maps = []
    for c in range(a.shape[2]):
        x, y = a[:, :, c], b[:, :, c]
        mu_x, mu_y = _blur64(x, k), _blur64(y, k)
        mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
        sx = _blur64(x * x, k) - mu_x2
        sy = _blur64(y * y, k) - mu_y2
        sxy = _blur64(x * y, k) - mu_xy
        num = (2 * mu_xy + C1) * (2 * sxy + C2)
        den = (mu_x2 + mu_y2 + C1) * (sx + sy + C2)
        maps.append(num / den)
    return np.mean(maps, axis=0)


def ssim(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    """Gaussian-windowed SSIM, averaged over colour channels. Range [-1, 1]."""
    return float(np.mean(ssim_map(a, b, data_range)))


def worst_region_ssim(a: np.ndarray, b: np.ndarray, block: int = 16) -> float:
    """SSIM of the WORST block of the image.

    Whole-image mean SSIM hides localised defects: a candidate that leaves a
    person-shaped ghost in one corner still averages ~0.99 across the frame. That
    is not a hypothetical - the first version of this evaluator scored such a
    candidate at 47x and let it through. Grading the worst block instead of the
    whole frame is what closes that hole.
    """
    m = ssim_map(a, b)
    h, w = m.shape
    bh, bw = h // block, w // block
    if bh == 0 or bw == 0:
        return float(np.mean(m))
    # Trim to a whole number of blocks, then average within each block.
    m = m[: bh * block, : bw * block]
    blocks = m.reshape(bh, block, bw, block).mean(axis=(1, 3))
    return float(np.min(blocks))
