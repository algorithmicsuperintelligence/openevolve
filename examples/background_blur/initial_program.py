"""
Portrait-mode background blur - the hot function from a video pipeline.

For every frame we blur the background and composite the (sharp) person back on
top, using a soft segmentation mask that is handed to us:

    blurred = gaussian_blur(frame, sigma)
    out     = mask * frame + (1 - mask) * blurred

The seed below is CORRECT but SLOW: it convolves with the full 2D Gaussian
directly, which costs O(k^2) passes per frame. It is the honest, obvious
implementation you would write first - and it is the thing to beat.

The scoring function does not care how you get there. It only cares that the
output still looks like this one (SSIM gate) and that it is faster.
"""

import numpy as np


# EVOLVE-BLOCK-START
def gaussian_kernel_1d(sigma: float) -> np.ndarray:
    """Truncated (3 sigma) normalised 1D Gaussian."""
    radius = int(np.ceil(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x**2) / (2.0 * sigma**2))
    return k / k.sum()


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Blur an (H, W, 3) image with a Gaussian of the given sigma.

    Naive implementation: builds the full 2D kernel and accumulates one shifted
    copy of the image per kernel tap. That is (2*radius+1)^2 passes over the
    whole image - correct, but very slow.
    """
    k1 = gaussian_kernel_1d(sigma)
    k2 = np.outer(k1, k1)  # full 2D kernel
    r = k2.shape[0] // 2

    img = image.astype(np.float64)
    padded = np.pad(img, ((r, r), (r, r), (0, 0)), mode="reflect")

    out = np.zeros_like(img)
    h, w = img.shape[0], img.shape[1]
    for dy in range(k2.shape[0]):
        for dx in range(k2.shape[1]):
            out += k2[dy, dx] * padded[dy : dy + h, dx : dx + w, :]
    return out


def blur_background(frame: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Blur the background of one frame, keeping the person sharp."""
    blurred = gaussian_blur(frame, sigma)
    m = mask[:, :, None].astype(np.float64)
    return (m * frame.astype(np.float64) + (1.0 - m) * blurred).astype(np.float32)


def process_sequence(frames, masks, sigma: float):
    """Process a whole sequence of frames.

    frames: list of (H, W, 3) float32 arrays in [0, 1]
    masks:  list of (H, W)    float32 arrays in [0, 1], 1 = person
    returns: list of (H, W, 3) float32 arrays

    The background genuinely changes from frame to frame (the person moves and the
    lighting drifts), so results must stay faithful on EVERY frame, not just on
    average.
    """
    return [blur_background(f, m, sigma) for f, m in zip(frames, masks)]


# EVOLVE-BLOCK-END
