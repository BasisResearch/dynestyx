#!/usr/bin/env python3
"""
WARNING: This script is primarily LLM-generated (with lots of feedback and iteration).
It would be nice to rewrite this as a proper dynestyx model and sample from the SDE that way :).

dynestyx logo generator (BOUNDARY ORBIT version):
- particles initialized uniformly over the whole window
- each particle is assigned a fixed letter id/color based on its STARTING point
  (nearest-letter boundary field)
- particles are attracted to a thin ring around the UNION of letter boundaries
  and swirl tangentially along those boundaries (orbit)
- all particles are rendered at all times in their fixed color
- white background (ink subtracted from white)
- outputs a GIF
"""

import os
import math
import argparse
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.ndimage import binary_erosion


# -----------------------------
# Fonts (cross-platform)
# -----------------------------


def find_font_candidates():
    return [
        "RobotoMono-VariableFont_wght.ttf",
        # macOS common
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Verdana.ttf",
        "/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Verdana.ttf",
        # Linux common
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        # Windows common (if ever)
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/verdana.ttf",
    ]


def load_font(font_size: int, font_path: str | None):
    candidates = []
    if font_path:
        candidates.append(font_path)
    candidates += find_font_candidates()

    for p in candidates:
        if p and os.path.exists(p):
            try:
                return ImageFont.truetype(p, font_size)
            except OSError:
                continue

    raise FileNotFoundError(
        "Could not load a TrueType font. Provide --font_path pointing to a .ttf/.otf "
        "(recommended for deterministic logo rendering)."
    )


# -----------------------------
# Geometry helpers
# -----------------------------


def bilinear_sample(arr: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear interpolation for arr[y,x], x,y in pixel coordinates."""
    H, W = arr.shape
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = np.clip(x0, 0, W - 1)
    x1 = np.clip(x1, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1)
    y1 = np.clip(y1, 0, H - 1)

    Ia = arr[y0, x0]
    Ib = arr[y1, x0]
    Ic = arr[y0, x1]
    Id = arr[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return Ia * wa + Ib * wb + Ic * wc + Id * wd


def compute_centered_text_origin(
    text: str, font: ImageFont.FreeTypeFont, canvas: tuple[int, int]
):
    """
    Compute a centered placement origin (x0,y0) robustly by rendering onto a big scratch canvas.
    """
    W, H = canvas
    scratch = Image.new("L", (3 * W, 3 * H), 0)
    draw = ImageDraw.Draw(scratch)

    ox, oy = W, H
    draw.text((ox, oy), text, fill=255, font=font)

    bbox = scratch.getbbox()
    if bbox is None:
        raise ValueError("Text rendered empty; check font/text.")

    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]

    x0 = (W - bw) // 2 - (bbox[0] - ox)
    y0 = (H - bh) // 2 - (bbox[1] - oy)
    return int(x0), int(y0)


def build_boundary_fields(
    text: str, font: ImageFont.FreeTypeFont, canvas: tuple[int, int]
):
    """
    Builds boundary-based orbit fields:
      dist_grid: distance to UNION boundary (float32)
      vhatx_grid, vhaty_grid: unit vector pointing toward nearest boundary pixel
      letter_id_field: nearest-letter id at each pixel (0..len(text)-1), based on per-letter boundaries
    """
    W, H = canvas
    x0, y0 = compute_centered_text_origin(text, font, canvas)

    dummy = Image.new("L", (1, 1), 0)
    ddraw = ImageDraw.Draw(dummy)

    boundary_union = np.zeros((H, W), dtype=bool)
    boundary_label = -np.ones((H, W), dtype=np.int32)

    prefix = ""
    for li, ch in enumerate(text):
        dx = float(ddraw.textlength(prefix, font=font))
        prefix += ch

        img_ch = Image.new("L", (W, H), 0)
        draw_ch = ImageDraw.Draw(img_ch)
        draw_ch.text((x0 + dx, y0), ch, fill=255, font=font)
        mask = np.array(img_ch, dtype=np.uint8) > 0

        b = mask ^ binary_erosion(mask)
        boundary_union |= b
        boundary_label[b] = li

    if boundary_union.sum() == 0:
        raise ValueError(
            "Union boundary is empty. Increase font size/canvas or check font."
        )

    # Distance + nearest boundary indices
    dist_grid, inds = ndimage.distance_transform_edt(
        ~boundary_union, return_indices=True
    )
    dist_grid = dist_grid.astype(np.float32)
    iy = inds[0].astype(np.int32)
    ix = inds[1].astype(np.int32)

    # Unit vectors to nearest boundary point
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    vx = ix.astype(np.float32) - xx
    vy = iy.astype(np.float32) - yy
    vnorm = np.sqrt(vx * vx + vy * vy) + 1e-6
    vhatx_grid = (vx / vnorm).astype(np.float32)
    vhaty_grid = (vy / vnorm).astype(np.float32)

    # Nearest letter id everywhere (use nearest boundary pixel to pick letter id)
    letter_id_field = boundary_label[iy, ix].astype(np.int32)
    if (letter_id_field < 0).any():
        raise RuntimeError(
            "letter_id_field has -1 entries; boundary_label incomplete (unexpected)."
        )

    return dist_grid, vhatx_grid, vhaty_grid, letter_id_field


def palette_by_character(text: str):
    """Curated palette; repeated characters share a color."""
    base = np.array(
        [
            [0.35, 0.70, 0.98],  # sky
            [0.62, 0.45, 0.98],  # violet
            [0.98, 0.48, 0.70],  # pink
            [0.98, 0.76, 0.35],  # amber
            [0.42, 0.88, 0.60],  # mint
            [0.25, 0.85, 0.92],  # cyan
            [0.92, 0.35, 0.55],  # rose
            [0.78, 0.80, 0.86],  # cool gray
            [0.55, 0.92, 0.35],  # lime
            [0.98, 0.62, 0.25],  # orange
        ],
        dtype=np.float32,
    )

    uniq = []
    for ch in text:
        if ch not in uniq:
            uniq.append(ch)

    char_color = {ch: base[i % len(base)] for i, ch in enumerate(uniq)}
    pal = np.zeros((len(text), 3), dtype=np.float32)
    for i, ch in enumerate(text):
        pal[i] = char_color[ch]
    return pal


# -----------------------------
# Simulation + rendering
# -----------------------------


def simulate_boundary_orbit_gif(
    text: str,
    out_gif: str,
    canvas: tuple[int, int],
    font_path: str | None,
    font_size: int,
    n_particles: int,
    frames: int,
    steps_per_frame: int,
    dt: float,
    # orbit dynamics
    kappa: float,  # ring attraction strength
    omega: float,  # tangential swirl strength
    ring: float,  # target distance from boundary
    swirl_width: float,  # tangential localization width (pixels)
    sigma: float,  # diffusion
    ramp_power: float,  # >1 slows early convergence
    # rendering
    blur_sigma: float,
    gamma: float,
    ink_strength: float,
    fps: int,
    seed: int,
):
    W, H = canvas
    font = load_font(font_size, font_path)

    dist_grid, vhatx_grid, vhaty_grid, letter_id_field = build_boundary_fields(
        text=text, font=font, canvas=canvas
    )
    pal = palette_by_character(text)

    # Uniform init over window
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, W - 1, size=n_particles).astype(np.float32)
    y = rng.uniform(0, H - 1, size=n_particles).astype(np.float32)

    # Fixed particle color assignment based on STARTING point nearest-boundary-letter field
    xi0 = np.clip(x.astype(np.int32), 0, W - 1)
    yi0 = np.clip(y.astype(np.int32), 0, H - 1)
    particle_letter = letter_id_field[yi0, xi0].astype(np.int32)

    pr = pal[particle_letter, 0].astype(np.float32)
    pg = pal[particle_letter, 1].astype(np.float32)
    pb = pal[particle_letter, 2].astype(np.float32)

    rng_dyn = np.random.default_rng(seed + 123)
    sqrt_dt = math.sqrt(dt)

    frames_list: list[np.ndarray] = []

    for frame_idx in range(frames):
        prog = frame_idx / (frames - 1) if frames > 1 else 1.0
        ramp = prog**ramp_power

        kappa_t = kappa * ramp
        omega_t = omega * ramp
        sigma_t = sigma * (1.15 - 0.65 * ramp)  # noisier early

        for _ in range(steps_per_frame):
            dist = bilinear_sample(dist_grid, x, y)
            vhatx = bilinear_sample(vhatx_grid, x, y)
            vhaty = bilinear_sample(vhaty_grid, x, y)

            # Radial term: vhat points toward boundary; drive dist -> ring
            bx = (kappa_t * (dist - ring)) * vhatx
            by = (kappa_t * (dist - ring)) * vhaty

            # Tangential term: J vhat = (-vhaty, vhatx), localized near ring
            tx = -vhaty
            ty = vhatx
            wloc = omega_t * np.exp(
                -((dist - ring) ** 2) / (2.0 * swirl_width * swirl_width)
            )
            bx += wloc * tx
            by += wloc * ty

            # Euler–Maruyama
            x += bx * dt + sigma_t * sqrt_dt * rng_dyn.normal(size=n_particles).astype(
                np.float32
            )
            y += by * dt + sigma_t * sqrt_dt * rng_dyn.normal(size=n_particles).astype(
                np.float32
            )

            # Reflect at window boundaries
            x = np.where(x < 0, -x, x)
            y = np.where(y < 0, -y, y)
            x = np.where(x > W - 1, 2 * (W - 1) - x, x)
            y = np.where(y > H - 1, 2 * (H - 1) - y, y)

        # ---------------- Render once per frame ----------------
        xi = np.clip(x.astype(np.int32), 0, W - 1)
        yi = np.clip(y.astype(np.int32), 0, H - 1)

        counts_rgb = np.zeros((H, W, 3), dtype=np.float32)
        np.add.at(counts_rgb[..., 0], (yi, xi), pr)
        np.add.at(counts_rgb[..., 1], (yi, xi), pg)
        np.add.at(counts_rgb[..., 2], (yi, xi), pb)

        if blur_sigma and blur_sigma > 0:
            for c in range(3):
                counts_rgb[..., c] = ndimage.gaussian_filter(
                    counts_rgb[..., c], blur_sigma
                )

        brightness = counts_rgb.sum(axis=2)
        den = np.percentile(brightness, 99.7) + 1e-6
        ink = np.clip(counts_rgb / den, 0.0, 1.0) ** gamma
        ink = np.clip(ink_strength * ink, 0.0, 1.0)

        # White background: subtract ink
        frame = np.clip(1.0 - ink, 0.0, 1.0)
        frames_list.append((frame * 255).astype(np.uint8))

    imageio.mimsave(out_gif, frames_list, duration=1.0 / fps)
    print(f"Wrote {out_gif} ({frames} frames @ {fps} fps)")


# -----------------------------
# CLI
# -----------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, default="dynestyx")
    ap.add_argument("--out", type=str, default="dynestyx_boundary.gif")
    ap.add_argument("--W", type=int, default=900)
    ap.add_argument("--H", type=int, default=240)
    ap.add_argument("--font_path", type=str, default=None)
    ap.add_argument("--font_size", type=int, default=180)

    # simulation
    ap.add_argument("--n_particles", type=int, default=10_000)
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--steps_per_frame", type=int, default=2)
    ap.add_argument("--dt", type=float, default=0.06)

    # orbit dynamics
    ap.add_argument("--kappa", type=float, default=1.0, help="ring attraction strength")
    ap.add_argument(
        "--omega", type=float, default=15.0, help="tangential swirl strength"
    )
    ap.add_argument(
        "--ring", type=float, default=2.8, help="target distance from boundary (pixels)"
    )
    ap.add_argument(
        "--swirl_width",
        type=float,
        default=2.0,
        help="swirl localization width (pixels)",
    )
    ap.add_argument("--sigma", type=float, default=4.0, help="diffusion strength")
    ap.add_argument(
        "--ramp_power", type=float, default=1.0, help=">1 slows early convergence"
    )

    # rendering
    ap.add_argument("--blur_sigma", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.85)
    ap.add_argument("--ink_strength", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)

    args = ap.parse_args()

    simulate_boundary_orbit_gif(
        text=args.text,
        out_gif=args.out,
        canvas=(args.W, args.H),
        font_path=args.font_path,
        font_size=args.font_size,
        n_particles=args.n_particles,
        frames=args.frames,
        steps_per_frame=args.steps_per_frame,
        dt=args.dt,
        kappa=args.kappa,
        omega=args.omega,
        ring=args.ring,
        swirl_width=args.swirl_width,
        sigma=args.sigma,
        ramp_power=args.ramp_power,
        blur_sigma=args.blur_sigma,
        gamma=args.gamma,
        ink_strength=args.ink_strength,
        fps=args.fps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
