#!/usr/bin/env python3
"""
dynestyx logo generator (boundary-orbit version).

Trajectory generation is handled by a dynestyx continuous-time model simulated
with SDESimulator:
- particles start uniformly over the full canvas
- each particle receives a fixed color from its initial nearest-letter boundary
- the SDE drift adds tangential orbiting around letter boundaries
- a potential term attracts particles to a thin boundary ring
- diffusion is time-constant
- frames are rendered as white background with subtracted colored ink
"""

import argparse
import os

import imageio.v2 as imageio
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro.distributions as dist
from numpyro.infer import Predictive
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.ndimage import binary_erosion

import dynestyx as dsx
from dynestyx import (
    ContinuousTimeStateEvolution,
    DiracIdentityObservation,
    DynamicalModel,
    SDESimulator,
)

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


def bilinear_sample_jax(
    arr: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray
) -> jnp.ndarray:
    """Bilinear interpolation for arr[y, x], where x/y are pixel coordinates."""
    H, W = arr.shape
    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = jnp.clip(x0, 0, W - 1)
    x1 = jnp.clip(x1, 0, W - 1)
    y0 = jnp.clip(y0, 0, H - 1)
    y1 = jnp.clip(y1, 0, H - 1)

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

    dist_grid_j = jnp.asarray(dist_grid)
    vhatx_grid_j = jnp.asarray(vhatx_grid)
    vhaty_grid_j = jnp.asarray(vhaty_grid)

    total_steps = frames * steps_per_frame
    obs_times = jnp.arange(total_steps + 1, dtype=jnp.float32) * dt

    def logo_model(obs_times=None, obs_values=None):
        def drift_fn(x, u, t):
            dist_to_boundary = bilinear_sample_jax(dist_grid_j, x[0], x[1])
            vhatx = bilinear_sample_jax(vhatx_grid_j, x[0], x[1])
            vhaty = bilinear_sample_jax(vhaty_grid_j, x[0], x[1])

            tx = -vhaty
            ty = vhatx
            wloc = omega * jnp.exp(
                -((dist_to_boundary - ring) ** 2) / (2.0 * swirl_width * swirl_width)
            )
            return jnp.array([wloc * tx, wloc * ty], dtype=jnp.float32)

        def potential_fn(x, u, t):
            dist_to_boundary = bilinear_sample_jax(dist_grid_j, x[0], x[1])
            return 0.5 * kappa * (dist_to_boundary - ring) ** 2

        def diffusion_fn(x, u, t):
            return sigma * jnp.eye(2, dtype=jnp.float32)

        dynamics = DynamicalModel(
            control_dim=0,
            initial_condition=dist.Uniform(
                low=jnp.array([0.0, 0.0], dtype=jnp.float32),
                high=jnp.array([W - 1.0, H - 1.0], dtype=jnp.float32),
            ).to_event(1),
            state_evolution=ContinuousTimeStateEvolution(
                drift=drift_fn,
                potential=potential_fn,
                use_negative_gradient=True,
                diffusion_coefficient=diffusion_fn,
            ),
            observation_model=DiracIdentityObservation(),
        )
        return dsx.sample("f", dynamics, obs_times=obs_times, obs_values=obs_values)

    with SDESimulator(dt0=dt):
        samples = Predictive(logo_model, num_samples=n_particles)(
            jr.PRNGKey(seed),
            predict_times=obs_times,
        )

    states = np.asarray(samples["f_states"], dtype=np.float32)  # (N, T, 2)

    # Fixed particle colors based on STARTING point nearest-letter field
    x0 = states[:, 0, 0]
    y0 = states[:, 0, 1]
    xi0 = np.clip(x0.astype(np.int32), 0, W - 1)
    yi0 = np.clip(y0.astype(np.int32), 0, H - 1)
    particle_letter = letter_id_field[yi0, xi0].astype(np.int32)

    pr = pal[particle_letter, 0].astype(np.float32)
    pg = pal[particle_letter, 1].astype(np.float32)
    pb = pal[particle_letter, 2].astype(np.float32)

    frames_list: list[np.ndarray] = []

    for frame_idx in range(frames):
        step_idx = (frame_idx + 1) * steps_per_frame
        x_frame = states[:, step_idx, 0]
        y_frame = states[:, step_idx, 1]
        xi = np.clip(x_frame.astype(np.int32), 0, W - 1)
        yi = np.clip(y_frame.astype(np.int32), 0, H - 1)

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
    ap.add_argument("--out", type=str, default="dynestyx.gif")
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
    ap.add_argument("--kappa", type=float, default=0.5, help="ring attraction strength")
    ap.add_argument(
        "--omega", type=float, default=24.0, help="tangential swirl strength"
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
        blur_sigma=args.blur_sigma,
        gamma=args.gamma,
        ink_strength=args.ink_strength,
        fps=args.fps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
