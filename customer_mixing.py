from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, exp, radians, sin, sqrt
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

RGB = Tuple[int, int, int]

# -----------------------------
# Fixed model / UI constants
# -----------------------------

FIXED_TARGET_DENIER = 25.0
FIXED_BEAM_WIDTH = 6
FIXED_MAX_FIBERS = 7
FIXED_THICKNESS_TRADEOFF = 20

# Denier only gives you a rough proxy for how much surface influence a fiber
# can have. A soft exponent is much closer to reality than 1 / denier.
DENIER_REFERENCE = 20.0
DENIER_EXPONENT = 0.58

# Blend model: mix mostly like a reflectance problem with a little arithmetic
# averaging to keep the output from becoming unrealistically dark.
REFLECTANCE_BLEND = 0.72
REFLECTANCE_EPS = 1e-6

# Penalize deviation from a thickness target if provided.
BICO_MIN_SHARE = 0.08
MIN_ACTIVE_RATIO = 0.01


@dataclass(frozen=True)
class BaseColor:
    name: str
    rgb: RGB
    denier: float
    fleck_factor: float = 1.0


# -----------------------------
# Color conversion
# -----------------------------

def hex_to_rgb(value: str) -> RGB:
    value = value.strip().lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Invalid hex color: {value!r}")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore


def parse_rgb_text(value: str) -> RGB:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise ValueError("RGB must be entered as R,G,B")
    rgb = tuple(int(p) for p in parts)
    for c in rgb:
        if not (0 <= c <= 255):
            raise ValueError(f"RGB channel out of range: {c}")
    return rgb  # type: ignore


def parse_user_color(value: str) -> RGB:
    value = value.strip()
    if value.startswith("#"):
        return hex_to_rgb(value)
    if "," in value:
        return parse_rgb_text(value)
    return hex_to_rgb(value)


def srgb_channel_to_linear(c: float) -> float:
    c = c / 255.0
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def linear_channel_to_srgb(c: float) -> int:
    if c <= 0.0031308:
        v = 12.92 * c
    else:
        v = 1.055 * (c ** (1 / 2.4)) - 0.055
    return int(round(max(0.0, min(1.0, v)) * 255))


def rgb_to_linear_rgb(rgb: RGB) -> np.ndarray:
    return np.array([srgb_channel_to_linear(c) for c in rgb], dtype=float)


def linear_rgb_to_rgb(lrgb: np.ndarray) -> RGB:
    return tuple(linear_channel_to_srgb(float(c)) for c in lrgb)  # type: ignore


def rgb_to_xyz(rgb: RGB) -> Tuple[float, float, float]:
    r, g, b = rgb
    r_lin = srgb_channel_to_linear(r)
    g_lin = srgb_channel_to_linear(g)
    b_lin = srgb_channel_to_linear(b)

    x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
    y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041
    return x, y, z


def f_xyz(t: float) -> float:
    delta = 6 / 29
    if t > delta**3:
        return t ** (1 / 3)
    return t / (3 * delta**2) + 4 / 29


def rgb_to_lab(rgb: RGB) -> Tuple[float, float, float]:
    x, y, z = rgb_to_xyz(rgb)
    xr = x / 0.95047
    yr = y / 1.0
    zr = z / 1.08883
    fx = f_xyz(xr)
    fy = f_xyz(yr)
    fz = f_xyz(zr)
    l = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return l, a, b


def delta_e_2000(lab1: Tuple[float, float, float], lab2: Tuple[float, float, float]) -> float:
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    C1 = sqrt(a1 * a1 + b1 * b1)
    C2 = sqrt(a2 * a2 + b2 * b2)
    avg_C = (C1 + C2) / 2.0

    G = 0.5 * (1 - sqrt((avg_C**7) / (avg_C**7 + 25**7))) if avg_C != 0 else 0
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2

    C1p = sqrt(a1p * a1p + b1 * b1)
    C2p = sqrt(a2p * a2p + b2 * b2)

    def hp_fun(x: float, y: float) -> float:
        if x == 0 and y == 0:
            return 0.0
        h = degrees(atan2(y, x))
        return h + 360 if h < 0 else h

    h1p = hp_fun(a1p, b1)
    h2p = hp_fun(a2p, b2)

    dLp = L2 - L1
    dCp = C2p - C1p

    if C1p * C2p == 0:
        dhp = 0.0
    else:
        dh = h2p - h1p
        if dh > 180:
            dh -= 360
        elif dh < -180:
            dh += 360
        dhp = dh

    dHp = 2 * sqrt(C1p * C2p) * sin(radians(dhp / 2))

    avg_Lp = (L1 + L2) / 2.0
    avg_Cp = (C1p + C2p) / 2.0

    if C1p * C2p == 0:
        avg_hp = h1p + h2p
    else:
        dh = abs(h1p - h2p)
        if dh > 180:
            avg_hp = (h1p + h2p + 360) / 2.0
        else:
            avg_hp = (h1p + h2p) / 2.0

    T = (
        1
        - 0.17 * cos(radians(avg_hp - 30))
        + 0.24 * cos(radians(2 * avg_hp))
        + 0.32 * cos(radians(3 * avg_hp + 6))
        - 0.20 * cos(radians(4 * avg_hp - 63))
    )

    delta_theta = 30 * exp(-(((avg_hp - 275) / 25) ** 2))
    Rc = 2 * sqrt((avg_Cp**7) / (avg_Cp**7 + 25**7)) if avg_Cp != 0 else 0
    Sl = 1 + ((0.015 * ((avg_Lp - 50) ** 2)) / sqrt(20 + ((avg_Lp - 50) ** 2)))
    Sc = 1 + 0.045 * avg_Cp
    Sh = 1 + 0.015 * avg_Cp * T
    Rt = -sin(radians(2 * delta_theta)) * Rc

    return sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


# -----------------------------
# Fiber model
# -----------------------------

def normalize_weights(weights: np.ndarray) -> np.ndarray:
    w = np.maximum(np.array(weights, dtype=float, copy=True), 0.0)
    s = float(w.sum())
    if s <= 0:
        return np.ones_like(w) / len(w)
    return w / s


def fiber_characteristics(rgb: RGB) -> Tuple[float, float]:
    """Return normalized lightness and chroma from the color itself."""
    l, a, b = rgb_to_lab(rgb)
    chroma = sqrt(a * a + b * b)
    return l / 100.0, chroma


def fiber_visibility_factor(denier: float, fleck_factor: float = 1.0, rgb: Optional[RGB] = None) -> float:
    """Soft inverse-denier visibility with small brightness/chroma bias."""
    denier = max(float(denier), 1e-9)
    fleck_factor = max(float(fleck_factor), 1e-9)

    vis = (DENIER_REFERENCE / denier) ** DENIER_EXPONENT

    if rgb is not None:
        lightness, chroma = fiber_characteristics(rgb)
        darkness_bias = 0.88 + 0.22 * (1.0 - lightness)
        chroma_bias = 0.92 + 0.18 * min(chroma / 90.0, 1.0)
        vis *= darkness_bias * chroma_bias

    return vis * fleck_factor


def effective_weights_from_recipe(recipe_weights: np.ndarray, palette: Sequence[BaseColor]) -> Tuple[np.ndarray, np.ndarray]:
    vis = np.array([fiber_visibility_factor(c.denier, c.fleck_factor, c.rgb) for c in palette], dtype=float)
    effective = normalize_weights(recipe_weights * vis)
    return effective, vis


def forward_mix_linear_rgb(effective_weights: np.ndarray, palette: Sequence[BaseColor]) -> np.ndarray:
    """Approximate fiber blend as mostly reflectance-style mixing."""
    e = normalize_weights(effective_weights)
    X = np.column_stack([np.clip(rgb_to_linear_rgb(c.rgb), REFLECTANCE_EPS, 1.0) for c in palette])
    logX = np.log(X)
    geo = np.exp(logX @ e)
    arith = X @ e
    mixed = REFLECTANCE_BLEND * geo + (1.0 - REFLECTANCE_BLEND) * arith
    return np.clip(mixed, 0.0, 1.0)


def predict_blend_rgb(recipe_weights: np.ndarray, palette: Sequence[BaseColor]) -> RGB:
    effective, _ = effective_weights_from_recipe(recipe_weights, palette)
    mixed_linear = forward_mix_linear_rgb(effective, palette)
    return linear_rgb_to_rgb(mixed_linear)


# -----------------------------
# Optimizer helpers
# -----------------------------

def project_to_simplex(v: np.ndarray) -> np.ndarray:
    if v.ndim != 1:
        raise ValueError("v must be a 1D vector")
    n = v.size
    if n == 0:
        return v

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0

    if not np.any(cond):
        return np.ones(n) / n

    rho = ind[cond][-1] - 1
    theta = cssv[rho] / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    s = w.sum()
    if s <= 0:
        return np.ones(n) / n
    return w / s


def thickness_penalty(avg_denier: float, thickness_target: Optional[float]) -> float:
    if thickness_target is None:
        return 0.0
    denom = max(float(thickness_target), 1.0)
    return abs(avg_denier - float(thickness_target)) / denom


def is_bico(color: BaseColor) -> bool:
    return "bico" in color.name.lower()


def enforce_min_share(weights: np.ndarray, index: int, minimum: float) -> np.ndarray:
    w = normalize_weights(weights)
    if index < 0 or index >= len(w):
        raise IndexError("index out of range")
    if len(w) == 1:
        out = np.zeros_like(w)
        out[0] = 1.0
        return out
    if w[index] >= minimum:
        return w

    other_indices = [i for i in range(len(w)) if i != index]
    other_sum = float(w[other_indices].sum())
    if other_sum <= 0:
        out = np.zeros_like(w)
        out[index] = 1.0
        return out

    deficit = minimum - float(w[index])
    scale = (other_sum - deficit) / other_sum
    scale = max(scale, 0.0)
    w[other_indices] *= scale
    w[index] = minimum
    return normalize_weights(w)


def lab_hue_degrees(rgb: RGB) -> float:
    _, a, b = rgb_to_lab(rgb)
    if a == 0 and b == 0:
        return 0.0
    return degrees(atan2(b, a)) % 360.0


def hue_conflict_penalty(weights: np.ndarray, palette: Sequence[BaseColor]) -> float:
    w = normalize_weights(weights)
    hues = np.array([lab_hue_degrees(c.rgb) for c in palette], dtype=float)

    penalty = 0.0
    for i in range(len(palette)):
        for j in range(i + 1, len(palette)):
            wi = float(w[i])
            wj = float(w[j])
            if wi <= 0 or wj <= 0:
                continue

            delta = abs(hues[i] - hues[j])
            delta = min(delta, 360.0 - delta)
            opposition = max(0.0, -cos(radians(delta)))
            penalty += opposition * wi * wj

    return float(penalty)


def mix_score_from_effective_weights(
    effective_weights: np.ndarray,
    palette: Sequence[BaseColor],
    target_rgb: RGB,
    thickness_target: Optional[float] = None,
    thickness_tradeoff: Optional[float] = None,
) -> Dict[str, float | RGB]:
    effective = normalize_weights(effective_weights)
    mixed_linear = forward_mix_linear_rgb(effective, palette)
    mixed_rgb = linear_rgb_to_rgb(mixed_linear)
    color_error = delta_e_2000(rgb_to_lab(target_rgb), rgb_to_lab(mixed_rgb))

    vis = np.array([fiber_visibility_factor(c.denier, c.fleck_factor, c.rgb) for c in palette], dtype=float)
    raw_recipe = normalize_weights(effective / np.maximum(vis, 1e-9))
    deniers = np.array([c.denier for c in palette], dtype=float)
    avg_denier = float(np.dot(deniers, raw_recipe))
    denier_error = thickness_penalty(avg_denier, thickness_target)

    if thickness_tradeoff is not None and thickness_target is not None:
        trade = float(np.clip(thickness_tradeoff, 0, 100))
        color_weight = (100.0 - trade) / 100.0
        thickness_weight = trade / 100.0
        score = (color_weight * color_error) + (thickness_weight * denier_error)
    else:
        score = color_error

    return {
        "mixed_rgb": mixed_rgb,
        "delta_e": float(color_error),
        "score": float(score),
        "avg_denier": avg_denier,
        "denier_error": float(denier_error),
    }


def solve_effective_mix_weights(
    target_rgb: RGB,
    palette: Sequence[BaseColor],
    thickness_target: Optional[float] = None,
    thickness_tradeoff: Optional[float] = None,
    max_iter: int = 250,
    tol: float = 1e-7,
) -> np.ndarray:
    """Optimize effective weights directly on the simplex."""
    if not palette:
        raise ValueError("Palette cannot be empty")

    n = len(palette)
    target_lab = rgb_to_lab(target_rgb)

    lab_colors = np.array([rgb_to_lab(c.rgb) for c in palette], dtype=float)
    dists = np.array([delta_e_2000(target_lab, tuple(lab_colors[i])) for i in range(n)], dtype=float)
    w = np.ones(n, dtype=float) / n
    w[np.argmin(dists)] = 0.55
    w = project_to_simplex(w)

    def objective(weights: np.ndarray) -> float:
        s = mix_score_from_effective_weights(weights, palette, target_rgb, thickness_target, thickness_tradeoff)
        return float(s["score"])

    best_w = w.copy()
    best_obj = objective(best_w)
    step = 0.18
    eps = 1e-4

    for _ in range(max_iter):
        grad = np.zeros(n, dtype=float)
        base = objective(w)
        for i in range(n):
            pert = np.zeros(n, dtype=float)
            pert[i] = eps
            w_plus = project_to_simplex(w + pert)
            w_minus = project_to_simplex(w - pert)
            f_plus = objective(w_plus)
            f_minus = objective(w_minus)
            grad[i] = (f_plus - f_minus) / (2 * eps)

        candidate = project_to_simplex(w - step * grad)
        cand_obj = objective(candidate)

        if cand_obj + tol < base:
            w = candidate
            if cand_obj < best_obj:
                best_obj = cand_obj
                best_w = candidate.copy()
            step *= 1.05
        else:
            step *= 0.5

        if step < 1e-4:
            break

    return best_w


def raw_recipe_from_effective(effective_weights: np.ndarray, palette: Sequence[BaseColor]) -> np.ndarray:
    visibility = np.array([fiber_visibility_factor(c.denier, c.fleck_factor, c.rgb) for c in palette], dtype=float)
    raw = effective_weights / np.maximum(visibility, 1e-9)
    return normalize_weights(raw)


# -----------------------------
# Core fitting
# -----------------------------

def fit_palette_subset(
    target_rgb: RGB,
    palette: Sequence[BaseColor],
    thickness_target: Optional[float] = None,
    thickness_tradeoff: Optional[float] = None,
    require_bico: bool = True,
) -> Dict:
    if not palette:
        raise ValueError("Palette cannot be empty")

    if require_bico and not any(is_bico(c) for c in palette):
        raise ValueError("At least one fiber with 'bico' in the name is required")

    effective_guess = solve_effective_mix_weights(
        target_rgb=target_rgb,
        palette=palette,
        thickness_target=thickness_target,
        thickness_tradeoff=thickness_tradeoff,
    )

    vis = np.array([fiber_visibility_factor(c.denier, c.fleck_factor, c.rgb) for c in palette], dtype=float)
    raw_recipe = raw_recipe_from_effective(effective_guess, palette)

    candidate_results = []
    if require_bico:
        bico_indices = [i for i, c in enumerate(palette) if is_bico(c)]
        for idx in bico_indices:
            candidate_raw = enforce_min_share(raw_recipe, idx, BICO_MIN_SHARE)
            candidate_effective = normalize_weights(candidate_raw * vis)
            score_pack = mix_score_from_effective_weights(
                candidate_effective,
                palette,
                target_rgb,
                thickness_target=thickness_target,
                thickness_tradeoff=thickness_tradeoff,
            )
            rows = []
            for i, c in enumerate(palette):
                rows.append(
                    {
                        "name": c.name,
                        "rgb": c.rgb,
                        "denier": c.denier,
                        "fleck_factor": c.fleck_factor,
                        "visual_weight": float(candidate_effective[i]),
                        "recipe_ratio": float(candidate_raw[i]),
                        "adjusted_ratio": float(candidate_effective[i]),
                        "visibility": float(vis[i]),
                        "is_bico": is_bico(c),
                    }
                )
            rows.sort(key=lambda x: x["adjusted_ratio"], reverse=True)
            candidate_results.append(
                {
                    "mixed_rgb": score_pack["mixed_rgb"],
                    "delta_e": float(score_pack["delta_e"]),
                    "weights": rows,
                    "avg_denier": float(score_pack["avg_denier"]),
                    "denier_error": float(score_pack["denier_error"]),
                    "score": float(score_pack["score"]),
                }
            )
        if not candidate_results:
            raise ValueError("At least one fiber with 'bico' in the name is required")
        best = min(candidate_results, key=lambda x: x["score"])
        return best

    score_pack = mix_score_from_effective_weights(
        effective_guess,
        palette,
        target_rgb,
        thickness_target=thickness_target,
        thickness_tradeoff=thickness_tradeoff,
    )
    rows = []
    for i, c in enumerate(palette):
        rows.append(
            {
                "name": c.name,
                "rgb": c.rgb,
                "denier": c.denier,
                "fleck_factor": c.fleck_factor,
                "visual_weight": float(effective_guess[i]),
                "recipe_ratio": float(raw_recipe[i]),
                "adjusted_ratio": float(effective_guess[i]),
                "visibility": float(vis[i]),
                "is_bico": is_bico(c),
            }
        )
    rows.sort(key=lambda x: x["adjusted_ratio"], reverse=True)

    return {
        "mixed_rgb": score_pack["mixed_rgb"],
        "delta_e": float(score_pack["delta_e"]),
        "weights": rows,
        "avg_denier": float(score_pack["avg_denier"]),
        "denier_error": float(score_pack["denier_error"]),
        "score": float(score_pack["score"]),
    }


def choose_best_fibers_beam_search(
    target_rgb: RGB,
    palette: List[BaseColor],
    max_fibers_limit: int,
    beam_width: int = 5,
    thickness_target: Optional[float] = None,
    thickness_tradeoff: Optional[float] = None,
) -> Dict:
    if not palette:
        raise ValueError("Palette cannot be empty")
    if not any(is_bico(c) for c in palette):
        raise ValueError("At least one fiber with 'bico' in the name is required")

    max_fibers_limit = max(1, min(max_fibers_limit, len(palette)))
    beam_width = max(1, min(beam_width, len(palette)))

    current_level = []
    initial_candidates = [c for c in palette if is_bico(c)]
    for c in initial_candidates:
        result = fit_palette_subset(
            target_rgb,
            [c],
            thickness_target=thickness_target,
            thickness_tradeoff=thickness_tradeoff,
            require_bico=True,
        )
        current_level.append((result["score"], [c], result))

    current_level.sort(key=lambda item: item[0])
    beam = current_level[:beam_width]

    best_score = beam[0][0]
    best_result = beam[0][2]

    for _subset_size in range(2, max_fibers_limit + 1):
        next_level = []
        seen = set()

        for _, subset, _ in beam:
            remaining = [c for c in palette if c not in subset]
            for candidate in remaining:
                new_subset = subset + [candidate]
                key = tuple(sorted(color.name for color in new_subset))
                if key in seen:
                    continue
                seen.add(key)

                result = fit_palette_subset(
                    target_rgb,
                    new_subset,
                    thickness_target=thickness_target,
                    thickness_tradeoff=thickness_tradeoff,
                    require_bico=True,
                )
                next_level.append((result["score"], new_subset, result))

        if not next_level:
            break

        next_level.sort(key=lambda item: item[0])
        beam = next_level[:beam_width]

        if beam[0][0] < best_score:
            best_score = beam[0][0]
            best_result = beam[0][2]

    filtered_weights = [w for w in best_result["weights"] if w["adjusted_ratio"] >= MIN_ACTIVE_RATIO]
    active_count = len(filtered_weights)

    return {
        "target_rgb": target_rgb,
        "mixed_rgb": best_result["mixed_rgb"],
        "delta_e": round(best_result["delta_e"], 4),
        "fibers_used": active_count,
        "weights": filtered_weights,
        "avg_denier": best_result["avg_denier"],
    }


# -----------------------------
# Streamlit helpers
# -----------------------------

def rgb_to_hex(rgb: RGB) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def swatch_html(label: str, rgb: RGB, text_color: str = "#111111") -> str:
    bg = rgb_to_hex(rgb)
    return f"""
    <div style="
        border: 1px solid #ddd;
        border-radius: 14px;
        padding: 14px;
        background: {bg};
        color: {text_color};
        min-height: 92px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    ">
        <div style="font-size: 0.9rem; font-weight: 700; opacity: 0.9;">{label}</div>
        <div style="font-size: 0.95rem;">{bg} · RGB {rgb}</div>
    </div>
    """


def df_to_palette(df: pd.DataFrame) -> List[BaseColor]:
    colors: List[BaseColor] = []
    for _, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        try:
            r = int(row.get("r", 0))
            g = int(row.get("g", 0))
            b = int(row.get("b", 0))
            denier = float(row.get("denier", 0))
            fleck_factor = float(row.get("fleck_factor", 1.0))
        except Exception:
            continue
        if not all(0 <= c <= 255 for c in (r, g, b)):
            continue
        if denier <= 0 or fleck_factor <= 0:
            continue
        colors.append(BaseColor(name, (r, g, b), denier, fleck_factor))
    return colors


def palette_to_df(palette: Sequence[BaseColor]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": c.name,
                "r": c.rgb[0],
                "g": c.rgb[1],
                "b": c.rgb[2],
                "denier": c.denier,
                "fleck_factor": c.fleck_factor,
            }
            for c in palette
        ]
    )


def load_palette_csv(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)

    rename_map = {}
    for col in df.columns:
        c = str(col).strip().lower()
        if c in {"name", "fiber", "color", "colour", "shade"}:
            rename_map[col] = "name"
        elif c in {"r", "red"}:
            rename_map[col] = "r"
        elif c in {"g", "green"}:
            rename_map[col] = "g"
        elif c in {"b", "blue"}:
            rename_map[col] = "b"
        elif c in {"denier", "d", "weight"}:
            rename_map[col] = "denier"
        elif c in {"fleck_factor", "fleck", "influence", "influence_factor"}:
            rename_map[col] = "fleck_factor"

    df = df.rename(columns=rename_map)

    required = ["name", "r", "g", "b", "denier"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    if "fleck_factor" not in df.columns:
        df["fleck_factor"] = 1.0

    df = df[["name", "r", "g", "b", "denier", "fleck_factor"]].copy()
    df["name"] = df["name"].astype(str).str.strip()
    for c in ["r", "g", "b"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["denier"] = pd.to_numeric(df["denier"], errors="coerce").fillna(0).astype(float)
    df["fleck_factor"] = pd.to_numeric(df["fleck_factor"], errors="coerce").fillna(1).astype(float)

    df = df[(df["name"] != "") & (df["denier"] > 0) & (df["fleck_factor"] > 0)]
    df = df[(df[["r", "g", "b"]] >= 0).all(axis=1) & (df[["r", "g", "b"]] <= 255).all(axis=1)]
    return df.reset_index(drop=True)


# -----------------------------
# App
# -----------------------------

def main() -> None:
    st.set_page_config(page_title="Fiber Count Convergence", layout="wide")
    st.title("Fiber Count Convergence")
    st.caption("Needle-punch blend helper using soft denier weighting and reflectance-style mixing.")

    target_hex = st.color_picker("Target color", "#A56C3D")
    target_rgb = hex_to_rgb(target_hex)

    st.subheader("Palette CSV")
    uploaded_palette = st.file_uploader(
        "Upload a CSV with columns like name, r, g, b, denier, fleck_factor",
        type=["csv"],
        help="Required columns: name, r, g, b, denier. fleck_factor is optional.",
    )

    if uploaded_palette is None:
        st.info("Upload a palette CSV to continue.")
        st.stop()

    try:
        palette_df = load_palette_csv(uploaded_palette)
    except Exception as e:
        st.error(f"Could not load palette file: {e}")
        st.stop()

    palette = df_to_palette(palette_df)
    if not palette:
        st.error("Palette is empty or invalid. Add at least one valid color row.")
        st.stop()

    run = st.button("Solve blend", type="primary", use_container_width=True)

    if not run:
        st.info("Pick a target color, upload a palette CSV, and click **Solve blend**.")
        return

    try:
        result = choose_best_fibers_beam_search(
            target_rgb=target_rgb,
            palette=palette,
            max_fibers_limit=FIXED_MAX_FIBERS,
            beam_width=FIXED_BEAM_WIDTH,
            thickness_target=FIXED_TARGET_DENIER,
            thickness_tradeoff=FIXED_THICKNESS_TRADEOFF,
        )

        mixed_rgb = result["mixed_rgb"]
        delta_e = result["delta_e"]
        weights = result["weights"]
        avg_denier = result.get("avg_denier", None)

        top_left, top_right = st.columns(2)
        with top_left:
            st.markdown(swatch_html("Target", target_rgb), unsafe_allow_html=True)
        with top_right:
            st.markdown(swatch_html("Mixed result", mixed_rgb), unsafe_allow_html=True)

        metrics = st.columns(4)
        metrics[0].metric("Fibers used", result["fibers_used"])
        metrics[1].metric("Delta E", f"{delta_e:.4f}")
        metrics[2].metric("Target hex", rgb_to_hex(target_rgb))
        metrics[3].metric("Avg denier", f"{avg_denier:.2f}" if avg_denier is not None else "-")

        st.subheader("Selected recipe")
        recipe_df = pd.DataFrame(weights)
        recipe_df = recipe_df[
            [
                "name",
                "rgb",
                "denier",
                "fleck_factor",
                "visual_weight",
                "recipe_ratio",
                "adjusted_ratio",
                "visibility",
                "is_bico",
            ]
        ].copy()
        recipe_df["visual_weight"] = recipe_df["visual_weight"].map(lambda x: f"{x:.4f}")
        recipe_df["recipe_ratio"] = recipe_df["recipe_ratio"].map(lambda x: f"{x:.4f}")
        recipe_df["adjusted_ratio"] = recipe_df["adjusted_ratio"].map(lambda x: f"{x:.4f}")
        recipe_df["visibility"] = recipe_df["visibility"].map(lambda x: f"{x:.4f}")
        recipe_df["fleck_factor"] = recipe_df["fleck_factor"].map(lambda x: f"{x:.2f}")
        st.dataframe(recipe_df[recipe_df["adjusted_ratio"] != "0.0000"], use_container_width=True, hide_index=True)

        st.subheader("Optical color contribution distribution")
        chart_df = pd.DataFrame(weights)
        chart_df = chart_df[chart_df["adjusted_ratio"] > 0].sort_values("adjusted_ratio", ascending=False)

        if not chart_df.empty:
            pie_colors = [(r / 255.0, g / 255.0, b / 255.0) for (r, g, b) in chart_df["rgb"]]
            fig, ax = plt.subplots(figsize=(8, 8))
            wedges, texts, autotexts = ax.pie(
                chart_df["adjusted_ratio"],
                colors=pie_colors,
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 1},
            )
            ax.set_title("Adjusted Weight Distribution")
            ax.axis("equal")
            ax.legend(wedges, chart_df["name"], title="Fibers", loc="center left", bbox_to_anchor=(1.02, 0.5))
            st.pyplot(fig)

            st.subheader("Mix ratio distribution")
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            wedges2, texts2, autotexts2 = ax2.pie(
                chart_df["recipe_ratio"],
                colors=pie_colors,
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 1},
            )
            ax2.set_title("Recipe Ratio Distribution (Input Mix)")
            ax2.axis("equal")
            ax2.legend(wedges2, chart_df["name"], title="Fibers", loc="center left", bbox_to_anchor=(1.02, 0.5))
            st.pyplot(fig2)
        else:
            st.info("No positive adjusted weights to display.")

    except Exception as e:
        st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
