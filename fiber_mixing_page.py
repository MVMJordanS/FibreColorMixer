from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from math import atan2, cos, degrees, exp, radians, sin, sqrt, log1p
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

RGB = Tuple[int, int, int]


@dataclass(frozen=True)
class BaseColor:
    name: str
    rgb: RGB
    denier: float


DEFAULT_PALETTE: List[BaseColor] = [
]


# -----------------------------
# Color conversion and parsing
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
# Optimization helpers
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


def denier_strength(denier: float, max_denier: float, influence: float = 0.35) -> float:
    if max_denier <= 0:
        return 1.0
    normalized = log1p(denier) / log1p(max_denier)
    return 1.0 + (normalized * influence)


def solve_visual_mix_weights(
    target_rgb: RGB,
    palette: Sequence[BaseColor],
    max_iter: int = 2500,
    tol: float = 1e-10,
) -> np.ndarray:
    if not palette:
        raise ValueError("Palette cannot be empty")

    X = np.column_stack([rgb_to_linear_rgb(c.rgb) for c in palette])
    t = rgb_to_linear_rgb(target_rgb)

    gram = X.T @ X
    eigmax = float(np.max(np.linalg.eigvalsh(gram)))
    lr = 1.0 / (2.0 * eigmax + 1e-12)

    w = np.ones(len(palette), dtype=float) / len(palette)
    for _ in range(max_iter):
        grad = 2.0 * (X.T @ (X @ w - t))
        new_w = project_to_simplex(w - lr * grad)
        if np.linalg.norm(new_w - w, ord=1) < tol:
            w = new_w
            break
        w = new_w

    return w


def visual_to_recipe_weights(visual_weights: np.ndarray, palette: Sequence[BaseColor], influence: float = 0.35) -> np.ndarray:
    max_denier = max(c.denier for c in palette)
    strengths = np.array([denier_strength(c.denier, max_denier=max_denier, influence=influence) for c in palette], dtype=float)

    raw_recipe = visual_weights / strengths
    raw_recipe = np.maximum(raw_recipe, 0.0)

    total = raw_recipe.sum()
    if total <= 0:
        return np.ones_like(raw_recipe) / len(raw_recipe)

    return raw_recipe / total


def fit_palette_subset(target_rgb: RGB, palette: Sequence[BaseColor], influence: float = 0.35) -> Dict:
    visual_weights = solve_visual_mix_weights(target_rgb, palette)
    recipe_weights = visual_to_recipe_weights(visual_weights, palette, influence=influence)

    max_denier = max(c.denier for c in palette)
    strengths = np.array([denier_strength(c.denier, max_denier=max_denier, influence=influence) for c in palette], dtype=float)
    effective = recipe_weights * strengths
    effective = effective / effective.sum()

    X = np.column_stack([rgb_to_linear_rgb(c.rgb) for c in palette])
    mixed_linear = X @ effective
    mixed_rgb = linear_rgb_to_rgb(mixed_linear)

    target_lab = rgb_to_lab(target_rgb)
    mixed_lab = rgb_to_lab(mixed_rgb)
    error = delta_e_2000(target_lab, mixed_lab)

    rows = []
    for i, c in enumerate(palette):
        rows.append(
            {
                "name": c.name,
                "rgb": c.rgb,
                "denier": c.denier,
                "visual_weight": float(visual_weights[i]),
                "recipe_ratio": float(recipe_weights[i]),
            }
        )

    rows.sort(key=lambda x: x["recipe_ratio"], reverse=True)
    return {"mixed_rgb": mixed_rgb, "delta_e": error, "weights": rows}


def choose_best_fibers_greedily(
    target_rgb: RGB,
    palette: List[BaseColor],
    max_fibers_limit: int,
    min_improvement: float,
    influence: float = 0.35,
) -> Dict:
    if not palette:
        raise ValueError("Palette cannot be empty")

    max_fibers_limit = max(1, min(max_fibers_limit, len(palette)))

    best_result = None
    best_subset: List[BaseColor] = []

    for c in palette:
        result = fit_palette_subset(target_rgb, [c], influence=influence)
        if best_result is None or result["delta_e"] < best_result["delta_e"]:
            best_result = result
            best_subset = [c]

    current_error = best_result["delta_e"]
    remaining = [c for c in palette if c not in best_subset]

    while len(best_subset) < max_fibers_limit and remaining:
        candidate_best_result = None
        candidate_best_subset = None

        for candidate in remaining:
            subset = best_subset + [candidate]
            result = fit_palette_subset(target_rgb, subset, influence=influence)
            if candidate_best_result is None or result["delta_e"] < candidate_best_result["delta_e"]:
                candidate_best_result = result
                candidate_best_subset = subset

        assert candidate_best_result is not None and candidate_best_subset is not None
        improvement = current_error - candidate_best_result["delta_e"]

        if improvement < min_improvement:
            break

        best_subset = candidate_best_subset
        best_result = candidate_best_result
        current_error = candidate_best_result["delta_e"]
        remaining = [c for c in palette if c not in best_subset]

    return {
        "target_rgb": target_rgb,
        "mixed_rgb": best_result["mixed_rgb"],
        "delta_e": round(best_result["delta_e"], 4),
        "fibers_used": len(best_subset),
        "weights": best_result["weights"],
    }


def choose_best_fibers_beam_search(
    target_rgb: RGB,
    palette: List[BaseColor],
    max_fibers_limit: int,
    beam_width: int = 5,
    influence: float = 0.35,
) -> Dict:
    """Search small multi-fiber combinations directly.

    This is much better than a greedy start for colors that are naturally
    created by mixing two or three fibers, such as purples and teals.
    """
    if not palette:
        raise ValueError("Palette cannot be empty")

    max_fibers_limit = max(1, min(max_fibers_limit, len(palette)))
    beam_width = max(1, min(beam_width, len(palette)))

    current_level = []
    for c in palette:
        result = fit_palette_subset(target_rgb, [c], influence=influence)
        current_level.append((result["delta_e"], [c], result))

    current_level.sort(key=lambda item: item[0])
    beam = current_level[:beam_width]

    best_delta = beam[0][0]
    best_result = beam[0][2]
    best_subset = beam[0][1]

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

                result = fit_palette_subset(target_rgb, new_subset, influence=influence)
                next_level.append((result["delta_e"], new_subset, result))

        if not next_level:
            break

        next_level.sort(key=lambda item: item[0])
        beam = next_level[:beam_width]

        if beam[0][0] < best_delta:
            best_delta = beam[0][0]
            best_subset = beam[0][1]
            best_result = beam[0][2]

    return {
        "target_rgb": target_rgb,
        "mixed_rgb": best_result["mixed_rgb"],
        "delta_e": round(best_result["delta_e"], 4),
        "fibers_used": len(best_subset),
        "weights": best_result["weights"],
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


def palette_to_df(palette: Sequence[BaseColor]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": c.name,
                "r": c.rgb[0],
                "g": c.rgb[1],
                "b": c.rgb[2],
                "denier": c.denier,
            }
            for c in palette
        ]
    )


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
        except Exception:
            continue
        if not all(0 <= c <= 255 for c in (r, g, b)):
            continue
        if denier <= 0:
            continue
        colors.append(BaseColor(name, (r, g, b), denier))
    return colors


def load_palette_file(uploaded_file) -> pd.DataFrame:
    """Load a palette from CSV or Excel into the canonical column layout."""
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Please upload a CSV or Excel file.")

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

    df = df.rename(columns=rename_map)

    required = ["name", "r", "g", "b", "denier"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df = df[required].copy()
    df["name"] = df["name"].astype(str).str.strip()
    for c in ["r", "g", "b"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["denier"] = pd.to_numeric(df["denier"], errors="coerce").fillna(0).astype(float)
    df = df[(df["name"] != "") & df["denier"] > 0]
    df = df[(df[["r", "g", "b"]] >= 0).all(axis=1) & (df[["r", "g", "b"]] <= 255).all(axis=1)]
    return df.reset_index(drop=True)


# -----------------------------
# App
# -----------------------------

def main() -> None:
    st.set_page_config(page_title="Fiber Count Convergence", layout="wide")
    st.title("Fiber Count Convergence")
    st.caption("Needle-punch blend helper for target-color matching and recipe estimation.")

    with st.sidebar:
        st.header("Inputs")

        target_hex = st.color_picker("Target color", "#A56C3D")
        target_rgb = hex_to_rgb(target_hex)

        max_fibers_limit = st.slider("Maximum fibers allowed", 1, 12, 6)
        search_mode = st.selectbox(
            "Search strategy",
            ["Greedy", "Beam search (better for mixed colors)"],
            index=1,
        )
        beam_width = st.slider("Beam width", 2, 10, 5)
        min_improvement = st.number_input("Minimum Delta E improvement to add a fiber", min_value=0.0, value=0.75, step=0.05)
        influence = st.slider("Denier influence", 0.0, 1.0, 0.35, 0.01)

        st.divider()
        st.subheader("Needle punch notes")
        st.write(
            "This app keeps the current palette/denier logic and makes it interactive. "
            "You can later add stock limits, locked fibers, or lots/batches for true plant-side use."
        )

    st.subheader("Palette")
    if "palette_df" not in st.session_state:
        st.session_state.palette_df = palette_to_df(DEFAULT_PALETTE)

    uploaded_palette = st.file_uploader(
        "Upload palette CSV or Excel to replace the default palette",
        type=["csv", "xlsx", "xls"],
        help="Expected columns: name, r, g, b, denier. Common aliases are accepted.",
    )
    if uploaded_palette is not None:
        try:
            st.session_state.palette_df = load_palette_file(uploaded_palette)
            st.success(f"Loaded {len(st.session_state.palette_df)} palette rows from upload.")
        except Exception as e:
            st.error(f"Could not load palette file: {e}")

    st.caption("Edit the table below, or upload a file to replace the palette.")

    edited_df = st.data_editor(
        st.session_state.palette_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "name": st.column_config.TextColumn("Name"),
            "r": st.column_config.NumberColumn("R", min_value=0, max_value=255, step=1),
            "g": st.column_config.NumberColumn("G", min_value=0, max_value=255, step=1),
            "b": st.column_config.NumberColumn("B", min_value=0, max_value=255, step=1),
            "denier": st.column_config.NumberColumn("Denier", min_value=0.0, step=1.0),
        },
        key="palette_editor",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Reset palette", use_container_width=True):
            st.session_state.palette_df = palette_to_df(DEFAULT_PALETTE)
            st.rerun()
    with col_b:
        apply_editor = st.button("Use edited palette", use_container_width=True)

    if apply_editor:
        st.session_state.palette_df = edited_df.copy()

    palette = df_to_palette(st.session_state.palette_df)
    if not palette:
        st.error("Palette is empty or invalid. Add at least one valid color row.")
        st.stop()

    run = st.button("Solve blend", type="primary", use_container_width=True)

    if run:
        try:
            if search_mode.startswith("Beam"):
                result = choose_best_fibers_beam_search(
                    target_rgb=target_rgb,
                    palette=palette,
                    max_fibers_limit=max_fibers_limit,
                    beam_width=beam_width,
                    influence=influence,
                )
            else:
                result = choose_best_fibers_greedily(
                    target_rgb=target_rgb,
                    palette=palette,
                    max_fibers_limit=max_fibers_limit,
                    min_improvement=min_improvement,
                    influence=influence,
                )

            mixed_rgb = result["mixed_rgb"]
            delta_e = result["delta_e"]
            weights = result["weights"]

            top_left, top_right = st.columns(2)
            with top_left:
                st.markdown(swatch_html("Target", target_rgb), unsafe_allow_html=True)
            with top_right:
                st.markdown(swatch_html("Mixed result", mixed_rgb), unsafe_allow_html=True)

            metrics = st.columns(3)
            metrics[0].metric("Fibers used", result["fibers_used"])
            metrics[1].metric("Delta E", f"{delta_e:.4f}")
            metrics[2].metric("Target hex", rgb_to_hex(target_rgb))

            st.subheader("Selected recipe")
            recipe_df = pd.DataFrame(weights)
            recipe_df = recipe_df[["name", "rgb", "denier", "visual_weight", "recipe_ratio"]]
            recipe_df["visual_weight"] = recipe_df["visual_weight"].map(lambda x: f"{x:.4f}")
            recipe_df["recipe_ratio"] = recipe_df["recipe_ratio"].map(lambda x: f"{x:.4f}")
            st.dataframe(recipe_df[recipe_df["recipe_ratio"] != "0.0000"], use_container_width=True, hide_index=True)

            st.subheader("All fitted weights")
            chart_df = pd.DataFrame(weights)
            chart_df = chart_df[chart_df["recipe_ratio"] > 0].sort_values("recipe_ratio", ascending=False)

            if not chart_df.empty:
                pie_colors = [
                    (r / 255.0, g / 255.0, b / 255.0)
                    for (r, g, b) in chart_df["rgb"]
                ]

                fig, ax = plt.subplots(figsize=(8, 8))
                wedges, texts, autotexts = ax.pie(
                    chart_df["recipe_ratio"],
                    colors=pie_colors,
                    autopct="%1.1f%%",
                    startangle=90,
                    wedgeprops={"edgecolor": "white", "linewidth": 1},
                )
                ax.set_title("Recipe Ratio Distribution")
                ax.axis("equal")

                ax.legend(
                    wedges,
                    chart_df["name"],
                    title="Fibers",
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                )

                st.pyplot(fig)
            else:
                st.info("No positive recipe ratios to display.")

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Choose your target color and click **Solve blend**.")


if __name__ == "__main__":
    main()
