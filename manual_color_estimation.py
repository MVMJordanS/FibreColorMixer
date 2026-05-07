from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, exp, radians, sin, sqrt
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

RGB = Tuple[int, int, int]


@dataclass(frozen=True)
class FiberRow:
    name: str
    rgb: RGB
    denier: float
    ratio: float


# -----------------------------
# Color helpers
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
    if not value:
        raise ValueError("Color value is empty")
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
# UI helpers
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


def default_manual_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"name": "Fiber A", "r": 165, "g": 120, "b": 80, "denier": 20.0, "ratio": 50.0},
            {"name": "Fiber B", "r": 220, "g": 200, "b": 180, "denier": 15.0, "ratio": 30.0},
            {"name": "Fiber C", "r": 90, "g": 70, "b": 50, "denier": 30.0, "ratio": 20.0},
        ]
    )


def normalize_manual_df(df: pd.DataFrame) -> pd.DataFrame:
    required = ["name", "r", "g", "b", "denier", "ratio"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    out = df[required].copy()
    out["name"] = out["name"].astype(str).str.strip()
    out["r"] = pd.to_numeric(out["r"], errors="coerce").fillna(0).astype(int)
    out["g"] = pd.to_numeric(out["g"], errors="coerce").fillna(0).astype(int)
    out["b"] = pd.to_numeric(out["b"], errors="coerce").fillna(0).astype(int)
    out["denier"] = pd.to_numeric(out["denier"], errors="coerce").fillna(0).astype(float)
    out["ratio"] = pd.to_numeric(out["ratio"], errors="coerce").fillna(0).astype(float)

    out = out[(out["name"] != "") & (out["denier"] > 0) & (out["ratio"] >= 0)]
    out = out[(out[["r", "g", "b"]] >= 0).all(axis=1) & (out[["r", "g", "b"]] <= 255).all(axis=1)]
    return out.reset_index(drop=True)


def estimate_manual_blend(df: pd.DataFrame, reference_rgb: RGB | None = None) -> dict:
    if df.empty:
        raise ValueError("At least one valid fiber row is required")

    ratios = df["ratio"].to_numpy(dtype=float)
    total = float(ratios.sum())
    if total <= 0:
        raise ValueError("At least one ratio must be greater than zero")

    weights = ratios / total
    rgb_matrix = np.array(
        [rgb_to_linear_rgb((int(r), int(g), int(b))) for r, g, b in df[["r", "g", "b"]].to_numpy()]
    )
    mixed_linear = weights @ rgb_matrix
    mixed_rgb = linear_rgb_to_rgb(mixed_linear)

    deniers = df["denier"].to_numpy(dtype=float)
    avg_denier = float(np.dot(deniers, weights))

    target_delta_e = None
    if reference_rgb is not None:
        target_delta_e = float(delta_e_2000(rgb_to_lab(reference_rgb), rgb_to_lab(mixed_rgb)))

    rows = []
    for i, row in df.iterrows():
        rows.append(
            {
                "name": row["name"],
                "rgb": (int(row["r"]), int(row["g"]), int(row["b"])),
                "denier": float(row["denier"]),
                "ratio": float(weights[i]),
                "ratio_percent": float(weights[i] * 100.0),
            }
        )

    rows.sort(key=lambda x: x["ratio"], reverse=True)
    return {
        "mixed_rgb": mixed_rgb,
        "avg_denier": avg_denier,
        "delta_e": target_delta_e,
        "weights": rows,
    }


# -----------------------------
# App
# -----------------------------

def main() -> None:
    st.set_page_config(page_title="Manual Color Estimation", layout="wide")
    st.title("Manual Color Estimation")
    st.caption("Enter fibers manually and estimate the blended output color, density proxy, and recipe proportions.")

    with st.sidebar:
        st.header("Settings")
        test_name = st.text_input("Test name", placeholder="Sample 014-A")
        use_reference = st.toggle("Compare against a reference color", value=False)
        reference_color = st.color_picker("Reference color", "#A56C3D", disabled=not use_reference)
        st.divider()
        st.write(
            "This page estimates the output from a user-entered recipe. It does not search for an optimal blend."
        )

    if "manual_df" not in st.session_state:
        st.session_state.manual_df = default_manual_df()
    if "estimate_history" not in st.session_state:
        st.session_state.estimate_history = []

    st.subheader("Enter fibers")
    uploaded = st.file_uploader(
        "Optional: upload CSV to replace the current table",
        type=["csv"],
        help="Expected columns: name, r, g, b, denier, ratio",
    )

    if uploaded is not None:
        try:
            loaded = pd.read_csv(uploaded)
            st.session_state.manual_df = normalize_manual_df(loaded)
            st.success(f"Loaded {len(st.session_state.manual_df)} rows from CSV.")
        except Exception as e:
            st.error(f"Could not load file: {e}")

    edited_df = st.data_editor(
        st.session_state.manual_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "name": st.column_config.TextColumn("Name"),
            "r": st.column_config.NumberColumn("R", min_value=0, max_value=255, step=1),
            "g": st.column_config.NumberColumn("G", min_value=0, max_value=255, step=1),
            "b": st.column_config.NumberColumn("B", min_value=0, max_value=255, step=1),
            "denier": st.column_config.NumberColumn("Denier", min_value=0.0, step=0.5),
            "ratio": st.column_config.NumberColumn("Ratio / Parts", min_value=0.0, step=1.0),
        },
        key="manual_color_editor",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Use edited table", use_container_width=True):
            try:
                st.session_state.manual_df = normalize_manual_df(edited_df)
                st.rerun()
            except Exception as e:
                st.error(f"Could not apply table: {e}")
    with col2:
        if st.button("Reset sample recipe", use_container_width=True):
            st.session_state.manual_df = default_manual_df()
            st.rerun()
    with col3:
        estimate = st.button("Estimate output", type="primary", use_container_width=True)

    try:
        preview_df = normalize_manual_df(edited_df)
    except Exception:
        preview_df = pd.DataFrame(columns=["name", "r", "g", "b", "denier", "ratio"])

    if not preview_df.empty:
        total_ratio = float(preview_df["ratio"].sum())
        st.caption(f"Current total parts: {total_ratio:.2f}")

    if estimate:
        try:
            df = normalize_manual_df(edited_df)
            reference_rgb = hex_to_rgb(reference_color) if use_reference else None
            result = estimate_manual_blend(df, reference_rgb=reference_rgb)

            top_left, top_right = st.columns(2)
            with top_left:
                st.markdown(swatch_html("Estimated blended output", result["mixed_rgb"]), unsafe_allow_html=True)
            with top_right:
                if use_reference:
                    st.markdown(swatch_html("Reference color", reference_rgb), unsafe_allow_html=True)
                else:
                    st.info("Reference comparison is disabled.")

            metrics = st.columns(4)
            metrics[0].metric("Fibers used", len(df))
            metrics[1].metric("Estimated hex", rgb_to_hex(result["mixed_rgb"]))
            metrics[2].metric("Avg denier", f"{result['avg_denier']:.2f}")
            metrics[3].metric(
                "Delta E vs reference",
                f"{result['delta_e']:.4f}" if result["delta_e"] is not None else "-",
            )

            st.subheader("Recipe breakdown")
            recipe_df = pd.DataFrame(result["weights"])
            recipe_df = recipe_df[["name", "rgb", "denier", "ratio_percent"]].copy()
            recipe_df["ratio_percent"] = recipe_df["ratio_percent"].map(lambda x: f"{x:.2f}%")
            st.dataframe(recipe_df, use_container_width=True, hide_index=True)

            chart_df = pd.DataFrame(result["weights"])
            chart_df = chart_df[chart_df["ratio"] > 0].sort_values("ratio", ascending=False)

            if not chart_df.empty:
                fig, ax = plt.subplots(figsize=(8, 8))
                pie_colors = [(r / 255.0, g / 255.0, b / 255.0) for (r, g, b) in chart_df["rgb"]]
                wedges, texts, autotexts = ax.pie(
                    chart_df["ratio_percent"],
                    colors=pie_colors,
                    autopct="%1.1f%%",
                    startangle=90,
                    wedgeprops={"edgecolor": "white", "linewidth": 1},
                )
                ax.set_title("Input Ratio Distribution")
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
                st.info("No positive ratios were available for plotting.")

            st.subheader("Exportable summary")
            summary_lines = [
                f"Estimated output: {rgb_to_hex(result['mixed_rgb'])} / RGB {result['mixed_rgb']}",
                f"Average denier: {result['avg_denier']:.2f}",
            ]
            if result["delta_e"] is not None:
                summary_lines.append(f"Delta E vs reference: {result['delta_e']:.4f}")
            st.code("\n".join(summary_lines), language="text")

            ratio_summary = ", ".join(
                f"{row['name']}: {row['ratio_percent']:.2f}%" for row in result["weights"]
            )
            history_row = {
                "timestamp": pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d %H:%M:%S"),
                "test_name": test_name.strip(),
                "estimated_hex": rgb_to_hex(result["mixed_rgb"]),
                "r": result["mixed_rgb"][0],
                "g": result["mixed_rgb"][1],
                "b": result["mixed_rgb"][2],
                "avg_denier": round(result["avg_denier"], 2),
                "delta_e": round(result["delta_e"], 4) if result["delta_e"] is not None else None,
                "fiber_count": int(len(df)),
                "fiber_names": ", ".join(df["name"].astype(str).tolist()),
                "fiber_ratios": ratio_summary,
            }
            st.session_state.estimate_history.insert(0, history_row)
            st.session_state.estimate_history = st.session_state.estimate_history[:25]

            st.subheader("Recorded estimates")
            history_df = pd.DataFrame(st.session_state.estimate_history)
            if not history_df.empty:
                display_cols = [
                    "timestamp",
                    "test_name",
                    "estimated_hex",
                    "r",
                    "g",
                    "b",
                    "avg_denier",
                    "delta_e",
                    "fiber_count",
                    "fiber_names",
                    "fiber_ratios",
                ]
                history_df = history_df[display_cols]
                st.dataframe(history_df, use_container_width=True, hide_index=True)
                st.caption("Copy cells directly from this table into Excel, Sheets, email, or a report.")
            else:
                st.info("No estimates recorded yet. Click **Estimate output** to add the first row.")

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Edit the table, then click **Estimate output**.")


if __name__ == "__main__":
    main()
