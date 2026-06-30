# app.py
import numpy as np
import streamlit as st
from PIL import Image, ImageFilter

st.set_page_config(page_title="Carpet Tile Color Extractor", layout="wide")

st.title("Carpet Tile Average Hex Color")
st.write(
    "Upload a carpet tile image. The app estimates the background from the image border, "
    "masks out background pixels, then removes the tile edge so the border mat is excluded "
    "from the final color."
)

# ----------------------------
# Helpers
# ----------------------------
def pil_to_rgb_np(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("RGB"), dtype=np.uint8)


def rgb_to_hex(rgb) -> str:
    r, g, b = [int(max(0, min(255, round(v)))) for v in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"


def trim_mean_rgb(pixels: np.ndarray, trim_percent: float = 5.0) -> np.ndarray:
    if pixels.size == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)

    pixels = pixels.astype(np.float64)

    if trim_percent <= 0:
        return pixels.mean(axis=0)

    lo = np.percentile(pixels, trim_percent, axis=0)
    hi = np.percentile(pixels, 100 - trim_percent, axis=0)
    keep = np.all((pixels >= lo) & (pixels <= hi), axis=1)
    kept = pixels[keep]
    if kept.size == 0:
        kept = pixels
    return kept.mean(axis=0)


def median_border_color(rgb_img: np.ndarray, border_pct: float = 8.0) -> np.ndarray:
    h, w = rgb_img.shape[:2]
    border = max(1, int(min(h, w) * border_pct / 100.0))

    top = rgb_img[:border, :, :]
    bottom = rgb_img[-border:, :, :]
    left = rgb_img[:, :border, :]
    right = rgb_img[:, -border:, :]

    border_pixels = np.concatenate(
        [
            top.reshape(-1, 3),
            bottom.reshape(-1, 3),
            left.reshape(-1, 3),
            right.reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(border_pixels, axis=0)


def mask_from_background_distance(
    rgb_img: np.ndarray,
    border_pct: float = 8.0,
    threshold: float = 35.0,
) -> np.ndarray:
    bg = median_border_color(rgb_img, border_pct=border_pct).astype(np.float32)
    img = rgb_img.astype(np.float32)

    # RGB distance from estimated background
    dist = np.linalg.norm(img - bg, axis=2)
    return dist > threshold


def bool_mask_to_pil(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")


def pil_mask_to_bool(mask_img: Image.Image) -> np.ndarray:
    return np.array(mask_img) > 0


def clean_mask(mask: np.ndarray, open_size: int = 3, close_size: int = 7) -> np.ndarray:
    """
    Clean a binary mask using only PIL filters.
    """
    m = bool_mask_to_pil(mask)

    if open_size >= 3:
        m = m.filter(ImageFilter.MinFilter(open_size))
        m = m.filter(ImageFilter.MaxFilter(open_size))

    if close_size >= 3:
        m = m.filter(ImageFilter.MaxFilter(close_size))
        m = m.filter(ImageFilter.MinFilter(close_size))

    return pil_mask_to_bool(m)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Simple flood-fill largest component using 8-connectivity.
    """
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)

    best = []
    best_size = 0

    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),   (1, 1),
    ]

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            component = []

            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))

                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))

            if len(component) > best_size:
                best_size = len(component)
                best = component

    out = np.zeros((h, w), dtype=bool)
    for y, x in best:
        out[y, x] = True
    return out


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill holes inside the foreground mask using border flood fill.
    """
    h, w = mask.shape
    inv = ~mask
    visited = np.zeros((h, w), dtype=bool)
    stack = []

    # Seed border background pixels
    for x in range(w):
        if inv[0, x] and not visited[0, x]:
            visited[0, x] = True
            stack.append((0, x))
        if inv[h - 1, x] and not visited[h - 1, x]:
            visited[h - 1, x] = True
            stack.append((h - 1, x))

    for y in range(h):
        if inv[y, 0] and not visited[y, 0]:
            visited[y, 0] = True
            stack.append((y, 0))
        if inv[y, w - 1] and not visited[y, w - 1]:
            visited[y, w - 1] = True
            stack.append((y, w - 1))

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while stack:
        cy, cx = stack.pop()
        for dy, dx in neighbors:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and inv[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                stack.append((ny, nx))

    holes = inv & (~visited)
    return mask | holes


def erode_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    """
    Erode a binary mask by repeatedly applying a 3x3 min filter.
    This is what removes the border mat from the edges.
    """
    if iterations <= 0:
        return mask

    m = bool_mask_to_pil(mask)
    for _ in range(iterations):
        m = m.filter(ImageFilter.MinFilter(3))
    return pil_mask_to_bool(m)


def resize_for_speed(rgb_img: np.ndarray, max_dim: int = 1400) -> np.ndarray:
    h, w = rgb_img.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale >= 1.0:
        return rgb_img
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return np.array(Image.fromarray(rgb_img).resize((new_w, new_h), Image.Resampling.LANCZOS), dtype=np.uint8)


def segment_carpet(rgb_img: np.ndarray, border_pct: float, threshold: float) -> np.ndarray:
    """
    1) Estimate background from image border
    2) Threshold pixels away from background
    3) Clean mask
    4) Keep largest component
    5) Fill holes
    """
    mask = mask_from_background_distance(rgb_img, border_pct=border_pct, threshold=threshold)
    mask = clean_mask(mask, open_size=3, close_size=7)
    mask = keep_largest_component(mask)
    mask = fill_holes(mask)
    return mask


# ----------------------------
# UI
# ----------------------------
uploaded = st.file_uploader("Upload a carpet tile image", type=["png", "jpg", "jpeg", "webp"])

col1, col2, col3 = st.columns(3)
with col1:
    border_pct = st.slider("Border used as background (%)", 2, 20, 8)
with col2:
    threshold = st.slider("Background distance threshold", 5, 100, 30)
with col3:
    trim_pct = st.slider("Trim outliers (%)", 0, 20, 5)

edge_exclude_pct = st.slider(
    "Exclude edge border (%)",
    0,
    40,
    14,
    help="This removes the outer edge of the carpet tile so the mat/border does not affect the average."
)

if uploaded:
    pil_img = Image.open(uploaded)
    rgb = resize_for_speed(pil_to_rgb_np(pil_img), max_dim=1400)

    st.subheader("Original Image")
    st.image(rgb, use_container_width=True)

    # First pass: find the whole tile / carpet area
    fg_mask = segment_carpet(rgb, border_pct=border_pct, threshold=threshold)

    # Second pass: shrink inward to exclude the tile edge / border mat
    h, w = rgb.shape[:2]
    shrink_px = max(0, int(round(edge_exclude_pct / 100.0 * min(h, w))))
    interior_mask = erode_mask(fg_mask, iterations=shrink_px)

    # Fallback if erosion removed too much
    fg_pixels = rgb[interior_mask]
    if len(fg_pixels) == 0:
        st.warning("The interior mask was too small, so using the un-errode mask instead.")
        fg_pixels = rgb[fg_mask]
        interior_mask = fg_mask

    avg_rgb = trim_mean_rgb(fg_pixels, trim_percent=trim_pct)
    hex_code = rgb_to_hex(avg_rgb)

    overlay = rgb.copy()
    overlay[~fg_mask] = (overlay[~fg_mask] * 0.12).astype(np.uint8)
    overlay[fg_mask & ~interior_mask] = np.array([120, 120, 120], dtype=np.uint8)  # show excluded border
    overlay[interior_mask] = (overlay[interior_mask] * 1.0).astype(np.uint8)

    st.subheader("Mask Preview")
    st.image(overlay, use_container_width=True)

    st.subheader("Result")
    c1, c2 = st.columns([1, 3])

    with c1:
        swatch = np.zeros((160, 160, 3), dtype=np.uint8)
        swatch[:, :] = np.array(avg_rgb, dtype=np.uint8)
        st.image(swatch, caption=hex_code, use_container_width=True)

    with c2:
        st.metric("Average Hex Color", hex_code)
        st.write(f"Estimated RGB: `{tuple(int(x) for x in np.round(avg_rgb))}`")
        st.write(f"Pixels used: **{len(fg_pixels):,}**")
        st.write(f"Tile pixels before edge exclusion: **{int(fg_mask.sum()):,}**")
        st.write(f"Tile pixels after edge exclusion: **{int(interior_mask.sum()):,}**")

else:
    st.info("Upload a carpet tile image to calculate its average color.")