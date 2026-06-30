import streamlit as st

st.set_page_config(page_title="Fiber Tools", layout="wide")

manual_page = st.Page(
    "manual_color_estimation.py",
    title="Manual Estimation",
    icon="🎨",
)

finder_page = st.Page(
    "fiber_mixing_page.py",
    title="Recipe Finder",
    icon="🔎",
)

customer_mixing_page = st.Page(
    "customer_mixing.py",
    title="Customer Mixing",
    icon="🧵",
)

color_extraction_page = st.Page(
    "ColorExtraction.py",
    title="Color Extraction",
    icon="📷",
)

page = st.navigation(
    [manual_page, finder_page, customer_mixing_page, color_extraction_page],
    position="sidebar",   # or "top"
)

page.run()
