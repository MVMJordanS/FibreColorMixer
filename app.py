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

page = st.navigation(
    [manual_page, finder_page],
    position="sidebar",   # or "top"
)

page.run()
