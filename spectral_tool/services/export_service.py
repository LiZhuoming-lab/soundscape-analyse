from __future__ import annotations

import io

import streamlit as st

from spectral_tool.symbolic_analysis import export_symbolic_score_file


@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def build_musicxml_export_bytes(score_bytes: bytes, source_name: str) -> tuple[bytes, str, str]:
    score_stream = io.BytesIO(score_bytes)
    score_stream.name = source_name
    return export_symbolic_score_file(score_stream, fmt="musicxml")

