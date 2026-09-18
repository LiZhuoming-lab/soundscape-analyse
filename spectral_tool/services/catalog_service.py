from __future__ import annotations

import pandas as pd
import streamlit as st

from spectral_tool.beethoven_sonatas import download_beethoven_sonata_score, list_beethoven_sonata_scores
from spectral_tool.when_in_rome import download_when_in_rome_score, list_when_in_rome_scores


@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def load_when_in_rome_catalog() -> pd.DataFrame:
    return list_when_in_rome_scores()


@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def load_when_in_rome_score_bytes(path: str) -> tuple[bytes, str]:
    return download_when_in_rome_score(path)


@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def load_beethoven_sonata_catalog() -> pd.DataFrame:
    return list_beethoven_sonata_scores()


@st.cache_data(show_spinner=False, ttl=6 * 60 * 60)
def load_beethoven_sonata_score_bytes(path: str) -> tuple[bytes, str]:
    return download_beethoven_sonata_score(path)

