from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import streamlit as st

from spectral_tool.ui.audio_workspace import render_audio_workspace
from spectral_tool.ui.score_workspace import render_score_workspace
from spectral_tool.ui.sidebar_audio import render_audio_sidebar
from spectral_tool.ui.sidebar_score import render_score_sidebar


def _background_image_data_uri() -> str:
    asset_path = Path(__file__).parent / "spectral_tool" / "assets" / "ink_landscape.svg"
    svg_text = asset_path.read_text(encoding="utf-8")
    return f"data:image/svg+xml;utf8,{quote(svg_text)}"


st.set_page_config(
    page_title="谱境研音 / Soundscape Analyse",
    page_icon="🎼",
    layout="wide",
)
BACKGROUND_DATA_URI = _background_image_data_uri()

st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background:
            linear-gradient(rgba(247, 245, 240, 0.92), rgba(247, 245, 240, 0.95)),
            url("{BACKGROUND_DATA_URI}") center 120px / min(120vw, 2200px) auto no-repeat fixed;
        background-color: #f7f5f0;
    }}
    [data-testid="stHeader"] {{
        background: rgba(247, 245, 240, 0.72);
        backdrop-filter: blur(8px);
    }}
    [data-testid="stSidebar"] > div:first-child {{
        background: rgba(252, 251, 247, 0.80);
        backdrop-filter: blur(10px);
    }}
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: rgba(255, 255, 255, 0.62);
        backdrop-filter: blur(6px);
        border-radius: 18px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("谱境研音 / Soundscape Analyse")
st.caption("保留原有声景 / 频谱事件分析，同时新增第一阶段乐谱符号分析工作台。")
st.markdown("### 音乐分析交流可联系V：`Mendel_Dog`")
if "workspace_mode" not in st.session_state:
    st.session_state["workspace_mode"] = "audio"

workspace_left, workspace_right = st.columns(2, gap="large")
with workspace_left:
    with st.container(border=True):
        st.markdown("### 左侧工作台")
        st.markdown("**声音频谱事件分析**")
        st.caption("适合做频谱、声景状态、自动事件点、局部试听与人工修订。")
        if st.session_state["workspace_mode"] == "audio":
            st.button("当前正在使用", key="workspace_audio_current", use_container_width=True, disabled=True)
        else:
            if st.button("进入左侧工作台", key="workspace_audio_switch", use_container_width=True):
                st.session_state["workspace_mode"] = "audio"
                st.rerun()

with workspace_right:
    with st.container(border=True):
        st.markdown("### 右侧工作台")
        st.markdown("**music21 符号分析**")
        st.caption("适合做音高、音级、音程、和声切片与主题 / 动机再现初筛。")
        if st.session_state["workspace_mode"] == "score":
            st.button("当前正在使用", key="workspace_score_current", use_container_width=True, disabled=True)
        else:
            if st.button("进入右侧工作台", key="workspace_score_switch", use_container_width=True):
                st.session_state["workspace_mode"] = "score"
                st.rerun()

workspace = str(st.session_state.get("workspace_mode", "audio"))

with st.sidebar:
    st.subheader("当前工作台")
    st.write("声音频谱事件分析" if workspace == "audio" else "music21 符号分析")
    st.caption("页面顶部的左右入口可以直接切换两个并列工作台。")

    if workspace == "audio":
        audio_sidebar_values = render_audio_sidebar()
        symbolic_config = None
    else:
        symbolic_config = render_score_sidebar()
        audio_sidebar_values = None

if workspace == "score":
    render_score_workspace(symbolic_config)
else:
    render_audio_workspace(audio_sidebar_values)
