from __future__ import annotations

import io
from pathlib import Path

import streamlit as st

from spectral_tool.beethoven_sonatas import BEETHOVEN_SONATAS_WEB_URL
from spectral_tool.services.catalog_service import (
    load_beethoven_sonata_catalog,
    load_beethoven_sonata_score_bytes,
    load_when_in_rome_catalog,
    load_when_in_rome_score_bytes,
)
from spectral_tool.services.export_service import build_musicxml_export_bytes
from spectral_tool.state.symbolic_state import (
    build_symbolic_analysis_signature,
    init_cadence_annotations,
    init_harmony_annotations,
    init_theme_annotations,
)
from spectral_tool.symbolic_analysis import analyze_symbolic_score, build_symbolic_export_payload, SymbolicAnalysisConfig
from spectral_tool.ui.cadence_editor import render_cadence_editor
from spectral_tool.ui.harmony_editor import render_harmony_editor
from spectral_tool.ui.theme_editor import render_theme_editor
from spectral_tool.when_in_rome import WHEN_IN_ROME_WEB_URL


def render_score_workspace(symbolic_config: SymbolicAnalysisConfig) -> None:
    st.markdown("### 乐谱 / 符号分析工作台")
    source_mode = st.radio(
        "乐谱来源",
        options=["local_upload", "when_in_rome", "beethoven_sonatas"],
        horizontal=True,
        format_func=lambda value: {
            "local_upload": "本地上传",
            "when_in_rome": "When-in-Rome 语料库",
            "beethoven_sonatas": "贝多芬钢琴奏鸣曲 32 首",
        }[value],
        key="symbolic_source_mode",
    )

    st.markdown(
        """
当前是“第一阶段可用版本”，它已经把项目从纯音频频谱工具往音乐分析平台推进了一步：

- 支持 `MusicXML / MXL / MIDI`
- 提取音高统计、音高类分布、音级分布
- 提取旋律音程与音程序类分布
- 生成和声切片与 Roman numeral 候选
- 做连续音窗口的主题 / 动机再现检索
- 允许人工修订和声标签与再现类型
"""
    )

    score_bytes: bytes | None = None
    score_source = None
    score_source_label = ""
    score_source_caption = ""

    if source_mode == "local_upload":
        score_file = st.file_uploader(
            "上传乐谱文件（支持 MusicXML、MXL、MIDI、KRN）",
            type=["musicxml", "xml", "mxl", "mid", "midi", "krn", "kern"],
            key="score_file_uploader",
        )
        if score_file is None:
            st.info("上传一份乐谱后，系统会生成和声、音高、音程和主题再现的第一轮分析结果。")
            return
        score_bytes = score_file.getvalue()
        score_source = score_file
        score_source_label = str(getattr(score_file, "name", "uploaded_score"))
        score_source_caption = "当前来源：本地上传"
    elif source_mode == "when_in_rome":
        st.markdown(f"**When-in-Rome 语料库**：[打开仓库]({WHEN_IN_ROME_WEB_URL})")
        st.caption("这里接入的是右侧符号分析工作台的外部语料库入口。你可以不上传本地乐谱，直接调用仓库里现成的 MusicXML / MXL / MIDI 文件。")

        try:
            catalog = load_when_in_rome_catalog()
        except RuntimeError as exc:
            st.warning(str(exc))
            return
        if catalog.empty:
            st.warning("暂时没有从 When-in-Rome 语料库读取到可用乐谱。")
            return

        category_options = ["全部"] + sorted(catalog["category_label"].dropna().unique().tolist())
        selected_category = st.selectbox("类别", options=category_options, key="wir_category")
        filtered_catalog = catalog.copy()
        if selected_category != "全部":
            filtered_catalog = filtered_catalog.loc[filtered_catalog["category_label"] == selected_category].copy()

        composer_options = ["全部"] + sorted(filtered_catalog["composer_label"].dropna().unique().tolist())
        selected_composer = st.selectbox("作曲家", options=composer_options, key="wir_composer")
        if selected_composer != "全部":
            filtered_catalog = filtered_catalog.loc[filtered_catalog["composer_label"] == selected_composer].copy()

        keyword = st.text_input("关键词筛选（作品 / 曲目）", key="wir_keyword").strip()
        if keyword:
            keyword_lower = keyword.lower()
            filtered_catalog = filtered_catalog.loc[
                filtered_catalog["display_name"].str.lower().str.contains(keyword_lower, na=False)
                | filtered_catalog["path"].str.lower().str.contains(keyword_lower, na=False)
            ].copy()

        if filtered_catalog.empty:
            st.warning("当前筛选条件下没有匹配到可调用的语料文件。")
            return

        st.caption(f"当前筛选到 {len(filtered_catalog)} 份可分析乐谱。")
        selected_path = st.selectbox(
            "选择语料文件",
            options=filtered_catalog["path"].tolist(),
            format_func=lambda value: filtered_catalog.loc[filtered_catalog["path"] == value, "display_name"].iloc[0],
            key="wir_path",
        )
        selected_row = filtered_catalog.loc[filtered_catalog["path"] == selected_path].iloc[0]
        st.caption(f"当前文件：{selected_row['path']}")

        try:
            score_bytes, score_name = load_when_in_rome_score_bytes(str(selected_path))
        except RuntimeError as exc:
            st.warning(str(exc))
            return
        score_stream = io.BytesIO(score_bytes)
        score_stream.name = str(score_name)
        score_source = score_stream
        score_source_label = str(selected_row["display_name"])
        score_source_caption = f"当前来源：When-in-Rome / {selected_row['path']}"
    else:
        st.markdown(f"**贝多芬钢琴奏鸣曲 32 首语料库**：[打开仓库]({BEETHOVEN_SONATAS_WEB_URL})")
        st.caption("这里接入的是 craigsapp 提供的 Beethoven piano sonatas GitHub 语料库，当前以 Humdrum / kern 文件为主。")

        try:
            catalog = load_beethoven_sonata_catalog()
        except RuntimeError as exc:
            st.warning(str(exc))
            return
        if catalog.empty:
            st.warning("暂时没有从贝多芬钢琴奏鸣曲语料库读取到可用乐谱。")
            return

        sonata_numbers = sorted(int(value) for value in catalog["sonata_number"].dropna().unique().tolist() if int(value) > 0)
        selected_sonata = st.selectbox(
            "选择奏鸣曲",
            options=sonata_numbers,
            format_func=lambda value: f"Sonata No.{int(value)}",
            key="beethoven_sonata_number",
        )
        filtered_catalog = catalog.loc[catalog["sonata_number"] == int(selected_sonata)].copy()
        movement_numbers = sorted(int(value) for value in filtered_catalog["movement_number"].dropna().unique().tolist() if int(value) > 0)
        selected_movement = st.selectbox(
            "选择乐章",
            options=movement_numbers,
            format_func=lambda value: f"Movement {int(value)}",
            key="beethoven_sonata_movement",
        )
        selected_row = filtered_catalog.loc[filtered_catalog["movement_number"] == int(selected_movement)].iloc[0]
        st.caption(f"当前文件：{selected_row['path']}")

        try:
            score_bytes, score_name = load_beethoven_sonata_score_bytes(str(selected_row["path"]))
        except RuntimeError as exc:
            st.warning(str(exc))
            return
        score_stream = io.BytesIO(score_bytes)
        score_stream.name = str(score_name)
        score_source = score_stream
        score_source_label = str(selected_row["display_name"])
        score_source_caption = f"当前来源：Beethoven Piano Sonatas / {selected_row['path']}"

    if score_bytes is None or score_source is None:
        return

    st.caption(score_source_caption)
    symbolic_analysis_key = build_symbolic_analysis_signature(score_bytes, symbolic_config)
    with st.spinner("正在解析乐谱、提取音高关系并构建主题再现候选..."):
        symbolic_result = analyze_symbolic_score(score_source, config=symbolic_config)

    harmony_state_key = init_harmony_annotations(symbolic_result, symbolic_analysis_key)
    cadence_state_key = init_cadence_annotations(symbolic_result, symbolic_analysis_key)
    theme_state_key = init_theme_annotations(symbolic_result, symbolic_analysis_key)
    harmony_annotations = st.session_state[harmony_state_key].copy()
    cadence_annotations = st.session_state[cadence_state_key].copy()
    theme_annotations = st.session_state[theme_state_key].copy()

    metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)
    metric_1.metric("乐谱标题", str(symbolic_result["score_title"]))
    metric_2.metric("总音高事件", int(symbolic_result["total_notes"]))
    metric_3.metric("不同音级类", int(symbolic_result["unique_pitch_classes"]))
    metric_4.metric("终止候选", int(len(symbolic_result["cadence_candidates"])))
    metric_5.metric("主题再现候选", int(len(symbolic_result["theme_matches"])))
    st.caption(f"当前分析对象：{score_source_label}")

    st.subheader("机器摘要")
    st.text("\n".join(symbolic_result["summary_lines"]))

    score_tab_1, score_tab_2, score_tab_3, score_tab_4, score_tab_5, score_tab_6, score_tab_7 = st.tabs(
        ["总览", "和声", "终止", "音高", "音程 / 关系", "主题 / 动机", "导出"]
    )

    with score_tab_1:
        st.markdown(f"**全局调性 / 调式估计：** {symbolic_result['global_key']}")
        st.markdown("**声部摘要**")
        st.dataframe(symbolic_result["part_summary"], width="stretch", hide_index=True)

        if not symbolic_result["measure_pitch_summary"].empty:
            st.markdown("**按小节的音高浓度变化**")
            measure_curve = symbolic_result["measure_pitch_summary"].set_index("measure_number")[
                ["note_count", "unique_pitch_classes"]
            ]
            st.line_chart(measure_curve, height=280)
            st.dataframe(symbolic_result["measure_pitch_summary"], width="stretch", hide_index=True)
        else:
            st.info("当前乐谱没有生成可展示的小节级音高摘要。")

    with score_tab_2:
        harmony_annotations = render_harmony_editor(harmony_annotations, symbolic_result, symbolic_analysis_key)
        st.session_state[harmony_state_key] = harmony_annotations

    with score_tab_3:
        cadence_annotations = render_cadence_editor(cadence_annotations, symbolic_analysis_key)
        st.session_state[cadence_state_key] = cadence_annotations

    with score_tab_4:
        if not symbolic_result["pitch_class_histogram"].empty:
            st.markdown("**音级类分布**")
            st.bar_chart(
                symbolic_result["pitch_class_histogram"].set_index("pitch_class_label")["count"],
                height=280,
            )
        if not symbolic_result["scale_degree_histogram"].empty:
            st.markdown("**音级分布**")
            degree_chart = symbolic_result["scale_degree_histogram"].copy()
            degree_chart["scale_degree"] = degree_chart["scale_degree"].astype(str)
            st.bar_chart(degree_chart.set_index("scale_degree")["count"], height=240)

        st.markdown("**音高总表（前 300 行）**")
        st.dataframe(symbolic_result["note_table"].head(300), width="stretch", hide_index=True)

    with score_tab_5:
        if symbolic_result["interval_table"].empty:
            st.warning("当前乐谱没有足够的连续旋律音高用于音程分析。")
        else:
            st.markdown("**音程序类分布**")
            st.bar_chart(
                symbolic_result["interval_class_histogram"].set_index("interval_class")["count"],
                height=260,
            )
            if not symbolic_result["directed_interval_histogram"].empty:
                st.markdown("**最常见定向音程**")
                directed_histogram = symbolic_result["directed_interval_histogram"].head(12).set_index("directed_name")["count"]
                st.bar_chart(directed_histogram, height=260)
            st.dataframe(symbolic_result["interval_table"].head(300), width="stretch", hide_index=True)

        if not symbolic_result["measure_consonance_summary"].empty:
            st.markdown("**按小节的音程协和走向**")
            st.caption("数值越小越协和。这里只统计同一时刻纵向同时发声的音，不把分解和弦算进去。")
            st.line_chart(
                symbolic_result["measure_consonance_summary"].set_index("measure_number")[["mean_consonance_rank"]],
                height=260,
            )
            st.dataframe(symbolic_result["measure_consonance_summary"], width="stretch", hide_index=True)

        if not symbolic_result["vertical_interval_histogram"].empty:
            st.markdown("**纵向音程分布**")
            vertical_histogram = symbolic_result["vertical_interval_histogram"].copy()
            vertical_histogram["interval_display"] = vertical_histogram.apply(
                lambda row: f"{int(row['consonance_rank'])} = {row['interval_name']}",
                axis=1,
            )
            st.bar_chart(vertical_histogram.set_index("interval_display")["count"], height=260)

        if not symbolic_result["vertical_interval_table"].empty:
            st.markdown("**纵向音程明细（前 300 行）**")
            st.dataframe(symbolic_result["vertical_interval_table"].head(300), width="stretch", hide_index=True)

    with score_tab_6:
        theme_annotations = render_theme_editor(theme_annotations, symbolic_analysis_key)
        st.session_state[theme_state_key] = theme_annotations

    with score_tab_7:
        exported_harmony = harmony_annotations.copy()
        exported_cadence = cadence_annotations.copy()
        exported_theme = theme_annotations.copy()
        if not exported_harmony.empty:
            exported_harmony = exported_harmony.loc[exported_harmony["export"]].copy()
        if not exported_cadence.empty:
            exported_cadence = exported_cadence.loc[exported_cadence["export"]].copy()
        if not exported_theme.empty:
            exported_theme = exported_theme.loc[exported_theme["export"]].copy()

        note_csv = symbolic_result["note_table"].to_csv(index=False).encode("utf-8-sig")
        harmony_csv = exported_harmony.to_csv(index=False).encode("utf-8-sig")
        cadence_csv = exported_cadence.to_csv(index=False).encode("utf-8-sig")
        interval_csv = symbolic_result["interval_table"].to_csv(index=False).encode("utf-8-sig")
        vertical_interval_csv = symbolic_result["vertical_interval_table"].to_csv(index=False).encode("utf-8-sig")
        measure_consonance_csv = symbolic_result["measure_consonance_summary"].to_csv(index=False).encode("utf-8-sig")
        theme_csv = exported_theme.to_csv(index=False).encode("utf-8-sig")
        summary_text = "\n".join(symbolic_result["summary_lines"]).encode("utf-8")
        analysis_json = build_symbolic_export_payload(
            symbolic_result,
            harmony_annotations=exported_harmony,
            cadence_annotations=exported_cadence,
            theme_annotations=exported_theme,
        )
        source_name = str(getattr(score_source, "name", "") or score_source_label or "score_input.musicxml")
        source_suffix = Path(source_name).suffix.lower()
        original_score_mime = {
            ".mxl": "application/vnd.recordare.musicxml",
            ".musicxml": "application/vnd.recordare.musicxml+xml",
            ".xml": "application/xml",
            ".mid": "audio/midi",
            ".midi": "audio/midi",
            ".krn": "text/plain",
            ".kern": "text/plain",
        }.get(source_suffix, "application/octet-stream")
        musicxml_bytes, musicxml_filename, musicxml_mime = build_musicxml_export_bytes(score_bytes, source_name)

        st.markdown("**分析结果导出**")
        result_export_col_1, result_export_col_2 = st.columns(2, gap="medium")
        with result_export_col_1:
            st.download_button("下载音高总表 CSV", data=note_csv, file_name="score_note_table.csv", mime="text/csv", use_container_width=True)
            st.download_button("下载和声分析 CSV", data=harmony_csv, file_name="harmony_annotations.csv", mime="text/csv", use_container_width=True)
            st.download_button("下载终止候选 CSV", data=cadence_csv, file_name="cadence_candidates.csv", mime="text/csv", use_container_width=True)
            st.download_button("下载音程分析 CSV", data=interval_csv, file_name="interval_table.csv", mime="text/csv", use_container_width=True)
        with result_export_col_2:
            st.download_button("下载纵向音程 CSV", data=vertical_interval_csv, file_name="vertical_interval_table.csv", mime="text/csv", use_container_width=True)
            st.download_button("下载协和走向 CSV", data=measure_consonance_csv, file_name="measure_consonance_summary.csv", mime="text/csv", use_container_width=True)
            st.download_button("下载主题再现 CSV", data=theme_csv, file_name="theme_matches.csv", mime="text/csv", use_container_width=True)
            st.download_button("下载摘要 TXT", data=summary_text, file_name="symbolic_summary.txt", mime="text/plain", use_container_width=True)
            st.download_button("下载完整分析 JSON", data=analysis_json, file_name="symbolic_analysis.json", mime="application/json", use_container_width=True)

        st.markdown("**乐谱文件导出**")
        st.caption("这一组适合把乐谱文件直接带到其他音频软件、乐谱软件或外部分析环境里继续查看。")
        score_export_col_1, score_export_col_2 = st.columns(2, gap="medium")
        with score_export_col_1:
            if source_suffix in {".mxl", ".musicxml", ".xml", ".krn", ".kern"}:
                st.download_button(
                    "下载原始乐谱文件",
                    data=score_bytes,
                    file_name=Path(source_name).name,
                    mime=original_score_mime,
                    use_container_width=True,
                )
        with score_export_col_2:
            st.download_button(
                "下载 MusicXML 乐谱",
                data=musicxml_bytes,
                file_name=musicxml_filename,
                mime=musicxml_mime,
                use_container_width=True,
            )

    st.markdown(
        """
提示：

- 如果主题再现候选过少，可以缩短“主题检索窗口”
- 如果候选过多，可以增大窗口长度，或者先把焦点放到一个主旋律声部
- Roman numeral 目前是第一阶段候选，并不等于最终和声结论
- 你在“和声”和“主题 / 动机”页中的人工修订会参与导出
"""
    )
