from __future__ import annotations

from spectral_tool.analysis import AnalysisConfig, CANDIDATE_BOUNDARY_LABEL

EVENT_MODEL_PRESETS = {
    "balanced_default": {
        "label": "平衡默认",
        "weights": (0.40, 0.25, 0.20, 0.15),
        "threshold_sigma": 1.0,
        "min_event_distance_sec": 5.0,
        "note": "当前的通用默认模式，适合先做一轮平衡检测。",
    },
    "traditional_structure": {
        "label": "传统结构模式",
        "weights": (0.30, 0.20, 0.30, 0.20),
        "threshold_sigma": 1.2,
        "min_event_distance_sec": 6.5,
        "note": "更强调起音、力度、织体和结构边界，适合古典、浪漫主义钢琴和交响作品。",
    },
    "timbre_soundscape": {
        "label": "音色 / 声景模式",
        "weights": (0.45, 0.30, 0.15, 0.10),
        "threshold_sigma": None,
        "min_event_distance_sec": None,
        "note": "更强调新材料、新音色和频谱状态变化，适合频谱音乐、当代音乐、声景音乐。",
    },
}

OPERATION_RESULT_OPTIONS = [
    "保留",
    "删除",
    f"改成{CANDIDATE_BOUNDARY_LABEL}",
    "改成局部事件",
    "暂不处理",
]


def format_weight_percent(weight: float) -> int:
    return int(round(float(weight) * 100))


def novelty_explanation(config: AnalysisConfig) -> str:
    extra_note = ""
    if config.event_model_preset == "timbre_soundscape":
        extra_note = """
在“音色 / 声景模式”里，上面的“频谱形态差异”这一项还会额外参考：

- `spectral centroid`：看频谱重心有没有整体移动
- `band energy ratio`：看不同频带之间的能量关系有没有改写
- `高频占比`：看高频材料是否明显抬升或退去
- `flatness`：看声音是否更趋向噪声质地

这样可以更敏感地抓到“新音色 / 新材质 / 新频谱状态”的进入。
"""

    return f"""
自动事件点来自“频谱新颖度检测”，它看的是相邻时刻之间的频谱是否发生了明显变化，而不是只看某一个单独音高。

- `{format_weight_percent(config.novelty_weight_cosine)}%` 频谱形态差异：比较前后频谱整体轮廓是否变了
- `{format_weight_percent(config.novelty_weight_flux)}%` 谱流量 `spectral flux`：看新的频率能量是否突然涌入
- `{format_weight_percent(config.novelty_weight_onset)}%` 起音强度 `onset strength`：看瞬时攻击和突发性是否增强
- `{format_weight_percent(config.novelty_weight_rms)}%` 能量突变 `RMS delta`：看整体响度是否突然变化

当这一条“新颖度曲线”超过阈值并形成峰值时，系统就会在那个时间点标成一个候选事件。

当前默认同时使用短、中、长三种频谱时间窗。事件详情中的“多尺度共识”表示同一变化是否在不同时间尺度上都能被观察到；
它和事件前后状态变化共同构成证据分数。证据分数是可解释的规则分数，不是统计概率。
{extra_note}
"""
