from __future__ import annotations

import json
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .analysis import CANDIDATE_BOUNDARY_LABEL, split_label_text

_FOCUS_PREFERENCE_OPTIONS = [
    {"key": "sustained", "label": "更重视持续变化"},
    {"key": "spike", "label": "更关注局部尖峰"},
]

_TERM_PREFERENCE_OPTIONS = ["材料聚集", "消散", "稳态持续", "高频扩展"]

_INTERACTION_PRIORITY_META = {
    "weak": {
        "label": "较小变化",
        "assistant_hint": "先看一眼就好",
        "depth": "light",
    },
    "medium": {
        "label": "一般变化",
        "assistant_hint": "可做一步跟进",
        "depth": "guided",
    },
    "strong": {
        "label": "重点变化",
        "assistant_hint": "建议重点讨论",
        "depth": "full",
    },
}

_SIMILARITY_FEATURE_COLUMNS = [
    "post_rms",
    "post_centroid_hz",
    "post_bandwidth_hz",
    "post_rolloff_hz",
    "post_low_ratio",
    "post_mid_ratio",
    "post_high_ratio",
    "post_flatness",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _energy_statement(rms_ratio: float) -> str:
    if rms_ratio >= 1.35:
        return "能量明显增强"
    if rms_ratio <= 0.80:
        return "能量明显减弱"
    return "能量变化还比较温和"


def _opening_prompt(labels: list[str], is_major_boundary: bool) -> str:
    if is_major_boundary:
        return "这个点确实有变化。我们先不要急着下结论，可以先看它像不像边界。"
    if "噪声侵入" in labels:
        return "这个点确实有变化。我们可以先看它是不是出现了更粗糙、更噪声化的纹理。"
    if "高频扩展" in labels:
        return "这个点确实有变化。我们可以先看它是不是变得更亮、更靠上。"
    if "能量增强" in labels or "能量减弱" in labels:
        return "这个点确实有变化。我们可以先看它是不是主要发生在能量层面。"
    return "这个点确实有变化。我们可以先从最稳的一步开始，不急着给它定性。"


def _timbre_phrase(labels: list[str], high_ratio: float, flatness: float) -> tuple[str, str]:
    if "噪声侵入" in labels or flatness >= 0.22:
        return "更粗糙、更摩擦化", "目前更像纹理变得更粗糙了一些。"
    if "高频扩展" in labels or high_ratio >= 0.28:
        return "更亮、更靠上", "目前更像高频边缘被往上拉开了一些。"
    return "音色轮廓有变化", "目前更像音色轮廓发生了变化。"


def _priority_meta(priority: str) -> dict[str, str]:
    return _INTERACTION_PRIORITY_META.get(priority, _INTERACTION_PRIORITY_META["medium"])


def annotate_event_interaction_levels(event_table: pd.DataFrame) -> pd.DataFrame:
    annotated = event_table.copy()
    if annotated.empty:
        annotated["interaction_score"] = pd.Series(dtype="float64")
        annotated["interaction_priority"] = pd.Series(dtype="object")
        annotated["interaction_priority_label"] = pd.Series(dtype="object")
        annotated["interaction_depth"] = pd.Series(dtype="object")
        return annotated

    strengths = annotated["strength"].astype(float).fillna(0.0)
    prominences = annotated["prominence"].astype(float).fillna(0.0)
    strength_rank = strengths.rank(method="average", pct=True)
    prominence_rank = prominences.rank(method="average", pct=True)
    label_source = annotated["effective_labels"] if "effective_labels" in annotated.columns else annotated["auto_labels"]
    label_bonus = label_source.astype(str).apply(
        lambda value: 0.12 if CANDIDATE_BOUNDARY_LABEL in value else 0.06 if any(token in value for token in ["高频扩展", "能量增强", "能量减弱"]) else 0.0
    )
    major_bonus = annotated["is_major_boundary"].astype(bool).astype(float) * 0.18
    interaction_score = 0.48 * strength_rank + 0.34 * prominence_rank + label_bonus + major_bonus
    annotated["interaction_score"] = interaction_score.round(4)

    priorities: list[str] = []
    if len(annotated) == 1:
        priorities = ["strong"]
    else:
        score_values = interaction_score.to_numpy(dtype=float)
        strong_cut = max(float(np.quantile(score_values, 0.70)), 0.72)
        weak_cut = min(float(np.quantile(score_values, 0.35)), 0.42)
        sorted_index = list(annotated.sort_values("interaction_score", ascending=False).index)
        top_count = max(1, int(round(len(annotated) * 0.25)))
        strong_indices = set(sorted_index[:top_count])

        for row_index, row in annotated.iterrows():
            labels = str(label_source.loc[row_index])
            score = float(row["interaction_score"])
            is_major = bool(row["is_major_boundary"])
            if is_major or row_index in strong_indices or score >= strong_cut:
                priorities.append("strong")
            elif score <= weak_cut and CANDIDATE_BOUNDARY_LABEL not in labels and not is_major:
                priorities.append("weak")
            else:
                priorities.append("medium")

    annotated["interaction_priority"] = priorities
    annotated["interaction_priority_label"] = annotated["interaction_priority"].map(
        lambda value: _priority_meta(str(value))["label"]
    )
    annotated["interaction_depth"] = annotated["interaction_priority"].map(
        lambda value: _priority_meta(str(value))["depth"]
    )
    return annotated


def annotate_event_similarity_groups(
    event_table: pd.DataFrame,
    threshold: float = 0.90,
) -> pd.DataFrame:
    annotated = event_table.copy()
    if annotated.empty:
        annotated["similarity_group_id"] = pd.Series(dtype="int64")
        annotated["similarity_group_label"] = pd.Series(dtype="object")
        annotated["similarity_group_size"] = pd.Series(dtype="int64")
        annotated["max_similarity_in_group"] = pd.Series(dtype="float64")
        return annotated

    available_columns = [column for column in _SIMILARITY_FEATURE_COLUMNS if column in annotated.columns]
    if not available_columns:
        annotated["similarity_group_id"] = np.arange(1, len(annotated) + 1, dtype=int)
        annotated["similarity_group_label"] = [f"相似组 {index}" for index in range(1, len(annotated) + 1)]
        annotated["similarity_group_size"] = 1
        annotated["max_similarity_in_group"] = 1.0
        return annotated

    feature_frame = annotated[available_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    feature_values = feature_frame.to_numpy(dtype=np.float64)
    means = np.mean(feature_values, axis=0)
    stds = np.std(feature_values, axis=0)
    safe_stds = np.where(stds < 1e-8, 1.0, stds)
    normalized = (feature_values - means) / safe_stds
    squared_norms = np.sum(np.square(normalized), axis=1)
    squared_distance = np.maximum(
        squared_norms[:, None]
        + squared_norms[None, :]
        - 2.0 * (normalized @ normalized.T),
        0.0,
    )
    normalized_distance = np.sqrt(squared_distance) / math.sqrt(max(1, normalized.shape[1]))
    similarity_matrix = np.exp(-normalized_distance)

    threshold = float(np.clip(threshold, 0.0, 1.0))
    group_ids = np.full(len(annotated), -1, dtype=int)
    group_index = 0
    for row_index in range(len(annotated)):
        if group_ids[row_index] != -1:
            continue
        group_index += 1
        stack = [row_index]
        group_ids[row_index] = group_index
        while stack:
            current = stack.pop()
            neighbors = np.where(similarity_matrix[current] >= threshold)[0]
            for neighbor in neighbors:
                if group_ids[neighbor] == -1:
                    group_ids[neighbor] = group_index
                    stack.append(int(neighbor))

    group_sizes = pd.Series(group_ids).map(pd.Series(group_ids).value_counts()).astype(int).to_numpy()
    max_similarity_in_group: list[float] = []
    for row_index, group_id in enumerate(group_ids):
        member_indices = np.where(group_ids == group_id)[0]
        if member_indices.size <= 1:
            max_similarity_in_group.append(1.0)
            continue
        member_scores = similarity_matrix[row_index, member_indices]
        peer_scores = member_scores[member_indices != row_index]
        max_similarity_in_group.append(float(np.max(peer_scores)) if peer_scores.size else 1.0)

    annotated["similarity_group_id"] = group_ids.astype(int)
    annotated["similarity_group_label"] = [f"相似组 {group_id}" for group_id in group_ids]
    annotated["similarity_group_size"] = group_sizes
    annotated["max_similarity_in_group"] = np.round(np.asarray(max_similarity_in_group, dtype=np.float64), 4)
    return annotated


def _build_primary_actions(
    labels: list[str],
    is_major_boundary: bool,
    interaction_priority: str,
) -> list[dict[str, object]]:
    if interaction_priority == "weak":
        return [
            {
                "key": "light_overview",
                "label": "先给我一个轻量观察",
                "secondary_actions": [
                    {"key": "overview_observation", "label": "先说最明显的一点"},
                ],
            },
            {
                "key": "light_state",
                "label": "只看前后有没有变",
                "secondary_actions": [
                    {"key": "state_diff", "label": "先看前后有没有明显不一样"},
                ],
            },
        ]

    if interaction_priority == "medium":
        follow_up_actions = [
            {"key": "state_diff", "label": "先看前后有没有明显不一样"},
            {"key": "energy", "label": "再看能量是不是变了"},
        ]
        if any(label in labels for label in ["噪声侵入", "音色突变", "高频扩展"]):
            follow_up_actions[1] = {"key": "timbre", "label": "再看音色是不是变了"}
        if is_major_boundary or CANDIDATE_BOUNDARY_LABEL in labels:
            return [
                {
                    "key": "overview",
                    "label": "先给我一个初步观察",
                    "secondary_actions": [
                        {"key": "overview_observation", "label": "先用一句话说说"},
                        {"key": "overview_evidence", "label": "告诉我最主要依据"},
                    ],
                },
                {
                    "key": "boundary_focus",
                    "label": "看它是不是边界",
                    "secondary_actions": [
                        {"key": "boundary", "label": "先看它像不像边界"},
                        {"key": "sustain", "label": "看这个变化有没有持续下去"},
                    ],
                },
            ]
        return [
            {
                "key": "overview",
                "label": "先给我一个初步观察",
                "secondary_actions": [
                    {"key": "overview_observation", "label": "先用一句话说说"},
                    {"key": "overview_evidence", "label": "告诉我最主要依据"},
                ],
            },
            {
                "key": "state_change",
                "label": "看前后有没有变",
                "secondary_actions": follow_up_actions,
            },
        ]

    actions = [
        {
            "key": "overview",
            "label": "先给我一个初步观察",
            "secondary_actions": [
                {"key": "overview_observation", "label": "先用一句话说说"},
                {"key": "overview_evidence", "label": "告诉我最主要依据"},
                {"key": "overview_next", "label": "我接下来先看什么"},
            ],
        },
        {
            "key": "state_change",
            "label": "看前后有没有变",
            "secondary_actions": [
                {"key": "state_diff", "label": "先看前后有没有明显不一样"},
                {"key": "energy", "label": "再看能量是不是变了"},
                {"key": "timbre", "label": "再看音色是不是变了"},
            ],
        },
    ]

    if is_major_boundary or CANDIDATE_BOUNDARY_LABEL in labels:
        actions.append(
            {
                "key": "boundary_focus",
                "label": "看它是不是边界",
                "secondary_actions": [
                    {"key": "boundary", "label": "先看它像不像边界"},
                    {"key": "sustain", "label": "看这个变化有没有持续下去"},
                    {"key": "state_diff", "label": "先看前后有没有明显不一样"},
                ],
            }
        )
    elif "能量增强" in labels or "能量减弱" in labels:
        actions.append(
            {
                "key": "energy_focus",
                "label": "看能量有没有明显变化",
                "secondary_actions": [
                    {"key": "energy", "label": "先看能量是不是变了"},
                    {"key": "state_diff", "label": "再看前后有没有明显不一样"},
                    {"key": "sustain", "label": "看这个变化有没有持续下去"},
                ],
            }
        )
    elif any(label in labels for label in ["噪声侵入", "音色突变", "高频扩展"]):
        actions.append(
            {
                "key": "timbre_focus",
                "label": "看音色是不是变了",
                "secondary_actions": [
                    {"key": "timbre", "label": "先看音色是不是变了"},
                    {"key": "state_diff", "label": "再看前后有没有明显不一样"},
                    {"key": "sustain", "label": "看这个变化有没有持续下去"},
                ],
            }
        )
    else:
        actions.append(
            {
                "key": "sustain_focus",
                "label": "看这个变化有没有持续下去",
                "secondary_actions": [
                    {"key": "sustain", "label": "先看这个变化有没有持续下去"},
                    {"key": "state_diff", "label": "再看前后有没有明显不一样"},
                    {"key": "boundary", "label": "再看它像不像边界"},
                ],
            }
        )

    return actions


def _build_path_results(
    *,
    time_label: str,
    labels: list[str],
    label_text: str,
    descriptor: str,
    channel_bias: str,
    dominant_freqs: str,
    strength: float,
    prominence: float,
    rms_ratio: float,
    centroid_hz: float,
    rolloff_hz: float,
    high_ratio: float,
    flatness: float,
    is_major_boundary: bool,
) -> dict[str, dict[str, object]]:
    timbre_label, timbre_hint = _timbre_phrase(labels, high_ratio, flatness)
    boundary_line = "它已经被系统列成候选段落边界。" if is_major_boundary else "它目前还没有被系统列成候选段落边界。"
    descriptor_line = descriptor or "当前规则摘要没有给出额外提示。"

    return {
        "overview_observation": {
            "label": "先用一句话说说",
            "current_observation": f"先看整体，{time_label} 附近确实出现了一个候选变化点，但现在还不急着把它说成最终结论。",
            "main_evidence": [
                f"系统在这里抓到了峰值：强度 {strength:.3f}，显著度 {prominence:.3f}。",
                f"自动标签目前更靠近“{label_text}”。",
            ],
            "next_step": "如果你想走最稳的一步，下一步先看前后有没有明显换状态。",
            "tentative": f"目前可以先把它记成“{time_label} 附近的候选变化点”。",
            "uncertainty": "现在只能说明“这里有变化”，还不能仅凭这一层就判断它属于哪一级结构。",
            "detail_draft": (
                f"在 {time_label} 附近，系统检测到一次可继续验证的候选变化。"
                f"目前自动标签更接近“{label_text}”，但其结构层级仍需继续核实。"
            ),
            "more_paths": ["看前后有没有变", "看这个变化有没有持续下去"],
        },
        "overview_evidence": {
            "label": "告诉我最主要依据",
            "current_observation": "如果只抓最主要的依据，目前最重要的是：这个点前后状态有切换迹象，而且系统把它抓成了明显峰值。",
            "main_evidence": [
                f"峰值强度 {strength:.3f}、显著度 {prominence:.3f}。",
                f"规则摘要提示：{descriptor_line}",
            ],
            "next_step": "下一步最值得看的，是这种变化到底只是一瞬间，还是会带着后面的状态一起走。",
            "tentative": "目前这更像一组支持“这里值得重点听”的证据，而不是直接成立的结论。",
            "uncertainty": "单看峰值和规则摘要，还不能替代你的局部试听与全曲判断。",
            "detail_draft": (
                f"从自动检测结果看，{time_label} 这一点具有较高的重点复核价值；"
                "但其最终意义仍需结合听觉与结构关系继续判断。"
            ),
            "more_paths": ["看它是不是边界", "看前后有没有变"],
        },
        "overview_next": {
            "label": "我接下来先看什么",
            "current_observation": "如果你现在只想走下一步，我建议先看前后有没有明显不一样，再决定要不要往边界或能量方向继续。",
            "main_evidence": [
                "这一步最稳，因为它先回答“到底有没有换状态”。",
                "只有先确认前后真的不一样，后面再谈边界、过程转折才更稳。 ",
            ],
            "next_step": "先点“看前后有没有变”下面的按钮，再回到局部试听和频谱一起看。",
            "tentative": "目前还不需要给它命名，先把证据顺序排好就够了。",
            "uncertainty": "如果一开始就急着命名，很容易把局部扰动误判成结构变化。",
            "detail_draft": (
                f"对 {time_label} 这一点，更合适的做法是先确认前后状态差异，"
                "再逐步推进到更高层级的结构解释。"
            ),
            "more_paths": ["看前后有没有变", "看它是不是边界"],
        },
        "state_diff": {
            "label": "先看前后有没有明显不一样",
            "current_observation": f"从机器检测的角度看，{time_label} 前后确实有状态差异，这也是它会被标出来的直接原因。",
            "main_evidence": [
                f"系统在这里检测到峰值，强度 {strength:.3f}、显著度 {prominence:.3f}。",
                f"自动标签目前更靠近“{label_text}”。",
            ],
            "next_step": "如果你听感上也觉得前后像换了一次状态，下一步就可以去看这个变化更偏能量、音色，还是结构层面的边界。",
            "tentative": "目前可以先把它记成“前后状态确实有切换的候选点”。",
            "uncertainty": "这里只能回答“有没有变”，还不足以直接回答“变到了什么程度”。",
            "detail_draft": (
                f"在 {time_label} 附近，作品前后音响状态出现了可被继续验证的差异；"
                "目前更适合先把它视为状态切换候选，而不是直接给出最终结构判断。"
            ),
            "more_paths": ["再看能量是不是变了", "看这个变化有没有持续下去"],
        },
        "energy": {
            "label": "再看能量是不是变了",
            "current_observation": f"从能量角度看，这里事件后的整体强度大约是前面的 {rms_ratio:.2f} 倍，{_energy_statement(rms_ratio)}。",
            "main_evidence": [
                f"事件后 RMS 与事件前的比值约为 {rms_ratio:.2f}。",
                f"当前规则摘要是：{descriptor_line}",
            ],
            "next_step": "如果你在波形和听感上都感到这里更“顶”或更“收”，下一步再看这种变化是不是持续了下去。",
            "tentative": "目前更适合把它理解成一个能量层面的候选变化。",
            "uncertainty": "单看能量还不够，它可能是结构转折，也可能只是局部材料一下子更密或更弱。",
            "detail_draft": (
                f"从能量变化看，{time_label} 附近的音响状态相较前一时段出现了可继续验证的强弱变化；"
                "其结构含义仍需结合持续性与音色变化继续判断。"
            ),
            "more_paths": ["看这个变化有没有持续下去", "再看音色是不是变了"],
        },
        "timbre": {
            "label": "再看音色是不是变了",
            "current_observation": f"从音色线索看，这里更像变得{timbre_label}。{timbre_hint}",
            "main_evidence": [
                f"事件后谱质心约为 {centroid_hz:.0f} Hz，滚降频率约为 {rolloff_hz:.0f} Hz。",
                f"事件后高频占比约为 {high_ratio:.1%}，谱平坦度约为 {flatness:.3f}。",
            ],
            "next_step": "如果你在局部听感里也觉得它更亮、更粗糙或更挤，下一步可以继续看这种变化有没有持续下去。",
            "tentative": "目前更像是一次音色层面的候选变化。",
            "uncertainty": "这里只能说明“音色好像变了”，还不够直接支持它一定是段落边界。",
            "detail_draft": (
                f"在 {time_label} 附近，音响的音色轮廓呈现出可继续验证的变化迹象；"
                "目前更适合先把它当作音色状态转换候选来处理。"
            ),
            "more_paths": ["看这个变化有没有持续下去", "看它是不是边界"],
        },
        "boundary": {
            "label": "先看它像不像边界",
            "current_observation": f"如果把问题收紧到“它像不像边界”，目前的机器证据给出的回答是：{boundary_line}",
            "main_evidence": [
                boundary_line,
                f"声道倾向是“{channel_bias}”，规则摘要提示：{descriptor_line}",
            ],
            "next_step": "下一步最好回到总览和局部试听，一起听它前后 5 到 10 秒的状态有没有真的换过去。",
            "tentative": "目前最多只能把它记成“边界候选”或“暂不成立的边界候选”。",
            "uncertainty": "边界判断一定要看更长时间范围，不能只盯着这一个点本身。",
            "detail_draft": (
                f"从当前自动结果看，{time_label} 这一点可被暂时视为边界候选；"
                "但其是否真正构成段落层级的边界，仍需回到全曲继续验证。"
            ),
            "more_paths": ["看这个变化有没有持续下去", "先看前后有没有明显不一样"],
        },
        "sustain": {
            "label": "看这个变化有没有持续下去",
            "current_observation": "如果你想判断它是不是更高层级的变化，最关键的问题不是“这一瞬间有没有变”，而是“这个变化有没有持续下去”。",
            "main_evidence": [
                f"系统当前给出的整体提示是：{descriptor_line}",
                f"自动标签目前更靠近“{label_text}”。",
            ],
            "next_step": "下一步把时间窗放长一点，回到总览看它后面几秒是不是进入了新的状态。",
            "tentative": "目前这一步更像是在验证结构层级，而不是马上给结构名称。",
            "uncertainty": "如果后面很快又回到原来的状态，那它更可能只是局部扰动，不一定能支撑更大的结构判断。",
            "detail_draft": (
                f"关于 {time_label} 这一点，更关键的验证问题是其变化是否具有持续性；"
                "这一层判断将直接影响它能否被提升到更高的结构解释。"
            ),
            "more_paths": ["先看它像不像边界", "再看能量是不是变了"],
        },
    }


def build_event_assistant_payload(
    event_row: Mapping[str, Any],
    effective_label_text: str | None = None,
) -> dict[str, Any]:
    labels = split_label_text(effective_label_text or str(event_row.get("auto_labels", "")))
    if not labels:
        labels = ["新事件出现"]

    pre_rms = _safe_float(event_row.get("pre_rms"), 1e-8)
    post_rms = _safe_float(event_row.get("post_rms"))
    rms_ratio = post_rms / max(pre_rms, 1e-8)
    strength = _safe_float(event_row.get("strength"))
    prominence = _safe_float(event_row.get("prominence"))
    centroid_hz = _safe_float(event_row.get("post_centroid_hz"))
    rolloff_hz = _safe_float(event_row.get("post_rolloff_hz"))
    high_ratio = _safe_float(event_row.get("post_high_ratio"))
    flatness = _safe_float(event_row.get("post_flatness"))
    channel_bias = str(event_row.get("channel_bias", "平衡"))
    time_label = str(event_row.get("time_label", "--"))
    dominant_freqs = str(event_row.get("dominant_freqs", ""))
    descriptor = str(event_row.get("descriptor", ""))
    is_major_boundary = bool(event_row.get("is_major_boundary", False))
    interaction_priority = str(event_row.get("interaction_priority", "strong" if is_major_boundary else "medium"))
    priority_meta = _priority_meta(interaction_priority)

    role_text = CANDIDATE_BOUNDARY_LABEL if is_major_boundary else "局部变化候选"
    label_text = "、".join(labels)
    opening_prompt = _opening_prompt(labels, is_major_boundary)
    primary_actions = _build_primary_actions(labels, is_major_boundary, interaction_priority)
    path_results = _build_path_results(
        time_label=time_label,
        labels=labels,
        label_text=label_text,
        descriptor=descriptor,
        channel_bias=channel_bias,
        dominant_freqs=dominant_freqs,
        strength=strength,
        prominence=prominence,
        rms_ratio=rms_ratio,
        centroid_hz=centroid_hz,
        rolloff_hz=rolloff_hz,
        high_ratio=high_ratio,
        flatness=flatness,
        is_major_boundary=is_major_boundary,
    )

    all_secondary_labels = [
        secondary["label"]
        for action in primary_actions
        for secondary in action["secondary_actions"]
    ]

    evidence_points = [
        f"{time_label} 的事件强度约为 {strength:.3f}，显著度约为 {prominence:.3f}。",
        f"事件后 RMS 约为事件前的 {rms_ratio:.2f} 倍，说明 {_energy_statement(rms_ratio)}。",
        f"事件后谱质心约为 {centroid_hz:.0f} Hz，滚降频率约为 {rolloff_hz:.0f} Hz。",
        f"事件后高频占比约为 {high_ratio:.1%}，谱平坦度约为 {flatness:.3f}，声道倾向为“{channel_bias}”。",
    ]
    if dominant_freqs:
        evidence_points.append(f"自动提取到的主导频率包括：{dominant_freqs}。")
    if descriptor:
        evidence_points.append(f"当前规则摘要提示：{descriptor}。")

    draft = (
        f"在 {time_label} 附近，系统检测到一个可继续验证的“{role_text}”。"
        f"目前自动标签更接近“{label_text}”，但其最终结构意义仍需结合局部试听与全曲关系继续核实。"
    )

    result_placeholder = "先从上面选一个方向，我再一步步展开，不会一开始把所有东西都堆给你。"
    secondary_placeholder = "先点一个最核心的问题，我再把下一步的按钮展开给你。"
    collaboration_note = "我会按步骤陪你看：先选方向，再选更具体的问题，最后再决定要不要展开更多细节。"
    if interaction_priority == "weak":
        collaboration_note = "这个点机器听到了变化，但它目前更像较小变化，所以我会先给你一个轻量回应，不默认展开完整讨论。"
        result_placeholder = "这是一个较小变化。你可以先点一个最轻的入口，我再给你一句观察，不会默认展开完整助手流程。"
        secondary_placeholder = "这个点先保持轻量处理。点一下上面的按钮，我就给你最必要的一层观察。"
    elif interaction_priority == "medium":
        collaboration_note = "这个点值得做一步跟进，但暂时还不需要进入最重的讨论深度。"

    return {
        "assistant_invite": "AI 小助手",
        "assistant_invite_hint": str(priority_meta["assistant_hint"]),
        "time_label": time_label,
        "labels": labels,
        "label_text": label_text,
        "question": opening_prompt,
        "opening_prompt": opening_prompt,
        "primary_actions": primary_actions,
        "result_paths": path_results,
        "suggested_checks": all_secondary_labels,
        "evidence_points": evidence_points,
        "draft": draft,
        "role_text": role_text,
        "interaction_priority": interaction_priority,
        "interaction_priority_label": str(priority_meta["label"]),
        "interaction_depth": str(priority_meta["depth"]),
        "collaboration_note": collaboration_note,
        "settings_title": "AI 设置",
        "result_placeholder": result_placeholder,
        "secondary_placeholder": secondary_placeholder,
        "preference_focus_options": _FOCUS_PREFERENCE_OPTIONS,
        "preference_term_options": _TERM_PREFERENCE_OPTIONS,
    }

def build_assistant_overlay_component(
    active_event_id: int,
    payload: Mapping[str, Any],
) -> str:
    assistant_data = {
        "active_event_id": int(active_event_id),
        "assistant_invite": str(payload.get("assistant_invite", "AI 小助手")),
        "assistant_invite_hint": str(payload.get("assistant_invite_hint", "可协助验证此点")),
        "time_label": str(payload["time_label"]),
        "role_text": str(payload["role_text"]),
        "interaction_priority": str(payload.get("interaction_priority", "medium")),
        "interaction_priority_label": str(payload.get("interaction_priority_label", "一般变化")),
        "interaction_depth": str(payload.get("interaction_depth", "guided")),
        "labels": [str(label) for label in payload["labels"]],
        "opening_prompt": str(payload.get("opening_prompt") or payload.get("question") or ""),
        "primary_actions": payload.get("primary_actions", []),
        "result_paths": payload.get("result_paths", {}),
        "collaboration_note": str(payload.get("collaboration_note", "")),
        "settings_title": str(payload.get("settings_title", "AI 设置")),
        "result_placeholder": str(payload.get("result_placeholder", "")),
        "secondary_placeholder": str(payload.get("secondary_placeholder", "")),
        "preference_focus_options": payload.get("preference_focus_options", _FOCUS_PREFERENCE_OPTIONS),
        "preference_term_options": payload.get("preference_term_options", _TERM_PREFERENCE_OPTIONS),
    }
    data_json = json.dumps(assistant_data, ensure_ascii=False)

    html = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <style>
      html, body {
        margin: 0;
        padding: 0;
        width: 0;
        height: 0;
        overflow: hidden;
        background: transparent;
      }
    </style>
  </head>
  <body>
    <script>
      (() => {
        const assistantData = __ASSISTANT_DATA__;
        const parentDoc = window.parent.document;
        const rootId = "event-ai-overlay-root";
        const styleId = "event-ai-overlay-style";
        const storageKey = "event_ai_overlay_position_v3";
        const openKey = "event_ai_overlay_open_v3";
        const prefsKey = "event_ai_overlay_prefs_v2";
        const eventStateKey = `event_ai_overlay_state_${assistantData.active_event_id}_v2`;

        if (!parentDoc.getElementById(styleId)) {
          const style = parentDoc.createElement("style");
          style.id = styleId;
          style.textContent = `
            #${rootId} {
              position: fixed;
              right: 20px;
              top: 88px;
              z-index: 99999;
              width: min(420px, calc(100vw - 24px));
              max-width: min(420px, calc(100vw - 24px));
              font-family: "Helvetica Neue", Arial, sans-serif;
              pointer-events: none;
            }
            #${rootId} * {
              box-sizing: border-box;
            }
            #${rootId} .assistant-launcher,
            #${rootId} .assistant-panel {
              pointer-events: auto;
            }
            #${rootId} .assistant-launcher {
              display: inline-flex;
              flex-direction: column;
              align-items: flex-start;
              gap: 0.2rem;
              margin-left: auto;
              border: 1px solid #c3d4ea;
              border-radius: 999px;
              background: linear-gradient(180deg, #ffffff 0%, #edf4fb 100%);
              color: #102a43;
              font-weight: 700;
              padding: 0.65rem 0.95rem;
              box-shadow: 0 12px 24px rgba(16, 42, 67, 0.14);
              cursor: grab;
              user-select: none;
            }
            #${rootId} .assistant-launcher-title {
              font-size: 0.88rem;
              line-height: 1.1;
            }
            #${rootId} .assistant-launcher-hint {
              font-size: 0.74rem;
              font-weight: 500;
              color: #486581;
              line-height: 1.2;
            }
            #${rootId} .assistant-panel {
              display: none;
              margin-top: 0.55rem;
              border: 1px solid #d8e0eb;
              border-radius: 18px;
              background: linear-gradient(180deg, #f8fbff 0%, #f4f7fb 100%);
              box-shadow: 0 16px 36px rgba(27, 53, 87, 0.14);
              overflow: hidden;
            }
            #${rootId}.open .assistant-panel {
              display: block;
            }
            #${rootId} .assistant-header {
              display: flex;
              justify-content: space-between;
              align-items: center;
              gap: 0.8rem;
              padding: 0.85rem 1rem 0.75rem 1rem;
              border-bottom: 1px solid #d8e0eb;
              background: rgba(255,255,255,0.7);
              cursor: grab;
              user-select: none;
            }
            #${rootId} .assistant-header-main {
              min-width: 0;
            }
            #${rootId} .assistant-kicker {
              font-size: 0.76rem;
              font-weight: 700;
              letter-spacing: 0.02em;
              color: #486581;
              text-transform: uppercase;
            }
            #${rootId} .assistant-title {
              font-size: 1.02rem;
              font-weight: 700;
              color: #102a43;
              margin-top: 0.16rem;
            }
            #${rootId} .assistant-subtitle {
              font-size: 0.84rem;
              color: #627d98;
              margin-top: 0.24rem;
            }
            #${rootId} .assistant-header-actions {
              display: inline-flex;
              align-items: center;
              gap: 0.45rem;
              flex-shrink: 0;
            }
            #${rootId} .assistant-close {
              border: 1px solid #c3d4ea;
              border-radius: 999px;
              background: #fff;
              color: #334e68;
              font-size: 0.8rem;
              padding: 0.3rem 0.65rem;
              cursor: pointer;
              white-space: nowrap;
            }
            #${rootId} .assistant-body {
              max-height: min(76vh, 720px);
              overflow-y: auto;
              padding: 0.95rem 1rem 1rem 1rem;
            }
            #${rootId} .assistant-chip-row {
              display: flex;
              flex-wrap: wrap;
              gap: 0.35rem;
              margin-top: 0.2rem;
            }
            #${rootId} .assistant-chip {
              display: inline-flex;
              align-items: center;
              border-radius: 999px;
              background: #e7eef8;
              color: #243b53;
              padding: 0.18rem 0.58rem;
              font-size: 0.78rem;
            }
            #${rootId} .assistant-section {
              margin-top: 1rem;
            }
            #${rootId} .assistant-section-title {
              font-size: 0.86rem;
              font-weight: 700;
              color: #243b53;
              margin-bottom: 0.38rem;
            }
            #${rootId} .assistant-copy {
              font-size: 0.92rem;
              line-height: 1.6;
              color: #102a43;
              margin: 0;
            }
            #${rootId} .assistant-meta {
              font-size: 0.84rem;
              line-height: 1.55;
              color: #52667a;
              margin: 0.35rem 0 0 0;
            }
            #${rootId} .assistant-option-grid {
              display: flex;
              flex-wrap: wrap;
              gap: 0.45rem;
              margin-top: 0.6rem;
            }
            #${rootId} .assistant-option {
              border: 1px solid #c6d3e1;
              border-radius: 12px;
              background: #fff;
              color: #243b53;
              font-size: 0.83rem;
              padding: 0.48rem 0.75rem;
              cursor: pointer;
              line-height: 1.3;
            }
            #${rootId} .assistant-option.active {
              border-color: #2680c2;
              background: #e7f2fb;
              color: #0f609b;
              font-weight: 700;
            }
            #${rootId} .assistant-note {
              background: #ffffff;
              border: 1px solid #d9e2ec;
              border-radius: 12px;
              padding: 0.75rem 0.8rem;
              font-size: 0.9rem;
              line-height: 1.65;
              color: #102a43;
            }
            #${rootId} .assistant-placeholder {
              border: 1px dashed #c5d2e0;
              border-radius: 12px;
              padding: 0.85rem 0.9rem;
              color: #486581;
              background: rgba(255,255,255,0.7);
              font-size: 0.88rem;
              line-height: 1.55;
            }
            #${rootId} .assistant-subsection-title {
              font-size: 0.82rem;
              font-weight: 700;
              color: #486581;
              margin: 0.75rem 0 0.35rem 0;
            }
            #${rootId} ul {
              margin: 0.25rem 0 0 1.15rem;
              padding: 0;
            }
            #${rootId} li {
              margin: 0.28rem 0;
              color: #334e68;
              line-height: 1.52;
            }
            #${rootId} details {
              border: 1px solid #d9e2ec;
              border-radius: 12px;
              background: #ffffff;
              margin-top: 0.8rem;
              overflow: hidden;
            }
            #${rootId} details summary {
              cursor: pointer;
              padding: 0.72rem 0.8rem;
              font-size: 0.86rem;
              font-weight: 700;
              color: #243b53;
              list-style: none;
            }
            #${rootId} details summary::-webkit-details-marker {
              display: none;
            }
            #${rootId} .assistant-detail-body {
              padding: 0 0.8rem 0.85rem 0.8rem;
            }
            #${rootId} .assistant-detail-note {
              margin-top: 0.45rem;
              font-size: 0.88rem;
              line-height: 1.6;
              color: #334e68;
            }
            #${rootId} .assistant-pref-note {
              margin-top: 0.5rem;
              color: #627d98;
              font-size: 0.8rem;
              line-height: 1.5;
            }
            @media (max-width: 980px) {
              #${rootId} {
                right: 12px;
                left: 12px;
                width: auto;
                max-width: none;
                top: 72px;
              }
            }
          `;
          parentDoc.head.appendChild(style);
        }

        let root = parentDoc.getElementById(rootId);
        if (!root) {
          root = parentDoc.createElement("div");
          root.id = rootId;
          parentDoc.body.appendChild(root);
        }

        const escapeHtml = (value) => String(value)
          .replaceAll("&", "&amp;")
          .replaceAll("<", "&lt;")
          .replaceAll(">", "&gt;")
          .replaceAll('"', "&quot;")
          .replaceAll("'", "&#39;");

        const loadJson = (key, fallbackValue) => {
          try {
            const raw = window.parent.localStorage.getItem(key);
            if (!raw) return fallbackValue;
            return JSON.parse(raw);
          } catch (error) {
            return fallbackValue;
          }
        };

        let prefs = loadJson(prefsKey, {
          focus: "sustained",
          terms: [],
        });
        let eventState = loadJson(eventStateKey, {
          selectedPrimary: "",
          selectedPath: "",
        });

        const chipsHtml = assistantData.labels
          .map((label) => `<span class="assistant-chip">${escapeHtml(label)}</span>`)
          .join("");

        root.innerHTML = `
          <div class="assistant-launcher" id="event-ai-launcher" title="点击打开，拖动可移动">
            <div class="assistant-launcher-title">${escapeHtml(assistantData.assistant_invite)}</div>
            <div class="assistant-launcher-hint">${escapeHtml(assistantData.assistant_invite_hint)}</div>
          </div>
          <div class="assistant-panel" id="event-ai-panel">
            <div class="assistant-header" id="event-ai-header">
              <div class="assistant-header-main">
                <div class="assistant-kicker">Collaborative Analysis</div>
                <div class="assistant-title">当前联动事件：事件 ${assistantData.active_event_id}</div>
                <div class="assistant-subtitle">时间 ${escapeHtml(assistantData.time_label)} · 系统候选 ${escapeHtml(assistantData.role_text)} · 交互层级 ${escapeHtml(assistantData.interaction_priority_label)}</div>
              </div>
              <div class="assistant-header-actions">
                <button type="button" class="assistant-close" id="event-ai-reset">重置位置</button>
                <button type="button" class="assistant-close" id="event-ai-close">收起</button>
              </div>
            </div>
            <div class="assistant-body">
              <div class="assistant-chip-row">${chipsHtml}</div>

              <div class="assistant-section">
                <div class="assistant-section-title">先从最核心的问题开始</div>
                <p class="assistant-copy">${escapeHtml(assistantData.opening_prompt)}</p>
                <p class="assistant-meta">${escapeHtml(assistantData.collaboration_note)}</p>
                <div class="assistant-option-grid" id="event-ai-primary-grid"></div>
              </div>

              <div class="assistant-section">
                <div class="assistant-section-title">下一步</div>
                <div id="event-ai-secondary-wrap"></div>
              </div>

              <div class="assistant-section">
                <div class="assistant-section-title">当前结果</div>
                <div id="event-ai-result"></div>
              </div>

              <details id="event-ai-settings">
                <summary>${escapeHtml(assistantData.settings_title)}</summary>
                <div class="assistant-detail-body">
                  <div class="assistant-subsection-title">你更希望 AI 更看重什么</div>
                  <div class="assistant-option-grid" id="event-ai-focus-grid"></div>
                  <div class="assistant-subsection-title">你更常用哪些术语</div>
                  <div class="assistant-option-grid" id="event-ai-term-grid"></div>
                  <div class="assistant-pref-note" id="event-ai-pref-note"></div>
                </div>
              </details>
            </div>
          </div>
        `;

        const launcher = parentDoc.getElementById("event-ai-launcher");
        const resetButton = parentDoc.getElementById("event-ai-reset");
        const closeButton = parentDoc.getElementById("event-ai-close");
        const dragHandle = parentDoc.getElementById("event-ai-header");
        const primaryGrid = parentDoc.getElementById("event-ai-primary-grid");
        const secondaryWrap = parentDoc.getElementById("event-ai-secondary-wrap");
        const resultEl = parentDoc.getElementById("event-ai-result");
        const focusGrid = parentDoc.getElementById("event-ai-focus-grid");
        const termGrid = parentDoc.getElementById("event-ai-term-grid");
        const prefNote = parentDoc.getElementById("event-ai-pref-note");

        const persistPrefs = () => {
          window.parent.localStorage.setItem(prefsKey, JSON.stringify(prefs));
        };

        const persistEventState = () => {
          window.parent.localStorage.setItem(eventStateKey, JSON.stringify(eventState));
        };

        const renderButtonGroup = (container, items, activeValues, dataKey, onClick) => {
          const activeSet = new Set(Array.isArray(activeValues) ? activeValues : [activeValues]);
          container.innerHTML = items
            .map((item) => `
              <button
                type="button"
                class="assistant-option ${activeSet.has(item.key) ? "active" : ""}"
                data-${dataKey}="${escapeHtml(item.key)}"
              >
                ${escapeHtml(item.label)}
              </button>
            `)
            .join("");
          container.querySelectorAll(`[data-${dataKey}]`).forEach((button) => {
            button.addEventListener("click", () => {
              const value = button.dataset[dataKey];
              onClick(value);
            });
          });
        };

        const findPrimary = (key) => assistantData.primary_actions.find((item) => item.key === key) || null;

        const renderPreferences = () => {
          renderButtonGroup(
            focusGrid,
            assistantData.preference_focus_options,
            prefs.focus,
            "focus",
            (value) => {
              prefs.focus = value;
              persistPrefs();
              renderPreferences();
              renderResult();
            }
          );

          renderButtonGroup(
            termGrid,
            assistantData.preference_term_options.map((item) => ({ key: item, label: item })),
            prefs.terms,
            "term",
            (value) => {
              if (prefs.terms.includes(value)) {
                prefs.terms = prefs.terms.filter((item) => item !== value);
              } else {
                prefs.terms = [...prefs.terms, value];
              }
              persistPrefs();
              renderPreferences();
              renderResult();
            }
          );

          const focusLabel = assistantData.preference_focus_options.find((item) => item.key === prefs.focus)?.label || "更重视持续变化";
          const termText = prefs.terms.length ? prefs.terms.join("、") : "暂未设定";
          prefNote.textContent = `当前 AI 设置：${focusLabel}；常用术语：${termText}。这些偏好默认藏在这里，不会打断主流程。`;
        };

        const renderPrimary = () => {
          renderButtonGroup(
            primaryGrid,
            assistantData.primary_actions,
            eventState.selectedPrimary,
            "primary",
            (value) => {
              const primary = findPrimary(value);
              eventState.selectedPrimary = value;
              const allowedKeys = primary ? primary.secondary_actions.map((item) => item.key) : [];
              if (!allowedKeys.includes(eventState.selectedPath)) {
                eventState.selectedPath = "";
              }
              persistEventState();
              renderSecondary();
              renderResult();
              renderPrimary();
            }
          );
        };

        const renderSecondary = () => {
          const primary = findPrimary(eventState.selectedPrimary);
          if (!primary) {
            secondaryWrap.innerHTML = `<div class="assistant-placeholder">${escapeHtml(assistantData.secondary_placeholder)}</div>`;
            return;
          }

          secondaryWrap.innerHTML = `
            <div class="assistant-note">${escapeHtml(`你刚刚选的是：${primary.label}`)}</div>
            <div class="assistant-option-grid" id="event-ai-secondary-grid"></div>
          `;

          const secondaryGrid = parentDoc.getElementById("event-ai-secondary-grid");
          renderButtonGroup(
            secondaryGrid,
            primary.secondary_actions,
            eventState.selectedPath,
            "path",
            (value) => {
              eventState.selectedPath = value;
              persistEventState();
              renderSecondary();
              renderResult();
            }
          );
        };

        const renderResult = () => {
          const selected = assistantData.result_paths[eventState.selectedPath];
          if (!selected) {
            resultEl.innerHTML = `<div class="assistant-placeholder">${escapeHtml(assistantData.result_placeholder)}</div>`;
            return;
          }

          const evidenceHtml = (selected.main_evidence || [])
            .slice(0, 2)
            .map((item) => `<li>${escapeHtml(item)}</li>`)
            .join("");
          const morePaths = (selected.more_paths || []).length
            ? `<div class="assistant-detail-note">如果你还想继续，可以再看：${escapeHtml(selected.more_paths.join("、"))}。</div>`
            : "";
          const prefLead = prefs.focus === "sustained"
            ? "按当前 AI 设置，我会更先提醒你看变化有没有持续下去。"
            : "按当前 AI 设置，我会更先提醒你留意局部尖峰是不是足够关键。";
          const termLead = prefs.terms.length
            ? `后面如果要写分析，我会优先沿用这些术语：${escapeHtml(prefs.terms.join("、"))}。`
            : "如果你有更常用的术语，可以在 AI 设置里记住。";

          if (assistantData.interaction_depth === "light") {
            resultEl.innerHTML = `
              <div class="assistant-note">${escapeHtml(selected.current_observation)}</div>
              <div class="assistant-subsection-title">主要依据</div>
              <ul>${evidenceHtml}</ul>
              <div class="assistant-subsection-title">下一步</div>
              <div class="assistant-note">${escapeHtml(selected.next_step)}</div>
              <div class="assistant-pref-note">这是一个较小变化，所以我先只给你最必要的一层回应。如果你觉得它比系统判断更重要，可以再打开“更多小变化”继续复核。</div>
            `;
            return;
          }

          if (assistantData.interaction_depth === "guided") {
            resultEl.innerHTML = `
              <div class="assistant-note">${escapeHtml(selected.current_observation)}</div>
              <div class="assistant-subsection-title">主要依据</div>
              <ul>${evidenceHtml}</ul>
              <div class="assistant-subsection-title">下一步</div>
              <div class="assistant-note">${escapeHtml(selected.next_step)}</div>
              <details>
                <summary>再展开一点</summary>
                <div class="assistant-detail-body">
                  <div class="assistant-subsection-title">暂定判断</div>
                  <div class="assistant-detail-note">${escapeHtml(selected.tentative)}</div>
                  <div class="assistant-subsection-title">还不确定的地方</div>
                  <div class="assistant-detail-note">${escapeHtml(selected.uncertainty)}</div>
                  <div class="assistant-pref-note">${prefLead}</div>
                </div>
              </details>
            `;
            return;
          }

          resultEl.innerHTML = `
            <div class="assistant-note">${escapeHtml(selected.current_observation)}</div>
            <div class="assistant-subsection-title">主要依据</div>
            <ul>${evidenceHtml}</ul>
            <div class="assistant-subsection-title">下一步</div>
            <div class="assistant-note">${escapeHtml(selected.next_step)}</div>
            <details>
              <summary>展开更详细的分析</summary>
              <div class="assistant-detail-body">
                <div class="assistant-subsection-title">暂定判断</div>
                <div class="assistant-detail-note">${escapeHtml(selected.tentative)}</div>
                <div class="assistant-subsection-title">还不确定的地方</div>
                <div class="assistant-detail-note">${escapeHtml(selected.uncertainty)}</div>
                <div class="assistant-subsection-title">分析文字草稿（暂定）</div>
                <div class="assistant-detail-note">${escapeHtml(selected.detail_draft)}</div>
                ${morePaths}
                <div class="assistant-pref-note">${prefLead} ${termLead}</div>
              </div>
            </details>
          `;
        };

        const clampPosition = (left, top) => {
          const rect = root.getBoundingClientRect();
          const maxLeft = Math.max(8, window.parent.innerWidth - rect.width - 8);
          const maxTop = Math.max(8, window.parent.innerHeight - rect.height - 8);
          return {
            left: Math.min(Math.max(8, left), maxLeft),
            top: Math.min(Math.max(8, top), maxTop),
          };
        };

        const snapToDefaultPosition = () => {
          root.style.left = "auto";
          root.style.right = "20px";
          root.style.top = "88px";
        };

        const ensureVisible = () => {
          const rect = root.getBoundingClientRect();
          const outOfBounds =
            rect.right < 36 ||
            rect.left > window.parent.innerWidth - 36 ||
            rect.bottom < 36 ||
            rect.top > window.parent.innerHeight - 36;
          if (outOfBounds) {
            snapToDefaultPosition();
          }
        };

        const savedPosition = window.parent.localStorage.getItem(storageKey);
        if (savedPosition) {
          try {
            const pos = JSON.parse(savedPosition);
            if (typeof pos.left === "number" && typeof pos.top === "number") {
              const next = clampPosition(pos.left, pos.top);
              root.style.left = `${next.left}px`;
              root.style.top = `${next.top}px`;
              root.style.right = "auto";
            } else {
              snapToDefaultPosition();
            }
          } catch (error) {
            snapToDefaultPosition();
          }
        } else {
          snapToDefaultPosition();
        }

        const setOpen = (isOpen) => {
          root.classList.toggle("open", isOpen);
          window.parent.localStorage.setItem(openKey, isOpen ? "1" : "0");
        };

        const wasOpen = window.parent.localStorage.getItem(openKey) === "1";
        setOpen(wasOpen);

        renderPreferences();
        renderPrimary();
        renderSecondary();
        renderResult();
        ensureVisible();

        let dragState = null;
        let suppressLauncherClick = false;

        const startDrag = (event) => {
          dragState = {
            startX: event.clientX,
            startY: event.clientY,
            rect: root.getBoundingClientRect(),
            moved: false,
          };
          event.preventDefault();
        };

        launcher.onmousedown = startDrag;
        dragHandle.onmousedown = startDrag;

        launcher.onclick = (event) => {
          if (suppressLauncherClick) {
            suppressLauncherClick = false;
            event.preventDefault();
            return;
          }
          setOpen(!root.classList.contains("open"));
        };

        resetButton.onclick = () => {
          window.parent.localStorage.removeItem(storageKey);
          snapToDefaultPosition();
        };
        closeButton.onclick = () => setOpen(false);

        const onMouseMove = (event) => {
          if (!dragState) return;
          const deltaX = event.clientX - dragState.startX;
          const deltaY = event.clientY - dragState.startY;
          if (Math.abs(deltaX) > 3 || Math.abs(deltaY) > 3) {
            dragState.moved = true;
          }
          const next = clampPosition(dragState.rect.left + deltaX, dragState.rect.top + deltaY);
          root.style.left = `${next.left}px`;
          root.style.top = `${next.top}px`;
          root.style.right = "auto";
        };

        const onMouseUp = () => {
          if (!dragState) return;
          suppressLauncherClick = dragState.moved;
          const rect = root.getBoundingClientRect();
          window.parent.localStorage.setItem(
            storageKey,
            JSON.stringify({ left: rect.left, top: rect.top })
          );
          dragState = null;
        };

        if (window.parent.__eventAiOverlayMouseMove) {
          window.parent.removeEventListener("mousemove", window.parent.__eventAiOverlayMouseMove);
        }
        if (window.parent.__eventAiOverlayMouseUp) {
          window.parent.removeEventListener("mouseup", window.parent.__eventAiOverlayMouseUp);
        }
        window.parent.addEventListener("mousemove", onMouseMove);
        window.parent.addEventListener("mouseup", onMouseUp);
        window.parent.addEventListener("resize", ensureVisible);
        window.parent.__eventAiOverlayMouseMove = onMouseMove;
        window.parent.__eventAiOverlayMouseUp = onMouseUp;
      })();
    </script>
  </body>
</html>
"""
    return html.replace("__ASSISTANT_DATA__", data_json)
