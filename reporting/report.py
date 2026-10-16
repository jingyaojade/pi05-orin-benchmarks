#!/usr/bin/env python3
"""Generate a Chinese Markdown report from baseline benchmark artifacts.

Standard library only. Missing values stay missing; the generator never runs
inference, changes benchmark settings, or estimates unrecorded measurements.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import statistics
import struct
from typing import Any
import zipfile


MISSING = "未记录"
PHASE_FULL = "official_policy_uninstrumented"
PHASE_MODEL = "model_uninstrumented"
PHASE_STAGES = "instrumented_decomposed"
STAGE_LABELS = {
    "model_preprocess": "模型内部预处理",
    "prefix_embedding_total": "完整前缀嵌入",
    "vision_encode": "视觉编码（全部视角）",
    "language_token_embedding": "语言 token 查表",
    "vlm_prefix_kv_prefill": "VLM 前缀 KV 预填充",
    "denoise_iteration": "动作去噪迭代",
    "action_suffix_embedding": "动作与时间后缀嵌入",
    "attention_mask_prepare": "注意力掩码数值转换",
    "action_expert_transformer": "动作专家 Transformer",
    "action_input_projection": "动作输入投影",
    "action_output_projection": "动作输出投影",
    "time_mlp_total": "完整时间 MLP（含两次 SiLU）",
    "time_mlp_in_linear": "时间 MLP 输入线性层",
    "time_mlp_out_linear": "时间 MLP 输出线性层",
}
IMAGE_LABELS = {
    "base_0_rgb": "主相机",
    "left_wrist_0_rgb": "左腕相机",
    "right_wrist_0_rgb_padded": "右腕占位图像（mask=false）",
}


def number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def fmt(value: Any, digits: int = 3) -> str:
    converted = number(value)
    return f"{converted:.{digits}f}" if converted is not None else MISSING


def text(value: Any) -> str:
    if value is None or value == "":
        return MISSING
    return str(value).replace("\r", " ").replace("\n", " ")


def code(value: Any) -> str:
    return "`" + text(value).replace("`", "'") + "`" if value is not None and value != "" else MISSING


def table(headers: list[str], rows: list[list[Any]]) -> str:
    def row(values):
        return "| " + " | ".join(text(value).replace("|", "\\|") for value in values) + " |"
    return "\n".join([row(headers), row(["---"] * len(headers)), *(row(values) for values in rows)])


def reject_constant(value):
    raise ValueError(f"Non-finite JSON constant is not supported: {value}")


def load_json(path: Path, required: bool = False) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return {}
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def validate_statistics(summary: dict[str, Any], source: Path) -> None:
    phases = summary.get("phases")
    if not isinstance(phases, dict) or not phases:
        raise ValueError(f"No phase statistics found in {source}")
    collections = [*phases.values()]
    collections.extend(summary.get(name, {}) for name in
                       ("stage_request_totals", "stage_per_call", "vision_per_image", "denoise_per_iteration"))
    for collection in collections:
        if not isinstance(collection, dict):
            raise ValueError(f"Statistics collection is not an object: {source}")
        for metric, values in collection.items():
            if not isinstance(values, dict):
                raise ValueError(f"Statistics for {metric} are not an object: {source}")
            for key in ("mean_ms", "median_ms", "p95_ms", "min_ms", "max_ms", "std_ms"):
                if key in values and number(values[key]) is None:
                    raise ValueError(f"Invalid numeric statistic {metric}.{key} in {source}")
            if "n" in values and (not isinstance(values["n"], int) or isinstance(values["n"], bool) or values["n"] < 1):
                raise ValueError(f"Invalid sample count for {metric} in {source}")


def bundle(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    summary_path = path if path.is_file() else path / "summary.json"
    summary = load_json(summary_path, required=True)
    validate_statistics(summary, summary_path)
    directory = summary_path.parent
    return {"directory": directory, "summary_path": summary_path, "summary": summary,
            "raw": load_json(directory / "raw.json"), "provenance": load_json(directory / "provenance.json"),
            "accuracy": load_json(directory / "accuracy.json")}


def stats(result, phase, metric):
    return result["summary"].get("phases", {}).get(phase, {}).get(metric, {})


def mean(result, phase, metric):
    return number(stats(result, phase, metric).get("mean_ms"))


def percentage(value, total):
    numerator, denominator = number(value), number(total)
    return f"{100 * numerator / denominator:.2f}%" if numerator is not None and denominator is not None and denominator > 0 else MISSING


def samples(result, phase):
    found = result["raw"].get("phases", {}).get(phase, [])
    return found if isinstance(found, list) else []


def stage_totals(result, stage):
    return result["summary"].get("stage_request_totals", {}).get(stage, {})


def distribution_row(label, values):
    return [label, values.get("n", MISSING), *(fmt(values.get(key)) for key in
            ("mean_ms", "median_ms", "p95_ms", "min_ms", "max_ms", "std_ms"))]


def npy_metadata(stream):
    if stream.read(6) != b"\x93NUMPY":
        raise ValueError("Invalid NPY magic header")
    version = stream.read(2)
    if len(version) != 2 or version[0] not in (1, 2, 3):
        raise ValueError("Unsupported NPY format version")
    length_size = 2 if version[0] == 1 else 4
    length_bytes = stream.read(length_size)
    if len(length_bytes) != length_size:
        raise ValueError("Truncated NPY header length")
    length = struct.unpack("<H" if length_size == 2 else "<I", length_bytes)[0]
    if length > 1024 * 1024:
        raise ValueError("Unexpectedly large NPY header")
    header = stream.read(length)
    if len(header) != length:
        raise ValueError("Truncated NPY header")
    value = ast.literal_eval(header.decode("utf-8" if version[0] == 3 else "latin-1").strip())
    if not isinstance(value, dict):
        raise ValueError("Invalid NPY metadata")
    return {"shape": value.get("shape"), "dtype": value.get("descr"), "fortran_order": value.get("fortran_order")}


def fixture_metadata(directory):
    result = {}
    archive = directory / "fixed_inputs.npz"
    if archive.is_file():
        with zipfile.ZipFile(archive) as stream:
            for name in ("image", "wrist_image", "state", "noise", "prompt"):
                if name + ".npy" in stream.namelist():
                    with stream.open(name + ".npy") as array:
                        result[name] = npy_metadata(array)
    reference = directory / "reference_actions.npy"
    if reference.is_file():
        with reference.open("rb") as stream:
            result["reference_actions"] = npy_metadata(stream)
    return result


def stage_counts(result, name):
    measured = samples(result, PHASE_STAGES)
    if not measured:
        return MISSING, MISSING
    counts = [sum(stage.get("stage") == name for stage in sample.get("stages", [])) for sample in measured]
    label = str(counts[0]) if min(counts) == max(counts) else f"{min(counts)}–{max(counts)}（均值 {statistics.mean(counts):.2f}）"
    return label, sum(counts)


def raw_child_mean(result, parent, stage):
    measured = samples(result, PHASE_STAGES)
    if not measured:
        return None
    totals = []
    for sample in measured:
        values = [number(row.get("cuda_stream_elapsed_ms")) for row in sample.get("stages", [])
                  if row.get("stage") == stage and row.get("parent") == parent]
        values = [value for value in values if value is not None]
        if not values:
            return None
        totals.append(sum(values))
    return statistics.mean(totals)


def timestamp(value):
    if not isinstance(value, str):
        return MISSING
    try:
        instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if instant.tzinfo is None:
            return value + "（时区未记录）"
        return instant.astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S UTC+08:00")
    except ValueError:
        return value


def render(result, comparison=None):
    summary, raw, provenance = result["summary"], result["raw"], result["provenance"]
    settings = raw.get("settings", {})
    lines = ["# π0.5 Orin 基线性能报告", "",
             f"测量记录时间：{timestamp(provenance.get('utc_time'))}。数据目录：{code(result['directory'])}。", ""]
    full = mean(result, PHASE_FULL, "full_policy_wall_ms")
    model = mean(result, PHASE_MODEL, "model_wall_ms")
    lines.append(f"未插桩完整策略平均耗时 **{fmt(full)} ms/请求**；CUDA 输入就绪到输出计算完成的模型平均墙钟耗时 **{fmt(model)} ms**。")
    if full is not None and full > 0:
        lines.append(f"由完整策略均值换算的顺序请求吞吐约 **{1000 / full:.3f} 次/秒**，不是机器人的控制频率。")
    lines.extend(["", "这些结果使用固定合成输入与显式固定噪声；未运行机器人任务成功率评估。完整策略基线不含网络通信。", "",
                  "## 测试设置与数值校验", ""])
    configuration = [
        ["种子", settings.get("seed", provenance.get("fixture", {}).get("seed", MISSING))],
        ["每阶段配置的测量次数 / 排除的预热次数", f"{settings.get('iterations_per_phase', MISSING)} / {settings.get('warmup_per_phase', MISSING)}"],
        ["批大小 / 动作预测步数 / 去噪迭代次数", f"{settings.get('batch_size', MISSING)} / {settings.get('action_horizon', MISSING)} / {settings.get('denoising_steps', MISSING)}"],
        ["编码图像数 / 有效图像数", f"{settings.get('image_views_encoded', MISSING)} / {settings.get('valid_image_views', MISSING)}"],
        ["torch.compile 配置", "关闭（None）" if "pytorch_compile_mode" in settings and settings["pytorch_compile_mode"] is None else text(settings.get("pytorch_compile_mode"))],
        ["文本指令", provenance.get("fixture", {}).get("prompt", MISSING)],
    ]
    lines.append(table(["项目", "记录"], configuration))
    lines.extend(["", "10 步动作与 10 次去噪是两个独立设置。占位相机虽被注意力掩码屏蔽，仍执行视觉编码；本基线保留该行为。", ""])
    phase_rows = []
    for name, phase, metric in (("官方完整策略（未插桩）", PHASE_FULL, "full_policy_wall_ms"),
                                ("模型（无内部插桩）", PHASE_MODEL, "model_wall_ms"),
                                ("分解测量（已插桩）", PHASE_STAGES, "decomposed_policy_wall_ms")):
        warmups = raw.get("warmup", {}).get(phase)
        phase_rows.append([name, len(samples(result, phase)) if phase in raw.get("phases", {}) else stats(result, phase, metric).get("n", MISSING),
                           len(warmups) if isinstance(warmups, list) else MISSING])
    lines.append(table(["阶段", "实际测量样本数", "记录的预热样本数（排除）"], phase_rows))
    validations = [sample.get("validation", {}) for phase in raw.get("phases", {}) for sample in samples(result, phase)]
    if validations:
        shape_counts = Counter(str(value.get("shape", MISSING)) for value in validations)
        finite = Counter("true" if value.get("finite") is True else "false" if value.get("finite") is False else MISSING for value in validations)
        equality = Counter("true" if value.get("bitwise_equal_reference") is True else "false" if value.get("bitwise_equal_reference") is False else MISSING for value in validations)
        lines.extend(["", f"原始记录中的动作形状计数：{code(dict(shape_counts))}；有限值校验：{code(dict(finite))}；与参考动作逐位相等校验：{code(dict(equality))}。"])
    lines.extend(["", f"汇总中的全部动作逐位相等标记：{code(summary.get('all_actions_bitwise_equal'))}。包装恢复后验证：{code(raw.get('restoration_validation'))}。"])
    fixtures = fixture_metadata(result["directory"])
    if fixtures:
        lines.extend(["", "保存的数组文件头（只读取元数据，不反序列化对象）：", "",
                      table(["数组", "形状", "dtype"], [[name, code(value["shape"]), code(value["dtype"])] for name, value in fixtures.items()])])

    lines.extend(["", "## 未插桩基线分布", "",
                  table(["计时区间", "n", "均值 ms", "中位数 ms", "p95 ms", "最小 ms", "最大 ms", "标准差 ms"],
                        [distribution_row("完整策略墙钟", stats(result, PHASE_FULL, "full_policy_wall_ms")),
                         distribution_row("模型同步墙钟", stats(result, PHASE_MODEL, "model_wall_ms")),
                         distribution_row("模型 CUDA 流事件区间", stats(result, PHASE_MODEL, "model_cuda_stream_elapsed_ms"))]), "",
                  "完整策略计时包含 CPU 输入处理、输入传输、模型采样、输出传回与动作解码。模型墙钟计时前输入已在 CUDA 上就绪，计时结束前同步整个设备；因此其中也包含 Python 发射与循环控制开销。"])
    lines.extend(["", "## 插桩开销与完整分解", ""])
    overhead_rows = []
    for label, baseline, instrumented in (
        ("模型墙钟", model, mean(result, PHASE_STAGES, "model_wall_ms")),
        ("完整策略 vs 同步分解总墙钟", full, mean(result, PHASE_STAGES, "decomposed_policy_wall_ms")),
    ):
        difference = instrumented - baseline if baseline is not None and instrumented is not None else None
        ratio = instrumented / baseline if baseline is not None and baseline > 0 and instrumented is not None else None
        overhead_rows.append([label, fmt(baseline), fmt(instrumented), fmt(difference),
                              f"{ratio:.4f}×" if ratio is not None else MISSING,
                              percentage(difference, baseline)])
    lines.append(table(["比较", "未插桩均值 ms", "已插桩均值 ms", "差值 ms", "耗时比", "均值增幅"], overhead_rows))
    lines.extend(["", "完整分解额外在输入传输结束及完整模型结束处同步。它与未插桩策略的差异同时包含事件包装、边界同步及分阶段运行时的温度/频率漂移，不能全部解释为包装函数的成本。", ""])
    decomposed_metrics = [("CPU 预处理", "cpu_preprocess_wall_ms"),
                          ("输入 H2D 与 Observation 构建", "h2d_and_observation_wall_ms"),
                          ("模型同步墙钟", "model_wall_ms"),
                          ("输出 D2H 与后处理", "d2h_and_postprocess_wall_ms"),
                          ("分解总墙钟", "decomposed_policy_wall_ms")]
    lines.append(table(["分解区间", "n", "均值 ms", "中位数 ms", "p95 ms", "最小 ms", "最大 ms", "标准差 ms"],
                       [distribution_row(label, stats(result, PHASE_STAGES, metric)) for label, metric in decomposed_metrics]))

    lines.extend(["", "## 模型顶层互不重叠区间", "",
                  "以下百分比以已插桩模型的 CUDA 流事件区间均值为分母，只选择互不重叠的顶层区间。残余区间来自实测总区间减去这四组顶层区间。", ""])
    denominator = mean(result, PHASE_STAGES, "model_cuda_stream_elapsed_ms")
    top_rows = []
    for stage in ("model_preprocess", "prefix_embedding_total", "vlm_prefix_kv_prefill", "denoise_iteration"):
        value = number(stage_totals(result, stage).get("mean_ms"))
        top_rows.append([STAGE_LABELS[stage], fmt(value), percentage(value, denominator)])
    residual = mean(result, PHASE_STAGES, "model_remaining_cuda_stream_elapsed_ms")
    top_rows.append(["剩余区间（掩码/位置、Euler 更新、循环与空隙等）", fmt(residual), percentage(residual, denominator)])
    top_rows.append(["模型总 CUDA 流事件区间", fmt(denominator), "100.00%" if denominator is not None and denominator > 0 else MISSING])
    lines.append(table(["互不重叠区间", "每请求区间均值 ms", "占模型流区间"], top_rows))
    lines.extend(["", "CUDA Event 测量当前流上两次事件之间的经过时间，其中可能包含等待 CPU 发射造成的空隙。它不是 GPU 内核忙碌时间或利用率；模型总区间也不应被称为纯 GPU 运算时间。"])

    lines.extend(["", "## 嵌套区间层级", "",
                  "子区间已包含在父区间中，跨层级不可相加。表中父区间占比使用每请求累计区间均值，而不是单次调用均值。", ""])
    hierarchy = {
        "prefix_embedding_total": ("vision_encode", "language_token_embedding"),
        "denoise_iteration": ("action_suffix_embedding", "attention_mask_prepare", "action_expert_transformer", "action_output_projection"),
        "action_suffix_embedding": ("action_input_projection", "time_mlp_total"),
        "time_mlp_total": ("time_mlp_in_linear", "time_mlp_out_linear"),
    }
    hierarchy_rows = []
    for parent, children in hierarchy.items():
        parent_mean = number(stage_totals(result, parent).get("mean_ms"))
        known_values = []
        for child in children:
            # Raw parent filtering is essential: one attention mask conversion
            # occurs outside denoise, and must not be charged to denoise.
            value = raw_child_mean(result, parent, child)
            hierarchy_rows.append([STAGE_LABELS[parent], STAGE_LABELS[child], fmt(value), percentage(value, parent_mean)])
            known_values.append(value)
        if parent_mean is not None and all(value is not None for value in known_values):
            remaining = parent_mean - sum(known_values)
            hierarchy_rows.append([STAGE_LABELS[parent], "该父区间其余部分", fmt(remaining), percentage(remaining, parent_mean)])
    lines.append(table(["父区间", "子区间", "每请求累计均值 ms", "占父区间"], hierarchy_rows))
    lines.extend(["", "时间 MLP 总区间包含两次 SiLU，单独线性层区间不包含激活。其余部分还可能含 Python 发射与事件空隙，不能直接当作某个未单独测量算子的耗时。去噪迭代区间不含后续 Euler 更新及循环条件检查。"])

    lines.extend(["", "## 阶段调用次数与分布", "",
                  "调用次数来自已插桩原始记录。单次调用分布与每请求累计区间分开列出；例如视觉每请求执行三次，去噪每请求执行十次。", ""])
    stages = set(summary.get("stage_request_totals", {})) | set(summary.get("stage_per_call", {}))
    ordering = list(STAGE_LABELS)
    ordered_stages = sorted(stages, key=lambda stage: (ordering.index(stage) if stage in ordering else len(ordering), stage))
    rows = []
    for name in ordered_stages:
        calls, total = stage_counts(result, name)
        per_call = summary.get("stage_per_call", {}).get(name, {})
        per_request = stage_totals(result, name)
        rows.append([STAGE_LABELS.get(name, name), calls, total, fmt(per_call.get("mean_ms")),
                     fmt(per_call.get("median_ms")), fmt(per_call.get("p95_ms")), fmt(per_request.get("mean_ms"))])
    if rows:
        lines.append(table(["区间", "调用/请求", "总调用数", "单次均值 ms", "单次中位数 ms", "单次 p95 ms", "每请求累计均值 ms"], rows))
    else:
        lines.append(MISSING)
    for heading, collection, labels in (
        ("逐相机视觉编码", summary.get("vision_per_image", {}), IMAGE_LABELS),
        ("逐次去噪迭代", summary.get("denoise_per_iteration", {}), {}),
    ):
        lines.extend(["", f"### {heading}", ""])
        keys = sorted(collection, key=lambda key: int(key) if str(key).isdigit() else list(collection).index(key))
        if keys:
            lines.append(table(["区间", "n", "均值 ms", "中位数 ms", "p95 ms", "最小 ms", "最大 ms", "标准差 ms"],
                               [distribution_row(labels.get(key, f"第 {key} 次"), collection[key]) for key in keys]))
        else:
            lines.append(MISSING)

    lines.extend(["", "## 环境与来源", ""])
    cuda = provenance.get("cuda", {})
    versions = provenance.get("versions", {})
    memory = number(cuda.get("total_memory_bytes"))
    peak = number(cuda.get("max_tensor_allocated_bytes"))
    env_rows = [["设备", cuda.get("device", MISSING)], ["系统", provenance.get("platform", MISSING)],
                ["计算能力", code(cuda.get("compute_capability"))],
                ["统一/设备内存总量 GiB", fmt(memory / 2**30 if memory is not None else None)],
                ["峰值 CUDA 张量分配 GiB", fmt(peak / 2**30 if peak is not None else None)],
                ["CUDA / cuDNN", f"{text(cuda.get('runtime'))} / {text(cuda.get('cudnn'))}"],
                ["Python", provenance.get("python", MISSING)],
                ["PyTorch / Transformers", f"{text(versions.get('torch'))} / {text(versions.get('transformers'))}"],
                ["JAX / Flax / NumPy", f"{text(versions.get('jax'))} / {text(versions.get('flax'))} / {text(versions.get('numpy'))}"],
                ["参数元素 dtype 分布", code(provenance.get("parameter_elements_by_dtype"))],
                ["PyTorch 执行设置", code(provenance.get("torch_settings"))]]
    lines.append(table(["项目", "记录"], env_rows))
    lines.extend(["", "峰值 CUDA 张量分配不是整机统一内存使用量。功耗模式、频率、只读命令结果与环境变量见 provenance.json；本报告不假设运行时锁频。", ""])
    repository_rows = []
    for label, repository in provenance.get("repositories", {}).items():
        repository_rows.append([label, code(repository.get("commit")), repository.get("remote", MISSING),
                                "干净" if repository.get("status") == "" else code(repository.get("status"))])
    lines.append(table(["仓库", "提交", "远程", "工作区状态"], repository_rows) if repository_rows else MISSING)
    checkpoint = provenance.get("checkpoint", {})
    lines.extend(["", f"权重目录：{code(checkpoint.get('directory'))}。", "",
                  table(["资产", "SHA-256"], [["model.safetensors", code(checkpoint.get("model_sha256"))],
                                               ["norm_stats.json", code(checkpoint.get("norm_stats_sha256"))],
                                               ["tokenizer model", code(checkpoint.get("tokenizer_model_sha256"))]])])
    source_rows = [[label, code(item.get("path")), code(item.get("sha256"))]
                   for label, item in provenance.get("source_files", {}).items()]
    if source_rows:
        lines.extend(["", "实际载入源码：", "", table(["源码", "路径", "SHA-256"], source_rows)])
    if comparison is not None:
        lines.extend(render_comparison(result, comparison))
    lines.extend(["", "## 解读范围", "",
                  "- 主结论使用未插桩基线；插桩区间用于定位耗时，不能替代未插桩总耗时。",
                  "- 本机策略耗时排除 WebSocket、SSH、相机采集与机器人执行耗时。",
                  "- 固定合成输入可以检验数值一致性与时间分布，不能证明真实场景任务成功率。",
                  "- 各阶段顺序执行，温度、功耗与频率变化可能影响差异；不同阶段不是同时测得。",
                  "- p95 是有限样本的经验分位数；缺少的字段保持“未记录”，不会补成零或推断值。", "",
                  "原始依据：`summary.json`、`raw.json`、`provenance.json`、`samples.csv`、`stages.csv`、`fixed_inputs.npz` 与 `reference_actions.npy`。", ""])
    return "\n".join(lines)


def render_comparison(baseline, comparison):
    lines = ["", "## 可选对照", "", f"对照目录：{code(comparison['directory'])}。"]
    rows = []
    flashrt_mapping = "flashrt_policy_uninstrumented" in comparison["summary"].get("phases", {})
    for label, phase, metric in (("完整策略均值", PHASE_FULL, "full_policy_wall_ms"),
                                 ("模型同步墙钟均值", PHASE_MODEL, "model_wall_ms"),
                                 ("模型 CUDA 流事件区间均值", PHASE_MODEL, "model_cuda_stream_elapsed_ms")):
        first = mean(baseline, phase, metric)
        if flashrt_mapping and phase == PHASE_FULL:
            second = mean(comparison, "flashrt_policy_uninstrumented", "policy_wall_ms")
            source = "flashrt_policy_uninstrumented.policy_wall_ms"
        else:
            second = mean(comparison, phase, metric)
            source = f"{phase}.{metric}"
        speedup = first / second if first is not None and second is not None and second > 0 else None
        rows.append([label, fmt(first), fmt(second), f"{speedup:.4f}×" if speedup is not None else MISSING, code(source)])
    lines.extend(["", table(["匹配的测量指标", "基线 ms", "对照 ms", "基线/对照耗时比", "对照数据来源"], rows), "",
                  "对照仅使用表中明确对应的已测指标；没有对应字段时不计算。耗时比没有自动证明模型语义、精度或执行负载等价。"])
    if flashrt_mapping:
        lines.extend(["", "FlashRT 完整策略映射：`flashrt_policy_uninstrumented.policy_wall_ms` 对应本地完整策略请求，包含输入处理、传输、生产图执行与输出后处理。"])
        model_wall = mean(comparison, PHASE_MODEL, "model_wall_ms")
        model_event = mean(comparison, PHASE_MODEL, "model_cuda_stream_elapsed_ms")
        if model_wall is not None or model_event is not None:
            lines.extend(["", "FlashRT 模型项直接读取独立的生产模型测量 `model_uninstrumented.model_wall_ms` 与 `model_cuda_stream_elapsed_ms`，并与基线同一边界的实测值比较。计时前输入已在 CUDA 上就绪，结束前完成设备同步；墙钟仍包含生产图 replay 发射开销，CUDA Event 仍是流上经过时间，两者都不是 GPU 内核忙碌时间。"])
        else:
            lines.extend(["", "FlashRT 独立生产模型测量未记录，模型项保持“未记录”，不从完整策略或诊断阶段耗时推算。"])
        lines.extend(["", "`flashrt_diagnostic_stage_graphs` 单独捕获并分别发射诊断阶段图，具有不同的 launch 边界。它不是生产完整图的精确耗时分解，不映射到上表的生产模型项，也不据此计算模型加速比。"])
    checks = []
    for label, keys in (("固定输入数组 SHA-256", ("fixture", "array_sha256")),
                        ("权重 SHA-256", ("checkpoint", "model_sha256")),
                        ("归一化资产 SHA-256", ("checkpoint", "norm_stats_sha256")),
                        ("tokenizer SHA-256", ("checkpoint", "tokenizer_model_sha256"))):
        first, second = baseline["provenance"], comparison["provenance"]
        for key in keys:
            first = first.get(key) if isinstance(first, dict) else None
            second = second.get(key) if isinstance(second, dict) else None
        status = MISSING if first is None or second is None else "一致" if first == second else "不一致"
        checks.append([label, status])
    lines.extend(["", table(["可核对的来源字段", "结果"], checks), "",
                  f"对照逐位相等汇总标记：{code(comparison['summary'].get('all_actions_bitwise_equal'))}。该标记通常仅指对照运行内部自洽，跨后端等价性仍需独立的动作误差验证。"])
    if flashrt_mapping:
        baseline_hash = baseline["provenance"].get("checkpoint", {}).get("model_sha256")
        flashrt_hash = comparison["provenance"].get("hashes", {}).get("weights")
        weight_match = MISSING if not baseline_hash or not flashrt_hash else "一致" if baseline_hash == flashrt_hash else "不一致"
        lines.extend(["", f"FlashRT `hashes.weights` 与基线权重 SHA-256 核对：{weight_match}。其他未采用统一来源结构的字段保留“未记录”。"])
    accuracy = comparison.get("accuracy", {})
    metrics = accuracy.get("metrics", comparison["raw"].get("accuracy_against_openpi", {}))
    if metrics:
        lines.extend(["", "### 对照动作误差（独立于耗时）", "",
                      table(["指标", "实测记录"], [["形状", code(metrics.get("shape"))],
                                                   ["全部有限值", code(metrics.get("all_finite"))],
                                                   ["逐位相等", code(metrics.get("exact_match"))],
                                                   ["余弦相似度", fmt(metrics.get("cosine"), 9)],
                                                   ["RMSE", fmt(metrics.get("rmse"), 9)],
                                                   ["最大绝对误差", fmt(metrics.get("max_abs"), 9)],
                                                   ["平均绝对误差", fmt(metrics.get("mean_abs"), 9)]]), "",
                      f"误差验证范围：{text(accuracy.get('scope'))}。噪声精度契约：{text(accuracy.get('noise_contract'))}。"])
    caveats = comparison["summary"].get("caveats", [])
    if caveats:
        lines.extend(["", "对照实现记录的限制（原文）：", "", *(f"- {text(value)}" for value in caveats)])
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", "--baseline", dest="results", type=Path, required=True,
                        help="Baseline artifact directory, or its summary.json")
    parser.add_argument("--compare", type=Path, help="Optional other result directory or summary.json using the same schema")
    parser.add_argument("--output", type=Path, help="Default: BASELINE_DIRECTORY/report_zh.md")
    args = parser.parse_args()
    baseline = bundle(args.results)
    comparison = bundle(args.compare) if args.compare else None
    output = (args.output or baseline["directory"] / "report_zh.md").expanduser().resolve()
    if output.suffix.lower() not in (".md", ".markdown"):
        parser.error("--output must be a Markdown file (.md or .markdown)")
    if output in (baseline["summary_path"], baseline["directory"] / "raw.json", baseline["directory"] / "provenance.json"):
        parser.error("The report cannot overwrite benchmark JSON inputs")
    content = render(baseline, comparison)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(json.dumps({"status": "written", "output": str(output), "compare": args.compare is not None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
