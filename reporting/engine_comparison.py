#!/usr/bin/env python3
"""Render measured engine comparisons from a schema-1 route manifest.

Relative measurement paths are relative to this repository, not to the cwd.
Missing/blocked values stay unmeasured. This script never invents samples.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MIB = 1024 ** 2


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}


def get(value, *keys):
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def stats(values):
    values = sorted(values)
    if not values:
        return {}
    position = (len(values) - 1) * .95
    low, high = math.floor(position), math.ceil(position)
    return {"mean_ms": sum(values) / len(values),
            "p95_ms": values[low] + (values[high] - values[low]) * (position - low)}


def cell(value, digits=2):
    return "N/A" if value is None else f"{value:,.{digits}f}"


def safe(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def relative(path, base):
    try:
        return os.path.relpath(path, base).replace("\\", "/")
    except ValueError:  # Windows output and inputs can be on different drives.
        return Path(path).as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read(args.manifest.resolve())
    if manifest.get("schema_version", manifest.get("schema")) != 1 or not isinstance(manifest.get("routes"), list):
        parser.error("manifest requires schema_version: 1 and routes: []")
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, sources, routes = [], [], []

    def source(path, fragment=""):
        name = relative(path, REPO)
        return name + ("#" + fragment if fragment else "")

    def metric(route, category, name, value, unit, origin, note=""):
        if route["status"] == "blocked" and route.get("reason"):
            note = (note + "; " if note else "") + route["reason"]
        rows.append({"route": route["id"], "label": route["label"], "status": route["status"],
                     "category": category, "metric": name, "value": value, "unit": unit,
                     "unmeasured": value is None, "source": origin, "note": note})
        return value

    ids = [item.get("id") for item in manifest["routes"]]
    if None in ids or len(ids) != len(set(ids)):
        parser.error("route ids must be present and unique")
    for item in manifest["routes"]:
        route = dict(item)
        if route.get("status") not in ("runnable", "numeric_rejected", "blocked"):
            parser.error("route status must be runnable, numeric_rejected or blocked")
        route.setdefault("label", route["id"])
        route["data"] = {}
        for key in ("latency", "memory", "pipeline"):
            if route["status"] != "blocked" and route.get(key):
                path = Path(route[key])
                path = path.resolve() if path.is_absolute() else (REPO / path).resolve()
                route[key + "_path"] = path
                route["data"][key] = read(path / "summary.json")
                if not route["data"][key]:
                    parser.error(f"missing or empty {key} summary: {path}")
                if key == "memory" and get(route["data"][key], "metadata", "completed") is not True:
                    parser.error(f"memory pass is incomplete; refusing to publish partial metrics: {path}")
                sources.append((route["label"], key, path / "summary.json"))
        routes.append(route)

    def location(route, key, fragment=""):
        return source(route[key + "_path"] / "summary.json", fragment) if key + "_path" in route else ""

    baseline = next((route for route in routes if route.get("kind") == "baseline" and route["status"] == "runnable"), None)
    baseline_mean = get(baseline["data"].get("latency", {}), "phases", "model_uninstrumented", "model_wall_ms", "mean_ms") if baseline else None
    text = ["# π0.5 Orin 推理引擎实测对比", "",
            "数值全部由 manifest 指向的原始结果生成；N/A 表示未测量或未获得可比较结果，不能解释为零。"
            "延迟单位 ms，内存单位 MiB（1 MiB = 1,048,576 bytes；1 GiB = 1,024 MiB）。", "",
            "## 原始模型与完整 policy 延迟", "",
            "模型边界为 CUDA 输入已就绪至 CUDA 输出完成，不含输入上传、D2H 与动作反归一化。"
            "完整 policy 包含服务进程内前后处理，未包含网络通信；FlashRT 的固定 prompt 已预先缓存，baseline public policy 每请求处理 prompt。"
            "加速比以本 manifest 首个可运行 baseline 的模型 mean 为比较基准。", "",
            "| 路线 | 状态 / 分支 | 模型 mean | 模型 p95 | 完整 policy mean | baseline / 模型 | 固定输入 cosine / RMSE |",
            "|---|---|---:|---:|---:|---:|---:|"]
    status_zh = {"runnable": "可运行", "numeric_rejected": "误差过大，不采用", "blocked": "当前栈未跑通"}
    for route in routes:
        latency = route["data"].get("latency", {})
        phase = get(latency, "phases", "model_uninstrumented", "model_wall_ms") or {}
        origin = location(route, "latency", "phases.model_uninstrumented.model_wall_ms")
        mean = metric(route, "model", "mean", phase.get("mean_ms"), "ms", origin)
        p95 = metric(route, "model", "p95", phase.get("p95_ms"), "ms", origin)
        policy_keys = ("official_policy_uninstrumented", "full_policy_wall_ms") if route.get("kind") == "baseline" else ("flashrt_policy_uninstrumented", "policy_wall_ms")
        full = metric(route, "policy", "mean", get(latency, "phases", *policy_keys, "mean_ms"), "ms", location(route, "latency", "phases." + ".".join(policy_keys)))
        ratio = baseline_mean / mean if baseline_mean is not None and mean is not None and mean > 0 else None
        metric(route, "model", "baseline_speedup", ratio, "x", origin + ("; " + location(baseline, "latency") if baseline else ""),
               "numeric_rejected routes are experimental and excluded from usable-engine ranking")
        accuracy_path = route.get("latency_path", REPO / "__unmeasured__") / "accuracy.json"
        accuracy = read(accuracy_path).get("metrics", {}) if route["status"] != "blocked" else {}
        cosine = metric(route, "accuracy", "cosine", accuracy.get("cosine"), "ratio", source(accuracy_path) if accuracy else "")
        rmse = metric(route, "accuracy", "rmse", accuracy.get("rmse"), "action_unit", source(accuracy_path) if accuracy else "")
        text.append(f"| {safe(route['label'])} | {status_zh[route['status']]} / `{safe(route.get('branch', 'N/A'))}` | {cell(mean)} | {cell(p95)} | {cell(full)} | {cell(ratio)} | {cell(cosine, 6)} / {cell(rmse, 6)} |")
    text += ["", "INT8 因固定输入动作偏差过大而不采用；未宣称通过或违反预先规定的任务精度门限。"
             "其延迟仅作实验记录，不作为可替代 baseline 的可用方案。"
             "余弦/RMSE 仅覆盖同一合成观测与固定噪声，不代表机器人任务成功率。", "",
             "## 分步骤延迟", "", "表内为 mean / p95。Vision 包括实际计算的 3 路图像，denoise 包括 10 次迭代。"
             "baseline 来自 CUDA event 插桩；FlashRT 来自独立诊断阶段图，不能将各阶段相加宣称等于未经插桩生产图。"
             "baseline denoise 区间不含随后 Euler 更新和循环控制。输入/输出边界为另行同步诊断数据。", "",
             "| 路线 | 步骤 | mean / p95 (ms) |", "|---|---|---:|"]
    for route in routes:
        latency, pipeline = route["data"].get("latency", {}), route["data"].get("pipeline", {})
        is_base = route.get("kind") == "baseline"
        if is_base:
            raw = read(route.get("latency_path", REPO / "__unmeasured__") / "raw.json") if route["status"] != "blocked" else {}
            samples = get(raw, "phases", "instrumented_decomposed") or []
            combined = stats([row["timings"]["cpu_preprocess_wall_ms"] + row["timings"]["h2d_and_observation_wall_ms"]
                              for row in samples if all(name in row.get("timings", {}) for name in ("cpu_preprocess_wall_ms", "h2d_and_observation_wall_ms"))])
            steps = [("输入准备+上传", combined, "same-sample sum of raw CPUpre + H2D"),
                     ("其中 CPU preprocess", get(latency, "phases", "instrumented_decomposed", "cpu_preprocess_wall_ms"), ""),
                     ("其中 H2D+Observation", get(latency, "phases", "instrumented_decomposed", "h2d_and_observation_wall_ms"), ""),
                     ("Vision ×3", get(latency, "stage_request_totals", "vision_encode"), ""),
                     ("Prefill", get(latency, "stage_request_totals", "vlm_prefix_kv_prefill"), ""),
                     ("Denoise ×10", get(latency, "stage_request_totals", "denoise_iteration"), ""),
                     ("D2H+postprocess", get(latency, "phases", "instrumented_decomposed", "d2h_and_postprocess_wall_ms"), "")]
        else:
            stages = get(latency, "phases", "flashrt_diagnostic_stage_graphs") or {}
            steps = [("输入准备+上传", pipeline.get("input_preparation_and_upload_wall_ms"), "CPU conversion and H2D/D2D combined; prompt cached"),
                     ("其中 CPU preprocess", None, "unchanged frontend has no independently measured split"),
                     ("其中 H2D+Observation", None, "unchanged frontend has no independently measured split"),
                     ("Vision ×3", stages.get("vision_cuda_stream_elapsed_ms"), "diagnostic stage graph"),
                     ("Prefill", stages.get("prefill_cuda_stream_elapsed_ms"), "diagnostic stage graph"),
                     ("Denoise ×10", stages.get("denoise_cuda_stream_elapsed_ms"), "diagnostic stage graph"),
                     ("D2H+postprocess", pipeline.get("download_and_postprocess_wall_ms"), "separate pipeline pass")]
        for name, value, note in steps:
            value = value or {}
            origin = location(route, "pipeline" if not is_base and name in ("输入准备+上传", "D2H+postprocess") else "latency")
            if is_base and name == "输入准备+上传" and "latency_path" in route:
                origin = source(route["latency_path"] / "raw.json", "phases.instrumented_decomposed")
            metric(route, "stage_latency", name + "_mean", value.get("mean_ms"), "ms", origin, note)
            metric(route, "stage_latency", name + "_p95", value.get("p95_ms"), "ms", origin, note)
            if route["status"] != "blocked":
                text.append(f"| {safe(route['label'])} | {name} | {cell(value.get('mean_ms'))} / {cell(value.get('p95_ms'))} |")
        if route["status"] == "blocked":
            text.append(f"| {safe(route['label'])} | 全部步骤，未完成模型推理 | N/A |")

    text += ["", "## 内存：生产稳态与完整模型 pass", "",
             "各 backend 为独立新进程。稳态取 `steady_state_after_warmup`，发生在额外诊断图捕获之前。"
             "NvMap 为当前 Python PID 的 Jetson 驱动记账；RSS/PSS/USS 为进程主机映射。"
             "Jetson 统一内存的这些记账域存在重叠，不能相加。Torch 峰值只涵盖 Torch allocator，"
             "FlashRT 原生 cudaMalloc 与图/库缓冲区可能在其外。", "",
             "| 路线 | 稳态 NvMap PID | RSS / PSS / USS | 稳态 Torch live / reserved | 模型 Torch allocator 峰值 | 模型 NvMap 观察最大值（下界） | 全设备 used（背景参考） |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for route in routes:
        memory = route["data"].get("memory", {})
        steady = next((value for value in memory.get("checkpoints", []) if value.get("label") == "steady_state_after_warmup"), {})
        whole = get(memory, "stages", "model_total") or {}
        values = {}
        for name in ("nvmap_self_bytes", "process_rss_bytes", "process_pss_bytes", "process_uss_bytes", "cuda_device_used_bytes", "torch_allocated_bytes", "torch_reserved_bytes"):
            values[name] = metric(route, "memory_steady", name, steady.get(name), "bytes", location(route, "memory", "checkpoints.steady_state_after_warmup"))
        for name in ("torch_peak_allocated_bytes", "max_observed_nvmap_self_bytes"):
            values[name] = metric(route, "memory_model_total", name, get(whole, name, "max_bytes"), "bytes", location(route, "memory", "stages.model_total"), "sampled NvMap is a lower bound" if "nvmap" in name else "Torch allocator only")
        mb = lambda key: cell(values[key] / MIB) if values[key] is not None else "N/A"
        text.append(f"| {safe(route['label'])} | {mb('nvmap_self_bytes')} | {mb('process_rss_bytes')} / {mb('process_pss_bytes')} / {mb('process_uss_bytes')} | {mb('torch_allocated_bytes')} / {mb('torch_reserved_bytes')} | {mb('torch_peak_allocated_bytes')} | {mb('max_observed_nvmap_self_bytes')} | {mb('cuda_device_used_bytes')} |")
    text += ["", "模型总峰值与阶段峰值来自不同 pass。总模型 pass 内不插入任何 stage peak reset；"
             "加载、图准备/捕获、model_total 的主机/NvMap 轮询等待间隔为 100 ms，读取本身另需约 40–52 ms，实际采样间隔更长；"
             "其他细分阶段只读取同步前后边界。这些观察最大值可能漏过瞬态尖峰，仅为峰值下界。"
             "全设备 used 包含其他进程和驱动；不代表当前模型占用，也不能用它的差值宣称精确进程峰值。"
             "内存 pass 含逐阶段同步、读取 /proc、sudo 与 profiler 本身开销，其时间不用于延迟结论。", "",
             "## 各阶段内存", "", "各行均是绝对占用或峰值高于该阶段起点的增量，单位 MiB。"
             "阶段共享权重、KV 和 scratch；预分配引擎阶段间增量接近零是正常现象，不能将行相加。"
             "backend_load / prepare_and_capture 包含加载与生产图准备；diagnostic_graph_capture 及其后的阶段含额外诊断图对象。", "",
             "| 路线 | 阶段 | NvMap PID 观察最大绝对值 | RSS / PSS 观察最大值 | Torch allocator 最大绝对峰值 | Torch 峰值高于阶段起点 |",
             "|---|---|---:|---:|---:|---:|"]
    for route in routes:
        stages = route["data"].get("memory", {}).get("stages", {})
        for stage, group in stages.items():
            values = []
            for name in ("max_observed_nvmap_self_bytes", "max_observed_process_rss_bytes", "max_observed_process_pss_bytes", "torch_peak_allocated_bytes", "torch_peak_above_start_bytes"):
                value = metric(route, "stage_memory", stage + "." + name, get(group, name, "max_bytes"), "bytes", location(route, "memory", "stages." + stage))
                values.append(cell(value / MIB) if value is not None else "N/A")
            text.append(f"| {safe(route['label'])} | `{safe(stage)}` | {values[0]} | {values[1]} / {values[2]} | {values[3]} | {values[4]} |")
        if not stages:
            text.append(f"| {safe(route['label'])} | 未获得阶段内存测量 | N/A | N/A | N/A | N/A |")
    text += ["", "## 状态与数据来源", ""]
    for route in routes:
        if route.get("reason"):
            text.append(f"- **{safe(route['label'])}**：{safe(route['reason'])}")
    for label, key, path in sources:
        link = relative(path, out.parent)
        text.append(f"- {safe(label)} {key}：[summary.json]({link})（`{source(path)}`；同目录保留 raw/provenance）。")
    text += ["", "CSV 为长表，保留每个指标的数值、单位、`unmeasured`、相对源路径与说明。"
             "缺失列保持空值；Blocked 路线不读取不存在的测量目录。正式技术路径及失败原因详见仓库 `summary.md`。", ""]
    out.write_text("\n".join(text), encoding="utf-8")
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=("route", "label", "status", "category", "metric", "value", "unit", "unmeasured", "source", "note"))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(out), "csv": str(out.with_suffix('.csv')), "routes": len(routes), "metrics": len(rows)}))


if __name__ == "__main__":
    main()
