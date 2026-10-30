"""Render saved quantization/candidate experiments without running inference.

Usage: python selective_quantization/report.py --results RUN/results --output RUN/report
The report preserves failed/partial runs and keeps camera groups and held-out
numerical agreement separate. It never interprets cosine as task success.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil


GIB = 1024 ** 3
RESERVED = {"references", "screening"}
BASELINES = {2: "bf16_2valid_968", 3: "bf16_3valid_968"}
LABELS = {
    "bf16_2valid_968": "BF16 基线：2路有效，968行",
    "bf16_2valid_968_end_control": "BF16 末尾控制：2路有效，968行",
    "int8_encoder_all": "Encoder 全层 W8A8",
    "int8_decoder_all": "Decoder 全层 W8A8",
    "int8_both_all": "Encoder＋Decoder 全层 W8A8",
    "int8_encoder_ffn": "Encoder FFN W8A8",
    "int8_decoder_ffn": "Decoder FFN W8A8",
    "int8_encoder_screened": "筛选 Encoder 层 W8A8",
    "int8_decoder_screened": "筛选 Decoder 层 W8A8",
    "int8_both_screened": "筛选 Encoder＋Decoder 层 W8A8",
    "bf16_2valid_text_compact_775": "仅删除文本 padding：775行",
    "bf16_2valid_full_compact_519": "删除文本 padding＋masked 相机：519行",
    "bf16_2valid_steps8": "BF16：8轮去噪",
    "bf16_2valid_steps5": "BF16：5轮去噪",
    "bf16_3valid_968": "BF16 基线：3路有效，968行",
    "bf16_3valid_compact_775": "3路有效，删除文本 padding：775行",
    "int8_screened_compact_519": "筛选 W8A8＋519行组合",
    "int8_decoder_ffn_compact_519": "Decoder FFN W8A8＋519行",
    "int8_decoder_ffn_3valid_compact_775": "3路有效：Decoder FFN W8A8＋775行",
    "int8_encoder_strict": "保守 Encoder 15/16/17 层 W8A8",
    "int8_strict_encoder_decoder_ffn_compact_519": "保守 Encoder＋Decoder FFN＋519行",
}


def display_label(name):
    match = re.match(r"^(.*)_retry\d*$", name)
    if match:
        return LABELS.get(match[1], match[1]) + "（独立重试）"
    return LABELS.get(name, name)


def load_json(path, problems):
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, ValueError) as error:
        problems.append(f"读取 {path} 失败：{error}")
        return None


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fmt(value, digits=2):
    return f"{value:.{digits}f}" if number(value) else "N/A"


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", "")


def table(headers, rows):
    lines = ["| " + " | ".join(map(cell, headers)) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(map(cell, row)) + " |" for row in rows]
    return "\n".join(lines)


def stage_mean(result, name):
    return result.get("stages", {}).get(name + "_cuda_stream_elapsed_ms", {}).get("mean_ms")


def stat_value(telemetry, field, name="max"):
    return telemetry.get(field, {}).get(name)


def checkpoint_max(telemetry, field):
    values = [row.get("torch", {}).get(field) for row in telemetry.get("memory_checkpoints", [])
              if isinstance(row.get("torch"), dict)]
    return max((value for value in values if number(value)), default=None)


def worst_case(metric):
    values = [row.get("cosine") for row in metric.get("per_case", [])]
    return min((value for value in values if number(value)), default=None)


def array_agreement(actual, expected):
    import numpy as np
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape or actual.ndim != 3 or actual.shape[1] != 10 or actual.shape[2] not in (6, 7):
        raise ValueError(f"invalid saved actions: {actual.shape}/{expected.shape}")
    if not actual.size or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("saved actions are empty or non-finite")
    a, b = actual.ravel(), expected.ravel()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    cosine = float(a @ b / denominator) if denominator else (1.0 if np.array_equal(a, b) else 0.0)
    per_case = []
    for aa, bb in zip(actual, expected):
        aa, bb = aa.ravel(), bb.ravel()
        den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
        per_case.append(float(aa @ bb / den) if den else (1.0 if np.array_equal(aa, bb) else 0.0))
    delta = actual - expected
    result = {"n_cases": len(actual), "cosine": max(-1.0, min(1.0, cosine)),
              "worst_case_cosine": max(-1.0, min(1.0, min(per_case))),
              "rmse": float(np.sqrt(np.mean(delta ** 2))), "max_abs": float(np.abs(delta).max())}
    if actual.shape[2] == 7:
        result["arm_only"] = array_agreement(actual[..., :6], expected[..., :6])
    return result


def split_agreements(run, results, screening, problems):
    """Recompute subset vector cosine, rather than average per-case cosines."""
    import numpy as np
    path = run["directory"] / "actions.npy"
    if not path.is_file():
        return {}
    try:
        actual = np.load(path, allow_pickle=False)
        groups = {"discovery": screening.get("discovery_indices"), "holdout": screening.get("holdout_indices")}
        for name, indices in groups.items():
            if indices is None:
                continue
            if not isinstance(indices, list) or len(set(indices)) != len(indices) or any(
                    not isinstance(i, int) or i < 0 or i >= len(actual) for i in indices):
                raise ValueError(f"invalid {name} indices")
        if set(groups.get("discovery") or ()) & set(groups.get("holdout") or ()):
            raise ValueError("discovery and holdout overlap")
        # The input manifest declares case 0 as the legacy synthetic fixture;
        # keep it separate because its action magnitude may dominate a single
        # cosine calculated across all cases.
        names = run["result"].get("case_names", [])
        if names and names[0] == "legacy_synthetic":
            groups["legacy"] = [0]
            groups["recorded"] = list(range(1, len(actual)))
        else:
            problems.append(f"{run['name']} 未确认 case 0 为 legacy_synthetic，未生成记录图像/合成输入分组。")
        valid = run["valid_views"]
        references = {"openpi": results / "references" / f"openpi_{valid}valid_10steps.npy",
                      "bf16": results / BASELINES.get(valid, "missing_baseline") / "actions.npy"}
        result = {}
        for reference_name, reference_path in references.items():
            if not reference_path.is_file():
                continue
            expected = np.load(reference_path, allow_pickle=False)
            if expected.shape != actual.shape:
                raise ValueError(f"reference shape mismatch: {reference_path}")
            result[reference_name] = {"all": array_agreement(actual, expected)}
            for split, indices in groups.items():
                if indices:
                    result[reference_name][split] = array_agreement(actual[indices], expected[indices])
        return result
    except (OSError, ValueError, TypeError) as error:
        problems.append(f"{run['name']} 划分精度计算失败：{error}")
        return {}


def gather(results, provenance, problems):
    records = {item["name"]: item for item in provenance.get("runs", []) if "name" in item}
    names = [name for name in records if name not in RESERVED]
    for directory in sorted(results.iterdir()):
        if directory.is_dir() and directory.name not in RESERVED and any(
                (directory / filename).is_file() for filename in
                ("result.json", "error.json", "latency_partial.json", "telemetry_partial.json")):
            if directory.name not in names:
                names.append(directory.name)
    runs = []
    for name in names:
        directory = results / name
        result, source = {}, None
        for filename in ("latency_partial.json", "telemetry_partial.json", "result.json"):
            current = load_json(directory / filename, problems)
            if current is not None:
                result, source = current, filename
        error = load_json(directory / "error.json", problems)
        record = records.get(name, {})
        config = result.get("config") or load_json(results.parent / "configs" / (name + ".json"), problems) or {}
        metadata = result.get("metadata", {})
        complete = result.get("completed") is True and source == "result.json"
        if error or record.get("returncode", 0) != 0:
            status = "失败（存在部分数据）" if result else "失败"
        elif complete:
            status = "完成"
        else:
            status = "未完成（部分数据）" if result else "未完成（无结果）"
        if complete and result.get("sustained", {}).get("telemetry", {}).get("sample_count", 0) < 2:
            complete = False
            if not error and record.get("returncode", 0) == 0:
                status = "部分完成（遥测无效）"
        valid = config.get("valid_num_views", metadata.get("valid_num_views", 3 if "3valid" in name else 2))
        runs.append({"name": name, "label": display_label(name), "directory": directory,
                     "result": result, "config": config, "metadata": metadata, "valid_views": valid,
                     "record": record, "error": error, "status": status,
                     "complete": complete and error is None and record.get("returncode", 0) == 0,
                     "source": source})
    return runs


def flatten(run, baselines):
    result, config, meta = run["result"], run["config"], run["metadata"]
    telemetry = result.get("sustained", {}).get("telemetry", {})
    latency = result.get("latency", {}).get("model_wall", {})
    base = baselines.get(run["valid_views"], {})
    base_mean = base.get("result", {}).get("latency", {}).get("model_wall", {}).get("mean_ms")
    mean = latency.get("mean_ms")
    reduction = (1 - mean / base_mean) * 100 if (
        run["complete"] and base.get("complete") and number(mean) and number(base_mean) and base_mean > 0) else None
    values = {"variant": run["name"], "label_zh": run["label"], "status": run["status"],
              "valid_num_views": run["valid_views"], "num_views": config.get("num_views", meta.get("num_views", 3)),
              "max_prompt_len": config.get("max_prompt_len", meta.get("max_prompt_len", 200)),
              "encoder_rows": meta.get("encoder_rows"), "num_steps": config.get("num_steps", meta.get("num_steps", 10)),
              "model_mean_ms": mean, "model_p95_ms": latency.get("p95_ms"),
              "model_samples": latency.get("n"), "reduction_vs_same_group_bf16_percent": reduction,
              "vision_mean_ms": stage_mean(result, "vision"), "prefill_mean_ms": stage_mean(result, "prefill"),
              "denoise_mean_ms": stage_mean(result, "denoise"),
              "nvmap_max_gib": stat_value(telemetry, "nvmap_self_bytes") / GIB if number(stat_value(telemetry, "nvmap_self_bytes")) else None,
              "rss_max_gib": stat_value(telemetry, "process_rss_bytes") / GIB if number(stat_value(telemetry, "process_rss_bytes")) else None,
              "system_available_min_gib": stat_value(telemetry, "system_available_bytes", "min") / GIB if number(stat_value(telemetry, "system_available_bytes", "min")) else None,
              "sustained_seconds": result.get("sustained", {}).get("seconds"),
              "telemetry_samples": telemetry.get("sample_count"),
              "telemetry_collection_mean_ms": stat_value(telemetry, "collection_ms", "mean"),
              "gpu_load_mean_percent": stat_value(telemetry, "gpu_load_percent", "mean"),
              "board_main_rails_mean_w": stat_value(telemetry, "main_rails_sum_w", "mean"),
              "error": (run["error"] or {}).get("error", ""),
              "returncode": run["record"].get("returncode"), "result_source": run["source"]}
    for field in ("allocated", "reserved", "peak_allocated", "peak_reserved"):
        value = checkpoint_max(telemetry, field + "_bytes")
        values["torch_" + field + "_checkpoint_max_gib"] = value / GIB if number(value) else None
    for counter in ("oc1_event_cnt", "oc2_event_cnt", "oc3_event_cnt"):
        first=telemetry.get("oc_counters_start",{}).get(counter)
        last=telemetry.get("oc_counters_end",{}).get(counter)
        values[counter+"_start"]=first
        values[counter+"_end"]=last
        values[counter+"_delta"]=(last-first if number(first) and number(last) else None)
    for sensor in ("gpu", "cpu"):
        distribution = telemetry.get("temperatures_c", {}).get(sensor, {})
        for stat in ("start", "end", "max", "mean"):
            values[f"{sensor}_temperature_{stat}_c"] = distribution.get(stat)
    for source in ("openpi", "bf16"):
        metric = result.get("accuracy_" + source, {})
        for key in ("cosine", "rmse", "max_abs"):
            values[source + "_" + key] = metric.get(key)
        values[source + "_worst_case_cosine"] = worst_case(metric)
        arm = run.get("splits", {}).get(source, {}).get("all", {}).get("arm_only", {})
        for key in ("cosine", "worst_case_cosine", "rmse", "max_abs"):
            values[f"{source}_arm_{key}"] = arm.get(key)
        for split in ("discovery", "holdout", "recorded", "legacy"):
            metric = run.get("splits", {}).get(source, {}).get(split, {})
            for key in ("n_cases", "cosine", "worst_case_cosine", "rmse", "max_abs"):
                values[f"{source}_{split}_{key}"] = metric.get(key)
            arm = metric.get("arm_only", {})
            for key in ("cosine", "worst_case_cosine", "rmse", "max_abs"):
                values[f"{source}_{split}_arm_{key}"] = arm.get(key)
    return values


def copy_layers(screening, source, output):
    target = output / "layers.csv"
    if source.is_file():
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
        return "已复制筛选阶段原始 layers.csv。"
    fields = ["domain", "layer", "cosine", "min_case_cosine", "rmse", "max_abs", "probe_mean_ms", "numerical_screen_pass"]
    with target.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in screening.get("layers", []):
            metric = item.get("accuracy", {})
            writer.writerow({"domain": item.get("domain"), "layer": item.get("layer"),
                             "cosine": metric.get("cosine"), "min_case_cosine": item.get("min_case_cosine"),
                             "rmse": metric.get("rmse"), "max_abs": metric.get("max_abs"),
                             "probe_mean_ms": item.get("probe_latency", {}).get("mean_ms"),
                             "numerical_screen_pass": item.get("numerical_screen_pass")})
    return "筛选阶段未保存完整 layers.csv；已从 screening.json 恢复现有记录，缺少记录未补造。"


def relative_link(path, output):
    try:
        return Path(os.path.relpath(path, output)).as_posix()
    except ValueError:
        return path.as_posix()


def render(results, output):
    output.mkdir(parents=True, exist_ok=True)
    problems = []
    provenance = load_json(results / "provenance.json", problems) or {}
    retries = load_json(results / "retry_provenance.json", problems) or {}
    screening = load_json(results / "screening" / "screening.json", problems) or {}
    manifest = load_json(results.parent / "inputs" / "manifest.json", problems) or {}
    if not (results / "provenance.json").is_file():
        problems.append("未发现 provenance.json，无法确认套件和服务恢复状态。")
    combined_provenance = dict(provenance)
    combined_provenance["runs"] = list(provenance.get("runs", [])) + list(retries.get("runs", []))
    runs = gather(results, combined_provenance, problems)
    baselines = {views: next((run for run in runs if run["name"] == name), {}) for views, name in BASELINES.items()}
    # Never replace the original action teacher; only the performance reference
    # can fall back to the independently measured, valid end control.
    end_control = next((run for run in runs if run["name"] == "bf16_2valid_968_end_control"), {})
    for views, original in BASELINES.items():
        if not baselines[views].get("complete"):
            completed_retries = [run for run in runs if run["name"].startswith(original + "_retry") and run["complete"]]
            if completed_retries:
                baselines[views] = completed_retries[-1]
            elif views == 2 and end_control.get("complete"):
                baselines[views] = end_control
    for run in runs:
        run["splits"] = split_agreements(run, results, screening, problems)
    flat = [flatten(run, baselines) for run in runs]
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = list(flat[0]) if flat else ["variant", "status"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    layer_note = copy_layers(screening, results / "screening" / "layers.csv", output)
    lines = ["# π0.5 选择性量化与补充候选实验报告", "",
             f"生成时间：{datetime.now(timezone.utc).isoformat()}。原始结果目录：`{results}`。", "",
             "本报告根据已保存的 GPU 实验记录生成；没有以估算填补缺失实测。Cosine 表示最终动作与数值参考的相似度，**不代表任务成功率**。", "",
             "## 实验完成情况", "",
             f"已发现 {len(runs)} 个独立候选进程结果，其中完成 {sum(run['complete'] for run in runs)} 个，失败或未完成 {sum(not run['complete'] for run in runs)} 个。套件 completed={provenance.get('completed', '未记录')}；调度结束不表示所有候选成功。"]
    if retries:
        lines += ["", f"独立补测 completed={retries.get('completed', False)}；计划补测：`{retries.get('pending', [])}`。补测使用新的 `_retry` 目录，原有精度 teacher、失败日志和无效遥测行均保留。"]
        unresolved = [run["name"] for run in runs if not run["complete"] and
                      not any(other["complete"] and other["name"].startswith(run["name"]+"_retry") for other in runs)]
        lines += ["", f"历史不完整尝试中，当前尚未通过独立重试补齐的配置：`{unresolved}`。主表只列完整数据集；历史缺失数据仍保留在 CSV、原始目录与失败附录中。"]
    if provenance.get("fatal_error"):
        lines += ["", f"套件中断原因：`{cell(provenance['fatal_error'])}`。"]
    lines += ["", "## 方法与比较边界", "",
              "- 主矩阵为两路有效相机；保留三路视觉计算的 968 行 BF16 是该组基线。",
              "- 三路有效组的第三路图像复制前视图，仅检验形状、mask 和数值路径；不是真实三相机任务数据。该组只与三路有效 BF16 比较，不跨组比较 cosine 或宣称加速。",
              "- `968→775` 在两路有效、仍计算三路视觉时只移除 193 行文本 padding；`968→519` 还移除了 masked 第三路视觉及其 256 行前缀，收益包含两项工作量减少。三路均有效时 `968→775` 只移除文本 padding。有效文本 token 不应被截断。",
              "- 去噪 10→8/5 改变积分次数及步长，action chunk 仍为 10 个动作；不是缩短 chunk，也不是直接截掉原来的循环尾部。",
              "- 8/5 轮候选的整模型节省可能超过按原去噪阶段平均单步成本作出的线性估算；需要同时查看 Vision、Prefill、Denoise 和 BF16 控制值，不能将其他阶段或 tactic/频率变化都归因于减少去噪步数。",
              "- 主延迟边界为 GPU 输入就绪→完整 CUDA Graph→GPU 同步完成，不包含输入 staging、下载和反归一化。阶段图在主测量之后捕获，阶段和不取代完整图延迟。",
              "- 延迟 pass 与带 OS 采样的持续负载 pass 分开。温度是约 30 秒窗口中的观测，不声称达到热稳态；起始温度、顺序、动态频率和 tactic 会影响结果。",
              "- 持续负载阶段为满速循环，较快配置会在相同窗口执行更多次；温度不能直接代表固定业务调用频率下的温度，也不用于给优化方案排序。",
              "- W8A8 代理仅替换选中 GEMM，保持原 BF16 GELU、Norm、残差与调度；未沿用旧 INT8 路径中 SiLU 替代 GELU 的计算。",
              "- 代理保留 BF16 权重并额外保存 INT8 副本与量化 scratch，属于数值/性能实验实现，不是释放 BF16 权重后的压缩部署。RSS、NvMap、Torch allocator 和 CUDA 设备记账重叠，**不可相加**。", ""]
    lines += ["性能基准：" + "；".join(f"{views}路有效组使用 `{baselines[views].get('name') if baselines[views].get('complete') else '无有效基线'}`" for views in (2, 3)) + "。精度参考仍固定为各组最初 BF16 actions.npy，末尾控制不会覆盖 teacher。", ""]
    initial_means = {}
    for views, name in BASELINES.items():
        initial = next((run for run in runs if run["name"] == name), {})
        initial_means[views] = initial.get("result", {}).get("latency", {}).get("model_wall", {}).get("mean_ms")
    if all(number(initial_means[views]) for views in (2, 3)):
        lines += [f"两组起始测量分别为：2路有效 {fmt(initial_means[2])} ms、3路有效 {fmt(initial_means[3])} ms。即使三路有效的观测值更低，也不能得出“增加有效相机会加速”的结论：两组运行时 tactic、频率和温度等没有被单独隔离，应只做组内比较。", ""]
    if manifest:
        lines += [f"输入：{manifest.get('cases', '未记录')} 个 case；{len(manifest.get('recorded_frames', []))} 组记录图像，另保留历史合成输入。固定 prompt 为 `{manifest.get('prompt', '未记录')}`。不同图像使用该固定测试指令，仅用于数值压力测试，不是对应任务的闭环执行评估。", ""]
    power = provenance.get("commands", {}).get("power_mode", {}).get("stdout", "").strip()
    if power:
        lines += [f"功耗模式查询：`{cell(power)}`。实验未修改时钟配置；完整时钟与依赖版本见 provenance.json。", ""]
    selected_names = ["bf16_2valid_968_retry", "int8_decoder_ffn", "bf16_2valid_full_compact_519",
                      "int8_decoder_ffn_compact_519", "int8_strict_encoder_decoder_ffn_compact_519",
                      "bf16_3valid_968", "bf16_3valid_compact_775", "int8_decoder_ffn_3valid_compact_775"]
    selected_rows = [row for name in selected_names for row in flat if row["variant"] == name and
                     any(run["name"] == name and run["complete"] for run in runs)]
    if selected_rows:
        lines += ["## 关键配置实测", "", "延迟为 ms，内存为 GiB，温度为 °C；cosine 均对照相同有效相机数量的原 OpenPI 10步输出。", "",
                  table(["配置", "均值 / p95", "NvMap / RSS max", "GPU / CPU最高温度", "全17样本 cosine"],
                        [[row["label_zh"], f"{fmt(row['model_mean_ms'])} / {fmt(row['model_p95_ms'])}",
                          f"{fmt(row['nvmap_max_gib'],3)} / {fmt(row['rss_max_gib'],3)}",
                          f"{fmt(row['gpu_temperature_max_c'])} / {fmt(row['cpu_temperature_max_c'])}",
                          fmt(row["openpi_cosine"],6)] for row in selected_rows]), "",
                  "两路有效时，优先进入后续任务验证的数值候选是 Decoder FFN W8A8＋519行；额外加入保守 Encoder 层的微小收益需结合控制波动判断。三路有效请使用独立775行结果，不能套用两路519行延迟。上述候选均未经过闭环任务成功率验收。", ""]
    for views, heading in ((2, "两路有效相机：主矩阵"), (3, "三路有效相机：复制前视图的形状验证")):
        subset = [(run, row) for run, row in zip(runs, flat) if run["valid_views"] == views and run["complete"]]
        if not subset:
            continue
        lines += ["## " + heading, "", "### 延迟", "",
                  "单位为 ms；改善百分比只相对于本组已完成的 BF16 基线。失败行即使有部分数据，也不计算有效改善。", "",
                  table(["配置", "状态", "模型均值 / p95", "Vision", "Prefill", "Denoise", "均值改善"],
                        [[run["label"], row["status"], f"{fmt(row['model_mean_ms'])} / {fmt(row['model_p95_ms'])}",
                          fmt(row["vision_mean_ms"]), fmt(row["prefill_mean_ms"]), fmt(row["denoise_mean_ms"]),
                          fmt(row["reduction_vs_same_group_bf16_percent"]) + "%" if number(row["reduction_vs_same_group_bf16_percent"]) else "N/A"]
                         for run, row in subset]), "", "### 内存与温度", "",
                  "内存为 GiB，温度为 °C。Torch 数字是显式 checkpoint 的最大值；峰值计数从持续负载前重置，不包含完整加载峰值。温度写为“开始→结束 / 最高”。", "",
                  table(["配置", "NvMap max", "RSS max", "Torch allocated / reserved", "Torch peak allocated / reserved", "GPU 温度", "CPU 温度"],
                        [[run["label"], fmt(row["nvmap_max_gib"], 3), fmt(row["rss_max_gib"], 3),
                          f"{fmt(row['torch_allocated_checkpoint_max_gib'],3)} / {fmt(row['torch_reserved_checkpoint_max_gib'],3)}",
                          f"{fmt(row['torch_peak_allocated_checkpoint_max_gib'],3)} / {fmt(row['torch_peak_reserved_checkpoint_max_gib'],3)}",
                          f"{fmt(row['gpu_temperature_start_c'])}→{fmt(row['gpu_temperature_end_c'])} / {fmt(row['gpu_temperature_max_c'])}",
                          f"{fmt(row['cpu_temperature_start_c'])}→{fmt(row['cpu_temperature_end_c'])} / {fmt(row['cpu_temperature_max_c'])}"]
                         for run, row in subset]), "", "### 全 case 最终动作数值一致性", "",
                  table(["配置", "本组 BF16 cosine", "最低 case cosine", "BF16 RMSE", "BF16 max abs", "OpenPI cosine", "OpenPI 最低 case cosine", "OpenPI RMSE", "OpenPI max abs"],
                        [[run["label"], fmt(row["bf16_cosine"], 6), fmt(row["bf16_worst_case_cosine"], 6),
                          fmt(row["bf16_rmse"], 6), fmt(row["bf16_max_abs"], 6),
                          fmt(row["openpi_cosine"], 6), fmt(row["openpi_worst_case_cosine"], 6),
                          fmt(row["openpi_rmse"], 6), fmt(row["openpi_max_abs"], 6)] for run, row in subset]), ""]
    lines += ["## 记录图像与历史合成输入分列", "",
              "全 17-case cosine 可能受到大幅值样本的权重影响，不能据此认定所有记录图像都出现相同程度的偏差。case 0 为 legacy synthetic；记录图像为其余 case（同一帧含不同噪声）。以下分别计算完整子集 cosine 与最低 case cosine。", ""]
    image_rows = []
    for run, row in zip(runs, flat):
        if not run["complete"]:
            continue
        for source, label in (("bf16", "本组 BF16"), ("openpi", "本组 OpenPI 10步")):
            if not run.get("splits", {}).get(source, {}).get("recorded"):
                continue
            prefix = source + "_"
            image_rows.append([run["label"], label,
                               fmt(row[prefix+"legacy_cosine"], 6),
                               f"{fmt(row[prefix+'recorded_cosine'],6)} / {fmt(row[prefix+'recorded_worst_case_cosine'],6)}",
                               fmt(row[prefix+"recorded_rmse"],6), fmt(row[prefix+"recorded_max_abs"],6)])
    lines += [table(["配置", "参考", "合成 case 0 cosine", "记录图像 cosine / 最低", "记录图像 RMSE", "记录图像 max abs"], image_rows)
              if image_rows else "缺少动作数组或 legacy case 身份记录，未补造分组结果。", ""]
    arm_rows = []
    for run, row in zip(runs, flat):
        if not run["complete"]:
            continue
        for source, label in (("bf16", "本组 BF16"), ("openpi", "本组 OpenPI 10步")):
            if not run.get("splits", {}).get(source, {}).get("recorded"):
                continue
            arm_rows.append([run["label"], label, fmt(row[source+"_arm_cosine"],6),
                             fmt(row[source+"_recorded_arm_cosine"],6),
                             fmt(row[source+"_recorded_arm_worst_case_cosine"],6),
                             fmt(row[source+"_recorded_arm_rmse"],6),
                             fmt(row[source+"_recorded_arm_max_abs"],6)])
    lines += ["### 前六维动作单独诊断", "",
              "下面移除第七维夹爪数值，避免常量或大幅值夹爪分量主导整体 cosine；它仍是数值指标，不是机器人末端位姿误差或成功率。", "",
              table(["配置", "参考", "全部 case 前6维 cosine", "记录图像前6维 cosine", "记录图像最低 case", "记录图像 RMSE", "记录图像 max abs"], arm_rows)
              if arm_rows else "缺少动作数组，未补造前六维诊断。", ""]
    lines += ["## 发现集与留出集", "",
              f"层选择仅使用 discovery indices={screening.get('discovery_indices', '未记录')}；留出评估使用 holdout indices={screening.get('holdout_indices', '未记录')}。",
              "留出集未参与层选择；同一帧的不同噪声可能分属两组，因此它不等价于独立任务或完全独立场景的泛化测试。下表 cosine 由各子集完整动作向量重新计算，并同时列出最差 case；不是把逐 case cosine 简单平均。", ""]
    split_rows = []
    for run, row in zip(runs, flat):
        if not run["complete"]:
            continue
        for source, label in (("bf16", "本组 BF16"), ("openpi", "本组 OpenPI 10步")):
            if not run.get("splits", {}).get(source):
                continue
            prefix = source + "_"
            split_rows.append([run["label"], label,
                               f"{fmt(row[prefix+'discovery_cosine'],6)} / {fmt(row[prefix+'discovery_worst_case_cosine'],6)}",
                               f"{fmt(row[prefix+'holdout_cosine'],6)} / {fmt(row[prefix+'holdout_worst_case_cosine'],6)}",
                               fmt(row[prefix+"holdout_rmse"],6), fmt(row[prefix+"holdout_max_abs"],6)])
    lines += [table(["配置", "参考", "发现集 cosine / 最低", "留出集 cosine / 最低", "留出 RMSE", "留出 max abs"], split_rows)
              if split_rows else "缺少动作数组、参考或划分记录，未补造子集精度。", "",
              "## 逐层筛选：诊断数据与最终候选分开", "",
              f"已保存 {len(screening.get('layers', []))}/36 个单层筛选记录，completed={screening.get('completed', False)}。选择规则：`{cell(json.dumps(screening.get('selection_rule', {}), ensure_ascii=False))}`。", "",
              f"通过筛选的 Encoder 层：`{screening.get('selected_layers', {}).get('encoder', '尚未完成选择')}`；Decoder 层：`{screening.get('selected_layers', {}).get('decoder', '尚未完成选择')}`。", "",
              "单层筛选将所有准备好的量化副本共同常驻，通过切换选择并重捕获图测数值误差；每层少量 probe 仅用于诊断，不能当成 fresh-process 部署延迟或内存。多个单层分别通过，不保证组合仍通过；组合结果必须查看主矩阵和留出集。", "",
              "追加保守 Encoder 集合为零起始索引[15,16,17]：对 discovery 每个样本要求cosine>=0.9999、RMSE<=0.005、max_abs<=0.025。该规则是在初始组合出现偏差后提出的探索规则；留出数据不是完全未见的确认性验收集，结果不代表任务泛化。", "",
              layer_note + " 完整记录见 [layers.csv](layers.csv)。", ""]
    start = next((row for row in flat if row["variant"] == "bf16_2valid_968"), None)
    end = next((row for row in flat if row["variant"] == "bf16_2valid_968_end_control"), None)
    if start and end and number(start["model_mean_ms"]) and number(end["model_mean_ms"]):
        drift = (end["model_mean_ms"] / start["model_mean_ms"] - 1) * 100
        control_low, control_high = sorted((start["model_mean_ms"], end["model_mean_ms"]))
        lines += ["## 首尾控制", "",
                  f"相同 BF16 配置的首尾均值：{fmt(start['model_mean_ms'])}→{fmt(end['model_mean_ms'])} ms，变化 {fmt(drift)}%；首尾 p95：{fmt(start['model_p95_ms'])}→{fmt(end['model_p95_ms'])} ms。控制均值观测范围为 [{fmt(control_low)}, {fmt(control_high)}] ms，跨度 {fmt(control_high-control_low)} ms。", "",
                  "这暴露了测量顺序、温度、频率或 tactic 的组合漂移，不足以给各因素分别归因。该范围也不是统计置信区间；小幅候选收益应结合控制漂移及 p95 判断，不能宣称已证明严格的单因素因果收益。", ""]
    lines += ["## 持续负载的硬件事件与功耗", "",
              "部分配置的OC3硬件事件计数在持续负载窗口增加；未进行逐次事件与主计时的对应分析，不能排除瞬态频率/功耗限制的影响。本轮结果代表当前MAXN与动态频率环境，不是固定满频性能上限。", "",
              table(["配置", "OC3 开始→结束 / 增量", "GPU平均活跃率%", "板载三轨平均W"],
                    [[row["label_zh"], f"{fmt(row['oc3_event_cnt_start'],0)}→{fmt(row['oc3_event_cnt_end'],0)} / {fmt(row['oc3_event_cnt_delta'],0)}",
                      fmt(row["gpu_load_mean_percent"]),fmt(row["board_main_rails_mean_w"])]
                     for run,row in zip(runs,flat) if run["complete"]]), "",
              "## 失败、未完成与数据缺口", ""]
    failures = []
    for run in runs:
        if not run["complete"]:
            message = (run["error"] or {}).get("error")
            if not message and run["status"] == "部分完成（遥测无效）":
                message = "模型/动作计算有结果，但遥测样本不足2条；保留为诊断，不作为完整性能基准。"
            if not message:
                message = f"未保存 completed=true 的 result.json；returncode={run['record'].get('returncode', '未知')}"
            failures.append([run["name"], run["status"], cell(message), run["record"].get("log", str(run["directory"]))])
    reported_failure_logs = {str(row[3]) for row in failures}
    for record in combined_provenance.get("runs", []):
        if record.get("returncode", 0) != 0 and str(record.get("log", "")) not in reported_failure_logs:
            failures.append([record.get("name", "未知"), "前置阶段失败" if record.get("name") in RESERVED else "历史失败尝试",
                             f"returncode={record['returncode']}", record.get("log", "")])
    lines += [table(["配置 / 阶段", "状态", "原因", "日志"], failures) if failures else "已发现的独立候选与前置阶段没有记录失败。", ""]
    for run, row in zip(runs, flat):
        if run["complete"]:
            missing = [key for key in ("model_mean_ms", "nvmap_max_gib", "rss_max_gib", "gpu_temperature_max_c", "cpu_temperature_max_c", "openpi_cosine") if row[key] is None]
            if missing:
                problems.append(run["name"] + " 缺少指标：" + ", ".join(missing))
            errors = run["result"].get("sustained", {}).get("telemetry", {}).get("errors", [])
            problems.extend(run["name"] + " telemetry：" + str(error) for error in errors)
    problems = list(dict.fromkeys(problems))
    lines += ["\n".join("- " + cell(problem) for problem in problems) if problems else "未发现额外 JSON 读取或核心指标缺口。N/A 始终表示未获得数据，不表示零。", ""]
    if provenance.get("aliases"):
        lines += ["配置去重别名（没有再次启动独立进程，不重复计作实验）：", "",
                  table(["别名", "实际配置"], provenance["aliases"].items()), ""]
    lines += ["## 服务恢复与原始数据", ""]
    before, after = provenance.get("service_state_before", {}), provenance.get("service_state_after", {})
    restore = provenance.get("restore", {})
    units = sorted(set(before) | set(after) | set(restore))
    if units:
        lines += [table(["服务", "实验前", "实验后", "恢复命令返回码"],
                        [[unit, before.get(unit, "未记录"), after.get(unit, "未记录"), restore.get(unit, {}).get("returncode", "不适用")]
                         for unit in units]), ""]
        bad = [unit for unit in units if before.get(unit) in ("active", "activating") and
               (after.get(unit) not in ("active", "activating") or restore.get(unit, {}).get("returncode", 0) != 0)]
        if bad:
            lines += ["**恢复状态异常，需检查：" + "、".join(bad) + "。**", ""]
    else:
        lines += ["未记录服务恢复状态。", ""]
    if retries:
        lines += ["### 独立补测的服务恢复", "",
                  table(["服务", "补测前", "恢复命令返回码", "补测后状态"],
                        [[unit, state, retries.get("restore", {}).get(unit, {}).get("returncode", "不适用"),
                          retries.get("service_state_after", {}).get(unit, "未记录，需另查健康状态")]
                         for unit, state in retries.get("service_state_before", {}).items()]), "",
                  "恢复命令返回码不等价于模型 HTTP 健康检查通过。", ""]
    environment = load_json(results / "environment_after.json", problems) or {}
    if environment:
        lines += [f"独立 HTTP 健康检查：restored_services_healthy={environment.get('restored_services_healthy')}；8081/8082状态：`{cell(json.dumps(environment.get('health', {}),ensure_ascii=False))}`。",
                  f"历史合成输入的新 OpenPI 参考与旧参考核对：`{cell(json.dumps(environment.get('historical_reference_check', {}),ensure_ascii=False))}`。",
                  f"软件、固定vendor补丁验证及native扩展SHA256见 [environment_after.json](<{relative_link(results / 'environment_after.json',output)}>)。", ""]
    lines += ["- [comparison.csv](comparison.csv)：完整候选指标，含 OpenPI/BF16 全集与 discovery/holdout 精度、内存和温度。",
              "- [layers.csv](layers.csv)：逐层诊断原始数据。",
              f"- [provenance.json](<{relative_link(results / 'provenance.json', output)}>)：命令、依赖、源码哈希、进程和服务状态。", "",
              "### 独立候选原始记录", ""]
    lines += [table(["配置", "结果", "错误"],
                    [[run["name"], f"[{run['source']}](<{relative_link(run['directory'] / run['source'], output)}>)" if run["source"] else "无结果 JSON",
                      f"[error.json](<{relative_link(run['directory'] / 'error.json', output)}>)" if run["error"] else "无"] for run in runs]), "",
              "## 解释限制", "",
              "数值门限是本实验的筛选规则，不是机器人可接受误差的通用标准。高 cosine 仍可能掩盖某个动作维度或夹爪的错误；逐维与夹爪诊断保存在各 result.json。量化收益、前缀压缩和去噪步数收益可能重叠，不能直接相加。真正的任务精度需要使用匹配指令和控制频率进行仿真或真机 rollout。", ""]
    if retries:
        lines += [f"补测调度与恢复记录：[retry_provenance.json](<{relative_link(results / 'retry_provenance.json', output)}>)。", ""]
    (output / "report_zh.md").write_text("\n".join(lines), encoding="utf-8")
    return {"report": str(output / "report_zh.md"), "runs": len(runs),
            "completed": sum(run["complete"] for run in runs), "problems": problems}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = args.results.expanduser().resolve()
    if not results.is_dir():
        parser.error("--results must be an existing directory")
    summary = render(results, args.output.expanduser().resolve())
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
