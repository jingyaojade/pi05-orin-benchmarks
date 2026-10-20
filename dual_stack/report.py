#!/usr/bin/env python3
"""Generate the combined latency/memory/thermal/power report from raw observations."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def stats(values):
    values=sorted(float(x) for x in values if x is not None)
    if not values:return {'n':0,'mean':None,'p95':None,'min':None,'max':None}
    index=(len(values)-1)*.95;low=math.floor(index);high=math.ceil(index)
    return {'n':len(values),'mean':sum(values)/len(values),'p95':values[low]+(values[high]-values[low])*(index-low),
            'min':values[0],'max':values[-1]}


def fmt(value,digits=2):return 'N/A' if value is None else f'{value:.{digits}f}'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('results',type=Path);args=p.parse_args()
    root=args.results
    meta=json.loads((root/'provenance.json').read_text(encoding='utf-8'))
    if meta.get('completed') is not True:
        raise RuntimeError('Incomplete suite; refusing to publish a completed benchmark')
    phases=json.loads((root/'phases.json').read_text(encoding='utf-8'))
    requests=[json.loads(x) for x in (root/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    telemetry=[json.loads(x) for x in (root/'telemetry.jsonl').read_text(encoding='utf-8').splitlines()]
    if meta.get('monitor_error'):
        raise RuntimeError('Telemetry collection failed: '+meta['monitor_error'])
    output={'schema_version':1,'metadata':meta,'phases':{}}
    csv_rows=[]
    names={'idle_system':'系统空闲基线（原π0.5已停）','vla_standalone':'VLA 单独常驻与推理',
           'vlm_standalone':'VLM 单独常驻与推理','dual_resident_idle':'双模型常驻空闲',
           'vla_dual_resident':'双常驻，仅 VLA 推理','vlm_dual_resident':'双常驻，仅 VLM 推理',
           'concurrent_saturated':'双模型同时持续推理'}
    for phase in phases:
        name=phase['name'];samples=[x for x in telemetry if phase['start_ns']<=x['monotonic_ns']<=phase['end_ns']]
        row={'duration_s':(phase['end_ns']-phase['start_ns'])/1e9,'telemetry_samples':len(samples),'requests':{}}
        for kind in ('vla','vlm'):
            req=[x for x in requests if x['phase']==name and x['kind']==kind and not x['warmup']]
            data={'n':len(req),'api_wall_ms':stats([x['api_wall_ms'] for x in req])}
            if kind=='vla':
                for key in ('http_decode_ms','queue_wait_ms','input_prepare_upload_ms','model_wall_ms','model_stream_ms',
                            'download_postprocess_ms','policy_wall_ms'):
                    data[key]=stats([x['response']['timings'][key] for x in req])
                data['all_outputs_instance_exact']=all(x['validation']['instance_exact_match'] for x in req) if req else None
            else:
                data['ttft_ms']=stats([x['ttft_ms'] for x in req])
                for key in ('prompt_ms','predicted_ms','prompt_n','predicted_n','cache_n','predicted_per_second'):
                    data[key]=stats([x['timings'].get(key) for x in req])
                seconds=sum(x['timings'].get('predicted_ms',0) for x in req)/1000
                # llama.cpp produces the first token during prompt evaluation;
                # predicted_ms covers the following N-1 decoding intervals.
                data['decode_tokens_per_s']=sum(max(x['timings'].get('predicted_n',0)-1,0) for x in req)/seconds if seconds else None
                data['length_limited_responses']=sum(x.get('finish_reason')=='length' for x in req)
                data['example_outputs']=[x['text'] for x in req[:2]]
            row['requests'][kind]=data
            for metric,value in data.items():
                if isinstance(value,dict) and 'n' in value:
                    for statistic in ('mean','p95','min','max'):
                        csv_rows.append({'phase':name,'group':kind,'metric':metric,'statistic':statistic,'value':value[statistic]})
        row['power_w']=stats([x.get('main_rails_sum_w') for x in samples])
        row['ram_used_mib']=stats([x.get('ram_used_mib') for x in samples])
        row['mem_available_gib']=stats([x.get('mem_available_bytes',0)/2**30 for x in samples])
        row['swap_used_mib']=stats([x.get('system_swap_used_bytes',0)/2**20 for x in samples])
        row['temperatures_c']={sensor:stats([x.get('temperatures_c',{}).get(sensor) for x in samples]) for sensor in ('cpu','gpu','tj','tboard')}
        row['rails_w']={rail:stats([x.get('power_rails_w',{}).get(rail) for x in samples])
                        for rail in ('VDD_GPU_SOC','VDD_CPU_CV','VIN_SYS_5V0','VDDQ_VDD2_1V8AO')}
        row['processes']={}
        for kind in ('vla','vlm'):
            row['processes'][kind]={metric:stats([x.get('processes',{}).get(kind,{}).get(metric) for x in samples])
                                     for metric in ('rss_bytes','pss_bytes','uss_bytes','nvmap_bytes','lifetime_rss_hwm_bytes','swap_bytes')}
        row['gpu_load_percent']=stats([x.get('gpu_load_percent') for x in samples])
        row['oc_event_delta'] = {
            key: samples[-1].get('oc_counters',{}).get(key,0)-samples[0].get('oc_counters',{}).get(key,0)
            for key in (samples[0].get('oc_counters',{}) if len(samples)>1 else {})}
        pairs=[(x['monotonic_ns']/1e9,x.get('main_rails_sum_w')) for x in samples if x.get('main_rails_sum_w') is not None]
        row['energy_sampled_j']=sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(pairs,pairs[1:]))
        row['energy_sampled_span_s']=pairs[-1][0]-pairs[0][0] if len(pairs)>1 else 0
        output['phases'][name]=row
        for metric in ('power_w','ram_used_mib','mem_available_gib','swap_used_mib','gpu_load_percent'):
            for statistic in ('mean','p95','min','max'):
                csv_rows.append({'phase':name,'group':'system','metric':metric,'statistic':statistic,'value':row[metric][statistic]})
        for sensor,value in row['temperatures_c'].items():
            for statistic in ('mean','p95','min','max'):
                csv_rows.append({'phase':name,'group':'temperature_c','metric':sensor,'statistic':statistic,'value':value[statistic]})
        for kind,metrics in row['processes'].items():
            for metric,value in metrics.items():
                for statistic in ('mean','p95','min','max'):
                    csv_rows.append({'phase':name,'group':kind,'metric':metric,'statistic':statistic,'value':value[statistic]})
        if name in names and (row['power_w']['n']<2 or row['temperatures_c']['tj']['n']<2):
            raise RuntimeError('Insufficient power/temperature samples in '+name)
    standalone=output['phases']['vla_standalone']['requests']['vla']['model_wall_ms']['mean']
    concurrent=output['phases']['concurrent_saturated']
    concurrent_vla=concurrent['requests']['vla']['model_wall_ms']['mean']
    overview=[
        '## 结论','',
        f"两套模型可同时常驻。只增加常驻而不同时计算时，延时接近各自单独运行；同时持续推理时，VLA模型均值从{standalone:.2f} ms升到{concurrent_vla:.2f} ms（{concurrent_vla/standalone:.2f}×）。",
        f"并发持续{concurrent['duration_s']:.1f}秒，VLM首字均值{concurrent['requests']['vlm']['ttft_ms']['mean']:.2f} ms、完整短回答均值{concurrent['requests']['vlm']['api_wall_ms']['mean']:.2f} ms。",
        f"并发系统RAM观测峰值{concurrent['ram_used_mib']['max']/1024:.2f} GiB，可用内存最低{concurrent['mem_available_gib']['min']:.2f} GiB；GPU最高{concurrent['temperatures_c']['gpu']['max']:.2f} °C；板载三轨功耗均值/采样最大值{concurrent['power_w']['mean']:.2f}/{concurrent['power_w']['max']:.2f} W。",
        f"全部已完成场景的系统Swap观测最大值为{max(r['swap_used_mib']['max'] or 0 for r in output['phases'].values()):.2f} MiB；这是系统总量，不能归为模型独占。",
        '本轮主要代价是同时计算时的延时增长；未进一步量化计算、内存带宽和GPU调度各自贡献。若动作更新延时优先，可先让VLM按事件/低频触发；这项调度建议尚未实施测试。','']
    text=['# π0.5 + Qwen3.5-9B：Orin 双模型部署实测','',*overview,
          '模型：π0.5 LIBERO / FlashRT BF16；Qwen3.5-9B / Q5_K_M 语言权重 + FP16 视觉模块。'
          'batch=1，VLM context=8192、并发槽位1、温度0、最多96个输出token、关闭thinking和prompt缓存。','',
          'VLA输入复用已提交的合成双相机/任务/固定噪声，输出10×7，10轮去噪。'
          'VLM使用两张512×384程序绘制桌面图交替请求；只用于性能和看图接口检查，不代表真实机器人成功率。','',
          '## 延时','',
          'API为Orin本机loopback HTTP。VLA纯模型边界是输入CUDA-ready到原完整生产图输出完成，'
          '不含上传、下载；服务为拆分计时增加边界同步，与历史未插桩policy并非同一采样口径。'
          'VLM TTFT从发送HTTP请求到收到第一个非空可见文本，包含图像处理、prefill、排队与传输。'
          'prompt/decode为llama.cpp回传内部timings，prompt_ms不能未经审计就当作独立视觉编码耗时。','',
           '相机采集和客户端PNG/base64编码均在测试前完成、不计入API时间。'
           'VLA API计时含客户端JSON序列化；VLM在构造好HTTP请求后开始计时，不含该序列化。','',
           'decode tok/s与该llama.cpp版本一致：首token在prompt阶段产生，用后续N−1个decode间隔除以predicted_ms。'
           'API完整回答时间同时保留，不能用纯decode速率推断整次请求耗时。','',
          '| 场景 | VLA n | VLA API mean / p95 ms | VLA 模型 mean / p95 ms | VLM n | VLM TTFT mean / p95 ms | VLM 完整 mean / p95 ms | decode tok/s |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    def pair(value):return fmt(value['mean'])+' / '+fmt(value['p95'])
    for name,label in names.items():
        r=output['phases'].get(name)
        if r is None:continue
        a,b=r['requests']['vla'],r['requests']['vlm']
        text.append(f"| {label} | {a['n']} | {pair(a['api_wall_ms'])} | {pair(a.get('model_wall_ms',stats([])))} | {b['n']} | {pair(b.get('ttft_ms',stats([])))} | {pair(b['api_wall_ms'])} | {fmt(b.get('decode_tokens_per_s'))} |")
    text+=['','## VLA 分步骤延时（mean / p95 ms）','',
           '| 场景 | HTTP解码 | 输入准备+上传 | 原模型 | 下载+动作处理 |','|---|---:|---:|---:|---:|']
    for name,label in names.items():
        a=output['phases'].get(name,{}).get('requests',{}).get('vla',{})
        if a.get('n'):
            text.append('| '+label+' | '+' | '.join(pair(a[k]) for k in ('http_decode_ms','input_prepare_upload_ms','model_wall_ms','download_postprocess_ms'))+' |')
    text+=['','## VLM 内部计时（mean / p95）','',
           '| 场景 | prompt ms | decode ms | prompt tokens | generated tokens | reused cache tokens | 长度截断次数 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,label in names.items():
        b=output['phases'].get(name,{}).get('requests',{}).get('vlm',{})
        if b.get('n'):
            text.append('| '+label+' | '+' | '.join(pair(b[k]) for k in ('prompt_ms','predicted_ms','prompt_n','predicted_n','cache_n'))+f" | {b['length_limited_responses']} |")
    text+=['','## 内存、温度与板载功耗','',
           '统一内存的RSS/PSS、NvMap、全设备/系统RAM记账存在重叠，不能相加。'
           '下表为1 Hz观察最大值，可能漏掉短暂分配；加载阶段另列。'
           'MemAvailable为Linux估计可用容量；RAM为tegrastats当前使用量，不是两模型独占物理内存。','',
           '| 场景 | 时长s / 采样数 | 系统RAM max GiB | MemAvailable min GiB | VLA NvMap / RSS max GiB | VLM NvMap / RSS max GiB | CPU / GPU / tj max °C | 三轨功耗 mean / max W |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for phase in phases:
        name=phase['name'];r=output['phases'][name]
        if name not in names and not name.startswith('load_'):continue
        def memory(kind,key):
            value=r['processes'][kind][key]['max'];return None if value is None else value/2**30
        temp=' / '.join(fmt(r['temperatures_c'][k]['max']) for k in ('cpu','gpu','tj'))
        text.append(f"| {names.get(name,name)} | {r['duration_s']:.1f} / {r['telemetry_samples']} | {fmt(r['ram_used_mib']['max']/1024 if r['ram_used_mib']['max'] is not None else None)} | {fmt(r['mem_available_gib']['min'])} | {fmt(memory('vla','nvmap_bytes'))} / {fmt(memory('vla','rss_bytes'))} | {fmt(memory('vlm','nvmap_bytes'))} / {fmt(memory('vlm','rss_bytes'))} | {temp} | {fmt(r['power_w']['mean'])} / {fmt(r['power_w']['max'])} |")
    text+=['','功耗定义：VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0 的当前读数合计。'
           '**VDDQ已包含在VIN_SYS_5V0，不重复相加；SUMMATION也不重复相加。**'
           '这是板载电源轨合计近似值，含系统背景负载，未用外接功率计测交流插座功耗。'
           'mean不是扣除空闲后的模型净功耗；逐轨数据和采样区间能量保留在summary.json。',
           '[NVIDIA AGX Orin电源轨定义](https://docs.nvidia.com/jetson/archives/r36.3/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#software-based-power-consumption-modeling)','',
           '## 硬件事件计数','',
           '记录soctherm-oc-event的原始OC计数，不能仅凭计数不增长推断完全没有热/电限频。'
           '每场景首末采样的差值保存在summary.json；跨场景边界还可核对environment-before/after.json。','',
           '| 场景 | OC1 增量 | OC2 增量 | OC3 增量 |','|---|---:|---:|---:|',
           *[f"| {names.get(name,name)} | {r['oc_event_delta'].get('oc1_event_cnt','N/A')} | {r['oc_event_delta'].get('oc2_event_cnt','N/A')} | {r['oc_event_delta'].get('oc3_event_cnt','N/A')} |" for name,r in output['phases'].items()],
           '',
           'AGX Orin 64GB的NVIDIA默认定义将OC3对应CPU_CV_GPU_SOC瞬时功耗限制事件；'
           '1Hz功耗采样可能漏掉触发事件的瞬时尖峰，计数不能换算为限频持续时间。'
           '[NVIDIA OC定义](https://docs.nvidia.com/jetson/archives/r36.3/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#overcurrent-event-status)','',
           '## 数值与运行边界','',
           'VLA每个进程分别建立first-output reference，所有测量输出要求与该实例逐元素一致。'
           '跨进程GEMM调优可能有数值差异，不要求两个实例互相bitwise一致；相对官方baseline的误差记录在provenance的vla_instances。'
           'VLM文字输出和实际输出token数均保存；这些合成图片没有构成任务精度评测。','']
    for name,label in names.items():
        b=output['phases'].get(name,{}).get('requests',{}).get('vlm',{})
        if b.get('example_outputs'):
            text.append('**'+label+' 示例输出：**')
            text += ['','```text',*b['example_outputs'],'```','']
    text+=['## 时间轴','', '![功耗、温度、内存和逐请求延时](timeline.png)','',
           '图表由plot_results.py读取原始采样生成，灰色区域为加载；不插值补造温度缺失值。','',
           '## 限制与复现','',
           '场景按固定顺序执行，未做随机交叉或恒温稳态控制；温度、频率可能随顺序漂移。'
           '同时持续推理使用默认GPU调度，无任务优先级保证；测试持续时间不代表长时间热稳态认证。'
           '后台监控统一为1 Hz，未单独校准其开销；未改功耗模式、锁频或风扇设置。',
           '完整命令、源码与模型SHA、原服务状态见provenance.json；requests.jsonl保存全部SSE/动作响应，'
           'telemetry.jsonl保存带单调时间的原始tegrastats、proc、NvMap、MemAvailable和Swap。',
           'GPU运行时脚本快照SHA见provenance；后处理代码随后完善，其实际SHA与输入SHA单独保存在postprocessing.json。',
           '新服务和复现说明见[dual_stack/README.md](../../dual_stack/README.md)。','']
    (root/'report_zh.md').write_text('\n'.join(text),encoding='utf-8')
    (root/'summary.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (root/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=['phase','group','metric','statistic','value']);writer.writeheader();writer.writerows(csv_rows)
    processing={'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'inputs_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                                 for name in ['provenance.json','phases.json','requests.jsonl','telemetry.jsonl']},
                'note':'Postprocessing is separate from GPU benchmark execution; this hash identifies the report generator.'}
    (root/'postprocessing.json').write_text(json.dumps(processing,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(root/'report_zh.md'),'phases':len(phases),'requests':len(requests),'telemetry':len(telemetry)}))


if __name__ == '__main__':main()
