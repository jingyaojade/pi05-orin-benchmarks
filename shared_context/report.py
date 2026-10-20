#!/usr/bin/env python3
"""Generate the shared-context evaluation from fresh matched raw measurements."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

REPO=Path(__file__).resolve().parents[1]
# Load the established statistics implementation without this file shadowing it.
import importlib.util
spec=importlib.util.spec_from_file_location('dual_statistics',REPO/'dual_stack/report.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
stats=base.stats;fmt=base.fmt


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def lines(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines()]
def pair(value):return fmt(value['mean'])+' / '+fmt(value['p95'])


def summarize(root):
    meta=read(root/'provenance.json')
    if not meta.get('completed') or meta.get('monitor_error'):raise RuntimeError('Incomplete variant: '+str(root))
    requests=lines(root/'requests.jsonl');telemetry=lines(root/'telemetry.jsonl');phases=read(root/'phases.json')
    output={'metadata':meta,'phases':{}}
    for phase in phases:
        name=phase['name'];samples=[r for r in telemetry if phase['start_ns']<=r['monotonic_ns']<=phase['end_ns']]
        row={'duration_s':(phase['end_ns']-phase['start_ns'])/1e9,'telemetry_samples':len(samples),'requests':{}}
        for kind in ['vla','vlm']:
            req=[r for r in requests if r['phase']==name and r['kind']==kind and not r['warmup']]
            data={'n':len(req),'completed_per_second':len(req)/row['duration_s'],
                  'api_wall_ms':stats([r['api_wall_ms'] for r in req])}
            if kind=='vla':
                for key in ['http_decode_ms','queue_wait_ms','setup_ms','input_prepare_upload_ms','model_wall_ms',
                            'model_stream_ms','download_postprocess_ms','policy_wall_ms']:
                    data[key]=stats([r['response']['timings'][key] for r in req])
                data['all_outputs_instance_exact']=all(r['validation']['instance_exact_match'] for r in req) if req else None
            else:
                for key in ['ttft_ms','headers_ms']:data[key]=stats([r[key] for r in req])
                for key in ['prompt_ms','predicted_ms','prompt_n','predicted_n','cache_n','predicted_per_second']:
                    data[key]=stats([r['timings'].get(key) for r in req])
                total_ms=sum(r['timings'].get('predicted_ms',0) for r in req)
                data['decode_tokens_per_second']=sum(max(0,r['timings'].get('predicted_n',0)-1) for r in req)/(total_ms/1000) if total_ms else None
                data['length_limited_responses']=sum(r['finish_reason']=='length' for r in req)
                data['example_outputs']=[r['text'] for r in req[:2]]
            data['trigger_lateness_ms']=stats([r.get('start_lateness_ms') for r in req])
            row['requests'][kind]=data
        row['ram_used_gib']=stats([r['ram_used_mib']/1024 for r in samples])
        row['mem_available_gib']=stats([r['mem_available_bytes']/2**30 for r in samples])
        row['swap_used_mib']=stats([r['system_swap_used_bytes']/2**20 for r in samples])
        row['power_w']=stats([r.get('main_rails_sum_w') for r in samples])
        row['gpu_active_percent']=stats([r.get('gpu_load_percent') for r in samples])
        cpu=[]
        for r in samples:
            match=re.search(r'CPU \[([^]]+)\]',r['raw'])
            if match:
                cores=[int(x) for x in re.findall(r'(\d+)%',match[1])]
                if cores:cpu.append(sum(cores)/len(cores))
        row['cpu_core_average_percent']=stats(cpu)
        row['temperatures_c']={key:stats([r['temperatures_c'].get(key) for r in samples]) for key in ['cpu','gpu','tj','tboard']}
        row['processes']={}
        for name_pid in meta.get('pids',{}):
            row['processes'][name_pid]={key:stats([r.get('processes',{}).get(name_pid,{}).get(key) for r in samples])
                                      for key in ['rss_bytes','pss_bytes','uss_bytes','nvmap_bytes','lifetime_rss_hwm_bytes','swap_bytes']}
        row['monitor_collection_ms']=stats([r.get('collection_ms') for r in samples])
        row['oc_event_delta']={key:samples[-1]['oc_counters'][key]-samples[0]['oc_counters'][key]
                               for key in samples[0].get('oc_counters',{})} if len(samples)>1 else {}
        if name in ['dual_resident_idle','vla_dual_resident','vlm_dual_resident','concurrent_saturated','periodic_2hz_0p2hz']:
            if row['power_w']['n']<2:raise RuntimeError('Missing power samples: '+name)
        output['phases'][phase['name']]=row
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('results',type=Path);args=p.parse_args();root=args.results
    meta=read(root/'provenance.json')
    if not meta.get('completed'):raise RuntimeError('Refusing to publish incomplete experiment')
    output={'metadata':meta,'variants':{name:summarize(root/name) for name in meta['variants']}}
    if (root/'shared_context_16k').exists():output['memory_probe_16k']=summarize(root/'shared_context_16k')
    names={'two_process':'双进程对照','shared_stream':'同进程共享上下文 / stream同步','shared_device':'同进程 / device同步诊断'}
    scenarios={'vla_dual_resident':'双常驻，仅VLA推理','vlm_dual_resident':'双常驻，仅VLM推理',
               'concurrent_saturated':'同时持续推理','periodic_2hz_0p2hz':'目标VLA 2Hz / VLM 0.2Hz'}
    a=output['variants']['two_process']['phases']['concurrent_saturated']
    b=output['variants']['shared_stream']['phases']['concurrent_saturated']
    speedup=a['requests']['vla']['model_wall_ms']['mean']/b['requests']['vla']['model_wall_ms']['mean']
    vlm_ratio=a['requests']['vlm']['api_wall_ms']['mean']/b['requests']['vlm']['api_wall_ms']['mean']
    periodic_a=output['variants']['two_process']['phases']['periodic_2hz_0p2hz']['requests']['vla']
    periodic_b=output['variants']['shared_stream']['phases']['periodic_2hz_0p2hz']['requests']['vla']
    text=['# π0.5 + Qwen3.5-9B：同进程共享CUDA上下文版测评','',
          f"本轮重新测量双进程对照及共享上下文实现。持续并发时VLA模型均值由{a['requests']['vla']['model_wall_ms']['mean']:.2f} ms变为{b['requests']['vla']['model_wall_ms']['mean']:.2f} ms，前者/后者为{speedup:.3f}；VLM完整短回答由{a['requests']['vlm']['api_wall_ms']['mean']:.2f} ms变为{b['requests']['vlm']['api_wall_ms']['mean']:.2f} ms，前者/后者为{vlm_ratio:.3f}。",
          '比值大于1代表新版本更快，小于1代表变慢；这些是本次设备/输入/顺序下的结果，不预设同进程一定加速。','',
          '## 实现与上下文证据','',
          '保留同一FlashRT BF16、Qwen3.5-9B Q5_K_M及FP16 mmproj、固定llama.cpp commit和模型SHA。'
          '共享版本通过ctypes.CDLL调用现有libllama-server-impl.so中的完整原生服务逻辑，阻塞原生调用释放GIL。'
          'C++服务线程和VLA工作线程显式绑定Torch建立的同一个CUDA上下文；仍提供相同HTTP/SSE接口，但没有启动独立llama-server模型进程。','',
          'VLA原完整CUDA Graph不变，将请求阶段设备级同步改为模型自身stream同步；下载及FP32转换也放入自身stream。'
          '现有C++后端内部同步未全面重写，因此共享context不等于保证所有kernel硬件重叠。','',
          'CUPTI诊断在单独的特权预检中记录实际VLA graph launch和Qwen launch的上下文身份；'
          '结果见evidence/context-validation.json及context-audit.json。主性能测试使用普通用户，不开启CUPTI callbacks。'
          '上下文身份记录证明使用同一个context，没有测出每个kernel的重叠率或Tensor Core/DRAM带宽利用率。','',
          '共享版本当前固定为fixture任务prompt，尚无更换任务的配置入口或维护调度API；单纯重启不会更换任务。'
          '相机图像和noise可逐请求更新，未提供noise时仍重新采样。该限制在接口显式返回409，未静默复用错误任务。','',
          '## 工作负载与边界','',
          '两路有效VLA 224×224 RGB、一路masked padding、文本容量200、10轮去噪、每次10×7动作；'
          'VLM交替两张512×384合成图片，temperature0、seed42、thinking关闭、prompt cache关闭、最多96输出tokens，context8192/单slot。'
          'VLA先内部预热10次；各变体再排除5次VLA与2次VLM客户端预热。','',
          '每个变体先空闲、加载和预热，再双常驻空闲、仅VLA、仅VLM、持续并发、指定触发频率。'
          '持续并发每模型最多一个在途请求，完成后立即触发下一次；指定频率为VLA 2Hz和VLM 0.2Hz，错过触发则跳过、不无限排队。'
          '这不是物理机器人的关节控制频率。模型不自动按该频率后台运行，频率由本轮客户端控制。','',
          'API计时为Orin loopback，预先完成PNG/base64编码，不含相机采集/客户端图像编码/外部网络RTT。'
          'VLA API含客户端JSON序列化，VLM在构造请求后计时；VLA模型为GPU输入就绪到原完整图完成，包含竞争时等待。'
          'TTFT从发送请求到首个非空可见content；prompt_ms不等于独立视觉编码耗时。decode速率按llama.cpp的N−1个生成间隔计。','',
          '## 延时与完成频率','',
          '| 实现 / 场景 | VLA n / 完成Hz | VLA模型 mean / p95 ms | VLA API mean / p95 ms | VLM n / 完成Hz | VLM首字 mean / p95 ms | VLM完整 mean / p95 ms | decode tok/s |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    for variant,data in output['variants'].items():
        for scenario,label in scenarios.items():
            r=data['phases'][scenario];v=r['requests']['vla'];l=r['requests']['vlm']
            text.append(f"| {names[variant]} / {label} | {v['n']} / {v['completed_per_second']:.3f} | {pair(v['model_wall_ms'])} | {pair(v['api_wall_ms'])} | {l['n']} / {l['completed_per_second']:.3f} | {pair(l['ttft_ms'])} | {pair(l['api_wall_ms'])} | {fmt(l['decode_tokens_per_second'])} |")
    text+=['','## 结果判断','',
           f"持续并发VLM完整回答耗时变化为{(1/vlm_ratio-1)*100:+.1f}%，decode由{a['requests']['vlm']['decode_tokens_per_second']:.2f}变为{b['requests']['vlm']['decode_tokens_per_second']:.2f} tok/s；VLA模型均值变化为{(1/speedup-1)*100:+.2f}%。"
           f"VLA模型p95由{a['requests']['vla']['model_wall_ms']['p95']:.2f}变为{b['requests']['vla']['model_wall_ms']['p95']:.2f} ms。指定2Hz/0.2Hz场景中，"
           f"VLA实得完成率由{periodic_a['completed_per_second']:.3f}变为{periodic_b['completed_per_second']:.3f} Hz。不能只用均值宣称控制循环更快。",'',
           f"共享版/对照并发三轨平均功耗分别{b['power_w']['mean']:.2f}/{a['power_w']['mean']:.2f}W；GPU最高{b['temperatures_c']['gpu']['max']:.2f}/{a['temperatures_c']['gpu']['max']:.2f}°C。"
           '固定先对照后共享的顺序使起始温度不同，不能把温差全部归因于代码。'
           '共享context与stream同步是组合改动，本轮没有拆分两者各自的因果贡献。'
           '共享版VLA并发模型耗时仍高于自身独自推理阶段，说明消除进程/上下文边界不足以消除竞争；'
           '计算单元、带宽及调度各自贡献仍需kernel级证据。','',
           '候选systemd服务已单独切换验证三个端口同一PID、HTTP/SSE/WS、固定输入一致性和图像变化响应，'
           '并验证不支持的任务变更返回409。测试后回退原双进程服务，恢复其动态任务接口；'
           '证据见evidence/lifecycle-validation.json、deployed-api-smoke.json和restored-api-smoke.json。','']
    text+=['','## VLA步骤（mean / p95 ms）','',
           '| 实现 / 场景 | HTTP解码 | 输入准备/上传 | 模型墙钟 | 模型stream事件跨度 | 下载/动作处理 |','|---|---:|---:|---:|---:|---:|']
    for variant,data in output['variants'].items():
        for scenario,label in scenarios.items():
            r=data['phases'][scenario]['requests']['vla']
            if r['n']:text.append('| '+names[variant]+' / '+label+' | '+' | '.join(pair(r[k]) for k in ['http_decode_ms','input_prepare_upload_ms','model_wall_ms','model_stream_ms','download_postprocess_ms'])+' |')
    text+=['','## VLM内部步骤（mean / p95）','',
           '| 实现 / 场景 | prompt ms | decode ms | prompt tokens | 输出tokens | 复用cache tokens | 长度截断数 |','|---|---:|---:|---:|---:|---:|---:|']
    for variant,data in output['variants'].items():
        for scenario,label in scenarios.items():
            r=data['phases'][scenario]['requests']['vlm']
            if r['n']:text.append('| '+names[variant]+' / '+label+' | '+' | '.join(pair(r[k]) for k in ['prompt_ms','predicted_ms','prompt_n','predicted_n','cache_n'])+f" | {r['length_limited_responses']} |")
    text+=['','## 内存、温度与功耗','',
           '共享进程只有一份PID记账，表中只记一次，不能把它再拆成两个进程相加。'
           '双进程NvMap与RSS列逐PID显示；它们和系统RAM、PSS、Torch allocator存在UMA重叠，不相加作为物理占用。'
           '均为1Hz采样，可能漏掉短时峰值。','',
           '| 实现 / 场景 | 时长s / 样本数 | RAM max / MemAvailable min GiB | PID NvMap max GiB | PID RSS max GiB | GPU / CPU max °C | 三轨功耗 mean / max W | GPU活跃 / CPU各核平均 % | OC3增量 |',
           '|---|---:|---:|---|---|---:|---:|---:|---:|']
    for variant,data in output['variants'].items():
        for phase,r in data['phases'].items():
            if phase not in scenarios and phase not in ['dual_resident_idle','load_models']:continue
            def mem(key):return '; '.join(k+'='+fmt(v[key]['max']/2**30 if v[key]['max'] is not None else None) for k,v in r['processes'].items())
            text.append(f"| {names[variant]} / {scenarios.get(phase,phase)} | {r['duration_s']:.1f} / {r['telemetry_samples']} | {fmt(r['ram_used_gib']['max'])} / {fmt(r['mem_available_gib']['min'])} | {mem('nvmap_bytes')} | {mem('rss_bytes')} | {fmt(r['temperatures_c']['gpu']['max'])} / {fmt(r['temperatures_c']['cpu']['max'])} | {fmt(r['power_w']['mean'])} / {fmt(r['power_w']['max'])} | {fmt(r['gpu_active_percent']['mean'])} / {fmt(r['cpu_core_average_percent']['mean'])} | {r['oc_event_delta'].get('oc3_event_cnt','N/A')} |")
    text+=['','三轨为VDD_GPU_SOC+VDD_CPU_CV+VIN_SYS_5V0，不重复加DDR VDDQ或SUMMATION；'
           '包含系统背景，是板载传感器合计，不是交流插座功率计。GR3D是GPU活跃时间比例，不是峰值FLOPs或Tensor Core利用率。'
           'CPU为各在线核心百分比的算术平均，频率动态变化。'
           '[NVIDIA指标定义](https://docs.nvidia.com/jetson/archives/r36.3/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html)','',
           '## 如何有效使用剩余GPU内存','',
           '**容量占用（GiB）与内存带宽利用率（GB/s或EMC负载）是不同指标。额外分配空tensor只提高占用数字，不提高速度。**','',
           '| 用途 | 可能获得的能力 | 代价/本轮状态 |','|---|---|---|',
           '| 增大KV/context容量 | 更长对话或输入上下文 | 本轮增加16K容量探针；短请求不会仅因此变快 |',
           '| 提高Qwen语言权重精度，例如Q8或FP16 | 减少量化误差的候选方向 | 更多权重读取和存储，需另测精度/速度；本轮未下载新权重 |',
           '| 更多VLM图像tokens/分辨率 | 保留更多图像细节 | 更多视觉/prefill计算；VLA仍使用已验证224×224规格 |',
           '| 批量/更多请求槽位 | 可能提高吞吐和权重复用 | 增加KV与排队，可能恶化VLA尾延时；本轮不把吞吐当单请求加速 |',
           '| 缓存不变前缀/可复用特征、预分配复用缓冲 | 减少重复计算或分配 | 必须正确失效，不能对变化的相机帧无条件复用 |','']
    if 'memory_probe_16k' in output:
        x=output['variants']['shared_stream']['phases']['dual_resident_idle']
        y=output['memory_probe_16k']['phases']['dual_resident_idle']
        kv8=x['processes']['shared']['nvmap_bytes']['max']/2**30;kv16=y['processes']['shared']['nvmap_bytes']['max']/2**30
        text += [f"相同单slot、同一Q5_K_M/FP16 vision下，8K/16K双模型常驻PID NvMap采样最大值分别为 **{kv8:.3f}/{kv16:.3f} GiB**，差值 **{kv16-kv8:.3f} GiB**。这是整个共享进程分配变化，包含KV和其他后端缓冲，并非逐buffer归因。",
                 '16K探针仍使用相同短图片问题，只验证加载和短请求；没有执行16K长度输入的最大长度正确性或速度评测。','']
    text+=['## 正确性、限制和复现','',
           '所有固定VLA输出在各实例内逐元素稳定，VLM无长度截断。逐次验证和各场景样本数见原始请求与summary.json。'
           '加载的原llama动态库SHA保存在evidence/loaded-libraries-sha256.json，可与原部署build-artifacts.json逐路径比对。','',
           '各VLA实例固定输入输出要求逐元素稳定；相对官方baseline的误差保存在每个变体provenance.vla_metadata.baseline_accuracy。'
           '跨进程/重启调优可能产生数值差异，不强制两个实例互相bitwise相同。VLM示例和逐token响应保留，合成图片不代表机器人任务成功率。','',
           '测试按固定顺序执行，没有随机交叉/固定温度/锁频；保留原功耗与风扇策略，不代表长期热稳态。'
           '预检与性能采样独立，主测未校准1Hz监控开销。内存峰值是采样下界，OC计数不增长不足以排除全部限频。','',
           '新实现与启动/恢复方式见[shared_context/README.md](../../shared_context/README.md)。'
           '原始请求、遥测、环境、源码SHA、加载命令、服务恢复状态及上下文诊断均保留。'
           '报告后处理源码和输入SHA单独保存在postprocessing.json。','',
           '## 时间轴','', '![双进程与共享上下文实测时间轴](timeline.png)','']
    (root/'report_zh.md').write_text('\n'.join(text),encoding='utf-8')
    (root/'summary.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    flat=[]
    def flatten(value,path):
        if isinstance(value,dict):
            for key,item in value.items():flatten(item,path+[key])
        elif isinstance(value,(int,float)) and not isinstance(value,bool):flat.append({'path':'.'.join(path),'value':value})
    flatten(output['variants'],['variants'])
    if 'memory_probe_16k' in output:flatten(output['memory_probe_16k'],['memory_probe_16k'])
    with (root/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['path','value']);writer.writeheader();writer.writerows(flat)
    paths=[root/'provenance.json']+[root/name/file for name in meta['variants']+(['shared_context_16k'] if 'memory_probe_16k' in output else [])
                                 for file in ['provenance.json','phases.json','requests.jsonl','telemetry.jsonl']]
    write_meta={'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'inputs_sha256':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}}
    (root/'postprocessing.json').write_text(json.dumps(write_meta,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(root/'report_zh.md'),'vla_ratio_old_over_new':speedup,'vlm_ratio_old_over_new':vlm_ratio}))


if __name__=='__main__':main()
