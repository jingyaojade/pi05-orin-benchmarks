"""Loopback API timing; TTFT starts at HTTP request and ignores empty role deltas."""
from __future__ import annotations
import json
import time
import urllib.request


def request_json(url, body=None, timeout=600):
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
    request = urllib.request.Request(url, data=data, headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def content_delta(chunk):
    choices = chunk.get('choices') or []
    return (choices[0].get('delta',{}).get('content') or '') if choices else ''


def vla_request(url, body):
    begin = time.monotonic_ns()
    response = request_json(url+'/infer', body)
    return {'start_ns':begin, 'end_ns':time.monotonic_ns(),
            'api_wall_ms':(time.monotonic_ns()-begin)/1e6, 'response':response}


def vlm_request(url, image_data, prompt, max_tokens=96):
    payload = {'model':'qwen3.5-9b', 'messages':[{'role':'user','content':[
        {'type':'image_url','image_url':{'url':'data:image/png;base64,'+image_data}},
        {'type':'text','text':prompt}]}], 'stream':True, 'stream_options':{'include_usage':True},
        'max_tokens':max_tokens, 'temperature':0, 'seed':42, 'cache_prompt':False,
        'chat_template_kwargs':{'enable_thinking':False}}
    encoded = json.dumps(payload,ensure_ascii=False).encode()
    request = urllib.request.Request(url+'/v1/chat/completions', data=encoded,
                                    headers={'Content-Type':'application/json'})
    begin = time.monotonic_ns()
    chunks, text, first, usage, timings, finish = [], [], None, {}, {}, None
    with urllib.request.urlopen(request, timeout=900) as response:
        headers_ns = time.monotonic_ns()
        for line in response:
            if not line.startswith(b'data: '):
                continue
            item = line[6:].strip()
            if item == b'[DONE]':
                break
            value = json.loads(item)
            now = time.monotonic_ns()
            delta = content_delta(value)
            if delta:
                if first is None:
                    first = now
                text.append(delta)
            chunks.append({'received_ns':now, 'data':value})
            usage = value.get('usage') or usage
            timings = value.get('timings') or timings
            for choice in value.get('choices',[]):
                finish = choice.get('finish_reason') or finish
    end = time.monotonic_ns()
    if first is None:
        raise RuntimeError('VLM generated no visible content; inspect reasoning configuration')
    return {'start_ns':begin, 'end_ns':end, 'api_wall_ms':(end-begin)/1e6,
            'headers_ms':(headers_ns-begin)/1e6, 'ttft_ms':(first-begin)/1e6,
            'text':''.join(text), 'usage':usage, 'timings':timings,
            'finish_reason':finish, 'chunks':chunks}
