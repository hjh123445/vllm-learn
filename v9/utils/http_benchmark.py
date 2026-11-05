"""Closed-loop HTTP streaming load; no model, torch, tokenizer or private data."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import math
from pathlib import Path
import time
from urllib.parse import urlsplit


def percentiles(values):
    values = sorted(values)
    return {str(q): values[round((len(values)-1)*q/100)] for q in (50, 95, 99)} if values else {}


def server_profile_status(base_url, timeout):
    """Our /health identifies diagnostic servers; other servers may not."""
    parsed=urlsplit(base_url)
    cls=http.client.HTTPSConnection if parsed.scheme=='https' else http.client.HTTPConnection
    conn=cls(parsed.hostname,parsed.port,timeout=timeout)
    try:
        conn.request('GET','/health')
        response=conn.getresponse()
        value=json.loads(response.read()).get('profiled_run')
        return value if type(value) is bool else None
    except (OSError,ValueError,AttributeError):
        return None
    finally:
        conn.close()


def request(base_url, payload, index, timeout):
    parsed = urlsplit(base_url)
    cls = http.client.HTTPSConnection if parsed.scheme == 'https' else http.client.HTTPConnection
    conn = cls(parsed.hostname, parsed.port, timeout=timeout)
    started = time.perf_counter()
    record = dict(index=index, request=payload, started=started, success=False)
    chunks = []
    try:
        endpoint = parsed.path.rstrip('/') + '/chat/completions'
        conn.request('POST', endpoint, json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                     {'Content-Type': 'application/json'})
        response = conn.getresponse()
        record['http_status'] = response.status
        if response.status != 200:
            record['error'] = response.read().decode('utf-8', errors='replace')
        else:
            done = False
            for line in response:
                if not line.startswith(b'data:'):
                    continue
                data = line[5:].strip()
                if data == b'[DONE]':
                    done = True
                    break
                item = json.loads(data)
                record['id'] = item.get('id', record.get('id'))
                if 'error' in item:
                    record['error'] = item['error']
                if item.get('usage') is not None:
                    record['usage'] = item['usage']
                choices = item.get('choices') or []
                text = choices[0].get('delta', {}).get('content', '') if choices else ''
                if text:
                    chunks.append(dict(at=time.perf_counter(), content=text))
            response.read()
            record['success'] = done and 'error' not in record
            if not done:
                record['error'] = 'stream ended without DONE'
    except Exception as exc:
        record['error'] = repr(exc)
    finally:
        record['finished'] = time.perf_counter()
        conn.close()
    record['e2e_s'] = record['finished'] - started
    record['content_chunks'] = chunks
    record['ttft_content_s'] = chunks[0]['at']-started if chunks else None
    record['chunk_itl_s'] = [b['at']-a['at'] for a,b in zip(chunks,chunks[1:])]
    # Tokens are taken only from server usage; a content chunk may contain
    # several tokens, or special tokens may emit no content at all.
    tokens = record.get('usage', {}).get('completion_tokens')
    record['output_tokens'] = tokens if type(tokens) is int and tokens >= 0 else None
    record['tpot_content_s'] = ((chunks[-1]['at']-chunks[0]['at'])/(tokens-1)
        if chunks and record['output_tokens'] is not None and tokens > 1 else None)
    return record


def summarize(records, wall_s, concurrency, profiled):
    good = [r for r in records if r['success']]
    counts_known = bool(good) and all(r['output_tokens'] is not None for r in good)
    metric = lambda key: percentiles([r[key] for r in good if r.get(key) is not None])
    return dict(requests=len(records), concurrency=concurrency, wall_s=wall_s,
        successes=len(good), failures=len(records)-len(good), profiled=profiled,
        benchmark_valid=not profiled, missing_usage=sum(r['output_tokens'] is None for r in good),
        output_tokens=sum(r['output_tokens'] for r in good) if counts_known else None,
        output_tok_s=(sum(r['output_tokens'] for r in good)/wall_s
                      if counts_known and not profiled else None),
        request_s=len(good)/wall_s if not profiled else None,
        e2e_s=metric('e2e_s') if not profiled else None,
        ttft_content_s=metric('ttft_content_s') if not profiled else None,
        tpot_content_s=metric('tpot_content_s') if not profiled else None,
        chunk_itl_s=percentiles([v for r in good for v in r['chunk_itl_s']]) if not profiled else None,
        percentile_method='nearest observed order statistic: round((n-1)*q/100); seconds',
        measurement='closed-loop HTTP; TTFT=first nonempty content; chunk ITL is not token ITL; '
                    'content TPOT uses first/last nonempty content and server token count',
        quality='model accuracy is not evaluated; failures remain in requests.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--model', default='qwen3-omni')
    parser.add_argument('--output', required=True)
    parser.add_argument('--requests', type=int, default=256)
    parser.add_argument('--concurrency', type=int, default=32)
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--max-tokens', type=int, default=32)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--inputs', help='public/self-generated JSON list of messages or text rows')
    parser.add_argument('--profile', action='store_true',
                        help='mark this load diagnostic only; start server separately with --profile')
    args = parser.parse_args()
    parsed = urlsplit(args.base_url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.query or parsed.fragment:
        parser.error('base-url must be an HTTP(S) API base URL without query/fragment')
    if min(args.requests,args.concurrency,args.max_tokens) < 1 or args.warmup < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('counts/timeout must be positive; warmup must be nonnegative')
    rows = json.loads(Path(args.inputs).read_text(encoding='utf-8')) if args.inputs else [
        {'messages': [{'role':'user', 'content':f'自生成性能请求{i}：简述GPU推理通信与计算的关系。'}]}
        for i in range(args.requests)]
    if not isinstance(rows, list) or not rows:
        parser.error('inputs must be a nonempty list')
    payloads=[]
    for i in range(args.requests):
        row=rows[i % len(rows)]
        if isinstance(row,str):
            row={'messages':[{'role':'user','content':row}]}
        if not isinstance(row,dict) or not row.get('messages'):
            parser.error('each input needs messages, or use a text string')
        payloads.append(dict(model=args.model, messages=row['messages'], stream=True,
            stream_options={'include_usage':True}, temperature=0., max_tokens=args.max_tokens))
    output=Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'config.json').write_text(json.dumps(vars(args),ensure_ascii=False,indent=2),encoding='utf-8')
    profile_status=server_profile_status(args.base_url,args.timeout)
    warmups=[request(args.base_url,payloads[i % len(payloads)],i,args.timeout) for i in range(args.warmup)]
    (output/'warmup.json').write_text(json.dumps(warmups,ensure_ascii=False,indent=2),encoding='utf-8')
    if any(not r['success'] for r in warmups):
        raise RuntimeError('Warmup failed; see warmup.json')
    started=time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        records=list(pool.map(lambda pair:request(args.base_url,pair[1],pair[0],args.timeout),enumerate(payloads)))
    summary=summarize(records,time.perf_counter()-started,args.concurrency,args.profile or profile_status is True)
    summary['server_profile_status']=profile_status
    for name,value in (('requests.json',records),('summary.json',summary)):
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if summary['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
