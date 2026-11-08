"""Audit raw token consistency and visible SSE content from actual load cases."""
from pathlib import Path
import json
import os
import sys

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/load_matrix'

if __name__=='__main__':
    servers={}
    tokenizer=None
    checks=[]
    baseline={}
    cross=[]
    cross_compared=0
    isolated={name:dict(compared=0,differences=[]) for name in ('concurrency','backend','graph_eager')}
    isolated_refs={name:{} for name in isolated}
    for path in sorted(OUT.glob('*/results.json')):
        case=path.parent
        if not case.name.startswith(('online_','offline_')):continue
        reps=json.loads(path.read_text(encoding='utf-8'))
        if not reps:continue
        first=reps[0];scope=first['scope'];online=first['backend']=='online'
        if online and scope not in servers:
            servers[scope]={}
            f=OUT/f'server_{scope}_graph/http_requests.jsonl'
            if f.exists():
                for line in f.read_text(encoding='utf-8').splitlines():
                    r=json.loads(line);servers[scope][r['id']]=r
        if online and tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained('/data/ifly_jyhe23/models_omni/Qwen3-Omni-30B-A3B-Instruct')
        ordinal_refs={}
        for rep in reps:
            duplicates={};bad_duplicate=[];bad_round=[];bad_sse=[];missing=[]
            n_duplicate=n_round=n_sse=0
            for q in rep['requests']:
                if 'error' in q:continue
                ordinal=q['request_index']
                record=servers[scope].get(q['request_id'],{}) if online else q
                ids=record.get('output_ids')
                if ids is None:missing.append(ordinal);continue
                if online:
                    n_sse+=1
                    expected=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    if expected!=q['answer'] or record.get('status')!='ok':bad_sse.append(ordinal)
                key=q['index']
                if key in duplicates:
                    n_duplicate+=1
                    if duplicates[key]!=ids:bad_duplicate.append(ordinal)
                duplicates.setdefault(key,ids)
                if ordinal in ordinal_refs:
                    n_round+=1
                    if ordinal_refs[ordinal]!=ids:bad_round.append(ordinal)
                ordinal_refs.setdefault(ordinal,ids)
                # Same data and length, across concurrency, backend and Graph flag.
                fullkey=(scope,first['max_pixels'],first['context_tokens'],first['target_output_tokens'],key)
                # Hold all other run options fixed when identifying the
                # specific factor under comparison. Categories can overlap.
                variants={
                    'concurrency':((fullkey,first['backend'],first['graph']),first['concurrency']),
                    'backend':((fullkey,first['concurrency'],first['graph'],ordinal),first['backend']),
                    'graph_eager':((fullkey,first['concurrency'],first['backend'],ordinal),first['graph'])}
                for name,(group,value) in variants.items():
                    reference=isolated_refs[name].get(group)
                    if reference is not None and reference['value']!=value:
                        isolated[name]['compared']+=1
                        if reference['ids']!=ids:
                            isolated[name]['differences'].append(dict(case=case.name,repeat=rep['repeat'],
                                request_index=ordinal,index=key,reference_case=reference['case']))
                    isolated_refs[name].setdefault(group,dict(value=value,ids=ids,case=case.name))
                if fullkey in baseline and baseline[fullkey]['case']!=case.name:
                    cross_compared+=1
                    if baseline[fullkey]['ids']!=ids:
                        cross.append(dict(case=case.name,repeat=rep['repeat'],request_index=ordinal,
                            index=key,reference_case=baseline[fullkey]['case']))
                baseline.setdefault(fullkey,dict(case=case.name,ids=ids))
            checks.append(dict(case=case.name,repeat=rep['repeat'],
                duplicate_input_compared=n_duplicate,duplicate_input_differences=bad_duplicate,
                repeat_compared=n_round,repeat_differences=bad_round,
                sse_content_compared=n_sse,sse_content_differences=bad_sse,
                missing_token_evidence=missing))
    result=dict(checks=checks,cross_case_compared=cross_compared,cross_case_differences=cross,isolated_comparisons=isolated,
        note='zero differences with zero compared is untested; cross-case key requires same scope, max_pixels, extra context, output length and dataset row')
    path=ROOT/'outputs/raw_output_audit.json';path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('AUDIT',len(checks),'rounds',len(cross),'cross-case differences')
    for q in checks:
        print(q['case'],q['repeat'],'duplicate',q['duplicate_input_compared'],len(q['duplicate_input_differences']),
            'repeat',q['repeat_compared'],len(q['repeat_differences']), 'SSE',q['sse_content_compared'],len(q['sse_content_differences']),
            'missing',len(q['missing_token_evidence']))
