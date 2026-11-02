import argparse,json,os,sys,time,math
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer
from samplingParams import SamplingParams
MODEL=os.environ.get('QWEN3_OMNI_MODEL','/data/jingjiang3/res/AI-ModelScope/Qwen3-Omni-30B-A3B-Instruct')
if not os.path.exists(os.path.join(MODEL,'config.json')): MODEL='/data/ifly_jyhe23/models_omni/Qwen3-Omni-30B-A3B-Instruct'
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',choices=['ep8','ep4','tp2','pp2'],default='ep8'); ap.add_argument('--output',default='/tmp/qwen_context.json'); ap.add_argument('--batch',type=int,default=8); ap.add_argument('--lengths',default='16,128,512,1024'); a=ap.parse_args()
 tok=AutoTokenizer.from_pretrained(MODEL,use_fast=True); tf=os.path.join(MODEL,'chat_template.json')
 if os.path.exists(tf): tok.chat_template=json.load(open(tf,encoding='utf-8'))['chat_template']
 base=tok.encode(tok.apply_chat_template([{'role':'user','content':'请详细分析分布式推理的通信与显存权衡。'}],tokenize=False,add_generation_prompt=True),add_special_tokens=False)
 prompts={}
 for n in [int(x) for x in a.lengths.split(',')]: prompts[n]=(base*((n+len(base)-1)//len(base)))[:n]
 from engine.core import EngineCore
 specs={'ep8':(1,8,[0,1,2,3,4,5,6,7]),'ep4':(1,4,[0,1,2,3]),'tp2':(2,1,[0,1]),'pp2':(1,1,[0,1])}
 tp,ep,gpus=specs[a.config]
 t=time.perf_counter(); core=EngineCore(MODEL,tp_size=tp,ep_size=ep,pp_size=(2 if a.config=='pp2' else 1),gpus=gpus,master_port=29971+len(gpus),config_kwargs={'max_num_seqs':64,'max_model_len':2048,'max_num_batched_tokens':2048,'gpu_memory_utilization':0.85,'enforce_eager':True}); init=time.perf_counter()-t
 sp=SamplingParams(temperature=0.0,max_tokens=8); rows=[]
 for n,p in prompts.items():
  sids=[]
  for i in range(a.batch): sids.append(core.add_request(p,sp).seq_id)
  t=time.perf_counter(); out={}
  while not core.is_finished():
   for sid,toks in core.step(): out[sid]=len(toks)
  elapsed=time.perf_counter()-t; nt=sum(out.values()); rows.append({'prompt_tokens':n,'batch':a.batch,'elapsed_s':round(elapsed,4),'generated_tokens':nt,'tok_s':round(nt/elapsed,3)})
  print(json.dumps(rows[-1]),flush=True)
 result={'config':a.config,'init_s':round(init,3),'rows':rows}
 json.dump(result,open(a.output,'w'),indent=2); print(json.dumps(result),flush=True); core.shutdown()
if __name__=='__main__': main()
