"""KV pressure must never invalidate the batch already scheduled this step."""
import pathlib,sys,types,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from engine.scheduler import Scheduler
from engine.sequence import Sequence,SequenceStatus
from samplingParams import SamplingParams
class PreemptionTests(unittest.TestCase):
 def test_victims_exclude_scheduled_and_stale_snapshot_entries(self):
  cfg=types.SimpleNamespace(max_num_seqs=8,max_num_batched_tokens=3,eos=-1,kvcache_block_size=256,num_kvcache_blocks=3,enable_prefix_caching=False)
  scheduler=Scheduler(cfg);seqs=[Sequence([i+1]*256,SamplingParams(max_tokens=10)) for i in range(3)]
  for s in seqs:
   scheduler.block_manager.allocate(s,0);s.num_cached_tokens=256;s.is_prefill=False;s.status=SequenceStatus.RUNNING
   s.append_token(5);scheduler.running.append(s)
  scheduled,_=scheduler.schedule()
  self.assertEqual(scheduled,[seqs[0]])
  self.assertEqual(seqs[0].status,SequenceStatus.RUNNING)
  self.assertEqual(len(seqs[0].block_table),2)
  self.assertEqual(len(set(scheduler.waiting)),len(scheduler.waiting))
  self.assertTrue(all(s not in scheduler.waiting for s in scheduled))
  owned=[b for s in scheduler.running for b in s.block_table]
  self.assertEqual(len(owned),len(set(owned)))
  self.assertEqual(set(owned),scheduler.block_manager.used_block_ids)
if __name__=='__main__':unittest.main()
