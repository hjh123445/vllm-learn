"""Numerical contracts for fused kernels, with model-dtype rounding."""
import pathlib,sys,torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from layers.kernels import fused_add_rmsnorm,fused_qk_norm_rope_store,silu_and_mul
from layers.rotary_embedding import RotaryEmbeddding
torch.manual_seed(42);torch.set_num_threads(4)
def norm(x,w):
 v=x.float();return w*(v*torch.rsqrt(v.square().mean(-1,keepdim=True)+1e-6)).to(x.dtype)
records=[]
for dtype in (torch.bfloat16,torch.float32):
 for T in (1,17,129):
  x=torch.randn(T,2048,device='cuda',dtype=dtype);r=torch.randn_like(x);w=torch.randn(2048,device='cuda',dtype=dtype)
  residual=r+x;ref=norm(residual,w);out,rr=fused_add_rmsnorm(x,r.clone(),w,1e-6)
  torch.testing.assert_close(rr,residual,atol=0,rtol=0)
  torch.testing.assert_close(out,ref,atol=.016 if dtype==torch.bfloat16 else 2e-6,rtol=.01 if dtype==torch.bfloat16 else 2e-6)
  q=torch.randn(T,32,128,device='cuda',dtype=dtype);k=torch.randn(T,4,128,device='cuda',dtype=dtype);v=torch.randn_like(k)
  qw=torch.randn(128,device='cuda',dtype=dtype);kw=torch.randn_like(qw);pos=torch.arange(T,device='cuda')+4096
  rope=RotaryEmbeddding(8192,128,1000000.).to(device='cuda',dtype=dtype);cache=rope.cos_sin_cache[pos];c,s=cache.chunk(2,-1);c=torch.cat((c,c),-1);s=torch.cat((s,s),-1)
  def rotate(z,weight):
   z=norm(z,weight);a,b=z.chunk(2,-1);return z*c+torch.cat((-b,a),-1)*s
  qr,kr=rotate(q,qw),rotate(k,kw)
  kc=torch.zeros(1,256,4,128,device='cuda',dtype=dtype);vc=torch.zeros_like(kc);slots=torch.randperm(256,device='cuda')[:T];slots[0]=-1
  qo=fused_qk_norm_rope_store(q,k,v,qw,kw,rope.cos_sin_cache,pos,slots,kc,vc,1e-6)
  keep=slots>=0
  for actual,expected in ((qo,qr),(kc.view(-1,4,128)[slots[keep]],kr[keep])):
   torch.testing.assert_close(actual,expected,atol=.032 if dtype==torch.bfloat16 else 4e-6,rtol=.016 if dtype==torch.bfloat16 else 3e-6)
  torch.testing.assert_close(vc.view(-1,4,128)[slots[keep]],v[keep],atol=0,rtol=0)
  records.append(dict(dtype=str(dtype),tokens=T,q_max=float((qo-qr).abs().max()),norm_max=float((out-ref).abs().max())))
print(records)
