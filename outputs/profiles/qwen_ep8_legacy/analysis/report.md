# 多卡采集复核

基础采集检查：通过。这不等于完整性或性能认证。

| Rank | Stage | ATen | CUDA kernel | NCCL kernel | GPU层归属比例 | 同卡计算通信交集 ms |
|---:|---|---:|---:|---:|---:|---:|
|0|[0]|260360|32526|810|90.6%|0.000|
|1|[0]|263141|32515|810|90.6%|0.000|
|2|[0]|257973|32483|823|90.5%|0.000|
|3|[0]|265139|32505|813|90.6%|0.000|
|4|[0]|258377|32502|818|90.5%|0.000|
|5|[0]|259563|25989|650|90.6%|0.000|
|6|[0]|264342|32513|810|90.6%|0.000|
|7|[0]|263544|32519|810|90.6%|0.000|

缺口：rank0 stage0.microbatch3.lane1.decode: layer-attributed GPU coverage 19/48; inspect dropped/tail events; rank1 stage0.microbatch3.lane1.decode: layer-attributed GPU coverage 19/48; inspect dropped/tail events; rank2 stage0.microbatch3.lane1.decode: layer-attributed GPU coverage 21/48; inspect dropped/tail events; rank3 stage0.microbatch3.lane1.decode: layer-attributed GPU coverage 19/48; inspect dropped/tail events; rank4 stage0.microbatch3.lane1.decode: layer-attributed GPU coverage 20/48; inspect dropped/tail events; rank5 stage0.microbatch2.lane0.decode: layer-attributed GPU coverage 35/48; inspect dropped/tail events; rank6 stage0.microbatch3.lane1.decode: layer-attributed GPU coverage 19/48; inspect dropped/tail events; rank7 stage0.microbatch3.lane1.decode: layer-attributed GPU coverage 19/48; inspect dropped/tail events

逐层GPU时间、kernel热点、CPU操作数量、CUDA API耗时、最长可见空隙与每个消息名称见 summary.json。
不要用NCCL kernel时长推算传输带宽；不要把同卡NCCL/计算交集写成跨stage流水线利用率。
