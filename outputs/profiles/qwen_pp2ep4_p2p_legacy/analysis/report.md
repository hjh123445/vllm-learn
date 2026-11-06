# 多卡采集复核

基础采集检查：通过。这不等于完整性或性能认证。

| Rank | Stage | ATen | CUDA kernel | NCCL kernel | GPU层归属比例 | 同卡计算通信交集 ms |
|---:|---|---:|---:|---:|---:|---:|
|0|[0]|252750|32162|808|90.9%|0.000|
|1|[0]|252750|32141|813|90.8%|0.000|
|2|[0]|247500|32184|808|90.9%|0.000|
|3|[0]|251250|32166|808|90.9%|0.000|
|4|[1]|245711|32171|806|90.5%|0.000|
|5|[1]|250961|32078|830|90.2%|0.000|
|6|[1]|249461|32129|816|90.4%|0.000|
|7|[1]|252462|32137|811|90.4%|0.000|

缺口：rank0 stage0.microbatch6.lane0.decode: layer-attributed GPU coverage 15/24; inspect dropped/tail events; rank1 stage0.microbatch6.lane0.decode: layer-attributed GPU coverage 16/24; inspect dropped/tail events; rank2 stage0.microbatch6.lane0.decode: layer-attributed GPU coverage 15/24; inspect dropped/tail events; rank3 stage0.microbatch6.lane0.decode: layer-attributed GPU coverage 14/24; inspect dropped/tail events; rank4 stage1.microbatch6.lane0.decode: layer-attributed GPU coverage 14/24; inspect dropped/tail events; rank5 stage1.microbatch6.lane0.decode: layer-attributed GPU coverage 18/24; inspect dropped/tail events; rank6 stage1.microbatch6.lane0.decode: layer-attributed GPU coverage 16/24; inspect dropped/tail events; rank7 stage1.microbatch6.lane0.decode: layer-attributed GPU coverage 15/24; inspect dropped/tail events

逐层GPU时间、kernel热点、CPU操作数量、CUDA API耗时、最长可见空隙与每个消息名称见 summary.json。
不要用NCCL kernel时长推算传输带宽；不要把同卡NCCL/计算交集写成跨stage流水线利用率。
