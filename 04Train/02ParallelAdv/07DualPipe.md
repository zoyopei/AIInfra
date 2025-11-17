<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# 07. DualPipe原理(Continue)

> Author by：杨泓东

pipeline parallelism(pp)是介于data parallelism以及tensor parallelism之外的一种模型并行方式。pp的核心思想是通过将大模型分解成多个层，并将它们组成一个流水线的方式进行前向和反向传播，从而减少单卡的显存占用。主要相关的pp算法有1f1b, interval 1f1b, zero bubble, chimera, bitpipe。

接下来将深入解析DualPipe的核心原理与具体实现。从DualPipe的设计动机开始，阐述其具体实现，并引出其与megatron框架的结合，进行多机多卡实验，profile和流水线可视化。之后会重点介绍跨batch的overlap实现，分析如何重构 megatron GPT Model，实现fwd和bwd的chunk交错执行。随后，会进行Dualpipe参数气泡显存分析，并介绍dualPipe引入的变体

## 1. dualpipe设计动机
1. dualpipe的设计思路=zero bubble + 双流水线
2. 对于DeepSeek-V3，跨节点EP引入的通信开销导致计算与通信的效率比约为1:1。双流水线的稳定阶段提供了对来自两个方向的forward和backward之间实现overlap的机会。
3. 使用zero_bubble进一步减少气泡。整个dualpipe不仅实现了alltoall通讯和计算的overlap，而且使得dualpipe即使在没有繁重通讯负担的更一般情况下，仍展示出效率优势。
4. 虽然DualPipe需要保持模型参数的两个副本，但由于deepseekv3在训练期间使用了较大的EP尺寸，这并不会显著增加内存消耗。
5. 对于DualPipe来说，无论num_micro数量如何增加，气泡或激活内存都不会增加。

![dualpipe设计动机](./images/07DualPipe01.png)
![dualpipe设计动机](./images/07DualPipe02.png)

## 2. Dualpipe schedule 具体实现（micbatch层面）
### 2.1 流水线code实现逻辑
world_size: 每个pp group的长度
上半区：pp rank < pp_world_size / 2 
下半区：pp rank >= pp_world_size / 2 
abs_rank: 设rank为当前device在当前pp_group的rank(不是global rank)。如果rank处于下半区:
`abs_rank = rank < world_size // 2 ? rank : world_size - 1 - rank` 
F: forward
F0: 方向1的F，micro_batch_id = 0~9
F1: 方向2的F，micro_batch_id = 10 ～19
B: dgrad
W: wgrad
BW: dgrad + wgrad

| stage | loop_num                            | 上半区                                                    | 下半区                                                    | 通讯                                                       | 示意图 |   |   |   |   |
|-------|-------------------------------------|--------------------------------------------------------|--------------------------------------------------------|----------------------------------------------------------|-----|---|---|---|---|
| 1     | 2 * (world_size / 2 - 1 - abs_rank) | nF0                                                    | nF1                                                    | send_forward_recv_forward                                |     |   |   |   |   |
| 2     | 2 * (1 + abs_rank)                  | nF0F1                                                  | nF1F0                                                  | send_forward_recv_backward                               |     |   |   |   |   |
| 3     | (world_size / 2 - 1 - abs_rank)     | nB1W1F1                                                | nB0F0F1                                                | send_backward_recv_forward<br>send_forward_recv_backward |     |   |   |   |   |
| 4     | num_batch - world_size              | Overlap: nF0B1F1B0                                     | Overlap: nF1B0F0B1                                     | send_forward_recv_backward                               |     |   |   |   |   |
| 5     | 2 * abs_rank + 1                    | nBW1BW0(The second half of the chunks use zero bubble) | nBW0BW1(The second half of the chunks use zero bubble) | send_backward_recv_forward                               |     |   |   |   |   |
| 6     | (world_size / 2 - abs_rank)         | nB0W                                                   | nB1W                                                   | recv_backward<br>send_backward                           |     |   |   |   |   |
| 7     | pop_all                             | nW                                                     | nW                                                     |                                                          |     |   |   |   |   |
|       |                                     |                                                        |                                                        |                                                          |     |   |   |   |   |
|       |                                     |                                                        |                                                        |                                                          |     |   |   |   |   |

### 2.2与megatron框架的结合
#### 2.2.1 bd group和pp group
1. bd group用于实现双向model参数和梯度同步的通讯。如pp_group=[0,1,...,7]，则bd_group=[[0,7], [1, 6]...]。
2. 只维护一个pp_group，通讯时，根据global direction flag，交换get_pipeline_model_parallel_next_rank()和get_pipeline_model_parallel_prev_rank()的数值实现通讯方向的逆转。计算时，根据global direction flag获取需要使用的model和dataloader。

#### 2.2.2 Model generator
model构造为一个长度为2的list，model[0]存储的是方向1的model，model[1]存储的是方向2的model。
![dualpipe设计动机](./images/07DualPipe03.png)

#### 2.2.3 Model参数和梯度的同步
1. 创建完model list，将rank[i]上model[0]的参数broadcast到rank[pp_world_size - i]上的model[1]，确保初始值相同。
2. 每次流水线结束，在bd_group内进行梯度的all_reduce，确保每次迭代完毕后对应的model的梯度相同。
3. 设置另一份model的share参数为True，避免计算grad_norm时计入。
4. Load checkpoint时需要reload optimizer的parameters。

#### 2.2.4 dataloader
对于dualpipe，rank_0和rank_last的dataloader扩展为长度为2的list。dataloader[0]用于方向1，dataloader[1]用于方向2。dataloader包装的数据集完全相同，通过下图的指针控制dataloader获取不同方向的数据。
![dualpipe设计动机](./images/07DualPipe04.png)

#### 2.2.5 对pp <= num_micro < 2 * pp情况的支持
假设pp=8，num_micro=8，原来的stage的形状会变得极其不规则，采用传fake_tensor的方式解决
![dualpipe设计动机](./images/07DualPipe05.png)

### 2.3 profile和流水线可视化
pp=8, num_micro=20, on pp_rank=1
![dualpipe设计动机](./images/07DualPipe06.png)
pp=16, num_micro=40
![dualpipe设计动机](./images/07DualPipe07.png)
### 2.4 实验
#### 2.4.1 32c
橙色：dualpipe
蓝色：1F1B
dualpipe比1F1B慢1s左右。小参数量下，双向model带来的额外的梯度all_reduce成为了主要性能瓶颈。
![dualpipe设计动机](./images/07DualPipe08.png)
![dualpipe设计动机](./images/07DualPipe09.png)

#### 2.4.2 256c
红色：1F1B
蓝色：dualpipe
参数量上去后，速度上dualpipe已经反超了1F1B，减少气泡带来的收益能覆盖掉梯度all_reduce带来的额外开销。
![dualpipe设计动机](./images/07DualPipe10.png)
![dualpipe设计动机](./images/07DualPipe11.png)

## 3. Dualpipe参数气泡显存分析
### 3.1 Dualpipe
![dualpipe设计动机](./images/07DualPipe12.png)
![dualpipe设计动机](./images/07DualPipe02.png)
### 3.2 Bubble   

这里的气泡都是指device空闲时间,不是和计算时间的比例
![dualpipe设计动机](./images/07DualPipe13.png)

$$ s:seq len$$

$$h:hidden-size$$
$$b:micro-batch-size$$
下面的bubble计算都是单个devices
- 第一部分: 上图黄色框填充部分
$$bubble_{1}=(PP/2-1) \times F
$$
- 第二部分: 图中蓝色虚线框
设$$B_{X}=B-W$$为input的bwd, 即DGRAD,$W$为WGRAD。
$B_{X}$ 比 $F$ 需要的时间长，例如device6计算完了fwd 1,需要等device7完成绿色框的bwd for input 0

$$bubble_{2}=(PP/2-1)\times(B_{X}-F)=(PP/2-1)\times(B-W-F)


$$
- 第三部分：图中紫色框填充部分
$$bubble_{3}=(PP/2-1)\times(F\&B-B)$$
- 第四部分: 图中红色虚线框
$$bubble_{4}=(PP/2-1)\times(B_{X}-W)=(PP/2-1)\times(B-2W)


$$
$$bubble_{total}=bubble_{1}+bubble_{2}+bubble_{3}+bubble_{4}=(PP/2-1)\times(F\&B+B-3W)$$

### 3.3 Param

第一个正向流水线：                
- device 0: stage 0                     
- device 1: stage 1
- device 2: stage 2
- device 3: stage 3

第二个反向流水线
- device 3: stage 0
- device 2: stage 1
- device 1: stage 2
- device 0: stage 3
双向流水线一个device会保存两份stage的layer参数，所以是两倍

### 3.4 峰值显存
上表中激活的PP其实代表是$$PP*M_B$$，其中PP代表流水线stages,$$M_B$$代表backward阶段对于存储激活x所需要的显存
![dualpipe设计动机](./images/07DualPipe14.png)