import torch
import torch.distributed as dist
from deep_ep import Buffer

input_tensor = torch.ones((7289, 1024), dtype=torch.bfloat16, device='cuda')
expert_indices = [1, 5]

# 初始化通信缓冲区
buffer = Buffer(
    group=dist.group.WORLD,
    num_nvl_bytes=1e9,  # NVLink 缓冲区 1GB
    num_rdma_bytes=2e9   # RDMA 缓冲区 2GB
)

# MoE 分发数据
recv_data = buffer.dispatch(input_tensor, expert_indices, num_experts=64)

# 执行专家计算
expert_output = recv_data * 2

# 合并结果
combined_data = buffer.combine(expert_output)