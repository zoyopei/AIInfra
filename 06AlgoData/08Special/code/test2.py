import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, inputs):
        return self.fc(inputs)

class TopKRouter(nn.Module):
    def __init__(self, input_dim, num_experts, top_k):
        super().__init__()
        self.projection = nn.Linear(input_dim, num_experts)
        self.top_k = top_k

    def forward(self, inputs):
        # 计算每个专家的分数
        scores = self.projection(inputs)
        # 获取 Top-K 分数和对应的索引
        top_k_scores, top_k_indices = torch.topk(scores, self.top_k, dim=1)
        # 将 Top-K 分数转换为概率分布
        probabilities = F.softmax(top_k_scores, dim=1)
        return probabilities, top_k_indices

class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, expert_dim, top_k):
        super().__init__()
        self.router = TopKRouter(input_dim, num_experts, top_k)
        self.experts = nn.ModuleList([
            Expert(expert_dim, output_dim) for _ in range(num_experts)
        ])
        self.top_k = top_k

    def forward(self, inputs):
        # 根据输入数据获取路由概率和 Top-K 专家索引
        probabilities, expert_indices = self.router(inputs)
        # 将输入数据广播到所有专家
        inputs = inputs.unsqueeze(1).expand(-1, self.top_k, -1) # dispatch 过程
        # 根据 Top-K 索引选择对应的专家
        expert_outputs = torch.zeros_like(inputs)
        for i in range(self.top_k):
            expert_idx = expert_indices[:, i]
            expert_output = self.experts[expert_idx](inputs[:, i, :])
            expert_outputs[:, i, :] = expert_output
        # 加权求和得到最终输出
        final_output = torch.sum(expert_outputs * probabilities.unsqueeze(-1), dim=1) # combine 过程
        return final_output, probabilities

# 定义模型参数
input_dim = 10  # 输入特征维度
output_dim = 1  # 输出维度
num_experts = 5 # 专家数量
expert_dim = 10 # 每个专家处理的特征维度
top_k = 3       # Top-K 路由

# 创建 MoE 模型实例
model = MixtureOfExperts(input_dim, output_dim, num_experts, expert_dim, top_k)

# 创建一个假设的输入张量
inputs = torch.randn(2, input_dim)  # 假设 batch 大小为 2

# 获取模型输出
output, probabilities = model(inputs)
print("Model output:", output)
print("Routing probabilities:", probabilities)