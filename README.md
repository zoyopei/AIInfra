# AIInfra

文字课程内容正在一节节补充更新，尽可能抽空继续更新正在 [AIInfra](https://github.com/chenzomi12/AIInfra/)，希望您多多鼓励和参与进来！！！

文字课程开源在 [AISys](https://chenzomi12.github.io/)，系列视频托管[B 站 ZOMI 酱](https://space.bilibili.com/517221395)和[油管 ZOMI6222](https://www.youtube.com/@zomi6222/videos)，PPT 开源在 [AIInfra](https://github.com/chenzomi12/AIInfra/)，欢迎取用！！！

## 课程背景

这个开源项目英文名字叫做**AIInfra**，中文名字叫做**AI基础设施**。大模型是基于 AI 集群的全栈软硬件性能优化，通过最小的每一块 AI 芯片组成的 AI 集群，编译器使能到上层的 AI 框架，训练过程需要分布式并行、集群通信等算法支持，而且在大模型领域最近持续演进如智能体等新技术。

本开源课程主要是跟大家一起探讨和学习人工智能、深度学习的系统设计，而整个系统是围绕着 ZOMI 在工作当中所积累、梳理、构建 AI 大模型系统的基础软硬件栈，因此成为 AI 基础设施。希望跟所有关注 AI 开源课程的好朋友一起探讨研究，共同促进学习讨论。

与**AISystem**[https://github.com/chenzomi12/AISystem] 项目最大的区别就是 **AIInfra** 项目主要针对大模型，特别是大模型在分布式集群、分布式架构、分布式训练、大模型算法等相关领域进行深度展开。

![大模型系统全栈](statics/images/aifoundation01.jpg)

## 课程内容大纲

课程主要包括以下模块，内容陆续更新中，欢迎贡献：

| 序列 | 教程内容 | 简介 | 地址 | 状态 |
| --- | --------------- | ------------------------------------------------------------------------------------------------- | ---------------------------- | ---- |
| 00 | 大模型系统概述 | 大模型总体介绍与概览等。 | [[Slides](./00Summary/)] | 待更 |
| 01 | AI 计算集群 | 大模型虽然已经慢慢在端测设备开始落地，但是总体对云端的依赖仍然很重很重，AI 集群会介绍集群运维管理、集群性能、训练推理一体化拓扑流程等内容。 | [[Slides](./01AICluster/)] | 待更 |
| 02 | 通信与存储 | 大模型训练和推理的过程中都严重依赖于网络通信，因此会重点介绍通信原理、网络拓扑、组网方案、高速互联通信的内容。存储则是会从节点内的存储到存储 POD 进行介绍。 | [[Slides](./02StorComm/)] | DONE |
| 03 | 集群容器与云原生 | 从容器、云原生时代到 Docker 和 K8S 技术的应用。 | [[Slides](./03DockCloud/)] | 待更 |
| 04 | 大模型训练 | 大模型训练是通过大量数据和计算资源，利用 Transformer 架构优化模型参数，使其能够理解和生成自然语言、图像等内容，广泛应用于对话系统、文本生成、图像识别等领域。 | [[Slides](./04Train/)] | 更新中 |
| 05 | 大模型推理 | 大模型推理核心工作是优化模型推理，实现推理加速，其中模型推理最核心的部分是Transformer Block。本节会重点探讨大模型推理的算法、调度策略和输出采样等相关算法。 | [[Slides](./05Infer/)] | 更新中 |
| 06 | 大模型算法与数据 | Transformer起源于NLP领域，近期统治了 CV/NLP/多模态的大模型，我们将深入地探讨 Scaling Law 背后的原理。在大模型算法背后数据和算法的评估也是核心的内容之一，如何实现 Prompt 和通过 Prompt 提升模型效果。 | [[Slides](./06AlgoData/)] | 更新中 |
| 07 | 大模型应用 | 当前大模型技术已进入快速迭代期。这一时期的显著特点就是技术的更新换代速度极快，新算法、新模型层出不穷。因此本节内容将会紧跟大模型的时事内容，进行深度技术分析。 | [[Slides](./07Application/)] | 更新中 |

## 课程细节

## 课程设立目的

本课程主要为本科生高年级、硕博研究生、AI 大模型系统从业者设计，帮助大家：

1. 完整了解 AI 的计算机系统架构，并通过实际问题和案例，来了解 AI 完整生命周期下的系统设计。

2. 介绍前沿系统架构和 AI 相结合的研究工作，了解主流框架、平台和工具来了解 AI 大模型系统。

## 课程部分


### **[00. 大模型系统概述](./00Summary/)**

| 编号  | 名称       | 具体内容      | 状态      |
|:---:|:--- |:--- |:--- |
| 1      | [大模型系统概述](./00Summary/) | 大模型系统的总体介绍和概览  | 待更 |

### **[01. AI 计算集群](./01AICluster/)**

| 编号  | 名称       | 具体内容      | 状态      |
|:---:|:--- |:--- |:--- |
| 1      | [计算集群之路](./01AICluster/01Roadmap/) |    | 待更 |
| 2      | [集群建设之巅](./01AICluster/02TypicalRepresent/)   |         | 待更 |
| 3      | [集群性能分析](./01AICluster/03Analysis/)  |    | 待更 |
| 4      | [实践](./01AICluster/04Practices) |   | 待更 |

### **[02. 通信与存储](./02StorComm/)**

| 编号  | 名称       | 具体内容      | 状态      |
|:---:|:--- |:--- |:--- |
| 1      | [AI 集群组网之路](./02StorComm/01Roadmap/) | 集群组网的相关技术  | 待更 |
| 2      | [网络通信进阶](./02StorComm/02NetworkComm/) |    | 待更 |
| 3      | [集合通信原理](./02StorComm/03CollectComm/) | 通信域、通信算法、集合通信原语  | 待更 |
| 4      | [集合通信库](./02StorComm/04CommLibrary/)   |   | 待更 |
| 5      | [AI 集群存储之路](./02StorComm/05StorforAI/) | 数据存储、CheckPoint 梯度检查点等存储与大模型结合的相关技术  | 待更 |

### **[03. 集群容器与云原生](./03DockCloud/)**

| 编号  | 名称       | 具体内容      | 状态      |
|:---:|:--- |:--- |:--- |
| 1      | [容器时代](./03DockCloud/01Roadmap/) |    | 待更 |
| 2      | [Docker 与 K8S 初体验](./03DockCloud/02DockerK8s/) |    | 待更 |
| 3      | [深入 K8S](./03DockCloud/03DiveintoK8s/) |    | 待更 |
| 4      | [AI 集群云平台 Cloud for AI](./03DockCloud/04CloudforAI/) |    | 待更 |
| 5      | [实践](./03DockCloud/05Practices/) |    | 待更 |

### **[04. 大模型训练](./04Train/)**

| 编号  | 名称       | 具体内容      | 状态      |
|:---:|:--- |:--- |:--- |
| 1      | [分布式并行基础](./04Train/01ParallelBegin/) |    | 待更 |
| 2      | [大模型并行进阶](./04Train/02ParallelAdv/) |    | 待更 |
| 3      | [大模型训练加速](./04Train/03TrainAcceler/) |    | 待更 |
| 4      | [大模型后训练与强化学习](./04Train/04PostTrainRL/) |    | 待更 |
| 5      | [大模型微调 SFT](./04Train/05FineTune/) |    | 待更 |
| 6      | [大模型验证评估](./04Train/06VerifValid/) |    | 待更 |
| 7      | [实践](./04Train/07Practices/) |    | 待更 |

### **[05. 大模型推理](./05Infer/)**

| 编号  | 名称       | 具体内容      | 状态      |
|:---:|:--- |:--- |:--- |
| 1      | [大模型推理基本概念](./05Infer/01Foundation) |   | 待更 |
| 2      | [大模型推理加速](./05Infer/02InferSpeedUp) |  | 待更 |
| 3      | [架构调度加速 ](./05Infer/03SchedSpeedUp) |  | 待更 |
| 4      | [长序列推理](./05Infer/04LongInfer) |  | 待更 |
| 5      | [输出采样](./05Infer/05OutputSamp) |  | 待更 |
| 6      | [大模型压缩](./05Infer/06CompDistill) |  | 待更 |
| 7      | [推理框架架构分析](./05Infer/07Framework) |  | 待更 |
| 8      | [推理框架架构分析](./05Infer/08DeepSeekOptimize) |  | 待更 |
| 9      | [实践](./05Infer/09Practices) |  | 待更 |

### **[06. 大模型算法与数据](./06AlgoData/)**

大部分待更，欢迎参与，08 新算法根据时事热点不定期更新

| 编号  | 名称       | 具体内容      | 状态      |
|:---:|:--- |:--- |:--- |
| 1      | [Transformer 架构](./06AlgoData/01Transformer) | Transformer架构原理介绍 | 待更 |
| 2      | [MoE 架构 ](./06AlgoData/02MoE/) | MoE(Mixture of Experts) 模型架构原理与细节 |  待更 |
| 3      | [创新架构 ](./06AlgoData/03NewArch) | SSM、MMABA、RWKV、Linear Transformer 等新大模型结构 | 待更 |
| 4      | [图文生成与理解](./06AlgoData/04ImageTextGenerat) |   | 待更 |
| 5      | [视频大模型](./06AlgoData/05VideoGenerat) |  | 待更 |
| 6      | [语音大模型](./06AlgoData/06AudioGenerat) |   | 待更 |
| 7      | [数据工程](./06AlgoData/07DataEngineer) | 数据工程、Prompt Engine、Data2Vec 和 Tokenize 等相关技术 | 待更 |
| 8      | [实践](./06AlgoData/08Practices) |   | 待更 |

### **[07. 大模型应用](./07Application/)**

基本完结，01 根据时事热点不定期更新

| 编号  | 名称       | 具体内容      | 状态      |
|:---:|:--- |:--- |:--- |
| 1      | [AI Agent技术与实践](./07Application/01AIAgent/)   | AI Agent 智能体的原理、架构   | 待更 |
| 2      | [检索增强生成（ RAG ）](./07Application/02RAG/)   |  检索增强生成技术的介绍  | 待更 |
| 3      | [实践](./07Application/03Practices/)   |     | 待更 |
| 4      | [大模型热点](./07Application/10Others/)   |  OpenAI o1、WWDC 大会技术洞察   | 待更 |


## 知识清单

![大模型系统全栈](statics/images/aifoundation02.png)

## 备注

> 这个仓已经到达疯狂的 10G 啦（ZOMI 把所有制作过程、高清图片都原封不动提供），如果你要 git clone 会非常的慢，因此建议优先到  [Releases · chenzomi12/AIInfra](https://github.com/chenzomi12/AIInfra/releases) 来下载你需要的内容

> 非常希望您也参与到这个开源课程中，B 站给 ZOMI 留言哦！
> 
> 欢迎大家使用的过程中发现 bug 或者勘误直接提交代码 PR 到开源社区哦！
> 
> 请大家尊重开源和 ZOMI 的努力，引用 PPT 的内容请规范转载标明出处哦！
