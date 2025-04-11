---
layout: fast
title: EP02-vLLM源码讲解直播笔记-分布式通信与并行策略
date: 2025-03-25 17:32:10
summary: 
cover: posts_img/vLLM/cover.png
categories: 
  - Note
tags: 
  - vLLM
  - LLM
  - Inference
---

# [FIXME][EP02] vLLM 源码讲解直播笔记

## EP02: 分布式通信与并行策略

直播回看链接：https://www.youtube.com/watch?v=W83Zgbg8SkE&t=4s

特别鸣谢：月球大叔，Du Kuntai，Cheng Yihua 大佬带来的精彩讲解

### 📌 1. GroupCoordinator 类解析
'vllm/distributed/parallel_state.py'
```python
# 可以把GroupCoordinator想象成一个群聊
class GroupCoordinator:
    """
    # PyTorch ProcessGroup 的封装，管理进程组间的通信

    PyTorch ProcessGroup wrapper for a group of processes.
    PyTorch ProcessGroup is bound to one specific communication backend,
        e.g. NCCL, Gloo, MPI, etc.
    GroupCoordinator takes charge of all the communication operations among
        the processes in the group. It can route the communication to
        a specific implementation (e.g. switch allreduce implementation
        based on the tensor size and cuda graph mode).
    """

    # available attributes:
    
    # 这个群里表示的是我是谁
    rank: int  # global rank

    # 在这个群里加上我还有哪些人
    ranks: List[int]  # global ranks in the group

    # 在群里的人数
    world_size: int  # size of the group

    # difference between `local_rank` and `rank_in_group`:
    # if we have a group of size 4 across two nodes:
    # Process | Node | Rank | Local Rank | Rank in Group
    #   0     |   0  |  0   |     0      |       0
    #   1     |   0  |  1   |     1      |       1
    #   2     |   1  |  2   |     0      |       2
    #   3     |   1  |  3   |     1      |       3

    # 对应逻辑上的GPU id（比如一台8卡机上，0号GPU被占用了的话，SET_CUDA_DEVICE=1-6之后，内部映射为0-5）
    local_rank: int  # local rank used to assign devices

    rank_in_group: int  # rank inside the group

    # CPU的通信更加可控，更好做，用于主机端同步（如初始化阶段），支持任意通信后端（Gloo/MPI）
    cpu_group: ProcessGroup  # group for CPU communication

    # 用于设备间数据传输（必须使用支持GPU的后端，如NCCL）
    device_group: ProcessGroup  # group for device communication

    use_pynccl: bool  # a hint of whether to use PyNccl
    use_custom_allreduce: bool  # a hint of whether to use CustomAllreduce
    # communicators are only created for world size > 1
    pynccl_comm: Optional[Any]  # PyNccl communicator
    ca_comm: Optional[Any]  # Custom allreduce communicator
    mq_broadcaster: Optional[Any]  # shared memory broadcaster
```

### ⚡ 2. 并行策略详解

- TP（张量并行）
    - 需要allreduce，通信量大，对于通信需求较高
    - Infra端
        - 通信设备
            - NVLink: GPU之间的直接通信，常用于节点内通信
            - InfiniBand: 本质上也是硬件，常用于节点间通信
            - RDMA: RDMA网卡，最大的好处是跳过操作系统 / zero copy, RoCE
        - 通信库：'vllm/distributed/device_communicators'
            - PyNccl: Nvidia 之间的通信
            - Shared memory: 操作系统中不同进程之间数据共享
            - Custom allreduce: 专为all reduce操作的kernel 
            - torch.distributed: 广泛支持一系列的通信库
    - 算法端
        - 想了解任何通信方式可以了解：'vllm/model_executor/models/llama.py'，llama系模型支持各种并行方式，适合初学者学习架构
        - 'get_tp_group()'

- PP（流水线并行）
    - 通信量相对较小，对device--device通信需求较低
    - 不能降低延迟，但能提高吞吐
    - 算法端
        - 每个worker负责一个layers的子集
            - 'vllm/model_executor/models/llama.py' 中 self.start_layer --> self.end_layer
            - 在worker之间: communicate IntermediateTensor
            - 'vllm/worker/model_runner.py': 搜索 'get_pp_group()'

- EP（专家并行）& DP（数据并行）
    - 为什么要有EP？
        - Mistral / Mixtral / Deepseek 都是用MOE
        - MOE具有计算稀疏性，每个request只激活一小部分的expert
    - 将不同的expert放在不同的device上-->专家并行
    - 算法端
        - Shuffle (DeepEP communication kernel)
        - Forward
        - Shuffle back
    - 在Attention模块做TP，在FFN模块做EP
    - share expert负载较高，要做冗余

    - DP (数据并行)
        - 最大的TP << 所需要的EP（EP=320）
        - TP < # attention head
        - TP * DP == EP（通过请求并行的方式去拉满计算资源）
        - 在实践中难以应用
            - 对请求进行padding避免造成死锁


​    