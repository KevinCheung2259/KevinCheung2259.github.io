---
layout: fast
title: Llumnix-多实例LLM服务的请求动态调度
date: 2025-03-13 21:45:25
summary: 
cover: posts_img/Llumnix/7.png
categories: 
  - Paper Report
tags: 
  - OSDI 2024
  - LLM
  - Inference
---

Llumnix揭示了LLM服务与传统DNN推理的根本差异，提出动态调度必要性，设计了首个支持跨实例实时迁移的LLM服务系统，旨在解决大规模语言模型（LLM）推理服务中的请求高效调度问题。通过动态请求迁移和细粒度调度策略，Llumnix在降低延迟、提高优先级支持和降低成本方面表现出色。

论文：(OSDI 2024)[Llumnix: Dynamic Scheduling for Large Language Model Serving](https://www.usenix.org/system/files/osdi24-sun-biao.pdf)

代码：https://github.com/AlibabaPAI/llumnix

该文章参考自：
1. https://www.usenix.org/system/files/osdi24_slides-sun-biao.pdf
2. https://19shuidiph.github.io/2024/12/30/paperreading-llumnix/#wechat

# 背景

## LLM服务特性

- **异构性**：不同应用场景（如聊天、摘要、代码生成）导致请求的输入/输出长度、延迟需求（GPT Plus）差异显著。
    ![](posts_img/Llumnix/1.png)

- **不可预测性**：生成token数量未知，GPU内存占用动态增长，传统静态调度难以应对。
    ![](posts_img/Llumnix/2.png)
    在推理过程中遵从[Orca](https://kevin-zhang-sysu.github.io/2024/09/14/Orca/)中提出的选择性批处理机制，使用[vLLM](https://kevin-zhang-sysu.github.io/2024/09/14/vLLM/)中提出的动态内存分配机制不断地为新的KVcache分配新的内存。当GPU显存满载时，会将一部分的请求（蓝色的部分）从内存中驱逐，重新放回请求队列。

# 存在的问题与挑战

- 由于抢占带来的额外开销较大，导致P99延迟大，服务级别目标（SLO）难以满足。
  ![](posts_img/Llumnix/3.png)

- 请求之间的性能干扰。
  ![](posts_img/Llumnix/4.png)
  批处理的数量越多，模型参数量越大，干扰就越明显。

- 内存碎片。
  考虑到前两个挑战，应该将请求分散到不同的GPU，但这样容易造成显存的外部碎片化。导致外部请求（尤其是长请求）的延迟很高。
  ![](posts_img/Llumnix/5.png)

- 满足更高优先级
  现在的系统一般都是平等对待所有的请求，缺乏优先级支持。（这里的平等是对每个请求的优先级都一致，关于请求公平性方面的研究可以参考FairnessLLM这篇OSDI24的工作）

# 解决方案
Llumnix通过运行时跨多个模型实例重新调度请求来应对上述挑战。类似于现代操作系统中的上下文切换，通过高效且可扩展的请求和内存状态实时迁移机制实现重新调度，以改善负载均衡、减少资源碎片化、区分请求优先级和SLO，还能实现弹性伸缩(更快地耗尽要终止的实例或使新实例饱和)。

![](posts_img/Llumnix/6.png)

## 实时迁移
![](posts_img/Llumnix/7.png)

多阶段迁移：假设复制一个KVcache是0.5ms，计算一个KVcache是1ms。现在已经有了100个KVcache，一边算一边复制迁移，当源实例计算到第200个KVcache的时候，目标实例上也有199个KVcache了。然后要真正进行停等的KVcache就只有第200个这一个了，等待第200个计算完毕，复制迁移即可。

为了保证迁移的可靠性，Llumnix设计了一套handshake机制。

## 分布式调度架构
![](posts_img/Llumnix/8.png)

Llumnix采用分布式调度架构，结合全局调度器和实例级调度器（llumlet）。全局调度器负责根据实例负载进行新请求的分发、触发跨实例迁移和控制自动扩缩容；llumlet负责本地调度、迁移协调和执行。这种架构通过分离关注点，提高了调度的可扩展性。

## 动态调度策略
![](posts_img/Llumnix/9.png)

引入“虚拟使用量”概念，将不同调度目标（如负载均衡、去碎片化、优先级支持）统一为简单的实例负载度量。调度策略基于虚拟使用量进行启发式的负载均衡，同时通过规则设置不同场景下的虚拟使用量，例如为高优先级请求分配更多的虚拟使用量，确保其能够平稳无干扰地运行。

如果需要空出一台机器或一个模型实例，可以将其虚拟使用量设置为最大值。这样，该实例上的任务会被调度到其他实例上。
调度决策基于实例的自由度（Freeness）：
- 自由度计算公式：  
  **F = (M - ∑V) / B**  
  其中：
  - M = 总内存  
  - ∑V = 实例上所有请求的虚拟使用量之和  
  - B = 批大小  

调度时，优先将请求分配到自由度更高的实例上。

### 迁移策略
- 定期触发迁移操作，选择自由度最低的实例作为源实例，自由度最高的实例作为目标实例。
- 通过迁移请求实现负载均衡。

### 弹性伸缩
- 如果实例的自由度极高，考虑关闭该实例以节省资源。
- 如果实例的自由度极低，考虑增加一个新实例以分担负载。

# 算法实现
用3300行Python代码实现Llumnix，支持vLLM作为后端，利用Ray框架实现分布式协调。使用Gloo进行KVcache传输，而不是NCCL，因为后者并发调用不安全，将很多小的KVcache从GPU复制到CPU中，融合为一个大的，统一发送到目标实例。

# 评估
作者在16-GPU集群上使用真实（ChatGPT-4 conversation datasets, ShareGPT (GPT4) and BurstGPT (GPT4-Conversation)）和合成工作负载对Llumnix进行了评估，对比基线包括轮询调度、优化版INFaaS++等。

结果表明：Llumnix将尾延迟（P99）降低了高达15倍，将高优先级请求的延迟降低了1.5倍，并在保持类似尾延迟的情况下实现了高达36%的成本节约。

# 未来方向
1. 请求在异构实例之间的调度（不同的TP, PP等）。
2. 实例内调度技术（如抢占式调度）优化，设计early reject避免资源的浪费等。