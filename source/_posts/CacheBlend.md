---
layout: fast
title: CacheBlend-高效提高KVCache复用性的方法
date: 2025-03-29 18:24:59
summary: 
cover: posts_img/CacheBlend/1.png
categories: 
  - Paper Report
tags: 
  - EuroSys 2025
  - LLM
  - Inference
  - KVCache
---

CacheBlend，一种用于加速 LLM 服务的高效KV缓存融合方案，特别针对检索增强生成（RAG）等需要多文本块上下文的场景，能有效提高 KVCache 的复用性。

论文：(EuroSys 2025)[Fast Large Language Model Serving for RAG with Cached Knowledge Fusion](https://arxiv.org/abs/2405.16444)

代码：https://github.com/YaoJiayi/CacheBlend


# 问题背景

为了确保模型高质量的一致性的响应，RAG技术常被应用于LLM服务中，在这种情况下，多个从数据库检索来的文本片段会前置于用户的查询，形成LLM输入，这些长上下文片段显著地提高了LLM的TTFT时间。

为了提高prefill的时间，现有的优化通常重复使用存储的 KVCache，以避免重复的计算。目前有以下两类 KVCache重用的方法，但它们都存在局限性。


## 前缀缓存

仅能重用输入前缀的KV缓存，无法处理多文本块场景。


##  全量KV重用

当重用的文本不在输入前缀中时，仍然通过调整其位置嵌入来重用 KV 缓存（这里用到了 Rotary Position Embedding(RoPE) 编码相对位置的特性），从而使 LLM 生成有意义的输出（Prompt cache 2023）。但这样会忽略与之前文本块之间的**跨注意力**（cross-attention），导致生成质量显著下降。（如果这里不理解什么是cross-attention的话，可以重新回顾一下整个prefill阶段的计算流，tensor在不同的layer前向传递过程中，上下文的内容会不断融合）
   
![](posts_img/CacheBlend/2.png)

这个图表明了：
1. 随着选择的文本片段数量增加，生成质量显著提高，尽管包括在太多片段时会由于lost-in-the-middle问题而降低质量。
2. 随着选择的文本片段数量增加，由于交叉注意力缺失造成的性能损耗增大。

| ![Image 1](posts_img/CacheBlend/3.png) | ![Image 2](posts_img/CacheBlend/4.png) |
|:------------------------------:|:------------------------------:|

左图表示了，由于忽略交叉注意力可能导致错误的回答。为了理解这一点，可以仔细看一下两个文本片段之间的交叉注意力。可以看到，相比于完整prefill后的结果，全量KVCache复用时两个片段之间的交叉注意力被忽略了。

这里要说明一个点，当各片段之间的交叉注意力较低时，片段内自注意力影响较高时，完全重用 KV 是可以奏效的。这种情况在 PromptCache 的主要目标应用（gim2023prompt）的提示模板中较为常见。
   
## 挑战

如何在重用非前缀文本块的KV缓存时，保持生成质量与完全预填充（full prefill）一致，同时减少计算延迟？

![](posts_img/CacheBlend/1.png)

作者在这里做了一个trade-off，的核心思想是在 Full KV Reuse 的基础上再计算少量 token 的注意力以恢复文本块之间的交叉注意力。

#  核心方法：选择性KV重计算

## 选择性KV重计算（Selective KV Recomputation）

- **目标**：仅更新对生成质量影响最大的部分token的KV缓存，避免全量预填充的计算开销。
- **实现步骤**：在每一层（Layer）的输入中，通过掩码仅保留需要重计算的token（HKVD tokens），计算所需的QKV，其余token的KV值直接复用预计算的缓存。

![](posts_img/CacheBlend/5.png)

## 重计算Tokens的筛选
知道了要重计算关键token，那么如何选择这些token呢？

- **定义**：**高KV偏差（High KV Deviation, HKVD）Token**  
  指其预计算KV值与完全预填充（Full KV Recompute）结果的差异超过阈值的token。  
  - **量化指标**：  
    $$\Delta_{\text{kv}}(KV_i[j], KV_i^{\text{full}}[j]) = |KV_i[j] - KV_i^{\text{full}}[j]|$$  
    其中，$KV_i[j]$ 表示第i层第j个token的KV值。
## 筛选依据  
### 注意力稀疏性
仅约10-15%的token的KV偏差远高于其他令牌，对跨文本块注意力贡献显著。
![](posts_img/CacheBlend/6.png)

### 层间相关性
如何在不知道真实 KV 值或注意力矩阵的情况下识别 HKVD 令牌？简单来说，要识别 HKVD 令牌，首先必须知道每一层的完全重新计算的 KV，但这太过昂贵且违背了选择性 KV 重新计算的目的。相反，作者观察到不同层的 HKVD 令牌并不是独立的！

通过Spearman秩相关性分析，发现相邻层的HKVD tokens高度重叠，允许跨层渐进筛选，这是因为各层之间的 KV 缓存具有相似性。

![](posts_img/CacheBlend/7.png)

### 渐进式筛选流程（Gradual Filtering）
如果平均每层想要挑选 $r$% 个 HKVD 标记：
1. **首层筛选**：在第一层预填充时，计算所有token的KV偏差，选择偏差最高的前$r_1$%作为候选，例如20%）。  
2. **逐层过滤**：  
   - 对后续每一层，仅计算上一层的候选token的KV偏差，从中筛选偏差更高的\(r_i\%\)（逐步逼近目标比例\(r\)）。  
   - 最终每层仅重计算约10-15%的token，显著减少计算量。
  
![](posts_img/CacheBlend/8.png)


# 系统优化：延迟隐藏与动态控制

## 系统架构
![](posts_img/CacheBlend/10.png)

## 流水线并行（Pipelining）
- **KV加载与重计算并行**：  
  - 在GPU计算当前层的选择性KV重计算时，异步加载下一层的预计算KV缓存（从SSD/内存到GPU）。  
  - 若加载时间 ≥ 重计算时间，额外延迟被完全隐藏。  
- **优势**：支持将KV缓存存储在低速设备（如SSD）中，节省内存成本，同时不影响实时性。

![](posts_img/CacheBlend/11.png)

## 动态控制器（Loading Controller）
- **动态调整重计算比例**：  
  - 根据存储设备速度（如SSD吞吐量）和模型规模，计算最大可容忍的重计算比例\(r\%\)，使得：  
    $$T_{\text{recompute}}(r\%) \leq T_{\text{load}}(storage\_device)$$  
  - 确保总延迟不增加，同时质量损失可控（经验默认r = 15\%\)）。
- **存储设备选择**：  
  在成本与延迟间权衡，优先选择满足延迟约束的最廉价存储设备（如SSD而非GPU显存）。

![](posts_img/CacheBlend/9.png)

# **实验结果**
- **性能提升**：  
  - **服务质量**：相比完全预填充，首token时间（TTFT）减少 **2.2-3.3倍**，吞吐量提升 **2.8-5倍**。  
  - **生成质量**：与完全预填充相比，F1/Rouge-L分数下降小于 **0.02**，显著优于完全KV重用（质量提升最高达0.35）。  
- **适用性**：在多个模型（Mistral-7B、Yi-34B、Llama-70B）和任务（QA、摘要）中验证有效性，支持不同文本块长度和批量大小。

![](posts_img/CacheBlend/12.png)


# **贡献与意义**
- **理论价值**：揭示了注意力稀疏性与KV偏差的关系，为高效缓存复用提供了新视角。  
- **工程价值**：通过流水线和动态控制，实现了低成本、高质量的KV缓存复用，兼容现有LLM服务框架（如vLLM）。  
- **应用场景**：适用于需要多上下文交互的RAG、长文本生成等任务，显著降低服务延迟与计算开销。

# **局限与展望**
- 目前仅验证了Transformer架构，未来需适配Mamba等新型模型。  
- 可进一步结合KV压缩技术（如量化、剪枝）减少存储开销。