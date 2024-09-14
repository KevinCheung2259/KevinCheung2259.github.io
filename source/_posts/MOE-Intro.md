---
layout: fast
title: MOE基础介绍
date: 2024-09-13 16:53:07
summary: 简单介绍MOE的优缺点、模型框架等
cover: posts_img/MOE-Intro/1.png
categories:
  - Intro
tags: 
  - LLM
  - MOE
---

MOE重要性：坊间一直流传GPT-4是MoE模型
本文主要参考自：https://huggingface.co/blog/zh/moe

# 什么是MOE
基于 Transformer 架构的模型，混合专家模型主要由两个关键部分组成:

● **稀疏 MoE 层**: 这些层代替了传统 Transformer 模型中的前馈网络 (FFN) 层。MoE 层包含若干“专家”(例如 8 个)，每个专家本身是一个独立的神经网络。在实际应用中，这些专家通常是前馈网络 (FFN)，但它们也可以是更复杂的网络结构，甚至可以是 MoE 层本身，从而形成层级式的 MoE 结构。

● **门控网络或路由**: 这个部分用于决定哪些令牌 (token) 被发送到哪个专家。例如，在下图中，“More”这个令牌可能被发送到第二个专家，而“Parameters”这个令牌被发送到第一个专家。有时，一个令牌甚至可以被发送到多个专家。令牌的路由方式是 MoE 使用中的一个关键点，因为路由器由学习的参数组成，并且与网络的其他部分一同进行预训练。

![](posts_img/MOE-Intro/1.png)
<div style="text-align: center;">
  <a href="https://arxiv.org/pdf/1701.06538">Outrageously Large Neural Network 论文中的 MoE layer</a>
</div>


![](posts_img/MOE-Intro/2.png)
<div style="text-align: center;">
  <a href="https://arxiv.org/pdf/2101.03961">Switch Transformers paper 论文中的 MoE layer</a>
</div>

# 混合专家模型（MoEs）简短总结

## 优点
● 与稠密模型相比，**预训练速度更快**
● 与具有相同参数数量的模型相比，具有更快的 **推理速度**
● 与具有相同激活参数的稠密模型模型相比，具有更高的 **推理精度**

## 缺点
● 需要 **大量显存**，因为所有专家系统都需要加载到内存中（现有offload技术）
● 在 **微调方面** 存在诸多挑战，但 [近期的研究](https://arxiv.org/pdf/2305.14705) 表明，对混合专家模型进行指令调优具有很大的潜力
● 不同的专家倾向于专注于不同的 **语义**，而不是特定 **领域**
● 由于设备间需要传递数据，网络带宽常常成为性能瓶颈，应采取适当的并行化策略（3D并行+专家并行）

# 参考文献
[1] Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (2017)
[2] Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity (Jan 2022)
