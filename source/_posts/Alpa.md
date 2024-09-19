---
layout: fast
title: Alpa-自动生成DL/LLM模型并行策略
date: 2024-09-19 00:41:49
summary: 
cover: posts_img/Alpa/cover.png
categories: 
  - Paper Report
tags: 
  - LLM
  - DL
  - Train
---

只要输入DL模型 computation graph 和 device cluster，Alpa 通过生成统一**数据**、**运算**和**流水线**并行性的执行计划，在**可接受**的时间内**自动化**了大型深度学习（DL）模型的模型并行训练。

现有的模型并行训练系统要么要求用户手动创建并行化计划，要么从有限的模型并行配置空间中自动生成一个。它们不足以在分布式计算设备上扩展复杂的DL模型。Alpa通过将并行性视为两个层次来分配大型DL模型的训练：**运算内并行性(inter-operator)**和**运算间并行性(intra-operator)**。

基于此，Alpa为大规模模型并行执行计划构建了一个新的层次空间。Alpa设计了许多编译过程，以在每个并行级别自动导出高效的并行执行计划。Alpa实现了高效的运行时，以协调分布式计算设备上的两级并行执行。评估表明，Alpa生成的并行化计划与手动调优的模型并行训练系统相匹配或优于后者，即使在它们设计的模型上也是如此。与专用系统不同，Alpa还可以推广到具有异构架构的模型和没有手动设计计划的模型。

论文：(OSDI 2022)[Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning](https://www.usenix.org/system/files/osdi23-li-zhuohan.pdf)

源代码：https://github.com/alpa-projects/alpa.

文章参考自：
1. https://zhuanlan.zhihu.com/p/487588274
2. https://research.google/blog/alpa-automated-model-parallel-deep-learning/


# Alpa概念

作者首先将现有的机器学习并行化策略分为两类：

1. 运算内并行-Intra-Operator Parallelism：将Tensor按某些维度切裂，放到不同Device上计算的并行方式。如张量并行（如Megatron-LM）、数据并行（如Deepspeed Zero）、专家并行（如GShard MoE）。
2. 运算间并行-Inter-Operator Parallelism：即流水线并行

两类并行方式的特点：
1. Intra-op Parallelism：通讯量较大，可充分利用带宽，切分带来的通信基本属于高效的集合通信。
2. Inter-op Parallelism：只在两个设备间传输数据，通讯量较小，若切点寻找的合适，则通信较小，但同步版本的策略无可避免的会引来Bubble。

可以利用cluster的非对称特性，将Intra-op Parallelism映射到高带宽互联的devices上；将Inter-op Parallelism映射到低带宽互联的devices上。如此组合，就能释放更大的算力。Alpa会自动探索这些策略及组合情况。

在GPU集群中，节点内的GPU具有更高的通信带宽，可以适应运算内的并行性。然而，不同节点上的GPU通常以较低的带宽连接（例如以太网），因此首选运算间并行。

之所以要做这两种区分，目的是为了在不同的level上做策略搜索，然后将二者组合起来，生成大一统的并行方式的执行计划。总结起来就是以下两个点：

将并行度视为两级：intra-operator和inter-operator，构建分级的解空间
在两级空间中**分别**探索最优解，然后提供高效的Runtime将二者编排起来（总体上不能保证全局最优，但是实践证明在大模型上有很强的性能提升）。

# Alpa的Workflow

Alpa工作在DL编译层（基于XLA，在HLO上进行策略探索），所以文中也称之为——自动生成分布式策略的DL编译器。当用户给出模型计算图和设备集群时，它会进行各种操作流。

1. 运算间并行操作流-Intra-Operator Pass：传递将计算图切分为子图，将设备集群切片为子表（即分区设备集群），并确定将子图与子表的最佳配对方式。
2. 运算内并行操作流-Inter-Operator Pass：为运算间并行的每个流水线阶段找到最佳的运算内并行计划。
3. 运行时编排操作流-Runtime Orchestration pass：生成一个静态计划，对计算和通信进行排序，并在实际设备集群上执行分布式计算图（做runtime缝合的事情。比如stage调度，通信优化等，显然这个pass和策略搜索没什么关系）。

在宏观上做DP，微观上ILP，Inter-op和Intra-op两个pass不断迭代最终获得最佳方案。

下图中，在切片子图中，红色和蓝色表示运算的划分方式，灰色表示复制的运算，绿色代表实际设备（例如GPU）：

![Alpa概述。在切片子图中，红色和蓝色表示运算的划分方式，灰色表示复制的运算，绿色代表实际设备（例如GPU）。](posts_img/Alpa/1.gif)


## 运算内并行操作流（Intra-Operator Pass）

与之前的研究（例如Mesh TensorFlow和GSPMD）类似，运算内并行性在设备网格上划分张量。下图显示了Transformer模型中具有给定批次、序列和隐藏维度的典型3D张量。批处理维度沿着设备网格维度0（mesh0）进行划分，隐藏维度沿着网格维度1（mesh1）进行分区，序列维度被复制到每个处理器。

在2D设备网格上划分的3D张量：
![在2D设备网格上划分的3D张量](posts_img/Alpa/2.png)

通过Alpa中张量的划分，进一步为计算图中的每个独立运算定义了一组并行化策略。在下图中展示了矩阵乘法的并行化策略示例。在运算符上定义并行化策略可能会导致张量分区上的冲突，因为一个张量既可以是一个运算符的输出，也可以是另一个运算的输入。若分区方式不匹配，两个运算之间需要重新分区，这会产生额外的通信成本。

矩阵乘法的并行化策略：
![矩阵乘法的并行化策略](posts_img/Alpa/3.png)


给定每个运算和re-partition成本，作者将运算内操作流表示为整数线性规划（ILP）问题。对于每个运算，定义一个one-hot向量来枚举分区策略。ILP的目标是最小化计算和通信成本（节点成本）和重新划分通信成本（边成本）的总和。ILP的解决方案转化为一种特定的方法来分割原始计算图。

![](posts_img/Alpa/4.png)

## 运算间并行操作流（Inter-Operator Pass）

运算间传递对计算图和设备集群进行切片，以实现流水线并行性。如下图所示，方框表示输入的微批，流水线阶段表示执行子图的子网格。水平维度表示时间，并显示执行微批处理的传递阶段。运算间传递的目标是最大限度地减少总执行延迟，即设备上整个工作负载执行的总和。Alpa使用动态规划（DP）算法来最小化总延迟。计算图首先被展平，然后执行Intra-Operator Pass，对设备集群到子表的所有可能分区的性能进行分析。

对于给定的时间，此图显示了分区设备集群和计算图切片（例如，阶段1、2、3）正在处理的微批（彩色框）：
![流水线并行。对于给定的时间，此图显示了分区设备集群和计算图切片（例如，阶段1、2、3）正在处理的微批（彩色框）。](posts_img/Alpa/6.png)

# 评估

使用8个AWS p3.16xlarge实例测试Alpa，每个实例有8个16GB V100 GPU，总共64个GPU。研究了在增加GPU数量的同时增加模型大小的弱缩放结果。我们评估了三种模型：
1. 标准Transformer模型（GPT）；
2. GShard MoE模型，一种混合了专家层的Transformer；
3. Wide ResNet，一个明显不同的模型，没有现有的专家设计的模型并行化策略。性能是通过集群上每秒实现的peta浮点运算（PFLOPS）来衡量的。

作者证明，对于GPT，Alpa输出的并行化策略与现有最佳框架Megatron ML计算的策略非常相似，并与其性能相匹配。对于GShard MoE来说，Alpa在GPU（即Deepspeed）上的表现比专家设计的最佳基线高出8倍。Wide ResNet的结果表明，Alpa可以为专家尚未研究的模型生成最佳并行化策略。本文还展示了线性缩放数以供参考。

![](posts_img/Alpa/7.png)

在16个GPU上，Alpa将模型分为3个阶段，并分别为第1、2、3阶段分配4、4、8个GPU。在前两个阶段，数据并行性是首选，因为激活张量大于权重张量。在第三阶段，ILP求解器找到了一种划分卷积算子的非平凡方法。结果表明，对于像Wide ResNet这样的异构模型，即使对于领域专家来说，手动创建这样的策略也可能是困难的。

Alpa在16个GPU上为WideResNet找到的并行化策略：
![Alpa在16个GPU上为WideResNet找到的并行化策略](posts_img/Alpa/8.png)

# 结论

为分布式模型并行深度学习设计有效的并行化计划的过程历来是一项困难且劳动密集型的任务。Alpa是一个新的框架，它利用运算内和运算间的并行性进行自动化模型并行分布式训练。相信Alpa将使分布式模型并行学习规范化，并加速大型深度学习模型的开发。