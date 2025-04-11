---
layout: fast
title: EP04-vLLM源码讲解直播笔记-Speculative Decoding
date: 2025-04-05 17:44:31
summary: 
cover: posts_img/vLLM/cover.png
categories: 
  - Note
tags: 
  - vLLM
  - LLM
  - Inference
---

# [FIXME][EP04] vLLM 源码讲解直播笔记

## EP04: Speculative Decoding

直播回看链接：https://www.youtube.com/watch?v=WF5xaQqtKUE

特别鸣谢：月球大叔，Du Kuntai, Cheng Yihua 大佬带来的精彩讲解

### 📌 1. 为什么需要 Speculative decoding（推测解码）
- LLM的decode过程是GPU-Memory-Bound (GPU内存受限) 的
    - 寻找一种方法能够增加计算次数，但不显著增加对GPU内存的访问次数

- 解决方法：在decode生成token的时候 --> 用小模型猜多几个token并验证
    - 在 token 生成的每1次迭代中
        - 猜3个token，接受率为2/3
        - 2个token是猜测正确的，LLM推理每次还会生成1个新的token --> 3 tokens
    - 一次的迭代所需要的时间
        - 计算量：(1 + 3)x
        - 内存量
            - 没有 Speculative decoding 时：模型参数（8x2 GB）+ KVCache（n * 100 KB）
            - 有 Speculative decoding 时：模型参数（8x2 GB）+ KVCache（（n+3） * 100 KB）
        - 一次迭代的时间不变，吞吐量增加3倍

- 衡量一个操作是 computation-bound（计算受限型）还是 memory-bound（内存受限型）的指标是 arithmetic intensity（计算强度）：
    - 定义：FLOPS（每秒浮点运算次数）/ MIPs（内存指令次数）

- Speculative decoding虽然是个很好的优化点，但在实际落地的过程中还面临很多工程上的困难

### ⚡ 2. 怎么猜测 token?
- N-gram
    - 构造一个mapping：如果前3个tokens是A, B, C，接下来两个tokens是D, E
    - 示例：
        - 如果前3个tokens是 `To be or`，接下来两个tokens是 `not to`
        - 如果前3个tokens是 `be or not`，接下来两个tokens是 `to be`
        - ...
        - 如果前3个tokens是 `, this is`，接下来两个tokens是 `a question`

    - 从请求输入中构建N-gram，使用这个N-gram来猜测tokens：
        - 接下来是莎士比亚的一些名言：

            ...
            `To be or not to be, this is a question`
            ...
        
        里面最经典的一句名言是什么?
        - 假设LLM已经生成：
            - `Sure! We recommend you this quote: "To be or`
            - 猜测：接下来的两个tokens是 `not to`
            - 验证：正确
            - 输出：`Sure! We recommend you this quote: "To be or not to be"`

- Model-based（draft model）
    - Parallel guessing（并行猜测）
        - 优点：快
        - 缺点：在猜测第二个token的时候不知道第一个token是什么
    - Autoregression guessing（自回归猜测）
        - 优点：在猜测第二个token的时候知道第一个token
        - 缺点：慢

- Deployment（尤其是model-based）存在的问题
    - 小模型需要KVCache，应该怎么放置？
    - 小模型小，需要不同的并行策略
        - 假设小模型不并行，小模型在0号GPU + vLLM强制不同的GPU有一致的GPU内存利用率（同一个并行组内）--> 会造成其他GPU内存的浪费
    - 要为guessed tokens提前allocate KVCache
        - 如果allocate KVCache要跨越vLLM的block边界怎么办
        - 需要discard的token
    - 从 Sampling --> verification 阶段的转变
    - 最小化 overhead (Ngram)
    - 怎么确定每一次应该guess多少个tokens
    - 怎么在不同requests之间区分
        - 不同的request：不同的token数量，它们其中的一部分不进行spec decode

### 📍 3. 怎么验证 token 的正确性?
- Tree verification（树验证）
    - `To be or` --> `not to`, `sleep in`, `go to`
    - `To be or` 有很多种猜法: `not to`, `sleep in`, `go to`

- LLM怎么验证预测的是正确的?
    - Deterministic sampling（确定性采样）(spec decode bad case)
    - Random sampling（随机性采样），当 guess probability > threshold 就正确

- 示例：
    - 输入：`To be or` (already-decoded output) `not to` (guessed token)
        - `To be or not to`
        - `To -> be`
        - `be -> or`
        - `or -> not` 我们的猜测 "not" 是正确的
        - `not -> to` 我们的猜测 "to" 是正确的
        - `to -> be` "be" 是正确的下一个token

    - 输入：`To be or` (already-decoded output) `not be` (guessed token)
        - `To be or not be`
        - `To -> be`
        - `be -> or`
        - `or -> not` 我们的猜测 "not" 是正确的
        - `not -> be` 我们的猜测 "be" 是错误的，应该是 "to"
        发现有错误的token后，后面的预测会舍弃


