---
layout: fast
title: EP03-vLLM源码讲解直播笔记-PD分离
date: 2025-03-30 19:22:07
summary: 
cover: posts_img/vLLM/cover.png
categories: 
  - Note
tags: 
  - vLLM
  - LLM
  - Inference
---

# [FIXME][EP03] vLLM 源码讲解直播笔记

## EP03: PD分离

直播回看链接：https://www.youtube.com/watch?v=ih6fcJnhoJI

特别鸣谢：月球大叔，Cheng Yihua 大佬带来的精彩讲解

### 📌 1. 上周回顾（分布式通信与并行策略）

- TP，all_gather
    - Linear M x N x K -> M x K
    - M * N', N' * K -> M * K
    在MHA中，每个头被均匀地分在不同的worker上，在进入之后的线性层前要做一次all_gather（这里不理解的可以看看-lm张量并行的方法）
    - "vllm\model_executor\models\llama.py"
    ```python
    # llama前向传播的代码
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output
    ```

### ⚡ 2. PD分离

- 什么是PD分离（Prefill和Decode）
    - prefill: 处理输入的prompt，生成KVCache
    - decode: 根据KVCache连续自回归生成一个一个的token
- 为什么要PD分离
    - prefill: attention N tokens QKV，与序列长度n的平方成正比，需要相当多时间
    - decode: attention N KV, 1Q，生成新的token，速度较快
    - 最初的逻辑：prefill优先
    - 问题：当新的一个request到来时，进行的prefill会使其他正在decode的request停住
    - 解决方法：
        - PD分离（PD disaggregation）
            - 挑战是P，D的数量
        - 分块预填充（chunked prefill），已在vllm v1版本中默认使用
            - 挑战是chunked_size的设置，这里有一个prefill和decode中的trade-off
- PD分离的关键问题
    - 怎么传输KVCache
        - 两种模式：pooling模式，P2P模式
        - LMCache都支持上面两种模式，Mooncake(pooling)，NIXL(p2p)
    - 怎么从vllm提取（注入）KVCache
        - connector API
            - 在model_runner中被调用，"vllm\worker\model_runner.py"
            - 在模型forward前：尝试接收KVCache并注入到到vllm的pages memory中
            - 模型执行
            - 在模型forward后，将KVCache从pages memory中并将它发送出去
              ![](posts_img/vLLM-EP03/1743330247803.png)
                - 两个函数的详细设计在"vllm\distributed\kv_transfer\kv_transfer_agent.py"中
                ```python
                # 本质上是根据model input计算出KVCache放在page memory中的什么地方
                def recv_kv_caches_and_hidden_states(
                    self, model_executable: torch.nn.Module,
                    model_input: "ModelInputForGPUWithSamplingMetadata",
                    kv_caches: List[torch.Tensor]
                ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
                        "ModelInputForGPUWithSamplingMetadata"]:
              
                    return self.connector.recv_kv_caches_and_hidden_states(
                        model_executable, model_input, kv_caches)
                ```
                这里不懂的可以去看"vllm\distributed\kv_transfer\kv_connector\simple_connector.py"
    - 什么时候将request从P node传输到D node
        - 先P后D（production stack）
        - 先D后P（D node收到后先检查是否有KVCache，没有的话再转给P node去做，这个思路主要考虑的是TTFT）


