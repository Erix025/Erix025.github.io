# An Introduction to Efficient LLM Inference

> 本文为一次面向 [ZJUSCT](https://www.zjusct.io/) 内部分享的内容整理，旨在介绍大模型推理中常用的高效推理手段，包括量化、并行、稀疏注意力等技术。

## LLM Inference Overview

### Core in LLM Inference

#### Overview of AR

首先我们来梳理一下 LLM 推理中的关键过程。在大家使用 LLM 的时候，会发现他是一个一个词逐渐输出的。

这是因为 LLM 的整个过程就是一个 Next-Token-Prediction。每一次前向其实是把一个序列输入进入，预测整个序列的下一个词，然后不断重复这个过程。

$$
X_{t+1} = f(X_t)\\
X_t=[x_0, \cdots, x_p, x_{p+1}, \cdots, x_{p+t}]
$$

每一个 $x_i$ 都是一个 token 对应的向量 (embedding)。这种通过不断调用 $f$ 来实现序列理解生成的模式我们也叫做 **自回归** (Auto Regressive)。

#### Fundamental Components in LLM

目前主流的 LLM 由以下几个组件构成：

1. Tokenizer & Input Embedding
2. Transformer
3. `lm_head` and Sampler

一个文本序列经过 Tokenizer 分成一个个 token 得到 token id 序列，然后经过 Input Embedding 得到各自的词向量。然后通过最核心的 Transformer，最终经过 `lm_head` 得到词表中每个词的概率并由 Sampler 进行采样得到最终的输出 token。

而其中绝大部分的参数量和计算量都集中在 Transformer 这个部分。因此我们今天也主要针对这部分优化进行展开。

Transformer 大家想必不陌生，最初 Transformer 的设计是由 Encoder 和 Decoder 两个部分组成。而目前主流的 LLM 都是采用 Decoder Only 的设计。

![Transformer 示意图](Attention is All You Need)

Transformer 有若干层，其中每一层都有一个 Self Attention 和一个 MLP。

对于 Self Attention，最核心的就是 Attention 机制。他是通过计算 Q 和 K 的相关性得到 attention score，然后加权求和 V 得到最终的输出。

而 Self Attention 他的 Self 关键在于怎么得到 QKV。QKV 在这里都是通过 x 经过三个 proj 得到的。因而叫做 self attention。与之相对的 Cross Attention 则是 Q 和 KV 来源不相同，这里不过多展开。

![Self Attention 示意图](把 lab5 的抄过来)

Attention 计算的复杂度是我们比较关心的，已知 QKV 的形状都是 [n, d]，那么如果只关注矩阵乘的开销，我们会得到一个 $O(n^2d)$ 的复杂度，当 $n \gg d$ 时（即面对序列长度很大的场景），Attention 的复杂度将变为 $O(n^2)$，这带来了大量的计算开销，也是我们后续优化的重点。

这里还要补充一下 Causal Mask 的内容。因为在 LLM 中，我们希望模型根据前面的内容预测后面的知识，因此不希望其看到后面的内容。所以我们使用一个下三角矩阵作为 mask，使 Q 中的每一个 token 只能看到 K 中在他之前的 token。

![Attention 示意图](展现出 QK 的 attention matrix)

而 MLP 部分则是大家熟悉的两层 Linear 加一个激活函数的组合。

近年来尤其是 DeepSeek-V3/R1 以来，MoE(Mixture of Expert)模型成为主流。即使用多个 MLP 来存储模型知识，而在实际推理时只调用其中 k 个 MLP，实现了在模型规模增大的同时不等比例增加推理开销。

![MoE 示意图]

### Prefilling and Decoding

理解了 Transformer 的结构后，我们来看实际推理时是如何运行的。由于 LLM 是 自回归（Auto-Regressive） 模型，生成 $N$ 个 token 就需要调用 $N$ 次模型。但在这 $N$ 次调用中，第一步和后续步骤的计算逻辑存在巨大的非对称性，因此我们将其分为两个阶段：Prefilling & Decoding

#### Prefilling

当你输入一段 Prompt（长度为 $n$）时，模型需要一次性处理所有输入 token。

- 特点：这是一个计算密集型的过程，其在整个 transformer 中流动的 x 形状是 [n, d]。虽然我们最终只需要最后一个 token 预测出的概率分布来生成第一个新 token，但为了得到这个结果，前面的 $n-1$ 个 token 的计算是必经之路。
- 产物：在这个阶段结束时，我们不仅得到了第一个生成的 token，还得到了输入序列所有 token 的 $K$ 和 $V$ 向量。

#### Decoding

从生成第二个新 token 开始，推理进入了逐个生成的循环。

理论上，生成第 $n+i$ 个 token 时，我们需要输入前面所有的 $n+i-1$ 个 token。但你会发现一个明显的计算冗余：前 $n+i-2$ 个 token 的 $K$ 和 $V$ 矩阵在之前的循环中已经计算过了，且由于 Causal Mask 的存在，旧 token 的特征并不会受到新 token 的影响。因此，我们不再重新计算整个序列，而是每次只输入上一步产出的那一个 token。

这个阶段的特点是，由于只需要输入一个 token，此时 x 的形状将会是 [1, d]，因此通常情况下这个阶段是 memory-bound 的。

#### KV Cache

这里就引出了推理优化的第一个关键点。如果在 Decoding 阶段只输入一个 token，Attention 层如何计算它与历史 token 的相关性？为了解决这个问题，我们需要在 Prefilling 阶段将所有 token 计算出的 $K$ 和 $V$ 向量缓存在显存中，这就是 KV Cache。在 Decoding 的每一步，模型只需要计算当前新 token 的 $K_{new}$ 和 $V_{new}$。将其追加（Append）到 KV Cache 中。依靠缓存的“历史记忆”，当前 token 就能通过 Attention 看到整个上下文。

### Challenges in LLM Inference

随着模型规模的增大，LLM 推理面临着以下几个挑战：

1. 模型参数量大：在模型大小不断 scale up 的过程下，当前主流模型参数已经向 >1T 级别迈进，这一方面在有限显存资源下的推理变得困难，另一方面在前向计算时对权重的访存开销也大幅增加。
2. 序列长度大：在实际应用中，上下文的长度需求不断增加。从应用场景看，一些如文档理解、代码生成等任务让 prefilling 阶段的输入长度达到 10k+ 已经变得常见。而在 decoding 阶段，自 DeepSeek-R1 以来 thinking model 和 agent 等应用场景也让生成长度不断增加。这些都让推理的计算和内存开销大幅增加。

围绕这两个问题，这几年大家在 LLM 推理中也做了很多一些工作，今天的分享我会抛砖引玉式地介绍一下他们，感兴趣的同学可以顺着这个脉络进一步了解。

## Model Compression and Parallelism

### Quantization

大家在之前的比赛中如果遇到 AI 相关的赛题，或多或少会接触到**量化 (Quantization)** 这个概念。

量化最开始的 Motivation 是，大家发现在深度学习当中，精度并不是那么的重要。一是决定一个模型 capacity 的要素除了每个权重的精度还有权重数量、模型结构等等，另一个是经过额外的训练后模型也能够弥补由于精度表示误差所带来的损失。因此低比特模型量化就成为目前大模型训练/推理加速中的一个重要手段。

量化最核心的概念就是用低比特数据类型来表示原来的数据。量化的过程就是将数值从高比特表示向低比特表示映射的过程，这是一个单向的、有损的过程。

![数据类型]()

目前主流的量化方法是采用线性映射的方式，这里会引入两个量化参数 scale 和 zero_point。通过这两个参数，我们能够将一组数值从高比特表示映射到低比特表示。并且能够通过“反量化”这个过程再将其还原到原来的表示（当然还原后肯定会引入误差，因为低比特表示的取值可能数少于高比特表示，这意味着高比特表示下两个不一样的数值有可能映射到同一个低比特表示，从而带来 error）

![线性映射过程示意图]()

因此量化的过程就是给定一组数据，计算他们的 scale 和 zero_point，然后据此计算低比特表示 x_q，最终返回 (scale, zero_point, x_q)。

所以这里我们会发现，从存储空间上看，量化后的压缩比并不完全等于前后数据类型的比特数之比，因为还有量化参数所带来的额外占用。那么这里就会引入量化的另一个内容：量化粒度（Granularity）。

量化粒度他决定了多少数值共用同一组量化参数。例如如果用 per-tensor 的粒度去做，那么一整个 weight tensor 只会共用同一组量化参数。这确实会引入较小的 overhead，但是与此同时带来的问题也是表示精度的下降。如下面这张图所示，这是某个 activation 矩阵在 dimension 维度和 token 的数值分布情况。可以看到整体的数值分布范围是非常大的，如果使用 per tensor 的粒度，那么映射之后一些数值差异小的地方会被映射到同一数值，这时候产生的 error 就会非常大。而我们如果逐个 channel 去看，会发现他们的数值范围其实只局限在一个更小的范围，如果我们 per-channel 而不是 per-tensor 地去做量化，虽然量化参数的数量增多了，但是同时也减少了量化前后的损失。因此量化精度的选择实际上是一个量化参数 overhead 和量化精度损失之间的一个取舍。

当然这里的 per-channel 只是举一个例子，实际的量化粒度会更复杂，例如目前主流的方案其实是通过 per-group 的粒度去做，例如将 128 个数值划分为一组去共享一个量化参数来去做。

![Smooth Quant 示意图来说明粒度的影响。]

讲完量化的一个基本概念，那么我们来看看他是怎么用到推理加速当中的。

#### Weight-Only Quantization

在早期，大家最开始在权重上做量化探索，即只量化权重而不量化 activation。这样的好处是可以显著减小大模型权重的显存占用，同时能够减少前向过程中对权重的访存量。

但与此同时，如果只对权重做量化，在实际计算中其实无法利用低比特的计算单元，因为此时 activation 还是 fp16/bf16 的。因此在 weight only 下，会先将权重进行反量化，得到高比特表示的 weight，再和 activation 在高比特下进行乘法。

目前在 LLM 上主流的量化方法已经可以做到 training-free 的 W4A16 近乎无损。

#### Weight-Activation Quantization

在 weight only 下取得巨大成功后，大家就开始继续在 activation 上进行探索。因为如果将 activation 同时进行量化，那么就可以充分地利用硬件上的低比特计算单元，从而在计算上获得更好的加速效果。

例如我们当时 ASC25 DeepSeek 在 CPU 上的方案就是一个 W4A8 的方案。即 weight 使用 4-bit 量化，然后在计算时 dequant 到 8-bit，利用 8-bit 的 AMX/AVX 计算单元进行计算。

当然对 activation 的量化也会引入更大的精度损失。目前主流效果较好的量化方法还是基于 W4A8 在做，W4A4 目前也已经有一些工作在推进，尤其是在 NVFP4 这种数据类型出来之后。

#### Nowadays' Quantization

上面介绍的量化貌似其实是一个很简单的东西，只要挑好数据类型和量化粒度就能开干了，那么学术界的一堆量化工作在做什么呢？正在追求如何“既要又要”。

在 LLM 上，大家发现有很多的 outlier，即那些数值离群点。他们会影响到 scale/zeropoint 的计算，进而带来更大的量化损失。因此大家推出了一系列的工作来想方法处理 outlier 减少量化损失。例如提出缩放来解决的 SmoothQuant 和利用旋转矩阵来解决的 SpinQuant 等。这里我不再过多展开，大家可以进一步了解。

### Parallelism

讲完了量化，我们来讲另一个维度加速的方法。围绕着刚刚提的模型大的问题，量化是从压缩模型大小的角度，而并行则是从将模型推理过程分散到多个设备来利用更多计算/存储资源的角度来加速模型推理。

推理阶段的并行策略主要有以下几种：

- **数据并行（Data Parallelism）**：将数据分成多个 batch，在不同的设备上处理。
- **流水线并行（Pipeline Parallelism）**：将模型分成多个阶段，每个阶段在不同的设备上处理。
- **张量并行（Tensor Parallelism）**：将模型的参数分布到不同的设备上，每个设备处理一部分参数。
- **序列并行（Sequence Parallelism）**：将长序列分成多个短序列，在不同的设备上处理。
- **专家并行（Expert Parallelism）**：将模型分成多个专家，每个专家在不同的设备上处理。针对于 MoE（Mixture of Experts）模型。

由于推理阶段只需要考虑模型的 forward 过程，因此并行策略的实现会相对简单。

#### Data Parallelism

推理阶段的数据并行其实比较简单。他的关键是以 request 作为并行拆分的最小单位，将不同的 request 分布到不同的 device 上进行推理。

- Memory 占用：每个 device 上有完整的一份模型副本。
- Communication：只在每条 request 的推理开始 / 结束会有通信，而且只需要对最终的 input / output 进行通信，通信量小。
- Throughput：当同时需要推理的 request 数接近于 infinity 时，增大并行度对 throughput 的增益是接近线性的（通信量小）
- Latency：并行度的增加对 latency 不会有正面效果，因为每个 request 分到的计算资源是一致的。

#### Pipeline Parallelism

流水线并行是将单次前向过程划分为多个 stage，以 stage 作为单位进行拆分，将不同的 stage 分布到不同的 device 上进行推理。

- Memory 占用：每个 device 上只有其所负责的 stage 对应的模型参数。
- Communication：在单个机器所负责的 device 完成后会发生通信，需要将 hidden states 传递给后续阶段的 device。
- Throughput：在理想状态下，某一时刻流水线中所有 device 都在进行计算，总吞吐量会增加，类比 CPU 的流水线。
- Latency：Latency 会下降，因为对于每个 request 而言，每次都只有一个 device 的计算资源参与前向过程，并且还有 device 间通信带来的 overhead。

相比于 Data Parallelism，其对于 throughput 的提升会相对较低。但 PP 的优势是他能够拆分模型权重到不同的 device 上，从而能够在多个 device 上部署规模较大的模型。

#### Tensor Parallelism

![Tensor Parallelism](https://docs.nvidia.com/nemo-framework/user-guide/latest/_images/tp2.png)

和 PP 一样，Tensor Parallelism 也是一种拆分模型权重的并行方法（Model Parallelism），但是它拆分的粒度会更细。TP 是针对每个模型权重张量 $W$ 进行拆分。

![TP in GEMM](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tgi/TP.png)

以一个简单的 Linear 为例，TP 是将 weight 在 output channel 这个维度上进行切分并分布到多个 device 上，计算出结果后再将结果 gather 起来。

- Memory 占用：能够将模型权重拆分到多个 device 上，且拆分的粒度比 PP 更细。
- Communication：在 TP 中存在大量的 collective communication，因为往往输出需要在 output channel 这个维度 gather 在一起。
- Throughput：TP 能够通过利用更多计算资源对单个 request 进行计算，从而减小单个 request 的时间，提升 throughput。
- Latency：如上述，单个 request 利用的计算资源更多了，latency 会大大减小。

纵然 TP 能够在 throughput 和 latency 两个维度都有很好的提升，但这个前提是建立在通信带宽和延迟非常优秀的情况下。由于 TP 会有非常大的通信开销，往往需要使用高速互联网络（如 Infini bands, NV links）等方式才能够在 scaling out 上使用 TP 取得很好的提升。

### Sequence Parallelism

Sequence Parallelism 在 TP 的基础上继续在 input channel 维度进行切分。相比于 TP 在 output channel 进行切分后需要对 result 进行 gather 以参与下一次运算（例如 MLP 的两层 Linear），SP 可以在保持 input channel 被切分的情况下继续参与下一次运算，从而减小集合通信的次数。

以 Transformer Decoder Block 为例，若在 Prefilling 阶段采用 SP，那么就可以在 MLP 中完全不进行集合通信，每个 partition 单独经过 Linear, BatchNorm, Dropout 等层，直到需要进行 Attention 计算时再通过集合通信 Concatenate，相比 TP 减小了集合通信量。

然而 SP 由于需要对 input channel 进行拆分，即拆分 activation。因此和 input shape 强相关，从而对并行调度和对不同场景的适配性不如 TP 稳定（因为 output channel 通常是由模型决定固定不变的）。例如在 LLM Inference 当中，prefilling 阶段能够很好的在 token 维度使用 SP 进行切分，但在 decoding 阶段却很难进行切分，因为 token 维度始终为 1。

- Memory 占用：减小 activation 的峰值占用。对模型的内存占用没有影响。
- Communication：同样需要对 activation 进行集合通信，但相比 TP 次数减少。
- Throughput & Latency：和 TP 一样同样能够利用更多计算资源进行计算，提升 throughput，减小 latency。

目前主流的实现有两种，一种是 Ulysses，通过在 Head 维度拆分来实现。他的好处是在 Attention 时不需要通信，分 head 去做就可以。另一种是 Ring，他是通过在 Sequence 维度拆分来实现，那么他的特点是在 Attention 时需要经过 N 次通信来实现全局 Attention 信息的计算。

### Expert Parallelism

Expert Parallelism 是针对 MoE（Mixture of Experts）模型的并行策略。MoE 模型通常包含多个专家（Expert），每个专家是一个 FFN（Feed Forward Network），在每次前向过程中只会激活其中的一部分专家。

![MoE](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/00_switch_transformer.png)

而在 MoE 模型中，主要的显存压力在于专家参数上。因此通过将专家分布到多个 device 上来实现 Expert Parallelism，从而减小显存占用。

- Memory 占用：能够将 MoE 模型权重拆分到多个 device 上
- Communication：在每次 FFN 专家层计算结束后需要进行 all-to-all
- Throughput&Latency：在专家层计算时利用更多的计算资源，减小专家层的计算时间，从而提高 Throughput 和 Latency。

## Long-Context Inference

在目前 thinking model 和 agent 等场景下，其实会遇到一个更加重要的问题就是 sequence length 越来越长了。一个是 input length 越来越长，在文档理解、agent 多轮交互等情况下对 prefilling 阶段的计算带来很大开销。另一个是 output length 越来越长，尤其是在 DeepSeek-R1 之后的 test-time scaling 范式下，long decoding 带来的更大 KV Cache 访存压力以及越来越大的计算开销也是不可忽视的。

因此需要有一系列优化手段来提升长序列推理能力。

### KV Cache Compression

我们知道 KV Cache 的存储实际上是和序列长度成 $O(n)$ 线性关系的。因此长序列的 KV Cache 存储和访存压力也是不可忽视的。为了解决这个问题，大家提出了一系列的 KV Cache Compression 方法。

#### More than MHA

为了缓解 KV Cache 的压力，大家提出了一系列 Attention 的变种来解决。

- MHA(baseline): 多个 head 各自一套 query, key, value。
- MQA(Multi-Query Attention): 所有 query head 共享同一个 key/value head。将 KV Cache 压缩到 $\frac 1 H$，但对性能影响较大。
- GQA(Group-Query Attention): 在 MHA 和 MQA 中间进行插值，即将 heads 划分成多个 group，在一个 group 内多个 query head 共享一个 key head。目前主流开源模型采用的方法。
- MLA(Multi-Latent Attention): 由 DeepSeek 提出，将 x -> KV 变成 x -> shared latent KV -> KV，从而极大降低 KV Cache 压力（详见 DeepSeek-V2 tech report）

#### KV Cache Eviction

最简单的思路是直接丢弃掉一些不重要的 KV Cache，从而减小 KV Cache 的大小。有一系列的工作（H2O, snapKV 等）都采用了这种思路，通过某种方法来评估每个 token 的重要性，然后丢弃掉不重要的 token 的 KV Cache，从而减小 KV Cache 的大小。

后来有一些工作（如 Quest）发现，对于先前 token 不重要的 KV 在后续的推理过程中可能会变得重要，因此直接丢弃掉不重要的 KV 会带来较大的精度损失。于是相比于直接丢弃掉不重要的 KV，Quest 提出将不重要的 KV 先 offload 到 CPU 上，然后在后续需要时再将其 load 回 GPU 上进行计算，从而在减小 GPU 端 KV Cache 占用的同时减少精度损失。

#### KV Cache Quantization

另一种思路是通过量化来减小 KV Cache 的大小。例如 KIVI 这篇工作就是将 KV Cache 压缩到 2-bit，从而大幅减小 KV Cache 的存储占用，同时也减少了 decoding 阶段对 KV Cache 的访存开销。当然他这一篇是只在 KV 上做量化，目前如果考虑 weight 和 activation 的话比较成熟的是 QServe 提出的 W4A8KV4 的方案。

我们在上一节中提到 KV 的重要性有所不同，因此为了减少量化损失，ZIPCache 等工作提出了对重要的 KV 使用更高精度的量化（如 8-bit），而对不重要的 KV 使用更低精度的量化（如 4-bit），从而在减小 KV Cache 占用的同时减少量化损失。

#### KV Cache Merging

还有一种思路是合并层间 KV Cache。MiniCache 这篇工作就发现在整个模型后半部分，层间的 KV Cache 差异并不大，因此可以将多个层的 KV Cache 进行合并，从而减小 KV Cache 的存储占用。

### Sparse Attention

#### Motivation

我们仔细看一下 Attention 的过程，首先 $Q$ 和 $K^T$ 进行乘法得到一个 $nq\times nk$ 的矩阵，这个矩阵我们也称为 attention score/map/weight。这里我们更关注他作为 weight 这个称呼，即他作为一个 weights，对一系列 value 进行加权最终得到最后的 output。

这里 attention weight 的计算是 $O(N^2)$ 的，随着序列长度的增长平方增加，给计算带来很大的开销，这是面临的核心问题。

然后我们对 attention weight 进行可视化，可以看到其实 attention weight 的分布并不是均匀的。由于过了一层 softmax，因此每一行经过 normalization 后和为 1。而这总和为 1 在一行中的分布并不是均匀的。大部分的数值集中在少部分的位置上。

![Attention 分布图]()

而如果 attention weight 非常小（接近于0），他乘上 value 后对最后 output 的贡献也是很小的，我们是否能够直接跳过这一部分 attention weight 的计算，从而减小 attention weight 的计算开销来加速 attention 计算呢？由此我们便引出了 Sparse Attention，通过部分计算来实现更高效的 Attention。

Sparse Attention 的核心思路可以从 attention mask 来理解。之前我们第一次接触到 attention mask 还是在语言模型的 causal mask 上，而在 sparse attention 中是通过 mask 将不重要的位置跳过，仅计算重要的位置。下面我们介绍的一系列 Sparse Attention 的工作，都可以看作围绕这个 sparse mask 展开。

#### Static Sparse Attention

我们通过刚刚的可视化分析，可以发现其实对于不同的模型、不同的输入，有一些共通的 Pattern。

![来一张 llm attention 的图片]()

- 对角线的 Attention 值很高
- 在某一个位置 t 的 token，他会非常关注前面 [t-p, t] 的token，存在局部性
- 前几个 token 的 attention score 非常高：Attention Sink

因此我们可以根据这些 pattern，设计出一个静态的 mask，apply 到每一个 attention 上来提升他的速度。

例如 StreamingLLM() 这篇文章，他就首次提出利用 attention sink 这个现象加上 sliding window 来加速 attention。

#### Dynamic Sparse Attention

当然静态的方法会有他的局限性，那就是他只能尽可能找到在各个配置下的最大公约数，而这样不够灵活的方式要么会遗漏掉重要的 attention 部分导致精度损失，要么会在一些不重要的 attention 上浪费计算资源。

因此大家提出使用 dynamic 的方式，根据每一个 attention 的输入 qkv 来决定保留的 attention。

一个最简单直接的想法就是对着 attention score 选数值高的部分，但做到精细选择的前提是先得到完整的 attention score，而我们本身就是想避免完整 attention score 的计算来提速，本末倒置了。同时以 token 为粒度进行选择往往是很分散的，在访存和计算上都会带来问题。

因此现在主流的做法是不以 token 粒度去选，而是先将 attention weights 划分成一个个 block，然后通过一些近似的方法来估计 block 的重要性，只选择一些重要的块来进行计算。

例如 SpargeAttention，就是通过平均池化来得到块重要性，然后选取重要的块来进行计算。

#### Future Works

最近几个月大家也都在许多地方探索 sparse attention 的更多可能，例如 DeepSeek Attention (DeepSeek-v3.2) 通过训练在 token 粒度而不是 block 粒度做 sparse attention，又或者是在视频生成领域 Sparse VideoGen 2 使用基于 k-means 的重排来将重要的 token 聚在一起计算等等。还有诸如 Sparse Linear Attention 和我们最近的一篇工作 Pyramid Sparse Attention 等在尝试打破 drop-or-keep 的固有模式。

同时目前其实在工业界大家也都在探索更多高效的 attention 机制，DeepSeek 关注于 Sparse Attention，而 Qwen 和 Kimi 则最近都在关注引入 Linear Attention 的 Hybrid Attention 来提升 attention 效率。在实现 AGI 的路上，我认为 如何实现更加强大而高效的 Attention 机制本身还是一个非常根本的问题，目前我的研究兴趣也是关注于 Efficient Attention with Algorithm-System Co-Design 上，感兴趣的同学可以后续交流。