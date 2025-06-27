# 大模型并行策略

这篇博客将从推理（Inference）和训练（Training）两个方面介绍大模型的并行策略。

## 推理

推理阶段的并行策略主要有以下几种：

- **数据并行（Data Parallelism）**：将数据分成多个 batch，在不同的设备上处理。
- **流水线并行（Pipeline Parallelism）**：将模型分成多个阶段，每个阶段在不同的设备上处理。
- **张量并行（Tensor Parallelism）**：将模型的参数分布到不同的设备上，每个设备处理一部分参数。
- **序列并行（Sequence Parallelism）**：将长序列分成多个短序列，在不同的设备上处理。
- **专家并行（Expert Parallelism）**：将模型分成多个专家，每个专家在不同的设备上处理。针对于 MoE（Mixture of Experts）模型。

由于推理阶段只需要考虑模型的 forward 过程，因此并行策略的实现会相对简单。

### Data Parallelism

推理阶段的数据并行其实比较简单。他的关键是以 request 作为并行拆分的最小单位，将不同的 request 分布到不同的 device 上进行推理。

- Memory 占用：每个 device 上有完整的一份模型副本。
- Communication：只在每条 request 的推理开始 / 结束会有通信，而且只需要对最终的 input / output 进行通信，通信量小。
- Throughput：当同时需要推理的 request 数接近于 infinity 时，增大并行度对 throughput 的增益是接近线性的（通信量小）
- Latency：并行度的增加对 latency 不会有正面效果，因为每个 request 分到的计算资源是一致的。

### Pipeline Parallelism

流水线并行是将单次前向过程划分为多个 stage，以 stage 作为单位进行拆分，将不同的 stage 分布到不同的 device 上进行推理。

- Memory 占用：每个 device 上只有其所负责的 stage 对应的模型参数。
- Communication：在单个机器所负责的 device 完成后会发生通信，需要将 hidden states 传递给后续阶段的 device。
- Throughput：在理想状态下，某一时刻流水线中所有 device 都在进行计算，总吞吐量会增加，类比 CPU 的流水线。
- Latency：Latency 会下降，因为对于每个 request 而言，每次都只有一个 device 的计算资源参与前向过程，并且还有 device 间通信带来的 overhead。

相比于 Data Parallelism，其对于 throughput 的提升会相对较低。但 PP 的优势是他能够拆分模型权重到不同的 device 上，从而能够在多个 device 上部署规模较大的模型。

### Tensor Parallelism

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

### Expert Parallelism

## 训练

训练阶段的并行策略主要有以下几种：

- **数据并行（Data Parallelism）**：将数据分成多个 batch，在不同的设备上处理。每个设备上运行相同的模型副本。
- **张量并行（Tensor Parallelism）**：将模型的参数分布到不同的设备上，每个设备处理一部分参数。适用于大模型。
- **流水线并行（Pipeline Parallelism）**：将模型分成多个阶段，每个阶段在不同的设备上处理。每个设备上运行模型的不同部分。

## References

- [Tensor Parallelism - Huggingface](https://huggingface.co/docs/text-generation-inference/en/conceptual/tensor_parallelism)
- [Parallelisms - NVIDIA NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/parallelisms.html#tensor-parallelism)
