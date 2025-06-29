# Basis of Diffusion Model

这篇博客记录我的 Diffusion 模型学习过程，主要关注在对 Diffusion 模型架构的理解。

## Derivation of Diffusion Model

简要概括 diffusion 的过程，就是分为 Forward 和 Reverse 两个过程。Forward 负责将图像不断加噪声，直到得到一张纯噪声图片。而 Reverse 过程则是通过 sample 一个噪声，然后不断去噪，直到得到一张完整的图片。

!!! question

    基于这个理解，我们接下来关注几个问题：

    1. 如何形式化描述 Forward 和 Reverse 这两个过程？
    2. 在 Training 和 Inference 中，Forward 和 Reverse 这两个阶段如何参与？

下面基于 [DDPM](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html) 来去理解一下 diffusion 的过程。

### Forward

首先我们看前向加噪声的过程。

对于一个输入图像 $x_0$，我们经过 $T$ 步加噪过程后，会得到一个长度为 $T$ 的序列 $[x_0, x_1, \cdots, x_T]$。其中第 $t$ 步的图像为 $x_t$。

我们将加噪过程描述为 $q(x_t|x_{t-1})$，即输入为 $x_{t-1}$，输出为 $x_{t}$。这个过程描述为一个正态分布：

$$
q(x_t|x_{t-1}) = \mathcal{N}(x_t;\sqrt{1-\beta_t x_{t-1}}, \beta_t I)
$$

其中这里的 $\beta_t$ 是一个控制每一步加噪过程方差的超参数。

接下来对这个加噪过程进行重参数化，得到：

$$
q(x_t|x_{t-1}) = \sqrt{1-\beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon
$$

其中 $\epsilon$ 是一个标准正态分布的噪声。

进一步的，我们可以写出

$$
q(x_{t-1}|x_{t-2}) = \sqrt{1-\beta_{t-1}} x_{t-2} + \sqrt{\beta_{t-1}} \epsilon
$$

将其代入上式，得到：

$$
q(x_t|x_{t-2}) = \sqrt{1-\beta_t} \sqrt{1-\beta_{t-1}} x_{t-2} + \sqrt{\beta_t} \epsilon_1 + \sqrt{\beta_{t-1}} \epsilon_2
$$

我们定义 $\alpha_t = 1-\beta_t$，则上式可以写为：

$$
q(x_t|x_{t-2}) = \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{1-\alpha_{t}} \epsilon_1 + \sqrt{\alpha_t(1-\alpha_{t-1})} \epsilon_2
$$

对于后两项噪声项，我们可以将其看成两个正态分布的线性组合，因此可以将其看成一个新的正态分布：

$$
q(x_t|x_{t-2}) = \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{1-\alpha_{t}\alpha_{t-1}} \epsilon
$$

不断递推至 $t=0$，我们可以得到：

$$
q(x_t|x_0) = \sqrt{\bar \alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon
$$

其中 $\bar \alpha_t = \prod_{i=1}^t \alpha_i$。

由于 $\beta_t$ 是一个超参数，因此我们就得到了一个加噪声的过程 $q(x_t|x_0)$，它是一个正态分布：

$$
q(x_t|x_0) = \mathcal{N}(x_t;\sqrt{\bar \alpha_t} x_0, (1-\bar \alpha_t) I)
$$

### Reverse

接下来我们看去噪的过程。去噪的过程是一个条件概率分布 $p_\theta(x_{t-1}|x_t)$，其参数 $\theta$ 是通过训练得到的。

我们可以将其写为一个正态分布：

$$
p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1};\mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$

其中 $\mu_\theta(x_t, t)$ 和 $\Sigma_\theta(x_t, t)$ 是通过神经网络得到的。

这里我们为了简化，假设 $\Sigma_\theta(x_t, t)$ 是一个常数 $\sigma^2 I$，则上式可以简化为：

$$
p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1};\mu_\theta(x_t, t), \sigma^2 I)
$$

我们的目标是，如果我们能够得到逆向的加噪声过程 $q_(x_{t-1}|x_t)$，那么就可以从一个纯噪声图像 $x_T$ 开始，经过 $T$ 步去噪，得到一张完整的图像 $x_0$。但是我们并不知道 $q(x_{t-1}|x_t)$ 的具体形式。

我们可以通过贝叶斯公式得到：

$$
q(x_{t-1}|x_t, x_0) = \frac{q(x_t|x_{t-1}, x_0) q(x_{t-1}|x_0)}{q(x_t|x_0)}
$$

这样就将所有逆向过程都又转化为了正向过程。经过推导我们可以得到这个正态分布的形式：

$$
q(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1}; \mu(x_t, x_0, t), \sigma^2 I)
$$

而在加噪过程中我们已经知道了 $q(x_t|x_0)$ 的形式，因此可以将其代入上式，得到：

$$
\mu_t(x_t) = \frac{1}{\sqrt{\bar \alpha_t}}( x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \epsilon_\theta(x_t, t))
$$

由此，对于每一步去噪过程的估计，就可以看作是对噪声 $\epsilon_\theta(x_t, t)$ 的估计。我们可以通过训练一个神经网络来学习这个噪声的估计。

### Training and Inference

在训练阶段，我们通过最小化以下损失函数来训练模型：

$$
L(\theta) = \mathbb{E}_{x_0, t, \epsilon} \left[ ||\epsilon - \epsilon_\theta(x_t, t)||^2 \right]
$$

将 $x_t$ 的形式代入上式，得到：
$L(\theta) = \mathbb{E}_{x_0, t, \epsilon} \left[ ||\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)||^2 \right]$

在推理阶段，我们从一个纯噪声图像 $x_T$ 开始，经过 $T$ 步去噪，得到一张完整的图像 $x_0$。每一步去噪都使用训练好的模型 $\epsilon_\theta(x_t, t)$ 来估计噪声。

![DDPM Algorithm](https://pic2.zhimg.com/v2-6a41afbb1bf22710efc37646b69ea085_1440w.jpg)

!!! question

    在学习了基本概念之后，接下来关注以下几个问题：

    1. 目前的 Diffusion Model 是如何对这个 denoising 过程进行建模的？对于一个 text-to-image 的任务来说，input text 携带的信息是如何参与这个 denoising 过程的？
    2. Denoising steps 这个参数是由模型决定，在训练阶段固定的吗？还是说可以 dynamic 调整？
