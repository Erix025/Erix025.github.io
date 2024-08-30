# Advanced Topics in Statistical Machine Learning
> 这是牛津大学的一门关于统计机器学习的课程，主要讲解一些高级的统计机器学习方法，是一门面向研究生的课程。我在 Oxford Study Abroad Program in Summer 暑期项目中学习了这门课程的一部分。
> 课程网站：https://www.robots.ox.ac.uk/~twgr/teaching/
> Instructor: Prof. Tom Rainforth

这篇笔记主要记录这门课程（我能听得懂）的一些内容，以及我对这些内容的理解。主要包括 Machine Learning 的基本思路，SVM，Kernel Methods 以及其与 Deep Learning 的关系等内容。

> 施工中...

## Basic Concepts in Machine Learning

### Empirical Risk Minimization

#### Loss Function & Risk

为了衡量模型的好坏，我们一个函数来评估模型预测的结果和真实结果之间的差异。这个函数就是 Loss Function。

$$
    \text{Loss Function} = L(y, f(x))
$$

Loss Function 通常是一个非负函数，当预测值和真实值相同时，Loss Function 取最小值。

Risk 是 Loss Function 在所有可能的输入上的期望值，即 Loss Function 的平均值。

省流：Loss Function 定义了模型的预测值与真实值之间的差异，Risk 是 Loss Function 的期望。

#### Empirical Risk

上述的 Risk 是在所有可能的输入上的期望值，但这个期望值是未知的（因为我们不可能在所有可能的输入上计算 Loss Function 的期望），这个 Risk 叫做 True Risk。因此我们通常用 Empirical Risk 来近似 True Risk。

Empirical Risk 是在训练集上计算的 Risk，即 Loss Function 在训练集上的平均值。
$$
    \text{Empirical Risk} = \frac{1}{N} \sum_{i=1}^{N} L(y_i, f(x_i))
$$

因此我们在训练模型的时候，实际上是在最小化 Empirical Risk。

接下来我们来分析 Empirical Risk:

首先我们基于一个假设：我们的数据集是从一个分布 $\mathcal{D}$ 中独立同分布采样得到的。因此我们可以认为训练集是从 $\mathcal{D}$ 中采样得到的。

当函数 $f$的选择和数据集 $\textbf{X}$ 的选择是独立的情况下，我们可以认为 Empirical Risk 是 True Risk 的无偏估计。

然而，在实际训练过程中，我们在不断基于训练集更新模型，使得在数据集上的 Empirical Risk 最小化，因此 $f$ 和 $\textbf{X}$ 是相关的，Empirical Risk 会低估 True Risk。

### Regularization

由于 Empirical Risk 会低估 True Risk，这会导致模型在训练集上表现很好，但在测试集上表现很差，这就引起了 Overfitting 问题。为了解决这个问题，我们需要引入 Regularization。

Regularization 本质上是为了限制模型的复杂度。当模型足够复杂，模型的拟合能力会很强，但这也会更容易造成 Overfitting 问题。因此我们需要在模型拟合能力和模型复杂度之间找到一个平衡。

Regularization 通常是在 Loss Function 中加入一个正则项，这个正则项通常是模型参数的范数。

$$
    \text{Regularized Loss Function} = \text{Loss Function} + \lambda \Omega(\theta)
$$

## Constrained Optimization

在进行 Empirical Risk Minimization 的过程中，我们通常会遇到一些约束条件（如 Regularizer 的大小约束等），需要一些手段来解决 Constrained Optimization 问题。

### Lagrange Multiplier

Lagrange Multiplier 是在微积分课程中学习过的解决**等式**约束条件下的最优化问题的方法。



### Primal and Dual Problem

## Support Vector Machine


## Kernel Methods

## Some interesting topics about Deep Learning

## Uncovered Topics

- Gaussian Processes
- Bayesian Optimization
- Variational Inference