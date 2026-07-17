# 从零开始的 GDN 推导

> 本文旨在把 GDN 中的推导细节揉碎了记录下来，基于最基本的线性代数知识，以供参考（防止以后忘记推导过程）

## Gated Delta Rule Chunk-wise Parallel 前向推导

### 记号定义

**关于状态下标：** 规定 $S_0$ 为尚未汇总当前 chunk 中任何 token 信息的初始状态。因此，$S_{t+1}$ 表示依次汇总 token $0,1,\ldots,t$ 后的状态。

**关于矩阵连乘：** 对于 $a\ge b$，定义逆序矩阵乘积为

$$
\prod_{i=a}^{b}A_i
\triangleq
A_aA_{a-1}\cdots A_b.
$$
### 从 S 的 recurrent form 进行展开

先从最开始的 recurrent form 开始：

$$
S_{t+1} = \alpha_t(I-\beta_tk_t^\top k_t)S_t + \beta_tk_t^\top v_t
$$

逐步展开 $S_t, S_{t-1}, \cdots$

$$
S_{t+1} = (\prod^{0}_{i=t}\alpha_i(I-\beta_i k_i^\top k_i))S_0 + \sum_{i=0}^{t} (\prod^{i+1}_{j=t} \alpha_j(I-\beta_j k_j^\top k_j)) \beta_i k_i^\top v_i \\
$$

这里可以看到 前面一项和后面一项的累乘有相似的结构，只是起止不一样，因此做构造：

令 $M = (\prod ^{0}_{i=t}\alpha_i)\prod^{0}_{i=t}(I-E_i), \quad E_i = \beta_i k_i^\top k_i$ 有：

$$
\begin{aligned}
\prod^{0}_{i=t}(I-E_i) &= (\prod^{1}_{i=t}(I-E_i))(I-E_{0}) \\
&=(\prod^{1}_{i=t}(I-E_i)) - \prod^{1}_{i=t}(I-E_i)E_0 \\
&=(\prod^{2}_{i=t}(I-E_i)) - \prod^{2}_{i=t}(I-E_i)E_1  - \prod^{1}_{i=t}(I-E_i)E_0 \\
&= \cdots \\
&= I - E_{t} -(I-E_{t})E_{t-1} - \cdots - \prod^{2}_{i=t}(I-E_i)E_1  - \prod^{1}_{i=t}(I-E_i)E_0\\
&= I - \sum^t_{i=0} \prod ^{i+1}_{j=t} (I-E_j)E_i
\end{aligned}
$$

所以有：

$$
\begin{aligned}
M &= (\prod ^{0}_{i=t}\alpha_i)\prod^{0}_{i=t}(I-E_i)\\
&=(\prod ^{0}_{i=t}\alpha_i)(I - \sum^t_{i=0} \prod ^{i+1}_{j=t} (I-E_j)E_i)\\
&=\prod ^{0}_{i=t}\alpha_iI-(\prod ^{0}_{i=t}\alpha_i)\sum^t_{i=0} \prod ^{i+1}_{j=t} (I-E_j)E_i\\
&=\prod ^{0}_{i=t}\alpha_iI-\sum^t_{i=0} (\prod ^{0}_{j=i}\alpha_j)\prod ^{i+1}_{j=t} \alpha_j(I-E_j)E_i
\end{aligned}
$$

令 $\gamma_i = \prod ^{0}_{j=i-1}\alpha_j, \gamma_0 = 1$，代回 $S$ 有：

$$
\begin{aligned}
S_{t+1} &= (\gamma_{t+1} I-\sum^t_{i=0} \gamma_{i+1} \prod ^{i+1}_{j=t} \alpha_j(I-\beta_j k_j^\top k_j)\beta_i k_i^\top k_i )S_0 + \sum_{i=0}^{t} (\prod^{i+1}_{j=t} \alpha_j(I-\beta_j k_j^\top k_j)) \beta_i k_i^\top v_i \\
&=(\gamma_{t+1} I-\sum^t_{i=0} \gamma_{i+1} z_i^t k_i )S_0 + \sum_{i=0}^{t} z^t_i v_i
\end{aligned}
$$

其中 $z^t_i = \prod ^{i+1}_{j=t} \alpha_j(I-\beta_j k_j^\top k_j)\beta_i k_i^\top$

### 推导 WY-representation

证明：$\sum_{i=0}^{t} z^t_i x_i = \sum_{i=0}^{t}\frac{\gamma_{t+1}}{\gamma_{i+1}} k_i^\top y_i$

当 $t=0$ 时，$z^0_0x_0=\beta_0k_0^\top x_0$，此时 $y_0 = \beta_0 x_0$

若 $t = n-1$ 时，$\sum_{i=0}^{n-1} z^{n-1}_i x_i = \sum_{i=0}^{n-1}\frac{\gamma_{n}}{\gamma_{i+1}} k_i^\top y_i$

则 $t=n$ 时有：

$$
\begin{aligned}
\sum_{i=0}^{n} z^n_i x_i &= \sum_{i=0}^{n} (\prod ^{i+1}_{j=n} \alpha_j(I-\beta_j k_j^\top k_j))\beta_i k_i^\top x_i\\
&=\sum_{i=0}^{n-1} (\prod ^{i+1}_{j=n} \alpha_j(I-\beta_j k_j^\top k_j))\beta_i k_i^\top x_i + \beta_{n}k_{n}^\top x_{n}\\
&=(\alpha_n(I-\beta_n k_n^\top k_n))\sum_{i=0}^{n-1} (\prod ^{i+1}_{j=n-1} \alpha_j(I-\beta_j k_j^\top k_j))\beta_i k_i^\top x_i + \beta_{n}k_{n}^\top x_{n}\\
&=(\alpha_n(I-\beta_n k_n^\top k_n))\sum_{i=0}^{n-1} z^{n-1}_ix_i + \beta_{n}k_{n}^\top x_{n}\\
&=(\alpha_n(I-\beta_n k_n^\top k_n))\sum_{i=0}^{n-1}\frac{\gamma_{n}}{\gamma_{i+1}} k_i^\top y_i + \beta_{n}k_{n}^\top x_{n}\\
&=\sum_{i=0}^{n-1}\frac{\gamma_{n+1}}{\gamma_{i+1}} k_i^\top y_i - \alpha_n\beta_n k_n^\top k_n \sum_{i=0}^{n-1}\frac{\gamma_{n}}{\gamma_{i+1}} k_i^\top y_i + \beta_{n}k_{n}^\top x_{n} \\
&=\sum_{i=0}^{n-1}\frac{\gamma_{n+1}}{\gamma_{i+1}} k_i^\top y_i + k_n^\top (\beta_n x_n - \beta_n k_n \sum_{i=0}^{n-1}\frac{\gamma_{n+1}}{\gamma_{i+1}} k_i^\top y_i) \\
&=\sum_{i=0}^{n-1}\frac{\gamma_{n+1}}{\gamma_{i+1}} k_i^\top y_i + \frac{\gamma_{n+1}}{\gamma_{n+1}} k_n^\top (\beta_n x_n - \beta_n k_n \sum_{i=0}^{n-1}\frac{\gamma_{n+1}}{\gamma_{i+1}} k_i^\top y_i ) \\
&=\sum_{i=0}^{n}\frac{\gamma_{n+1}}{\gamma_{i+1}} k_i^\top y_i\\
\end{aligned}
$$

其中
$$
y_n =\beta_n x_n - \beta_n k_n \sum_{i=0}^{n-1}\frac{\gamma_{n+1}}{\gamma_{i+1}} k_i^\top y_i
$$

### UT transform

UT transform 是将 $y_n =\beta_n x_n - \beta_n k_n \sum_{i=0}^{n-1}\frac{\gamma_{n+1}}{\gamma_{i+1}} k_i^\top y_i$ 转换成矩阵表达。

下面考虑一个长度为 $C$ 的 chunk，即下文 $r\in[0, C-1]$。

这里定义一下 $\beta$ 和 $\gamma$ 的矩阵形式。$B = \mathrm{diag}(\beta_0, \beta_1, \cdots, \beta_{C-1}), \Gamma = \mathrm{diag}(\gamma_1, \gamma_2, \cdots, \gamma_C)$，二者都是对角矩阵。

从 $y_n =\beta_n x_n - \beta_n k_n \sum_{i=0}^{n-1}\frac{\gamma_{n+1}}{\gamma_{i+1}} k_i^\top y_i$ 开始，表示成矩阵形式即为：

$$
\begin{aligned}
Y[r, :] &= \beta_r X[r,:] - \beta_r K[r,:]\sum_{i=0}^{r-1}\frac{\gamma_{r+1}}{\gamma_{i+1}} K^\top[:,i] Y[i,:]\\
&= \beta_r X[r,:] - \beta_r \gamma_{r+1} K[r,:]\sum_{i=0}^{r-1}\frac{1}{\gamma_{i+1}} K^\top[:,i] Y[i,:]\\
&= \beta_r X[r,:] - \beta_r (\Gamma K)[r,:]\sum_{i=0}^{r-1}(\Gamma^{-1} K)^\top[:,i] Y[i,:]\\
&= \beta_r X[r,:] - \beta_r (\Gamma K)[r,:]\sum_{i=0}^{r-1} (K^\top \Gamma^{-1})[:,i] Y[i,:]\\
&= \beta_r X[r,:] - \beta_r \sum_{i=0}^{r-1} (\Gamma KK^\top \Gamma^{-1})[r,i] Y[i,:]\\
&= \beta_r X[r,:] - \beta_r \sum_{i=0}^{C-1}\mathrm{StrictLower}(\Gamma KK^\top \Gamma^{-1})[r,i] Y[i,:]\\
&= \beta_r X[r,:] - \beta_r \mathrm{StrictLower}(\Gamma KK^\top \Gamma^{-1})[r,:] Y\\
&= (BX)[r,:] - \mathrm{StrictLower}(B\Gamma KK^\top \Gamma^{-1})[r,:] Y\\
\end{aligned}
$$

令 $L = \mathrm{StrictLower}(B\Gamma KK^\top \Gamma^{-1})$，所以有

$Y[r, :] = (BX)[r,:] - L[r,:] Y$，即：

$$Y = BX-LY$$

解得：

$$Y = (I + L) ^ {-1} BX$$

### S 的推导

回忆一下刚刚经过 WY-representation 和 UT transformation 之后我们得到了什么：

- $\sum_{i=0}^{t} z^t_i x_i = \sum_{i=0}^{t}\frac{\gamma_{t+1}}{\gamma_{i+1}} k_i^\top y_i$
- $Y = (I + L) ^ {-1} BX$

对于长度为 $C$ 的 chunk，其 token 下标为 $0, 1, \cdots, C-1$，因此令 $t=C-1$，将 WY-representation 也改为矩阵形式：

等式左边：$\sum_{i=0}^{C-1} Z_{[:,i]} X_{[i,:]}= ZX$, 其中 $Z\in \mathbb R^{d_k \times C}$, $X\in \mathbb R^{C \times d_x}$

等式右边：$\gamma_{C}\sum_{i=0}^{C-1}(\mathrm{diag}(\frac{1}{\gamma_{i+1}}) K)^\top_{[:,i]} Y_{[i,:]}=\gamma_{C}K^\top {\Gamma}^{-1}Y$, 其中 $K\in \mathbb R^{C \times d_k}$, $Y\in \mathbb R^{C \times d_x}$

所以有：

$$
\begin{aligned}
ZX&=\gamma_{C}K^\top {\Gamma}^{-1}Y\\
&=\gamma_{C}K^\top {\Gamma}^{-1}(I + L) ^ {-1} BX
\end{aligned}
$$

即 $Z = \gamma_{C}K^\top {\Gamma}^{-1} (I + L) ^ {-1} B$

然后回到 S 的公式中并改为矩阵形式。
这里我们定义 $A=(I+L)^{-1}$，整理 S 的公式得：

$$
\begin{aligned}
S_C &=(\gamma_C I-\sum^{C-1}_{i=0} \gamma_{i+1} z_i^{C-1} k_i )S_0 + \sum_{i=0}^{C-1} z_i^{C-1} v_i\\
&=(\gamma_C I-Z\Gamma K )S_0 + ZV\\
&=(\gamma_C I-\gamma_{C}K^\top {\Gamma}^{-1}(I + L) ^ {-1} B\Gamma K )S_0 + \gamma_{C}K^\top {\Gamma}^{-1}(I + L) ^ {-1} BV\\
&=(\gamma_C I-\gamma_{C}K^\top {\Gamma}^{-1}A B\Gamma K )S_0 + \gamma_{C}K^\top {\Gamma}^{-1}A BV\\
&=(\gamma_C I-\gamma_{C}K^\top {\Gamma}^{-1}W )S_0 + \gamma_{C}K^\top {\Gamma}^{-1}U\\
&=\gamma_C S_0+\gamma_{C}K^\top {\Gamma}^{-1}(U-WS_0)
\end{aligned}
$$

其中 $W = AB\Gamma K, U=ABV$

### O 的推导

先从向量形式开始：

$$
o_t = q_tS_{t+1}
$$

此时这里描述的一件事情是，$o_t$ 是由 $q_t$ 和汇总了 token $0, 1, \cdots, t$ 的 $S_{t+1}$ 决定的。

将 $S_{t+1}$ 代入得： 

$$
o_t = q_t(\gamma_{t+1} S_0+\gamma_{t+1}K_{[:t+1, :]}^\top {\Gamma_{[:t+1, :t+1]}}^{-1}(U_{[:t+1, :]}-W_{[:t+1, :]}S_0))
$$

注意这里的切片操作是因为 $S_{t+1}$ 只能看到 token $0, 1, \cdots, t$。

然后转为矩阵形式：

$$
O = \Gamma QS_0 + \Gamma\mathrm{Lower}(QK^\top){\Gamma}^{-1}(U-WS_0)
$$

其中这里的 Lower 是因为第 t 行的 token 只能看到它以及之前的内容，所以加了一个 causal mask，可以将向量形式中 $q_t$ $K_{[:t+1, :]}^\top$ 和 $(U-WS_0)_{[:t+1, :]}$ 三者的乘积画图捋一下形象感受这个 Lower 的由来。

可以考虑其第 $t$ 行来验证：

$$
O[t,:] = q_t(\gamma_{t+1} S_0+\gamma_{t+1}K_{[:t+1, :]}^\top {\Gamma_{[:t+1, :t+1]}}^{-1}(U_{[:t+1, :]}-W_{[:t+1, :]}S_0))
$$

### 总结

上述推导描述了单个 chunk 内的并行计算过程。对于多个 chunk，只需将前一个 chunk 的输出状态作为下一个 chunk 的初始状态，依次执行相同的计算。

令 $S_{[c]}$ 表示第 c 个 chunk 的输入状态，当前 chunk 的长度为 C。则 GDN chunk-wise parallel forward 可以分为以下步骤：

1. A 的计算：$A = (I+\mathrm{StrictLower}(B\Gamma KK^\top \Gamma^{-1}))^{-1}$
2. U 和 W 的计算：$U = A BV, W = A B\Gamma K$
3. S 的更新：$S_{[c+1]} = \gamma_C S_{[c]}+\gamma_{C}K^\top {\Gamma}^{-1}(U-WS_{[c]})$
4. O 的更新：$O_{[c]} = \Gamma QS_{[c]} + \Gamma\mathrm{Lower}(QK^\top){\Gamma}^{-1}(U-WS_{[c]})$