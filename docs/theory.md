# Neural Network Implementation: Advanced Theory & Architecture

## Table of Contents
1. [Foundational Concepts](#foundational-concepts)
2. [Mathematical Framework](#mathematical-framework)
3. [Architecture Design](#architecture-design)
4. [Forward Propagation](#forward-propagation)
5. [Backpropagation Algorithm](#backpropagation-algorithm)
6. [Activation Functions](#activation-functions)
7. [Loss Functions](#loss-functions)
8. [Optimization & Gradient Descent](#optimization--gradient-descent)
9. [Implementation Details](#implementation-details)
10. [Advanced Extensions](#advanced-extensions)

---

## Foundational Concepts

### Artificial Neural Networks
An artificial neural network (ANN) is a computational model inspired by biological neurons. It consists of interconnected layers of neurons that learn to map inputs to outputs through examples. The fundamental principle is **universal approximation**: a neural network with sufficient hidden units and non-linear activation functions can approximate any continuous function on compact domains.

### Tensor Abstraction
The core data structure is the `Tensor` class—a 2D matrix wrapper for efficient numerical computation. We use row-major storage:
```
_data = [element[0,0], element[0,1], ..., element[m,n]]
```
Access via `at(r, c) = _data[r * _cols + c]` provides O(1) lookup. This design balances cache locality with simplicity.

### Network Topology
A feedforward neural network forms a directed acyclic graph (DAG):
```
Input Layer → Hidden Layer 1 → Hidden Layer 2 → Output Layer
    784            128              64              10
```
Each layer applies a learned transformation: $y = \sigma(Wx + b)$

---

## Mathematical Framework

### Layer Transformation
Each layer performs an **affine transformation** followed by a **non-linearity**:

$$z_j = \sum_{i=0}^{n} w_{ij}x_i + b_j$$

$$a_j = \sigma(z_j)$$

Where:
- $W \in \mathbb{R}^{n \times m}$: weight matrix
- $b \in \mathbb{R}^{m}$: bias vector
- $\sigma$: activation function
- $z$: pre-activation (logits)
- $a$: post-activation (output)

### Composition Property
The power of neural networks emerges from **function composition**. For a 3-layer network:

$$f(x) = \sigma_3(W_3 \sigma_2(W_2 \sigma_1(W_1 x + b_1) + b_2) + b_3)$$

This hierarchical composition enables learning of increasingly abstract representations—a key insight from deep learning theory.

---

## Architecture Design

### Layer Class Structure
```cpp
class Layer {
  private:
    Tensor _w, _b;           // Parameters
    Tensor _out, _in;        // Cache for forward pass
    Tensor _gw, _gb, _gin;   // Gradients
    Activation _act;         // Activation function selector
};
```

**Design decision**: We cache inputs and outputs to enable **efficient backpropagation**. Without caching, we'd need to recompute activations during the backward pass, violating time-memory tradeoff principles.

### Network Class Structure
```cpp
class nn {
  private:
    std::vector<Layer> _layers;  // Composition of layers
    Loss _loss_fn;               // Loss function selector
};
```

By storing layers in a vector, we achieve:
- **Dynamic network depth**: layers can be added at runtime
- **Modular design**: each layer encapsulates its own computation
- **Sequential processing**: layers process data in topological order

---

## Forward Propagation

### Computational Flow
Forward propagation computes the network output given an input $x$:

```
Input: x (dimension: 1 × 784)
  ↓
Layer 1: z₁ = x·W₁ + b₁, a₁ = ReLU(z₁)
  ↓
Layer 2: z₂ = a₁·W₂ + b₂, a₂ = ReLU(z₂)
  ↓
Layer 3: z₃ = a₂·W₃ + b₃, a₃ = Softmax(z₃)
  ↓
Output: ŷ = a₃ (dimension: 1 × 10)
```

### Implementation Strategy
```cpp
Tensor &nn::forward(const Tensor &in) {
    const Tensor *cur = &in;
    for (size_t i = 0; i < _layers.size(); i++)
        cur = &_layers[i].forward(*cur);
    _out = *cur;
    return (_out);
}
```

**Key insight**: We maintain pointer reference `cur` to avoid copying intermediate tensors. Each layer processes the output of the previous layer, minimizing memory allocations.

### Complexity Analysis
- **Time**: $O(d_1 \cdot d_2 + d_2 \cdot d_3 + ... + d_n \cdot d_{out})$ = $O(\sum_{i} d_i \cdot d_{i+1})$
- **Space**: $O(\max(d_i))$ for layer activations (dominant factor)

For MNIST (784→128→64→10): ~180K multiplications per forward pass.

---

## Backpropagation Algorithm

### The Chain Rule in Backpropagation
Backpropagation computes gradients via the **chain rule**. For loss $L$ and layer parameters $\theta$:

$$\frac{\partial L}{\partial \theta} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial \theta}$$

Unwinding through layers:

$$\frac{\partial L}{\partial w_{ij}} = \frac{\partial L}{\partial a_3} \cdot \frac{\partial a_3}{\partial z_3} \cdot ... \cdot \frac{\partial z_2}{\partial a_1} \cdot \frac{\partial a_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial w_{ij}}$$

### Gradient Computation Per Layer
Starting with upstream gradient $\delta^{(l)}$ from the next layer:

**1. Compute local gradient**:
$$\delta^{(l)} = \sigma'(z^{(l)}) \odot \delta^{(l+1)} W^{(l+1)^T}$$

Where $\odot$ is element-wise multiplication (Hadamard product).

**2. Accumulate parameter gradients**:
$$\frac{\partial L}{\partial W^{(l)}} = (a^{(l-1)})^T \cdot \delta^{(l)}$$

$$\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}$$

**3. Propagate to previous layer**:
$$\delta^{(l-1)} = \delta^{(l)} W^{(l)^T}$$

### Implementation
```cpp
Tensor &Layer::backward(const Tensor &grad) {
    for (size_t j = 0; j < _out_sz; j++) {
        float z = compute_preactivation();  // Reuse cached input
        float d = grad.at(0, j) * activation_derivative(z);
        
        _gb.at(0, j) = d;  // Bias gradient
        
        for (size_t i = 0; i < _in_sz; i++) {
            _gw.at(i, j) = _in.at(0, i) * d;  // Weight gradient
            _gin.at(0, i) += _w.at(i, j) * d;  // Propagate upstream
        }
    }
    return (_gin);
}
```

### Why Caching Matters
During backprop, we need $z^{(l)}$ (pre-activations) to compute $\sigma'(z^{(l)})$. We cached these during forward pass instead of recomputing them, achieving:
- **Time savings**: Avoid $O(n^2)$ redundant exponentials/divisions
- **Numerical stability**: Use exact cached values instead of imprecise recomputation

---

## Activation Functions

### ReLU (Rectified Linear Unit)
$$\text{ReLU}(x) = \max(0, x)$$
$$\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{otherwise} \end{cases}$$

**Advantages**:
- **Non-saturation**: Unlike sigmoid, doesn't flatten gradients at extremes
- **Sparsity**: ~50% of neurons are inactive, enabling learned feature selection
- **Computational efficiency**: $O(1)$ evaluation vs. expensive sigmoid
- **Biological plausibility**: Resembles firing thresholds

**Disadvantage**: **Dying ReLU problem** when $x < 0$ for many samples. Mitigation: Leaky ReLU ($\text{LeakyReLU}(x) = \alpha x$ if $x < 0$).

### Sigmoid
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$
$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

Used for **binary classification** output layer. Maps $\mathbb{R} \to (0, 1)$.

**Why NOT for hidden layers**: Gradient vanishing problem—at extremes, $\sigma'(x) \approx 0$, causing exponentially small gradients flowing backward through many layers.

### Softmax (for Multi-class Output)
$$\text{Softmax}(z_j) = \frac{e^{z_j - \max(z)}}{\sum_k e^{z_k - \max(z)}}$$

**Key features**:
- **Probability distribution**: Outputs sum to 1
- **Numerical stability trick**: Subtracting $\max(z)$ prevents $e^x$ overflow
- **Temperature scaling**: Can be generalized to $\text{Softmax}(z_j / T)$ for entropy control

---

## Loss Functions

### Mean Squared Error (MSE)
$$L_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Gradient**: $\frac{\partial L}{\partial \hat{y}} = 2(\hat{y} - y)$

**Use case**: Regression, XOR (binary output)

**Drawback**: For classification, outliers heavily penalize via quadratic term, leading to suboptimal gradients.

### Cross-Entropy Loss
$$L_{\text{CE}} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

Where $y$ is one-hot encoded ground truth, $\hat{y}$ is softmax output.

**Gradient**: $\frac{\partial L}{\partial \hat{y}} = -\frac{y}{\hat{y}}$

**Advantages**:
- **Information-theoretic**: Minimizes KL divergence between predicted and true distributions
- **Natural with softmax**: Softmax + CE loss simplifies to linear gradient in logits
- **Robust to confidence**: Penalizes wrong predictions more severely than MSE

**Numerical stability**: Clamp predictions to $[\epsilon, 1-\epsilon]$ (e.g., $10^{-7}$) to avoid $\log(0)$.

---

## Optimization & Gradient Descent

### Stochastic Gradient Descent (SGD)
$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

Where $\eta$ is the **learning rate**.

**Hyperparameter tuning**:
- Too high $\eta$: Divergence, oscillation around optimum
- Too low $\eta$: Slow convergence, risk of local minima
- Typical range: $[10^{-5}, 10^{-1}]$

### Learning Rate Schedule
**Constant**: Simple but risks overshooting near optimum.

**Decay schedule**: $\eta_t = \eta_0 \cdot (1 - t/T)$ gradually reduces step size.

**Warmup + cosine annealing** (modern): Increases learning rate initially, then decays smoothly.

### Batch Updates (Our Implementation)
```cpp
void nn::train(const Tensor &in, const Tensor &exp, float lr) {
    forward(in);
    backward(exp);
    for (size_t i = 0; i < _layers.size(); i++)
        _layers[i].update(lr);
}
```

One sample per update. In production, we'd accumulate gradients over mini-batches for:
- **Noise reduction**: Average gradient over 32-256 samples reduces variance
- **Parallelization**: Process multiple samples simultaneously
- **Generalization**: Noise acts as regularization (SGDR effect)

### Convergence Analysis
Assuming convex loss (unrealistic for NNs), SGD converges at rate $O(1/\sqrt{T})$ in expectation.

For **non-convex neural network loss**, convergence guarantees don't hold, but empirically:
- Good generalization often achieved before theoretical convergence
- Early stopping essential to avoid overfitting

---

## Implementation Details

### Tensor Storage & Indexing
```cpp
float &Tensor::at(size_t r, size_t c) {
    return (_data[r * _cols + c]);  // Row-major
}
```

Row-major ordering matches typical matrix multiplication libraries (BLAS), enabling future GPU acceleration.

### Memory Initialization
```cpp
void Tensor::randomize(float min, float max) {
    r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    _data[i] = min + r * (max - min);
}
```

**Weight initialization** critically affects convergence:
- **Uniform [-1, 1]**: Ensures non-zero gradients initially
- **Advanced**: Xavier/Glorot init ($\text{U}[-\sqrt{6/(n_{in}+n_{out})}, +]$) balances variance across layers
- **Modern**: Kaiming init for ReLU networks

### Error Handling
```cpp
if (exp.cols() == 0) {
    return (std::cerr << "Error: ...\n", 0.0f);
}
float pred = _out.at(0, j);
if (pred < 1e-7f) pred = 1e-7f;  // Prevent log(0)
```

**Critical for robustness**:
- Dimension validation prevents silent buffer overruns
- Clipping prevents NaN propagation in loss computation

---

## Advanced Extensions

### 1. Batch Normalization
Normalize layer inputs: $\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$

**Benefits**:
- Reduces internal covariate shift
- Allows higher learning rates
- Acts as regularizer

### 2. Dropout Regularization
Randomly zero activations during training: $a_{\text{drop}} = a \cdot m$, where $m \sim \text{Bernoulli}(p)$.

**Effect**: Forces network to learn redundant representations, reducing co-adaptation.

### 3. Momentum & Adaptive Learning Rates (Adam)
$$v_t = \beta v_{t-1} + \nabla L$$
$$\theta_{t+1} = \theta_t - \eta v_t$$

Momentum accelerates convergence in consistent gradient directions.

**Adam**: Combines momentum with per-parameter adaptive learning rates via exponential moving averages of gradients and squared gradients.

### 4. Regularization Techniques
- **L2 (Ridge)**: Add $\lambda ||W||_2^2$ to loss, penalizing large weights
- **L1 (Lasso)**: Promotes sparsity via $\lambda ||W||_1$
- **Early stopping**: Monitor validation loss, stop when it increases

### 5. Convolutional Layers
Replace dense layers with **local connectivity + weight sharing**:
- Reduces parameters from $n \times m$ to $k^2 \times c$ (kernel size, channels)
- Preserves spatial structure for image tasks

### 6. Attention Mechanisms
**Self-attention**: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

Enables modeling long-range dependencies. Foundational to Transformers.

---

## Summary: Why This Architecture Works

1. **Universal Approximation**: ReLU hidden layers + linear output can approximate any function
2. **Efficient Learning**: Backpropagation computes all gradients in $O(\text{forward_cost})$ via chain rule
3. **Representation Learning**: Composition of layers learns hierarchical abstractions (MNIST: pixels → edges → shapes → digits)
4. **Empirical Success**: Despite non-convex loss landscape, SGD + careful initialization reaches good local minima

This implementation captures the **essential ideas** underlying modern deep learning: differentiable computation, gradient-based optimization, and learned feature hierarchies.

---

## References & Further Study
- **Backpropagation**: Rumelhart et al. (1986)
- **ReLU Networks**: Krizhevsky et al. (2012) - AlexNet
- **Information Theory**: Cross-entropy as KL divergence minimization
- **Optimization**: Adam, SGD with momentum (Kingma & Ba 2014)
- **Modern Theory**: Double descent phenomenon, loss landscape geometry
