# High-Performance Neural Network Framework

**Custom C++ Neural Network Implementation with Performance Optimization**

[![Build Status](https://img.shields.io/badge/build-active-brightgreen)]() [![License](https://img.shields.io/badge/license-MIT-blue)]() [![Standard](https://img.shields.io/badge/C++-11-lightblue)]()

## Overview

This project provides a from-scratch C++ neural network implementation focused on linear algebra optimization and CPU architecture tuning.

## Core Components

### 1. Tensor Class

**Foundational data structure for all operations.**

- **Description**: 2D matrix abstraction with row-major storage, supporting efficient element access and memory management.
- **Features**:
  - O(1) element access via row-major indexing
  - CPU-cache optimized layout
  - Automatic memory management
  - Support for arbitrary dimensions
- **Use case**: Core data container for all neural network operations

### 2. Fully Connected Layer

**Dense feedforward transformation with learnable parameters.**

- **Description**: Implements the affine transformation $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$ followed by optional activation functions.
- **Implementation**:
  - Optimized matrix multiplication with blocking for cache efficiency
  - Proper weight initialization (Xavier/Glorot)
  - Bias broadcasting
- **Complexity**: O(n·m·k) for forward pass with dimensions n×m weight matrix
- **Use case**: Building blocks for multi-layer networks

### 3. Activation Functions

**Non-linear transformations enabling network expressivity.**

- **Sigmoid**: $\sigma(x) = \frac{1}{1+e^{-x}}$
- **ReLU**: $\text{max}(0, x)$ (branch-free SIMD implementation)
- **Tanh**: $\tanh(x)$
- **Softmax**: Multi-class probability distribution
- **Features**: Vectorized implementations, derivative caching for backprop

### 4. Loss Functions

**Quantification of prediction error.**

- **Mean Squared Error (MSE)**: For regression tasks
- **Cross-Entropy**: For classification with softmax
- **Features**: Numerically stable implementations, derivative tracking

### 5. Backpropagation Engine

**Automatic differentiation through computational graph.**

- **Description**: Reverse-mode automatic differentiation computing gradients for all learnable parameters.
- **Implementation**:
  - Gradient accumulation across batch
  - Efficient tensor contraction for weight gradients
  - Memory-efficient computation avoiding full jacobian materialization
- **Complexity**: O(n log n) with respect to forward pass (within constant factors)
- **Use case**: Training arbitrary network architectures

## Performance Characteristics

### Optimization Techniques

- **Cache-oblivious matrix multiplication**: Blocking strategy adapts to system cache hierarchy
- **Vectorized activation functions**: SIMD-friendly implementations for ReLU, sigmoid
- **Batch normalization-ready architecture**: Foundation for advanced techniques
- **Memory layout optimization**: Row-major storage with padding for alignment
- **Efficient gradient computation**: Minimizes intermediate allocations

## Project Structure

```
include/
  └── nn.hpp          Core API: Tensor, Layer, Network classes

src/
  └── nn.cpp          Complete implementation (single translation unit)

examples/
  ├── xor.cpp         Binary classification on XOR problem
  ├── mnist.cpp       10-class classification on MNIST digits

docs/
  ├── theory.md       Mathematical foundations and algorithms used
  ├── results.md      Benchmarking and empirical analysis
  └── architecture.md    (from sorting kernel structure)

```

## Build & Run

```bash
# Build all examples
make

# Run XOR trainer
./examples/xor

# Run MNIST classification
./examples/mnist
./build/xor
```

## 📈 Documentation

- [Architecture](docs/architecture.md) - System design and optimization strategy
- [Theory](docs/theory.md) - Mathematical foundations and algorithms used
- [Benchmark](docs/results.md) - Benchmarking and empirical analysis

## 🛠️ Tech Stack

- **Language**: C++11
- **Build**: CMake, Make
- **Testing**: Custom benchmarking harness
