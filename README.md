# Autograd Engine & MLP From Scratch

A lightweight, pure-Python scalar autograd engine implementing a reverse-mode automatic differentiation graph alongside a Multi-Layer Perceptron (MLP) architecture. Built completely from scratch without external libraries like NumPy or PyTorch.

This project is a clean, minimal codebase designed to clearly showcase how automatic differentiation, topological sorting, computational graphs, and backpropagation work under the hood.

## 🚀 Features

- **Scalar Autograd Engine:** Tracks dynamic execution traces via an explicit computational directed acyclic graph (DAG).
- **Reverse-Mode Backpropagation:** Automatically computes exact gradients for mathematical expressions using the Chain Rule.
- **Topological Sorting:** Orders nodes automatically during backpropagation to resolve dependencies in evaluation sequence.
- **Deep Learning Objects:** Modular object-oriented design building up from basic functional expressions (`Value`) to nodes (`Neuron`), stacks (`Layer`), and neural architectures (`MLP`).
- **Optimization:** Native Stochastic Gradient Descent (SGD) setup for local parameter updates.

---

## 🛠️ Architecture Deep Dive

### 1. The Value Object
The atom of this engine is the `Value` class, which encapsulates a float parameter (`.data`) and tracks its accumulated derivative (`.grad`). Every mathematical operator overloaded in the class (`__add__`, `__sub__`, `__mul__`, `__pow__`) stores child pointers (`self.prev`) and hooks onto custom structural standard closures (`_backward`) to map out local gradient distributions.

### 2. Topological Graph Traversal
To calculate global derivatives systematically across arbitrary functional graphs, a recursive Depth-First Search (DFS) topology algorithm sorts operations cleanly in linear time before computing vector-Jacobian products in reverse sequence.

---

## 💻 Getting Started

### Prerequisites
- Python 3.6 or higher (No external installations required!)

### Installation
Clone this repository directly to your local workspace:
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME