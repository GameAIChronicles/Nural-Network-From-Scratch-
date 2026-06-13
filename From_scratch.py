import random
import math
import time


class Value:
    """
    Stores a single scalar value and its gradient.
    Builds a dynamic computational graph for automatic differentiation.
    """

    def __init__(self, data=None, children=()):
        self.data = float(data) if data is not None else 0.0
        self.grad = 0.0
        # Internal variable holding the function that propagates the gradient backward
        self._backward = lambda: None
        # Set of parent nodes used to build the graph
        self.prev = set(children)

    def __repr__(self):
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other))

        def backward():
            self.grad += out.grad
            other.grad -= out.grad  # Corrected: d/d(other) (self - other) = -1

        out._backward = backward
        return out

    def __rsub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(other.data - self.data, (self, other))

        def backward():
            self.grad -= out.grad  # Corrected: d/d(self) (other - self) = -1
            other.grad += out.grad

        out._backward = backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def abs(self):
        """Returns the absolute value."""
        out = Value(abs(self.data), (self,))

        def backward():
            sign = 1.0 if self.data >= 0 else -1.0
            self.grad += sign * out.grad

        out._backward = backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only supporting int/float powers for now"
        out = Value(self.data ** other, (self,))

        def backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = backward
        return out

    def tanh(self):
        """Applies the Hyperbolic Tangent activation function."""
        t = math.tanh(self.data)
        out = Value(t, (self,))

        def backward():
            self.grad += (1.0 - t ** 2) * out.grad

        out._backward = backward
        return out

    def backward(self):
        """
        Triggers backpropagation by dynamically sorting the computational graph
        topologically and executing the chain rule backward.
        """
        self.grad = 1.0
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        for value in reversed(topo):
            value._backward()


class Neuron:
    """A single artificial neuron containing weights, a bias, and a tanh activation."""

    def __init__(self, nin):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # Calculates: activation(w * x + b)
        act = sum((wi * xi for wi, xi in zip(self.weights, x)), self.bias)
        return act.tanh()

    def parameters(self):
        """Returns a flat list of all trainable parameter nodes in this neuron."""
        return self.weights + [self.bias]


class Layer:
    """A fully connected layer containing an array of independent neurons."""

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        """Returns a flat list of all parameters across all neurons in this layer."""
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    """A Multi-Layer Perceptron (Neural Network) engine built from scratch."""

    def __init__(self, nin, nlayers: list):
        dimensions = [nin] + nlayers
        self.layers = [Layer(dimensions[i], dimensions[i + 1]) for i in range(len(nlayers))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """Returns a flat list of all parameters in the entire network."""
        return [p for layer in self.layers for p in layer.parameters()]


if __name__ == '__main__':
    # Initialize an MLP with 2 inputs, two hidden layers of 8 neurons, and 1 output neuron
    nn = MLP(2, [8, 8, 1])

    # Generate synthetic training dataset
    # X context: 100 samples containing pairs of numbers between -1.0 and 1.0
    x_train = [[round(random.uniform(-1, 1), 2), round(random.uniform(-1, 1), 2)] for _ in range(100)]
    # Target function: simple addition. Note: Targets fall between -2.0 and 2.0.
    # Since tanh limits output to [-1, 1], this network will attempt to match it as closely as possible.
    y_train = [i[0] + i[1] for i in x_train]

    print("Starting optimization loop...")
    start_time = time.time()

    # Optimization/Training Loop
    for step in range(101):

        # 1. Forward pass: compute predictions and calculate Mean Squared Error loss
        ypred = [nn(i)[0] for i in x_train]
        loss = sum(((vout - ygt) ** 2 for ygt, vout in zip(y_train, ypred)), Value(0.0))

        # 2. Zero out gradients before performing backpropagation
        for p in nn.parameters():
            p.grad = 0.0

        # 3. Backward pass: Auto-differentiation using topological sort
        loss.backward()

        # 4. Optimization step: Stochastic Gradient Descent (SGD)
        learning_rate = 0.01
        for p in nn.parameters():
            p.data -= learning_rate * p.grad

        if step % 10 == 0:
            print(f"Step {step:3d} | Loss: {loss.data:.4f}")

    print(f"\nTraining completed in: {time.time() - start_time:.2f} seconds")