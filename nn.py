import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Generate synthetic data
def generate_data(n=100):
    np.random.seed(42)
    X = np.random.randn(n, 2)
    y = np.array([1 if x[0] * x[0] + x[1] * x[1] < 1 else 0 for x in X])
    return X, y

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Derivative of the sigmoid function
def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Neural network class
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

    def forward(self, X):
        self.hidden_layer = sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        self.output_layer = sigmoid(np.dot(self.hidden_layer, self.weights_hidden_output) + self.bias_output)
        return self.output_layer

    def backward(self, X, y):
        output_error = self.output_layer - y.reshape(-1, 1)
        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * sigmoid_derivative(self.hidden_layer)

        # Update weights and biases
        self.weights_hidden_output -= self.learning_rate * np.dot(self.hidden_layer.T, output_error)
        self.bias_output -= self.learning_rate * np.sum(output_error, axis=0, keepdims=True)
        self.weights_input_hidden -= self.learning_rate * np.dot(X.T, hidden_error)
        self.bias_hidden -= self.learning_rate * np.sum(hidden_error, axis=0, keepdims=True)

    def train(self, X, y, epochs=100):
        self.history = []
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)
            if epoch % 10 == 0:
                self.history.append(self.weights_input_hidden.copy())

    def predict(self, X):
        return self.forward(X)

# Visualization function
def visualize_training(X, y, nn, epochs=100):
    fig, ax = plt.subplots()
    ax.set_title('Neural Network Decision Boundary')

    # Create a mesh to plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    def animate(i):
        ax.clear()
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
        Z = nn.forward(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.8, levels=np.linspace(0, 1, 10), cmap='RdBu', vmin=0, vmax=1)
        ax.set_title(f'Epoch: {i * 10}')
    
    ani = FuncAnimation(fig, animate, frames=len(nn.history), interval=200, repeat=False)
    plt.show()

# Main function
if __name__ == "__main__":
    X, y = generate_data()
    nn = SimpleNN(input_size=2, hidden_size=3, output_size=1, learning_rate=0.1)
    nn.train(X, y, epochs=100)
    visualize_training(X, y, nn)
