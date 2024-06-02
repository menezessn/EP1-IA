import numpy as np  # type: ignore


# Implementação de uma rede neural MLP (Multi-Layer Perceptron)
# com uma camada oculta


class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, activation_function='relu'):
        # Definição de hiperparâmetros
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        """Inicialização de pesos aleatórios e bias zerados para as
        camadas ocultas e de saída"""
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

        # Setando qual função de ativação será utilizada e sua derivada
        self.activation_function = activation_function
        self.activation = self.get_activation_function(activation_function)
        self.activation_derivative = self.get_activation_derivative(activation_function)

    # Função que retorna a função de ativação escolhida
    def get_activation_function(self, activation_function):
        if activation_function == 'relu':
            return self.relu
        elif activation_function == 'sigmoid':
            return self.sigmoid
        elif activation_function == 'tanh':
            return self.tanh
        else:
            raise ValueError("Unsupported activation function")

    # Função que retorna a derivada da função de ativação escolhida
    def get_activation_derivative(self, activation_function):
        if activation_function == 'relu':
            return self.relu_derivative
        elif activation_function == 'sigmoid':
            return self.sigmoid_derivative
        elif activation_function == 'tanh':
            return self.tanh_derivative
        else:
            raise ValueError("Unsupported activation function")

    # Funções de ativação que podem ser utilizadas e suas derivadas
    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x)**2

    # Função softmax para a camada de saída
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    # Função de perda de entropia cruzada
    def cross_entropy_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m
        return loss

    # Etapa de feedforwarding da rede
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    # Etapa de backpropagation da rede e atualização dos pesos
    def backward(self, X, y_true):
        m = y_true.shape[0]

        # Gradientes da camada de saída
        dZ2 = self.a2 - y_true
        dW2 = np.dot(self.a1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Gradientes da camada escondida
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.activation_derivative(self.z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Atualizar pesos e bias com gradiente descendente
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    # Funcão de treinamento da rede
    def train(self, X_train, y_train, X_val, y_val, epochs=1000, patience=10, early_stopping=True):
        best_loss = np.inf
        epochs_without_improvement = 0
        loss_history = []

        for epoch in range(epochs):
            y_pred_train = self.forward(X_train)
            loss = self.cross_entropy_loss(y_pred_train, y_train)
            loss_history.append(loss)
            self.backward(X_train, y_train)

            y_pred_val = self.forward(X_val)
            val_loss = self.cross_entropy_loss(y_pred_val, y_val)

            if early_stopping:
                # Early stopping check
                if val_loss < best_loss - 0.1:
                    best_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    print(f'Early stopping at epoch {epoch} with validation loss {val_loss}')
                    break

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}, Validation Loss: {val_loss}')

        return loss_history

    # faz predições baseado em uma entrada X
    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)
