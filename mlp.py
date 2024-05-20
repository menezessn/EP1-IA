import numpy as np


class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size # Tamanho de cada uma entrada (Pixels)
        self.hidden_size = hidden_size # Número de neurônios na camada oculta
        self.output_size = output_size # Número de neurônios na camada de saída (quantidade de letras do alfabeto)
        
        # Inicialização dos pesos
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        
        # Inicialização dos viéses
        self.bias_hidden = np.random.randn(1, self.hidden_size)
        self.bias_output = np.random.randn(1, self.output_size)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, x):
        # Camada oculta
        self.hidden_output = self.sigmoid(np.dot(x, self.weights_input_hidden) + self.bias_hidden)
        
        # Camada de saída
        self.output = self.sigmoid(np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output)
        
        return self.output
    
    def backward(self, x, y, learning_rate):
        # Cálculo do erro na camada de saída
        error_output = y - self.output
        delta_output = error_output * self.sigmoid_derivative(self.output)
        
        # Cálculo do erro na camada oculta
        error_hidden = np.dot(delta_output, self.weights_hidden_output.T)
        delta_hidden = error_hidden * self.sigmoid_derivative(self.hidden_output)
        
        # Atualização dos pesos e viéses
        self.weights_hidden_output += np.dot(self.hidden_output.T, delta_output) * learning_rate
        self.weights_input_hidden += np.dot(x.reshape(-1,1), delta_hidden) * learning_rate 
        self.bias_output += np.sum(delta_output, axis=0) * learning_rate
        self.bias_hidden += np.sum(delta_hidden, axis=0) * learning_rate

        
    def train(self, X_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            
            if epoch % 100 == 0:
                loss = np.mean(np.square(y_train - self.forward(X_train)))
                print(f'Epoch {epoch}, Loss: {loss}')
            for i in range(len(X_train)):
                inputs = X_train[i]
                target = y_train[i]
                
                # Forward pass
                output = self.forward(inputs)
                
                # Backward pass
                self.backward(inputs, target, learning_rate)