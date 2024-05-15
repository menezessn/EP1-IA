import numpy as np


# Função para leitura dos dados de entrada, coloca em um array de arrays
def read_x(file_path):
    arrays = []
    with open(file_path, 'r') as file:
        for line in file:
            array = [int(value) for value in line.strip().split(',') if value.strip()]
            arrays.append(array)
    return arrays

# Função para leitura dos dados de saída, coloca em um array de letras
def read_Y(file_path):
    letters = []
    with open(file_path, 'r') as file:
        for line in file:
            letter = line.strip()
            letters.append(letter)
    return letters

# Função para transformar as letras em vetores one-hot
def array_letters(letras):
    resultado = []
    for letra in letras:
        vetor = np.zeros(26)
        indice = ord(letra.lower()) - ord('a')
        vetor[indice] = 1
        resultado.append(vetor)
    return resultado


# Leitura dos dados
file_path = 'Data/X.txt' 
X = np.array(read_x(file_path))


file_path = 'Data/Y_letra.txt' 
Y_letters = read_Y(file_path)
Y = np.array(array_letters(Y_letters))


class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size # Tamanho de cada uma entrada (Pixels)
        self.hidden_size = hidden_size # Número de neurônios na camada oculta
        self.output_size = output_size # Número de neurônios na camada de saída (quantidade de letras do alfabeto)
        
        # Inicialização dos pesos
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) 
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        
        # Inicialização dos viéses
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))
        
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
        self.weights_input_hidden += np.dot(x.reshape(-1, 1), delta_hidden) * learning_rate 
        self.bias_output += np.sum(delta_output, axis=0) * learning_rate
        self.bias_hidden += np.sum(delta_hidden, axis=0) * learning_rate

        
    def train(self, X_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            for i in range(len(X_train)):
                inputs = X_train[i]
                target = y_train[i]
                
                # Forward pass
                output = self.forward(inputs)
                
                # Backward pass
                self.backward(inputs, target, learning_rate)
                
            if epoch % 100 == 0:
                loss = np.mean(np.square(y_train - self.forward(X_train)))
                print(f'Epoch {epoch}, Loss: {loss}')

# Divisão dos dados de treinamento e teste
X_train = X[:-130]
y_train = Y[:-130]
X_test = X[-130:]
y_test = Y[-130:]

# Parâmetros da MLP
input_size = 120  # Número de pixels
hidden_size = 40  # Número de neurônios na camada oculta (pode ser ajustado experimentalmente)
output_size = 26  # Número de letras do alfabeto

# Criação e treinamento da MLP
mlp = MLP(input_size, hidden_size, output_size)
mlp.train(X_train, y_train, epochs=3000, learning_rate=0.1)

# Avaliação da MLP no conjunto de teste 
predictions = np.argmax(mlp.forward(X_test), axis=1)
y_test_labels = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions == y_test_labels)
print("Acurácia no conjunto de teste:", accuracy)