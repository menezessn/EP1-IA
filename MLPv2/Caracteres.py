import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dense import Dense
from activations import Tanh
from losses import mse, mse_prime
from network import train, predict

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
    return np.array(resultado)


# Leitura dos dados
file_path = 'Data/X.txt' 
X = np.array(read_x(file_path))
X = np.reshape(X, (1326, 120, 1))


file_path = 'Data/Y_letra.txt' 
Y_letters = read_Y(file_path)
Y = np.array(array_letters(Y_letters))
Y = np.reshape(Y, (1326, 26, 1))

input_neurons = 120
hidden_neurons = 60
output_neurons = 26

network = [
    Dense(input_neurons, hidden_neurons),
    Tanh(),
    Dense(hidden_neurons, output_neurons),
    Tanh()
]

# train


X_train = X[:-130]
y_train = Y[:-130]
X_test = X[-130:]
y_test = Y[-130:]
X_test = np.reshape(X_test, (130, 120, 1))
y_test = np.reshape(y_test, (130, 26, 1))

# Criação e treinamento da MLP


train(network, mse, mse_prime, X_train, y_train, epochs=1000, learning_rate=0.1)

# Avaliação da MLP no conjunto de teste 
predictions = np.argmax(predict(network, X_test), axis=1)
y_test_labels = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions == y_test_labels)
print("Acurácia no conjunto de teste:", accuracy)