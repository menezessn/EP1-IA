from mlp import *
from readFiles import *
import numpy as np


# Leitura dos dados
file_path = 'Data/X.txt' 
X = np.array(read_x(file_path))

file_path = 'Data/Y_letra.txt' 
Y_letters = read_Y(file_path)
Y = np.array(array_letters(Y_letters))


# Divisão dos dados de treinamento e teste
X_train = X[:-350]
y_train = Y[:-350]
X_test = X[-350:]
y_test = Y[-350:]

# Parâmetros da MLP
input_size = 120  # Número de pixels
hidden_size = 30  # Número de neurônios na camada oculta (pode ser ajustado experimentalmente)
output_size = 26  # Número de letras do alfabeto

# Criação e treinamento da MLP
mlp = MLP(input_size, hidden_size, output_size)
mlp.train(X_train, y_train, epochs=1000, learning_rate=0.1)

# Avaliação da MLP no conjunto de teste 
predictions = np.argmax(mlp.forward(X_test), axis=1)
y_test_labels = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions == y_test_labels)
print("Acurácia no conjunto de teste:", accuracy)
