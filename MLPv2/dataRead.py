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
Y = np.reshape(Y, (1326, 26, 1))

print(Y);