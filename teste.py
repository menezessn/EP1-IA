import numpy as np

entrada = np.array((1, 2, 3))
pesos = np.random.randn(3, 2)

     


entrada2 = np.array((1, 4, 3))
print(entrada.T.shape)

# print(matriz)
# print(sig)
print((np.mean(entrada == entrada2)))