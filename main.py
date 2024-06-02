import numpy as np  # type: ignore
from aux import (
                    train_test_split, accuracy, precision, recall, f1_score,
                    confusion_matrix, plot_confusion_matrix,
                    make_grid_search, plot_metrics)
from MLP import MLP

# Tamanho da entrada e saída da rede
input_size = 120
output_size = 26

# X deve ser do shape (num_samples, input_size)
# y deve ser do shape (num_samples, output_size), one-hot encoded
X = np.load("X.npy").reshape(-1, input_size)
Y = np.load("Y_classe.npy").reshape(-1, output_size)

# divide o dataset
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# grid
hidden_sizes = [20, 40, 60, 80, 100, 120, 150]
activation_functions = ['relu', 'sigmoid', 'tanh']
learning_rates = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

# acha os melhores hiperparâmetros
best_params, best_accuracy = make_grid_search(
        X_train, y_train, input_size, output_size, hidden_sizes,
        activation_functions, learning_rates, early_stopping=False
    )

# Treina o modelo final com os melhores hiperparâmetros
mlp = MLP(
    input_size, best_params['hidden_size'], output_size,
    best_params['learning_rate'], best_params['activation_function']
)

loss_history = mlp.train(
    X_train, y_train, X_test, y_test,
    epochs=1000, patience=10
)

# avalia modelo no modelo de teste
y_pred = mlp.predict(X_test)
y_test_labels = np.argmax(y_test, axis=1)

acc = accuracy(y_pred, y_test_labels)
prec = precision(y_pred, y_test_labels, output_size)
rec = recall(y_pred, y_test_labels, output_size)
f1 = f1_score(prec, rec)

print(f'Test Accuracy: {acc}')
print(f'Test Precision: {prec}')
print(f'Test Recall: {rec}')
print(f'Test F1 Score: {f1}')

# matriz de confusão 
cm = confusion_matrix(y_test_labels, y_pred, output_size)
plot_confusion_matrix(
    cm,
    classes=[chr(i) for i in range(ord('A'), ord('Z') + 1)]
)

# plot loss
plot_metrics([loss_history])
