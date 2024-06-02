import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from MLP import MLP

"""Funções auxiliares para avaliação do modelo e
criação de conjuntos de treino e teste"""


def train_test_split(X, y, test_size=0.2):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_index = int(X.shape[0] * (1 - test_size))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# Funções de avaliação do modelo

# Função que retorna a acurácia do modelo


def accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

# Função que retorna a precisão do modelo


"""Precisão é a proporção de verdadeiros
positivos sobre o total de predições positivas"""


def precision(y_pred, y_true, num_classes):
    precision_values = []
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        precision_values.append(tp / (tp + fp + 1e-9))
    return np.mean(precision_values)

# Função que retorna o recall do modelo
# é a proporção de verdadeiros positivos sobre o total de positivos reais


def recall(y_pred, y_true, num_classes):
    recall_values = []
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fn = np.sum((y_pred != c) & (y_true == c))
        recall_values.append(tp / (tp + fn + 1e-9))
    return np.mean(recall_values)

# Função que retorna o F1 Score do modelo
# F1 Score é a média harmônica entre precisão e recall


def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall + 1e-9)

# Função que retorna a matriz de confusão
# A matriz de confusão é uma tabela que mostra as frequências de
# classificação para cada classe do modelo comparando com as classes reais


def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

# Função que plota a matriz de confusão


def plot_confusion_matrix(cm, classes):
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Função de validação cruzada para avaliar o modelo


def cross_validation(
        X, y, input_size, output_size, k=5, epochs=1000,
        patience=10, learning_rate=0.01, hidden_size=64,
        activation_function='relu', early_stopping=True
        ):
    fold_size = X.shape[0] // k
    accuracies = []
    losses = []

    for i in range(k):
        # Separa o conjunto de dados em treino e validação
        start_val = i * fold_size
        end_val = start_val + fold_size

        X_val = X[start_val:end_val]
        y_val = y[start_val:end_val]

        X_train = np.concatenate([X[:start_val], X[end_val:]], axis=0)
        y_train = np.concatenate([y[:start_val], y[end_val:]], axis=0)

        # Cria um novo modelo para cada fold
        model = MLP(
            input_size, hidden_size, output_size,
            learning_rate, activation_function
            )

        #  Treina o modelo
        loss_history = model.train(
            X_train, y_train,
            X_val, y_val, epochs, patience, early_stopping
            )
        losses.append(loss_history)

        # Avalia o modelo no conjunto de validação
        y_pred = model.predict(X_val)
        y_val_labels = np.argmax(y_val, axis=1)
        acc = accuracy(y_pred, y_val_labels)
        accuracies.append(acc)
        print(f'Fold {i+1}, Accuracy: {acc}')

    # Retorna a média das acurácias e os históricos de perda de cada fold
    avg_accuracy = np.mean(accuracies)
    print(f'Average Accuracy: {avg_accuracy}')
    return avg_accuracy, losses

# Função que plota o histórico de perda


def plot_metrics(losses):
    plt.figure(figsize=(12, 4))

    for i, loss_history in enumerate(losses):
        plt.plot(loss_history, label=f'Fold {i+1} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# Função que realiza a busca em grade de hiperparâmetros


def make_grid_search(
        X_train, y_train, input_size, output_size,
        hidden_sizes, activation_functions, learning_rates, early_stopping=True
        ):
    best_accuracy = 0
    best_params = {}

    # Perform grid search
    for hidden_size in hidden_sizes:
        for activation_function in activation_functions:
            for learning_rate in learning_rates:
                print(f"Testing model with hidden_size={hidden_size}, \
                        activation_function={activation_function}, \
                        learning_rate={learning_rate}")

                avg_accuracy, losses = cross_validation(
                    X_train, y_train,  input_size, output_size,
                    k=5, epochs=1000, patience=10,
                    learning_rate=learning_rate, hidden_size=hidden_size,
                    activation_function=activation_function, 
                    early_stopping=early_stopping
                )
                if avg_accuracy > best_accuracy:
                    best_accuracy = avg_accuracy
                    best_params = {
                        'hidden_size': hidden_size,
                        'activation_function': activation_function,
                        'learning_rate': learning_rate
                    }

    print(f'Best params: {best_params}, Best accuracy: {best_accuracy}')
    return best_params, best_accuracy
