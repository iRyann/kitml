import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# Importer vos classes corrigées
from kitml.activations.softMax import SoftMax
from kitml.metrics.logLoss import LogLoss
from kitml.models.singleLayerPerceptron import SingleLayerPerceptron
from kitml.utilities.dataset import dataset_3_1, dataset_3_5

def test_single_layer_perceptron_iris():
    # Charger un dataset de test (iris)
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Prétraiter les données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Diviser en train/test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Créer et entraîner le modèle
    input_size = X_train.shape[1]  # Nombre de caractéristiques
    output_size = len(np.unique(y))  # Nombre de classes
    activation = SoftMax()
    metric = LogLoss()
    eta = 0.1  # Taux d'apprentissage
    nb_epoch = 1000

    # Initialiser le modèle
    model = SingleLayerPerceptron(input_size, output_size, activation, metric, eta, nb_epoch)

    # Entraîner le modèle
    cost_values, accuracy_values = model.train(X_train, y_train, error_threshold=0.01)

    # Évaluer sur l'ensemble de test
    y_pred = model.predict(X_test)
    test_accuracy = np.mean(y_pred == y_test)
    print(f"Précision sur l'ensemble de test: {test_accuracy:.4f}")

    # Visualiser l'évolution de l'entraînement
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(cost_values)
    plt.title('Évolution du coût')
    plt.xlabel('Itérations (x10)')
    plt.ylabel('LogLoss')

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_values)
    plt.title('Évolution de la précision')
    plt.xlabel('Itérations (x10)')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.show()


def test_single_layer_perceptron_3_1():
    x_train, y_train = dataset_3_1()

    # Créer et entraîner le modèle
    input_size = x_train.shape[1]  # Nombre de caractéristiques
    output_size = len(np.unique(y_train))  # Nombre de classes
    activation = SoftMax()
    metric = LogLoss()
    eta = 0.1  # Taux d'apprentissage
    nb_epoch = 1000

    # Initialiser le modèle
    model = SingleLayerPerceptron(input_size, output_size, activation, metric, eta, nb_epoch)

    # Entraîner le modèle
    cost, acc = model.train(x_train, y_train, error_threshold=0.01)

    plt.figure(figsize=(12, 5))
    absciss = np.linspace(0, len(acc) * 10 + 10, len(acc), dtype=int)
    plt.subplot(1, 2, 1)
    plt.plot(absciss, cost, label='Cost')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.title('Training Cost')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(absciss, acc, label='Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/slp_3_1_training.png')
    plt.show()

    # Afficher l'espace des décisions
    xx = np.linspace(-5, 5, 100)
    yy = np.linspace(0, 8, 100)
    xx, yy = np.meshgrid(xx, yy)
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_predictions = model.predict(grid)
    grid_predictions = grid_predictions.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, grid_predictions, alpha=0.8)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, edgecolors='k', marker='o')
    plt.title('Espace des décisions')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig('outputs/slp_3_1_decision_space.png')
    plt.show()
    
    # Afficher la matrice de confusion

    y_pred = model.predict(x_train)
    cm = confusion_matrix(y_train, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_train))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Matrice de confusion')
    plt.savefig('outputs/slp_3_1_confusion_matrix.png')
    plt.show()

def test_single_layer_perceptron_3_5():
    x_train, y_train = dataset_3_5()

    # Créer et entraîner le modèle
    input_size = x_train.shape[1]  # Nombre de caractéristiques
    output_size = len(np.unique(y_train))  # Nombre de classes
    activation = SoftMax()
    metric = LogLoss()
    eta = 0.1  # Taux d'apprentissage
    nb_epoch = 1000

    # Initialiser le modèle
    model = SingleLayerPerceptron(input_size, output_size, activation, metric, eta, nb_epoch)

    # Entraîner le modèle
    cost, acc = model.train(x_train, y_train, error_threshold=0.01)

    plt.figure(figsize=(12, 5))
    absciss = np.linspace(0, len(acc) * 10 + 10, len(acc), dtype=int)
    plt.subplot(1, 2, 1)
    plt.plot(absciss, cost, label='Cost')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.title('Training Cost')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(absciss, acc, label='Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/slp_3_5_training.png')
    plt.show()
    
    # Afficher la matrice de confusion
    y_pred = model.predict(x_train)
    cm = confusion_matrix(y_train, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_train))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Matrice de confusion')
    plt.savefig('outputs/slp_3_5_confusion_matrix.png')
    plt.show()

test_single_layer_perceptron_3_5()