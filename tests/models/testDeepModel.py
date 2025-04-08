import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import de nos classes personnalisées
from kitml.activations.activation import Activation
from kitml.activations.relu import ReLU
from kitml.activations.sigmoid import Sigmoid
from kitml.activations.softMax import SoftMax
from kitml.activations.linear import Linear
from kitml.metrics.logLoss import LogLoss
from kitml.metrics.meanQuadraticError import MeanQuadraticError
from kitml.models.layer import Layer
from kitml.models.deepModel import DeepModel

def test_classification():
    """Test du DeepModel sur un problème de classification"""
    print("\n=== Test de classification ===")
    
    # Génération d'un dataset de classification
    X, y = make_classification(n_samples=1000, n_classes=3, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
    
    # Affichage des données et des classes
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
    plt.title('Données de classification')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Classe')
    plt.show()

    # Split et normalisation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Construction du modèle
    layers = [
        Layer(2, 64, ReLU()),
        Layer(64, 32, ReLU()),
        Layer(32, 3, SoftMax())
    ]
    
    # Initialisation du modèle
    model = DeepModel(layers, learning_rate=0.1, loss=LogLoss(), metric=LogLoss())
    
    # Entraînement
    costs, accuracies = model.fit(X_train, y_train, epochs=500, error_threshold=0.01, one_hot_encoded=False)
    
    # Évaluation
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Précision finale sur le test set: {accuracy:.4f}")
    
    # Visualisation
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(costs)
    plt.title('Évolution du coût')
    plt.xlabel('Évaluation (tous les 10 epochs)')
    plt.ylabel('Coût')
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Évolution de la précision')
    plt.xlabel('Évaluation (tous les 10 epochs)')
    plt.ylabel('Précision')
    
    plt.tight_layout()
    plt.savefig('classification_results.png')
    plt.show()

    # Affichage des prédictions
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', edgecolor='k', s=50)
    plt.title('Prédictions du modèle')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Classe prédite')
    plt.savefig('predictions.png')
    plt.show()
    # Affichage de la matrice de confusion
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Matrice de confusion')
    plt.savefig('confusion_matrix.png')
    plt.show()
    print(f"Matrice de confusion:\n{cm}")
    print(f"Précision finale sur le test set: {accuracy:.4f}")

    return model

def test_regression():
    """Test du DeepModel sur un problème de régression"""
    print("\n=== Test de régression ===")
    
    # Génération d'un dataset de régression
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    y = y.reshape(-1, 1)  # Reshape pour avoir la forme (samples, 1)
    
    # Split et normalisation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)
    
    # Construction du modèle
    layers = [
        Layer(10, 64, ReLU()),
        Layer(64, 32, ReLU()),
        Layer(32, 1, Linear())
    ]
    
    # Initialisation du modèle
    model = DeepModel(layers, learning_rate=0.1, loss=MeanQuadraticError(), metric=MeanQuadraticError())
    
    # Entraînement
    costs, _ = model.fit(X_train, y_train, epochs=1000, error_threshold=0.01, one_hot_encoded=False)
    
    # Évaluation
    y_pred = model.predict(X_test)
    mse = np.mean((y_pred - y_test.flatten()) ** 2)
    print(f"MSE finale sur le test set: {mse:.4f}")
    
    # Visualisation
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(costs)
    plt.title('Évolution du coût (MSE)')
    plt.xlabel('Évaluation (tous les 10 epochs)')
    plt.ylabel('MSE')
    
    plt.subplot(2, 1, 2)
    indices = np.argsort(y_test.flatten())  # Pour un affichage ordonné
    plt.scatter(y_test.flatten()[indices], y_pred.T[indices], alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('Prédictions vs Valeurs réelles')
    plt.xlabel('Valeurs réelles')
    plt.ylabel('Prédictions')
    
    plt.tight_layout()
    plt.savefig('regression_results.png')
    plt.show()
    
    return model

def compare_activations():
    """Comparaison de différentes fonctions d'activation sur un problème de classification"""
    print("\n=== Comparaison des fonctions d'activation ===")
    
    # Génération d'un dataset de classification
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, n_informative=15, random_state=42)
    
    # Split et normalisation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Fonctions d'activation à tester
    activation_functions = {
        "ReLU": ReLU(),
        "Sigmoid": Sigmoid()
    }
    
    results = {}
    
    for name, activation in activation_functions.items():
        print(f"\nTest avec activation: {name}")
        
        # Construction du modèle
        layers = [
            Layer(20, 64, activation),
            Layer(64, 32, activation),
            Layer(32, 3, SoftMax())
        ]
        
        # Initialisation du modèle
        model = DeepModel(layers, learning_rate=0.01, loss=LogLoss(), metric=LogLoss())
        
        # Entraînement
        costs, accuracies = model.fit(X_train, y_train, epochs=100, error_threshold=0.01, one_hot_encoded=False)
        
        # Évaluation
        y_pred = model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        print(f"Précision finale avec {name}: {accuracy:.4f}")
        
        # Stockage des résultats
        results[name] = {
            "costs": costs,
            "accuracies": accuracies,
            "final_accuracy": accuracy
        }
    
    # Visualisation comparative
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    for name, data in results.items():
        plt.plot(data["costs"], label=name)
    plt.title('Comparaison de l\'évolution du coût')
    plt.xlabel('Évaluation (tous les 10 epochs)')
    plt.ylabel('Coût')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    for name, data in results.items():
        plt.plot(data["accuracies"], label=name)
    plt.title('Comparaison de l\'évolution de la précision')
    plt.xlabel('Évaluation (tous les 10 epochs)')
    plt.ylabel('Précision')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('activation_comparison.png')
    plt.show()
    
    return results


print("Début des tests du DeepModel...")

# Test de classification
classification_model = test_classification()

# Test de régression
# regression_model = test_regression()

# # Comparaison des fonctions d'activation
# activation_results = compare_activations()

print("\nTous les tests sont terminés.")