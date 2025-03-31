import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importer vos classes corrigées
from kitml.activations.softMax import SoftMax
from kitml.metrics.logLoss import LogLoss
from kitml.models.singleLayerPerceptron import SingleLayerPerceptron

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