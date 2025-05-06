"""
Test du DeepModel sur le jeu de données LangageDesSignes (classification 5 classes).
- Sépare 250 images pour l'apprentissage (50 par classe) et 50 pour la validation (10 par classe).
- Entraîne un réseau de neurones multicouches et évalue la précision sur le jeu de validation.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from kitml.models.layer import Layer
from kitml.models.deepModel import DeepModel
from kitml.activations.relu import ReLU
from kitml.activations.softMax import SoftMax
from kitml.metrics.logLoss import LogLoss
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

DATA_PATH = '../Datas/LangageDesSignes/data_formatted.csv'
data = pd.read_csv(DATA_PATH, header=None)
X = data.iloc[:, :-5].values 
y = np.argmax(data.iloc[:, -5:].values, axis=1)  

X_train, X_val, y_train, y_val = [], [], [], []
for label in range(5):
    idx = np.where(y == label)[0]
    idx = np.sort(idx)  # Assure un ordre cohérent
    X_train.append(X[idx[:50]])
    y_train.append(y[idx[:50]])
    X_val.append(X[idx[50:60]])
    y_val.append(y[idx[50:60]])
X_train = np.vstack(X_train)
y_train = np.hstack(y_train)
X_val = np.vstack(X_val)
y_val = np.hstack(y_val)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

y_train_oh = np.zeros((y_train.size, 5))
y_train_oh[np.arange(y_train.size), y_train] = 1
y_val_oh = np.zeros((y_val.size, 5))
y_val_oh[np.arange(y_val.size), y_val] = 1

layers = [
    Layer(42, 64, ReLU()),
    Layer(64, 32, ReLU()),
    Layer(32, 5, SoftMax())
]
model = DeepModel(layers, learning_rate=0.05, loss=LogLoss(), metric=LogLoss())

costs, accs = model.fit(X_train, y_train_oh, epochs=300, error_threshold=0.01, one_hot_encoded=True)

y_pred = model.predict(X_val)
accuracy = np.mean(y_pred == y_val)
print(f"Précision sur le validation set : {accuracy:.4f}")

cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["A", "B", "C", "D", "E"])
disp.plot(cmap=plt.cm.Blues)
plt.title('Matrice de confusion (validation LangageDesSignes)')
plt.show()
