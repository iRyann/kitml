# kitml

KitML est une boîte à outils pour expérimenter différents perceptrons et réseaux de neurones légers en Python.

## Fonctionnalités principales

- Perceptron binaire avec mode ADALINE, métriques configurables et suivi convergence.
- Perceptron multiclasses à une couche avec SoftMax et one-hot encoding.
- Réseau profond modulaire (`DeepModel`) supportant plusieurs couches, fonctions d'activation et métriques.
- Palette de fonctions d’activation (Sigmoid, ReLU, Tanh, Linear, SoftMax) et leurs dérivées.
- Métriques d’apprentissage supervisé (log-loss, MSE) avec gradients prêts pour la rétropropagation.
- Générateurs de jeux de données synthétiques pour tests rapides (AND, XOR, blobs, tables CSV issues des TP).
- Scripts de démonstration/tests (classification, régression, langage des signes) illustrant l’usage de la librairie.
