import numpy as np

# Exemple de matrice de caractéristiques
x = np.array([[1.0, 2.0],
              [2.0, 3.0],
              [3.0, 4.0]])

# Ajouter une colonne de 1 pour le biais
x_with_bias = np.c_[np.ones((x.shape[0], 1)), x]

print("Matrice originale :")
print(x)

print("\nMatrice avec biais :")
print(x_with_bias)
