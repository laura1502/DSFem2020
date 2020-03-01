
from sklearn.datasets import load_iris
iris = load_iris()

n_samples, n_features = iris.data.shape

print('Número de ejemplos:', n_samples)
print('Número de características:', n_features)
print('Ejemplo primer dato:',iris.data[0])

import numpy as np
np.bincount(iris.target)
print(iris.target_names)


import matplotlib.pyplot as plt
x_index = 3
colors = ['blue', 'red', 'green']
for label, color in zip(range(len(iris.target_names)),colors):
    plt.hist(iris.data[iris.target==label, x_index], 
             label=iris.target_names[label],
             color=color)
plt.xlabel(iris.feature_names[x_index])
plt.legend(loc='upper right')
plt.show()




X, y = iris.data, iris.target
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, 
                                                    train_size=0.5,
                                                    test_size=0.5,
                                                    random_state=123
                                                   #stratify=y
                                                    )
print("Etiquetas para los datos de entrenamiento y test")
print(train_y)
print(test_y)



from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

classifier = KNeighborsClassifier()

classifier.fit(train_X, train_y)

pred_y = classifier.predict(test_X)



print("Precisión:")
print(np.mean(pred_y == test_y))

print('Ejemplos correctamente clasificados:')
correct_idx = np.where(pred_y == test_y)[0]
print(correct_idx)

print('\nEjemplos incorrectamente clasificados:')
incorrect_idx = np.where(pred_y != test_y)[0]
print(incorrect_idx)


colors = ["darkblue", "darkgreen", "gray"]

for n, color in enumerate(colors):
    idx = np.where(test_y == n)[0]
    plt.scatter(test_X[idx, 1], test_X[idx, 2], color=color, label="Clase %s" % str(n))

plt.scatter(test_X[incorrect_idx, 1], test_X[incorrect_idx, 2], color="darkred")

plt.xlabel('sepal width [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc=3)
plt.title("Resultados de clasificación en iris con KNN")
plt.show()




"""
Parametros KNeighborsClassifier
(n_neighbors=5, #Número de vecinos
 weights='uniform', #función de peso utilizada en la predicción. 
 #Valores posibles: uniform , distance. 
 algorithm='auto', #Algoritmo utilizado para calcular los vecinos más cercanos:
 #‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’
)
