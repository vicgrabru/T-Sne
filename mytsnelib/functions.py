import sklearn.manifold as manif
import sklearn.manifold._utils as ut

import numpy as np
from numpy import random
import similarities


def joint_probabilities(distances):
    


    return 0

def joint_probabilities_nn(distances):



    return 0


def TSne(X) :
    distances = similarities.euclidean_distance(X)
    
    probabilities = joint_probabilities(distances)
    
    ut._binary_search_perplexity(distances, 30)

    return 0



"""
def suma(int1: int, int2: int) -> int:
    t = TSNE(
        n_components=2,
        init="random",
        random_state=0,
        perplexity=30,
        n_iter=300,
    )
    return int1 + int2
"""

""""
1.- calcular distancias entre puntos
2.- obtener la distribucion de probabilidad condicional para cada pareja de puntos (p_i|j, p_j|i)
3.- crear espacio de menos dimensiones (misma cantidad de entradas)
4.- obtener la distribucion de probabilidad del nuevo espacio (q)
5.- calcular la divergencia kullback-leibler entre p y q para cada par de puntos
6.- repetir desde el paso 3 (cambiando las posiciones de los puntos segun la divergencia) para disminuir la divergencia resultante

nota: usar la divergencia como referencia para ver los cambios que hay que hacer para reducir la divergencia
en la siguiente iteracion (es como un puntero que indica a donde hay que mover cada punto)



1. Calculate pairwise similarities or dissimilarities between data points in the high-dimensional space. This could be based on various metrics,
    such as Euclidean distance, cosine similarity, or other similarity measures.

2. Construct a probability distribution over these pairwise similarities or dissimilarities. This distribution
    represents the relationships between data points in the high-dimensional space.

3. Define a similar probability distribution over the lower-dimensional space (e.g., 2D or 3D).

4. Optimize the positions of data points in the lower-dimensional space so that the divergence between the two distributions
    (high-dimensional and lower-dimensional) is minimized. This is typically done using gradient descent optimization techniques.

5. Iterate the optimization process to achieve a stable mapping of data points in the lower-dimensional space.


"""