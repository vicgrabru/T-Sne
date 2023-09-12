import sklearn.manifold as manif
import sklearn.manifold._utils as ut

import numpy as np
#from numpy import random
import similarities


def gradient(high_dimension_probs, low_dimension_probs, embed):
    prob_diff = high_dimension_probs - low_dimension_probs

    embed_distances = similarities.euclidean_distance(embed, return_squared=True)
    

    n = high_dimension_probs.shape[0]

    gradient = np.zeros(shape=(n,embed.shape[1]))
    

    for i in range(0,n):
        y_diff = embed[i] - embed
        prob_diff = high_dimension_probs[i]-low_dimension_probs[i]
        dist = 1+embed_distances[i]
        gradient[i] = 4*np.sum(prob_diff*y_diff/dist, axis=0)
    return gradient


def kl_divergence(high_dimension_p, low_dimension_q):
    result = high_dimension_p * np.log(np.divide(high_dimension_p, low_dimension_q))
    return np.sum(result)




class TSne():

    def __init__(self, *, n_dimensions, perplexity=30, perplexity_tolerance=0.1,
                 metric='euclidean', init_method="random", init_embed=None, early_exaggeration=4,
                 learning_rate=200, max_iter=1000, momentum_params=[250,0.5,0.8], descent_mode="recursive"):
        #validacion de parametros
        if metric!='euclidean': raise ValueError("The only currently supported metric is Euclidean distance")
        if descent_mode not in ["recursive", "iterative"]: raise ValueError('descent_mode must be either "recursive" or "iterative"')

        #inicializacion de la clase
        self.n_dimensions = n_dimensions
        self.perplexity = perplexity
        self.perplexity_tolerance = perplexity_tolerance
        self.metric = metric
        self.init_method = init_method
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.embed = init_embed
        self.momentum_params = momentum_params
        self.early_exaggeration = early_exaggeration
        self.embedding_current_t = 0
        self.descent_mode = descent_mode


    def momentum(self,t):
        t_limit = self.momentum_params[0]
        m1 = self.momentum_params[1]
        m2 = self.momentum_params[2]
        if t<t_limit:
            return m1
        else:
            self.early_exaggeration=1
            return m2


    def initial_embed(self, *, data, zeros=False, entries=None):
        #TODO hacer de vd el embedding inicial
        n_points = data.shape()[0]
        embed = np.zeros(shape=[n_points, self.n_dimensions])
        if (data==None and entries==None) or zeros:
            return embed
        elif self.init_method=="random":
            if data==None:
                size_x = entries
            else:
                size_x = data.shape[0]
            self.embed = np.random.normal(loc=0.0, scale=1e-4, size=(size_x, self.n_dimensions))
        else:
            self.embed = embed
    
    

    def fit(self, X):
        
        if self.descent_mode=="recursive":
            embedded = self.gradient_descent_recursive(self.max_iter, X)
        elif self.descent_mode=="recursive":
            embedded = self.gradient_descent_iterative(self.max_iter, X)
        
        return embedded



    def gradient_descent_recursive(self, t, data):
        """Performs the gradient descent in a recursive manner

        Parameters
        ----------
        t: int
            The amount of iterations to perform.


        data : ndarray of shape (n_samples, n_features)
            An array of the data where each row is a sample and each column is a feature. 

        Returns
        -------
        embed : ndarray of shape (n_samples, n_dimensions) that contains the resulting embedding after t iterations.
        """
        distances_original = similarities.euclidean_distance(data)
        affinities_original = similarities.joint_probabilities(distances_original, self.perplexity, self.perplexity_tolerance)
        
        
        return self.__gradient_descent(t,affinities_original)

    def __gradient_descent(self,t,affinities_original):
        """Computational side of the gradient descent,computed as:
        gradient_descent(t) = gradient_descent(t-1) + learningRate*gradient(current_embedding) + momentum(t)*(gradient_descent(t-1)-gradient_descent(t-2))
        """

        if t<0:
            raise ValueError("t must be a positive Integer")

        if t == 0:
            new_embed = self.initial_embed(affinities_original, zeros=True)
        elif t == 1:
            if self.embed is not None:
                new_embed = self.embed
            else:
                new_embed = self.initial_embed(entries=affinities_original.shape[0])
        else:
            if self.embed==None:
                raise ValueError("Embed can only be None for t<2")
            
            if self.embedding_current_t==t-1:
                embed_t1 = self.embed
            else:
                embed_t1 = self.__gradient_descent(t-1,affinities_original)
            embed_t2 = self.__gradient_descent(t-2,affinities_original)
            
            distances_embed = similarities.euclidean_distance(embed_t1)
            affinities_current = similarities.joint_probabilities(distances_embed,self.perplexity,self.perplexity_tolerance)
            grad = gradient(affinities_original,self.early_exaggeration*affinities_current,self.embed)

            new_embed = embed_t1 + self.learning_rate * grad + self.momentum(t)*(embed_t1-embed_t2)
            self.embed = new_embed
        
        self.embedding_current_t = t
        return new_embed
    
    def gradient_descent_iterative(self, t, data):
        distances_original = similarities.euclidean_distance(data)
        affinities_original = similarities.joint_probabilities(distances_original, self.perplexity, self.perplexity_tolerance)

        n_points = data.shape[0]
        Y = np.zeros(shape=(t, n_points, self.n_dimensions))
        Y[0] = self.initial_embed(data, zeros=True)

        Y[1] = self.embed if self.embed is not None else self.initial_embed(entries=affinities_original.shape[0])
        
        for i in range(2,t):
            distances_embed = similarities.euclidean_distance(Y[i-1])
            affinities_current = similarities.joint_probabilities(distances_embed,self.perplexity,self.perplexity_tolerance)
            grad = gradient(affinities_original,self.early_exaggeration*affinities_current,Y[i-1])
            Y[i] = Y[i-1] + self.learning_rate*grad + self.momentum(i)*(Y[i-1]-Y[i-2])
            self.embedding_current_t = i
        
        return Y[t-1]

        
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