#import sklearn.manifold as manif
#import sklearn.manifold._utils as ut

import numpy as np
import matplotlib.pyplot as plt
from mytsnelib import similarities
import time
import warnings



def gradient(high_dimension_probs, low_dimension_probs, embed,*, n_neighbors=None, embed_distances=None):
    """Computes the gradient of the cost function.

    Parameters
    ----------
        high_dimension_probs: ndarray of shape (n_samples, n_samples)
            The joint probabilities for the samples in the original dimension.


        low_dimension_probs: ndarray of shape (n_samples, n_samples)
            The joint probabilities for the samples embedded in the lower dimension.
        
        embed: ndarray of shape (n_samples, n_dimensions)
            The samples embedded in the lower dimension.

    Returns
    -------
        gradient : ndarray of shape (n_samples, n_dimensions)
            The gradient of the cost function.
            It can be interpreted as a set of arrays that shows in what direction each point
            must be "pulled" towards for a better embedding.
    """
    if n_neighbors is None:
        n_neighbors = embed.shape[0]-1
    
    prob_diff = high_dimension_probs - low_dimension_probs
    
    if embed_distances is None:
        embed_distances = similarities.euclidean_distance_neighbors(embed,n_neighbors=n_neighbors)
    
    embed_distances = np.power(embed_distances,2)

    #1 + distancia euclidiana al cuadrado
    embed_distances += np.ones_like(embed_distances)
    np.fill_diagonal(embed_distances, np.inf)


    n = high_dimension_probs.shape[0]
    gradient = np.zeros_like(embed)
    
    # print(embed_distances)
    
    # for i in range(0,n):
    #     for j in range(0,n):
    #         aux = 4.0*prob_diff[i][j]/embed_distances[i][j]
    #         embed_diff = embed[i] - embed[j]
    #         gradient[i] += np.multiply(embed_diff, aux)

    for i in range(0,n):
        #p_ij - q_ij
        prob_diff = high_dimension_probs[i]-low_dimension_probs[i]
        #y_i - y_j
        embed_diff = embed[i] - embed
        
        # embed_i_rep = np.repeat(embed[i], embed.shape[0], axis=0)
        # embed_diff = embed_i_rep - embed

        # 1 + (||y_i - y_j||)^2
        dist = embed_distances[i]
        n_repeats = embed_diff.shape[1]
        #expand prob_diff and dist to match embed_diff's shape
        if embed_diff.shape != prob_diff.shape:
            prob_diff_temp = prob_diff
            prob_diff = np.repeat(np.expand_dims(prob_diff_temp, axis=1), n_repeats, axis=1)
        if embed_diff.shape != dist.shape:
            dist_temp = dist
            dist = np.repeat(np.expand_dims(dist_temp, axis=1), n_repeats, axis=1)
        # print(dist)
        mult = np.multiply(prob_diff, embed_diff)
        div = np.divide(mult, dist)
        gradient[i] = 4*np.sum(div, axis=0)
    
    return gradient

def gradient_no_neighbors(high_dimension_probs, low_dimension_probs, embed):
    
    embed_distances = similarities.pairwise_euclidean_distance(embed)

    #1 + distancia euclidiana al cuadrado
    embed_distances += np.ones_like(embed_distances)
    np.fill_diagonal(embed_distances, np.inf)


    n = high_dimension_probs.shape[0]
    result = np.zeros_like(embed)

    for i in range(0,n):
        #p_ij - q_ij
        prob_diff = high_dimension_probs[i]-low_dimension_probs[i]
        #y_i - y_j
        embed_diff = embed[i] - embed

        # 1 + (||y_i - y_j||)^2
        dist = embed_distances[i]
        n_repeats = embed_diff.shape[1]
        #expand prob_diff and dist to match embed_diff's shape
        if embed_diff.shape != prob_diff.shape:
            prob_diff_temp = prob_diff
            prob_diff = np.repeat(np.expand_dims(prob_diff_temp, axis=1), n_repeats, axis=1)
        if embed_diff.shape != dist.shape:
            dist_temp = dist
            dist = np.repeat(np.expand_dims(dist_temp, axis=1), n_repeats, axis=1)
        # print(dist)
        mult = np.multiply(prob_diff, embed_diff)
        div = np.divide(mult, dist)
        result[i] = 4*np.sum(div, axis=0)
    
    return result

def gradient_no_neighbors_2(high_dimension_probs, low_dimension_probs, embed):
    
    #embed_distances = pairwise_distances(embed)
    embed_distances = np.power(similarities.euclidean_distance(embed),2)

    n = high_dimension_probs.shape[0]
    result = np.zeros_like(embed)

    prob_diff = high_dimension_probs - low_dimension_probs
    embed_diff = np.expand_dims(embed,1) - np.expand_dims(embed,0)

    for i in range(0,n):
        #p_ij - q_ij
        pq_diff = prob_diff[i]
        #y_i - y_j
        emb_diff = embed_diff[i]

        # 1 + (||y_i - y_j||)^2
        dist = 1/ (1 + embed_distances[i])
        n_repeats = emb_diff.shape[1]
        #expand prob_diff and dist to match embed_diff's shape
        if emb_diff.shape != pq_diff.shape:
            pq_diff_temp = pq_diff
            pq_diff = np.repeat(np.expand_dims(pq_diff_temp, axis=1), n_repeats, axis=1)
        if emb_diff.shape != dist.shape:
            dist_temp = dist
            dist = np.repeat(np.expand_dims(dist_temp, axis=1), n_repeats, axis=1)
        # print(dist)
        mult = np.multiply(pq_diff, emb_diff)
        div = np.divide(mult, dist)
        result[i] = 4*np.sum(div, axis=0)
    
    return result

def gradient_no_neighbors_3(high_dimension_probs, low_dimension_probs, embed):
    
    #embed_distances = pairwise_distances(embed)
    embed_distances = similarities.pairwise_euclidean_distance(embed)

    n = high_dimension_probs.shape[0]
    result = np.zeros_like(embed)

    prob_diff = high_dimension_probs - low_dimension_probs
    embed_diff = np.expand_dims(embed,1) - np.expand_dims(embed,0)

    for i in range(0,n):
        #p_ij - q_ij
        pq_diff = prob_diff[i]
        #y_i - y_j
        emb_diff = embed_diff[i]

        # 1 + (||y_i - y_j||)^2
        dist = 1/ (1 + embed_distances[i])
        n_repeats = emb_diff.shape[1]
        #expand prob_diff and dist to match embed_diff's shape
        if emb_diff.shape != pq_diff.shape:
            pq_diff_temp = pq_diff
            pq_diff = np.repeat(np.expand_dims(pq_diff_temp, axis=1), n_repeats, axis=1)
        if emb_diff.shape != dist.shape:
            dist_temp = dist
            dist = np.repeat(np.expand_dims(dist_temp, axis=1), n_repeats, axis=1)
        # print(dist)
        mult = np.multiply(pq_diff, emb_diff)
        div = np.divide(mult, dist)
        result[i] = 4*np.sum(div, axis=0)
    
    return result

def gradient_no_neighbors_4(high_dimension_probs, low_dimension_probs, embed, embed_distances):
    
    result = np.zeros_like(embed)
    
    
    #aux1 = p - q ---> aux1[i][j] = p_ij - q_ij
    aux1 = high_dimension_probs - low_dimension_probs
    
    

    # x = np.array([1, 2])
    # x: [1,
    #     2]
    # x.shape: (2,)
    #
    # y1 = np.expand_dims(x, axis=0)
    # y1: [[1, 2]]
    # y1.shape: (1, 2)
    #
    # y2 = np.expand_dims(x, axis=1)
    # y2: [[1],
    #      [2]])
    # y2.shape: (2, 1)

    # aux2 = np.expand_dims(y,1) - np.expand_dims(y,0)
    np.exp
    for i in range(0, embed.shape[0]):
        #aux2= y_i - y_j
        aux2 = embed[i] - embed
        
        #aux3= 1/(1 + dist(y_i, y_j))
        aux3 = 1/(1 + embed_distances[i])

        #prod = aux1*aux2*aux3
        prod = np.multiply(aux1[i], aux2, aux3)
        sum = np.sum(prod, 1)
        result[i] = 4*sum

    return result


def kl_divergence(high_dimension_p, low_dimension_q):
    """Computes the Kullback-Leibler divergence
    Parameters
    ----------
        high_dimension_p: ndarray of shape (n_samples, n_samples)
            The joint probabilities for the samples in the original dimension.

        low_dimension_p: ndarray of shape (n_samples, n_samples)
            The joint probabilities for the samples embedded in the lower dimension.

    Returns
    -------
        divergence : double.
            The divergence.
    """
    
    # div = np.divide(high_dimension_p,low_dimension_q)
    # log = np.log(div)
    # result = np.multiply(high_dimension_p, log)
    # return np.sum(result)
    result = 0.0
    
    for i in range(0, high_dimension_p.shape[0]):
        for j in range(0, high_dimension_p.shape[1]):
            result += high_dimension_p[i][j]*np.log(high_dimension_p[i][j]/low_dimension_q[i][j])
    return result

def trustworthiness(data, embed, n_neighbors):
    dist_original = np.sqrt(similarities.pairwise_euclidean_distance(data))
    dist_embed = np.sqrt(similarities.pairwise_euclidean_distance(embed))

    v_original = similarities.get_neighbors_ranked_by_distance(dist_original)
    v_embed_index = similarities.find_nearest_neighbors_index(dist_embed, n_neighbors).astype(np.int32)
    
    n = data.shape[0]
    k = n_neighbors

    penalizacion = v_original - k


    sumatorio_doble = 0.0

    for i in range(0, v_original.shape[0]):
        for j in v_embed_index[i]:
            sumatorio_doble += max(0,penalizacion[i][j])
    

    div = n*k*(2*n-3*k-1)
    result = 1-(2*sumatorio_doble/div)
    
    return result

def trustworthiness_2(data, embed, n_neighbors):
    dist_original = np.sqrt(similarities.pairwise_euclidean_distance(data))
    dist_embed = np.sqrt(similarities.pairwise_euclidean_distance(embed))

    v_original = similarities.get_neighbors_ranked_by_distance(dist_original)
    v_embed_index = similarities.find_nearest_neighbors_index(dist_embed, n_neighbors).astype(np.int32)
    
    n = data.shape[0]
    k = n_neighbors

    penalizacion = v_original - k

    sumatorio_doble = 0.0

    for i in range(0, v_original.shape[0]):
        for j in v_embed_index[i]:
            sumatorio_doble += max(0,penalizacion[i][j])
    

    div = n*k*(2*n-3*k-1)
    result = 1-(2*sumatorio_doble/div)
    
    return result



class TSne():
    """Class for the performance of the T-Sne method.
    TODO escribir un tooltip en condiciones xd
    """

    def __init__(self, *, n_dimensions=2, perplexity=30, perplexity_tolerance=1e-10, n_neighbors = 10,
                 metric='euclidean', init_method="random", init_embed=None, early_exaggeration=4,
                 learning_rate=500, max_iter=1000, momentum_params=[250.0,0.5,0.8], seed=None, verbose=0):
        #validacion de parametros
        self._input_validation(n_dimensions, perplexity, perplexity_tolerance, n_neighbors, metric, init_method, init_embed,
                                early_exaggeration, learning_rate, max_iter, momentum_params, seed, verbose)

        if n_neighbors==None:
            n_neighbors = 3*perplexity + 1

        #inicializacion de la clase
        self.n_dimensions = n_dimensions
        self.perplexity = perplexity
        self.perplexity_tolerance = perplexity_tolerance
        self.metric = metric
        self.init_method = init_method
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.init_embed = init_embed
        self.momentum_params = momentum_params
        self.early_exaggeration = early_exaggeration
        self.n_neighbors = n_neighbors
        self.verbose = verbose

        #set parameters to use later
        self.element_classes = None
        self.Y = []
        self.embed_dist_history = []
        self.affinities_history = []
        self.cost_history = []
        self.trust_history = []
        self.best_cost = None
        self.best_trust = None
        self.best_iter_cost = None
        self.best_iter_trust = None

        #set the seed
        if seed is None:
            seed = int(time.time())
        self.random_state = np.random.RandomState(seed)

    def _input_validation(self,n_dimensions,perplexity,perplexity_tolerance,n_neighbors,metric,init_method,init_embed,
                           early_exaggeration,learning_rate,max_iter,momentum_params, seed, verbose):
        accepted_methods = ["random", "precomputed"]
        accepted_metrics=["euclidean"]
        accepted_momentum_param_types = [np.float64,np.float32]
        invalid_numbers = [np.nan, np.inf]
        accepted_verboses = range(0,2)

        #n_dimensions: int
        if n_dimensions is not None:
            if not isinstance(n_dimensions, int):
                raise ValueError("n_dimensions must be of int type")
            elif n_dimensions in invalid_numbers:
                raise ValueError("n_dimensions must be finite and not NaN")
            elif n_dimensions<1:
                raise ValueError("n_dimensions must be a positive number")
            elif n_dimensions>3:
                print("**Warning: If you use more than 3 dimensions, you will not be able to display the embedding**")

        # perplexity: int
        if perplexity is not None:
            if not isinstance(perplexity, int):
                raise ValueError("perplexity must be of int type")
            elif perplexity in invalid_numbers:
                raise ValueError("perplexity must be finite and not NaN")
            elif perplexity <1:
                raise ValueError("perplexity must be a positive number")

        # perplexity_tolerance: float
        if perplexity_tolerance is not None:
            if not isinstance(perplexity_tolerance, float):
                raise ValueError("perplexity_tolerance must be of float type")
            elif perplexity_tolerance in invalid_numbers:
                raise ValueError("perplexity_tolerance must be finite and not NaN")
            elif perplexity_tolerance < 0:
                raise ValueError("perplexity_tolerance must be a positive number or 0")

        # n_neighbors: int
        if n_neighbors is not None:
            if not isinstance(n_neighbors, int):
                raise ValueError("n_neighbors must be of int type")
            elif n_neighbors in invalid_numbers:
                raise ValueError("n_neighbors must be finite and not NaN")
            elif n_neighbors <0:
                raise ValueError("n_neighbors must be at least 0")
        
        # metric: str
        if metric is not None:
            if not isinstance(metric, str):
                raise ValueError("metric must be of str type")
            elif metric not in accepted_metrics:
                raise ValueError("Only currently accepted metric is euclidean")
        
        # init_method: str
        if init_method is not None:
            if not isinstance(init_method, str):
                raise ValueError("init_method must be of str type")
            else: 
                if init_method not in accepted_methods:
                    raise ValueError("Only init_method values accepted are random and precomputed")
        
        # init_embed*: ndarray of shape (n_samples, n_features)
        if init_embed is not None:
            if isinstance(init_embed, np.ndarray):
                if not isinstance(init_embed.ndtype, np.number):
                    raise ValueError("Data type of the initial embedding must be a number")
                elif np.where(init_embed in invalid_numbers, init_embed).count()>0:
                    raise ValueError("init_embed cant have NaN or an infinite number")
            else:
                raise ValueError("init_embed must be a ndarray")
        
        # early_exaggeration: int
        if early_exaggeration is not None:
            if not isinstance(early_exaggeration, int):
                raise ValueError("early_exaggeration must be of int type")
            elif early_exaggeration in invalid_numbers:
                raise ValueError("early_exaggeration must be finite and not NaN")
            elif early_exaggeration <1:
                raise ValueError("early_exaggeration must be a positive number")
        
        # learning_rate: int
        if learning_rate is not None:
            if not isinstance(learning_rate, int):
                raise ValueError("learning_rate must be of int type")
            elif learning_rate in invalid_numbers:
                raise ValueError("learning_rate must be finite and not NaN")
            elif learning_rate <1:
                raise ValueError("learning_rate must be a positive number")
        
        # max_iter: int
        if max_iter is not None:
            if not isinstance(max_iter, int):
                raise ValueError("max_iter must be of int type")
            elif max_iter in invalid_numbers:
                raise ValueError("max_iter must be finite and not NaN")
            elif max_iter <1:
                raise ValueError("max_iter must be a positive number")
        
        # momentum_params: ndarray of shape (3,)
        if momentum_params is not None:
            if not isinstance(momentum_params, np.ndarray):
                if np.asarray(momentum_params).shape!=(3,):
                    raise ValueError("momentum_params must be a ndarray of shape (3,)")
            elif np.where(momentum_params in invalid_numbers, momentum_params).count()>0:
                        raise ValueError("init_embed cant have NaN or an infinite number")
            elif momentum_params.dtype not in accepted_momentum_param_types:
                raise ValueError("The elements of momentum_params must be float(at least float32)")
            elif np.min(momentum_params)<=0:
                raise ValueError("All elements must be positive numbers")
            elif not (momentum_params[0]).is_integer():
                raise ValueError("The time threshold cant be a decimal number")

        # seed: int
        if seed is not None:
            if not isinstance(seed, int):
                raise ValueError("seed must be an integer")
            elif seed<0:
                raise ValueError("seed must be a positive integer")
        
        # verbose: int
        if verbose is not None:
            if not isinstance(verbose, int):
                raise ValueError("verbose must be an integer")
            elif verbose not in accepted_verboses:
                raise ValueError("verbose must be within the range [0,2)")

    def _array_validation(self, input):
        
        if not hasattr(input, "__len__"):
            raise ValueError("The given input is not array-like")
        
        if not isinstance(input, np.ndarray):
            result = np.array(input)
        else:
            result = input

        if result.shape[0]<10:
            raise ValueError("Not enough samples. Must be at least 10 samples.")
        
        if result.shape[0] <= self.n_neighbors:
            raise ValueError("The number of samples cannot be lower than the number of neighbors")

        return result

    def momentum(self,t):
        if t>self.momentum_params[0]:
            self.early_exaggeration=1
            result = self.momentum_params[2]
        else:
            result = self.momentum_params[1]
        return result

    def initial_embed(self, *, data):
        """Returns an initial embedding following the given parameters.

        Parameters
        ----------
        data: ndarray of shape (n_samples, n_features). Optional.
            The data to fit in the embedding.
        zeros: boolean. default=False.
            If True, sets all values in the embedding to be 0.
        entries: int. Optional.
            The number of samples to have in the embedding.
            Only taken into account if the initiation method is set to random.
        
        Returns
        ----------
        embed: ndarray of shape (n_samples, n_features).
            The calculated embedding.

        """
        assert data is not None

        result = self.random_state.standard_normal(size=(data.shape[0], self.n_dimensions))
        return result

    def fit(self, input, classes:np.ndarray=None):
        """Fit the given data and perform the embedding

        Parameters
        ----------
        X: array of shape (n_samples, n_features).
            The data to fit.
        classes: 1-D array of size (n_samples). Optional.
            Array with the class that each element in X belongs to.
        """

        X = self._array_validation(input)

        if self.verbose>0:
            t0 = time.time()
        
        self.learning_rate = max(self.learning_rate, np.floor([X.shape[0]/12])[0])


        #self.gradient_descent(self.max_iter, X)
        #self.gradient_descent_no_neighbors_2(self.max_iter, X)
        self.gradient_descent_no_neighbors_only_gradient(self.max_iter, X)

        if classes is not None:
            self.element_classes = classes


        if self.verbose>0:
            t1 = time.time()
            tdiff = t1-t0
            t_iter = tdiff/self.max_iter
            print("=================================================================")
            print("Embedding process finished")
            print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(tdiff)))
            print('Time/Iteration:', time.strftime("%H:%M:%S", time.gmtime(t_iter)))
            print("=================================================================")

    def gradient_descent(self, t, data):
        """Performs the gradient descent in an iterative manner

        Parameters
        ----------
        t: int
            The amount of iterations to perform.

        data : ndarray of shape (n_samples, n_features)
            An array of the data where each row is a sample and each column is a feature.
        
        Returns
        -------
        embed : ndarray of shape (n_samples, n_dimensions) or (t, n_samples, n_dimensions)
            Contains the resulting embedding after t iterations.
            If return_evolution is True, returns the entire history of embeddings.
        
        Note: Updates the value of the embedding_history parameter.
        """
        distances_original = similarities.euclidean_distance_neighbors(data,n_neighbors=self.n_neighbors)
        affinities_original = similarities.joint_probabilities(distances_original, self.perplexity, self.perplexity_tolerance)


        if self.init_embed is None:
            y = self.initial_embed(data=data)
        else:
            y = self.init_embed

        self.Y.append(y); self.Y.append(y)

        dist_embed = similarities.euclidean_distance_neighbors(y, n_neighbors=self.n_neighbors)
        self.embed_dist_history.append(dist_embed); self.embed_dist_history.append(dist_embed)

        affinities_embed = similarities.joint_probabilities(dist_embed,self.perplexity,self.perplexity_tolerance, distribution='t-student')
        self.affinities_history.append(affinities_embed); self.affinities_history.append(affinities_embed)

        cost = kl_divergence(affinities_original, affinities_embed)
        self.cost_history.append(cost); self.cost_history.append(cost)
        self.best_cost = cost
        self.best_iter_cost = 1
        
        trust = trustworthiness(data, y, self.n_neighbors)
        self.trust_history.append(trust); self.trust_history.append(trust)
        self.best_trust = trust
        self.best_iter_trust = 1

        for i in range(2,t):
            #grad = gradient_2(affinities_original,self.early_exaggeration*self.affinities_history[-1],self.Y[-1],n_neighbors=self.n_neighbors,embed_distances=self.embed_dist_history[-1])
            #y = self.Y[-1] - self.learning_rate*grad + self.momentum(i)*(self.Y[-1]-self.Y[-2])
            
            #grad2 = gradient_2(affinities_original,self.early_exaggeration*self.affinities_history[-1],self.Y[-1])
            #y = self.Y[-1] - self.learning_rate*grad2 + self.momentum(i)*(self.Y[-1]-self.Y[-2])

            grad_no_neighbors = gradient_no_neighbors_2(affinities_original,self.early_exaggeration*self.affinities_history[-1],self.Y[-1])
            y = self.Y[-1] - self.learning_rate*grad_no_neighbors + self.momentum(i)*(self.Y[-1]-self.Y[-2])
            
            self.Y.append(y)

            distances_embed = similarities.euclidean_distance_neighbors(self.Y[-1], n_neighbors=self.n_neighbors)
            self.embed_dist_history.append(distances_embed)
            
            affinities_current = similarities.joint_probabilities(distances_embed,self.perplexity,self.perplexity_tolerance, distribution='t-student')
            self.affinities_history.append(affinities_current)

            cost = kl_divergence(affinities_original, affinities_current)
            self.cost_history.append(cost)
            
            trust = trustworthiness(data, y, self.n_neighbors)
            self.trust_history.append(trust)
            
            if cost<self.best_cost:
                self.best_cost = cost
                self.best_iter_cost = i
            if trust>self.best_trust:
                self.best_trust = trust
                self.best_iter_trust = i

    def gradient_descent_no_neighbors(self, t, data):
        distances_original = similarities.euclidean_distance(data)
        affinities_original = similarities.joint_probabilities(distances_original, self.perplexity, self.perplexity_tolerance)

        if self.init_embed is None:
            y = self.initial_embed(data=data)
        else:
            y = self.init_embed

        self.Y.append(y); self.Y.append(y)

        dist_embed = similarities.euclidean_distance(y)
        self.embed_dist_history.append(dist_embed); self.embed_dist_history.append(dist_embed)

        affinities_embed = similarities.joint_probabilities(dist_embed,self.perplexity,self.perplexity_tolerance, distribution='t-student')
        self.affinities_history.append(affinities_embed); self.affinities_history.append(affinities_embed)

        cost = kl_divergence(affinities_original, affinities_embed)
        self.cost_history.append(cost); self.cost_history.append(cost)
        self.best_cost = cost
        self.best_iter_cost = 1
        
        trust = trustworthiness(data, y, self.n_neighbors)
        self.trust_history.append(trust); self.trust_history.append(trust)
        self.best_trust = trust
        self.best_iter_trust = 1

        for i in range(2,t):

            grad = gradient_no_neighbors_2(affinities_original,self.early_exaggeration*self.affinities_history[-1],self.Y[-1])
            y = self.Y[-1] - self.learning_rate*grad + self.momentum(i)*(self.Y[-1]-self.Y[-2])
            
            self.Y.append(y)

            distances_embed = similarities.euclidean_distance(self.Y[-1])
            self.embed_dist_history.append(distances_embed)
            
            affinities_current = similarities.joint_probabilities(distances_embed,self.perplexity,self.perplexity_tolerance, distribution='t-student')
            self.affinities_history.append(affinities_current)

            cost = kl_divergence(affinities_original, affinities_current)
            self.cost_history.append(cost)
            
            trust = trustworthiness(data, y, self.n_neighbors)
            self.trust_history.append(trust)
            
            if cost<self.best_cost:
                self.best_cost = cost
                self.best_iter_cost = i
            if trust>self.best_trust:
                self.best_trust = trust
                self.best_iter_trust = i

    def gradient_descent_no_neighbors_2(self, t, data):
        distances_original = similarities.pairwise_euclidean_distance(data)
        affinities_original = similarities.joint_probabilities_2(distances_original, self.perplexity, self.perplexity_tolerance)

        if self.init_embed is None:
            y = self.initial_embed(data=data)
        else:
            y = self.init_embed

        self.Y.append(y); self.Y.append(y)

        dist_embed = similarities.pairwise_euclidean_distance(y)
        self.embed_dist_history.append(dist_embed); self.embed_dist_history.append(dist_embed)

        affinities_embed = similarities.joint_probabilities_2(dist_embed,self.perplexity,self.perplexity_tolerance, distribution='t-student')
        self.affinities_history.append(affinities_embed); self.affinities_history.append(affinities_embed)

        cost = kl_divergence(affinities_original, affinities_embed)
        self.cost_history.append(cost); self.cost_history.append(cost)
        self.best_cost = cost
        self.best_iter_cost = 1
        
        trust = trustworthiness_2(data, y, self.n_neighbors)
        self.trust_history.append(trust); self.trust_history.append(trust)
        self.best_trust = trust
        self.best_iter_trust = 1

        for i in range(2,t):

            grad = gradient_no_neighbors_2(affinities_original,self.early_exaggeration*self.affinities_history[-1],self.Y[-1])
            y = self.Y[-1] - self.learning_rate*grad + self.momentum(i)*(self.Y[-1]-self.Y[-2])
            
            self.Y.append(y)

            distances_embed = similarities.pairwise_euclidean_distance(self.Y[-1])
            self.embed_dist_history.append(distances_embed)
            
            affinities_current = similarities.joint_probabilities_2(distances_embed,self.perplexity,self.perplexity_tolerance, distribution='t-student')
            self.affinities_history.append(affinities_current)

            cost = kl_divergence(affinities_original, affinities_current)
            self.cost_history.append(cost)
            
            trust = trustworthiness(data, y, self.n_neighbors)
            self.trust_history.append(trust)
            
            if cost<self.best_cost:
                self.best_cost = cost
                self.best_iter_cost = i
            if trust>self.best_trust:
                self.best_trust = trust
                self.best_iter_trust = i

    def gradient_descent_no_neighbors_only_gradient(self, t, data):
        distances_original = similarities.pairwise_euclidean_distance(data)
        affinities_original = similarities.joint_probabilities(distances_original, self.perplexity, self.perplexity_tolerance)

        if self.init_embed is None:
            y = self.initial_embed(data=data)
        else:
            y = self.init_embed

        
        dist_embed = similarities.pairwise_euclidean_distance(y)
        affinities_embed = similarities.joint_probabilities(dist_embed,self.perplexity,self.perplexity_tolerance, distribution='t-student')
        
        
        self.Y.append(y); self.Y.append(y)
        self.embed_dist_history.append(dist_embed); self.embed_dist_history.append(dist_embed)
        self.affinities_history.append(affinities_embed); self.affinities_history.append(affinities_embed)
        with warnings.catch_warnings(record=True) as war:
            warnings.simplefilter("always")
            for i in range(2,t):
                #grad = gradient_no_neighbors_3(affinities_original,self.early_exaggeration*self.affinities_history[-1],self.Y[-1])
                grad = gradient_no_neighbors_4(affinities_original,self.early_exaggeration*self.affinities_history[-1],self.Y[-1],self.embed_dist_history[-1])
                
                y = self.Y[-1] - self.learning_rate*grad + self.momentum(i)*(self.Y[-1]-self.Y[-2])
                distances_embed = similarities.pairwise_euclidean_distance(y) #alcanza overflow salta a except, y no se ejecutan
                affinities_current = similarities.joint_probabilities(distances_embed,self.perplexity,self.perplexity_tolerance, distribution='t-student')
                
                self.Y.append(y)
                self.embed_dist_history.append(distances_embed)
                self.affinities_history.append(affinities_current)

                if len(war)>0:
                    print("===============================")
                    print("len(war): {}".format(len(war)))
                    print("-------------------------------")
                    for j in range(0, len(war)):
                        if j<10:
                            print("war[{}].filename: {}".format(j, war[j].filename))
                            print("war[{}].line_number: {}".format(j, war[j].lineno))
                            print("war[{}].message: {}".format(j, war[j].message))
                            print("-------------------------------")
                    print("-------------------------------")
                    print("i: {}".format(i))
                    print("len(self.Y): {}".format(len(self.Y)))
                    print("max(y): {}".format(np.max(y)))
                    print("min(y): {}".format(np.min(y)))
                    print("===============================")
                    exit(0)


    def display_embed(self, *, display_best_iter_cost=False, display_best_iter_trust=False, t:int=-1):
        """Displays the resulting embedding.

        Parameters
        ----------
        display_best_iter: bool, Optional.
            Whether or not to display the iteration with the lowest cost.
            If True, the "t" parameter is ignored
        t: int, Optional.
            The embedding iteration to display.
        """
        if display_best_iter_cost:
            t = self.best_iter_cost
        elif display_best_iter_trust:
            t = self.best_iter_trust
        elif t not in range(-1,self.max_iter):
            raise ValueError("Cannot show embedding for values of t that are not within the range [-1, {})=".format(self.max_iter))
        
        embed = np.array(self.Y[t])
        embed_T = embed.T

        if self.element_classes is not None:
            labels = self.element_classes.astype(str)

        if self.n_dimensions>3:
            raise ValueError("Display of embedding not available for more than 3 dimensions. I am limited by the technology of my time")
        else:
            if self.n_dimensions==3:
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                x = embed_T[0]
                y = embed_T[1]
                z = embed_T[2]
                if self.element_classes is not None:
                    for i in range(0,x.shape[0]):
                        ax.plot(x[i],y[i],z[i], label=labels[i], marker='o',linestyle='', markersize=5)
                    
                    handles, labels = ax.get_legend_handles_labels()
                    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
                    ax.legend(*zip(*unique), draggable=True)
                else:
                    for i in range(0,x.shape[0]):
                        ax.plot(x[i],y[i],z[i], marker='o',linestyle='', markersize=8)
            
            else:
                if self.n_dimensions==1:
                    x = embed
                    y = np.ones_like(x)
                else:
                    x = embed_T[0]
                    y = embed_T[1]

                if self.element_classes is not None:
                    for i in range(0,x.shape[0]):
                        plt.plot(x[i],y[i],marker='o',linestyle='', markersize=5, label=labels[i])
                    handles, labels = plt.gca().get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    plt.legend(by_label.values(), by_label.keys(), draggable=True)
                else:
                    for i in range(0,x.shape[0]):
                        plt.plot(x[i],y[i],marker='o',linestyle='', markersize=8)
            


            if display_best_iter_cost:
                title = "Best embedding for kl divergence, achieved at t={} out of {} iterations".format(t, self.max_iter)
            elif display_best_iter_trust:
                title = "Best embedding for trustworthiness, achieved at t={} out of {} iterations".format(t, self.max_iter)
            elif t==-1:
                title = "Embedding at the last iteration, at t={}".format(self.max_iter)
            else:
                title = "Embedding at t={} out of {} iterations".format(t, self.max_iter)

            plt.title(title)
            plt.show()


#============================================================================================
#============================================================================================
#===============De la pagina, borrar despues=================================================
#============================================================================================
#============================================================================================



def pairwise_distances(X): #devuelve la distancia euclidiana sin hacer la raiz cuadrada
    return np.sum((X[None, :] - X[:, None])**2, 2)

def p_conditional(dists, sigmas):
    e = np.exp(-dists / (2 * np.square(sigmas.reshape((-1,1)))))
    np.fill_diagonal(e, 0.)
    e += 1e-8
    return e / e.sum(axis=1).reshape([-1,1])

def perp(condi_matr):
    ent = -np.sum(condi_matr * np.log2(condi_matr), 1)
    return 2 ** ent

def find_sigmas(dists, perplexity):
    found_sigmas = np.zeros(dists.shape[0])
    for i in range(dists.shape[0]):
        func = lambda sig: perp(p_conditional(dists[i:i+1, :], np.array([sig])))
        found_sigmas[i] = search(func, perplexity)
    return found_sigmas


def search(func, goal, tol=1e-10, max_iters=1000, lowb=1e-20, uppb=10000):
    for _ in range(max_iters):
        guess = (uppb + lowb) / 2.
        val = func(guess)

        if val > goal:
            uppb = guess
        else:
            lowb = guess

        if np.abs(val - goal) <= tol:
            return guess

    return guess


def q_joint(y): #joint probabilities with t-student distribution with 1 degree of freedom
    dists = pairwise_distances(y)
    nom = 1 / (1 + dists)
    np.fill_diagonal(nom, 0.)
    return nom / np.sum(np.sum(nom))

def gradient_2(P, Q, y):
    (n, no_dims) = y.shape
    pq_diff = P - Q
    y_diff = np.expand_dims(y,1) - np.expand_dims(y,0)

    dists = pairwise_distances(y)
    aux = 1 / (1 + dists)
    return 4 * (np.expand_dims(pq_diff, 2) * y_diff * np.expand_dims(aux,2)).sum(1)

def m(t):
    return 0.5 if t < 250 else 0.8

def p_joint(X, perp):
    N = X.shape[0]
    dists = pairwise_distances(X)
    sigmas = find_sigmas(dists, perp)
    p_cond = p_conditional(dists, sigmas)
    return (p_cond + p_cond.T) / (2. * N)

def tsne_2(X, ydim=2, T=1000, l=500, perp=30):
    N = X.shape[0]
    P = p_joint(X, perp)

    Y = []
    y = np.random.normal(loc=0.0, scale=1e-4, size=(N,ydim))
    Y.append(y); Y.append(y)

    for t in range(T):
        Q = q_joint(Y[-1])
        grad = gradient_2(P, Q, Y[-1])
        y = Y[-1] - l*grad + m(t)*(Y[-1] - Y[-2])
        Y.append(y)
        if t % 10 == 0:
            Q = np.maximum(Q, 1e-12)
    return y
