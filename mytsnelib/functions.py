#import sklearn.manifold as manif
#import sklearn.manifold._utils as ut

import numpy as np
import matplotlib.pyplot as plt
from mytsnelib import similarities
import time

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
        embed_distances = similarities.euclidean_distance_neighbors(embed, return_squared=False,n_neighbors=n_neighbors)
    

    np.fill_diagonal(embed_distances,0.)
    embed_distances += np.ones_like(embed_distances)


    

    n = high_dimension_probs.shape[0]
    gradient = np.zeros_like(embed)
    
    # print(embed_distances)
    
    for i in range(0,n):
        #y_i - y_j
        embed_diff = embed[i] - embed
        #p_ij - q_ij
        prob_diff = high_dimension_probs[i]-low_dimension_probs[i]
        # 1 + (||y_i - y_j||)^2
        dist = embed_distances[i]
        
        """
        print("================================")
        print("previo a ajuste de dimensiones:")
        print("embed_diff.shape: {}".format(embed_diff.shape))
        print("prob_diff.shape: {}".format(prob_diff.shape))
        print("dist.shape: {}".format(dist.shape))
        print("================================")
        """

        
        n_repeats = embed_diff.shape[1]
        if embed_diff.shape != prob_diff.shape:
            prob_diff_temp = prob_diff
            prob_diff = np.repeat(np.expand_dims(prob_diff_temp, axis=1), n_repeats, axis=1)
        
        #set the dimensions of dist to match embed_diff's
        if embed_diff.shape != dist.shape:
            dist_temp = dist
            dist = np.repeat(np.expand_dims(dist_temp, axis=1), n_repeats, axis=1)
        
        
        # print(dist)
        
        gradient[i] = 4*np.sum(prob_diff*embed_diff/dist, axis=0)

        # prod = np.multiply(prob_diff,embed_diff)
        # division = np.divide(prod,dist)
        # gradient[i] = 4*np.sum(division, axis=0)
    
    return gradient

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
    
    result = high_dimension_p * np.log(high_dimension_p/low_dimension_q)

    return np.sum(result)

def trustworthiness(data, embed, n_neighbors):
    dist_original = similarities.euclidean_distance(data, return_squared=True)
    dist_embed = similarities.euclidean_distance(embed, return_squared=True)

    v_in = similarities.get_neighbors_ranked_by_distance(dist_original)
    v_out = similarities.find_nearest_neighbors_index(dist_embed, n_neighbors)
    
    n = data.shape[0]
    k = n_neighbors

    # k_array = k * np.ones_like(v_in)

    penalizacion_temp = v_in - k
    penalizacion = np.where (penalizacion_temp<0, penalizacion_temp, 0)

    for i in range(0, v_in.shape[0]):
        for j in range(0, v_in.shape[0]):
            if j not in v_out[i]:
                penalizacion[i][j]=0

    sumatorio_doble = np.sum(penalizacion)
    
    div = n*k*(2*n-3*k-1)
    result = 1-(2*sumatorio_doble/div)
    
    return result



class TSne():
    """Class for the performance of the T-Sne method.
    TODO escribir un tooltip en condiciones xd
    """

    def __init__(self, *, n_dimensions=2, perplexity=30, perplexity_tolerance=0.1, n_neighbors = 10,
                 metric='euclidean', init_method="random", init_embed=None, early_exaggeration=4,seed=None,
                 learning_rate=200, max_iter=1000, momentum_params=[250.0,0.5,0.8], descent_mode="iterative"):
        #validacion de parametros
        self.__input_validation(n_dimensions, perplexity, perplexity_tolerance, n_neighbors, metric, init_method, init_embed,
                                early_exaggeration, learning_rate, max_iter, momentum_params, descent_mode, seed)


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
        self.embed = init_embed
        self.momentum_params = momentum_params
        self.early_exaggeration = early_exaggeration
        self.embedding_current_t = 0
        self.descent_mode = descent_mode
        self.n_neighbors = n_neighbors


        #set parameters to use later
        self.element_classes = None
        self.embedding_history = None
        self.best_cost = np.finfo(float).max
        self.best_iter = max_iter
        self.cost_history = None

        #set the seed
        self.random_state = np.random.RandomState(seed) if seed is not None else None

    def __input_validation(self,n_dimensions,perplexity,perplexity_tolerance,n_neighbors,metric,init_method,init_embed,
                           early_exaggeration,learning_rate,max_iter,momentum_params,descent_mode, seed):
        
        accepted_methods = ["random", "precomputed"]
        accepted_metrics=["euclidean"]
        accepted_modes = ["recursive", "iterative"]
        accepted_momentum_param_types = [np.float64,np.float32]
        invalid_numbers = [np.nan, np.inf]

        #n_dimensions: int
        if n_dimensions is not None:
            if not isinstance(n_dimensions, int):
                raise ValueError("n_dimensions must be of int type")
            elif n_dimensions in invalid_numbers:
                raise ValueError("n_dimensions must be finite and not NaN")
            elif n_dimensions<1:
                raise ValueError("n_dimensions must be a positive number")

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
        
        # descent_mode: str
        if descent_mode is not None:
            if not isinstance(descent_mode, str):
                raise ValueError("descent_mode must be of str type")
            elif descent_mode not in accepted_modes:
                raise ValueError("Only possible descent modes are recursive and iterative")

    def __momentum(self,t):
        t_limit = self.momentum_params[0]
        m1 = self.momentum_params[1]
        m2 = self.momentum_params[2]
        
        if t>t_limit:
            self.early_exaggeration=1
            return m2
        else:
            return m1

    def initial_embed(self, *, data=None, zeros=False):
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
        if data is None:
            if zeros:
                raise ValueError("Cannot generate any initial embed without generation parameters")
        else:
            target_shape = (data.shape[0], self.n_dimensions)
            if zeros:
                embed = np.zeros(shape=target_shape)
            elif self.random_state is not None:
                embed = self.random_state.standard_normal(size=target_shape)
            else:
                embed = np.random.standard_normal(size=target_shape)
            
            if not zeros and self.embed is None and self.init_method!="precomputed":
                self.embed = embed
            
            return embed

    def fit(self, X, classes:np.ndarray=None):
        """Fit the given data and perform the embedding

        Parameters
        ----------
        X: array of shape (n_samples, n_features).
            The data to fit.
        classes: 1-D array of size (n_samples). Optional.
            Array with the class that each element in X belongs to.
        """
    
    
        t0 = time.time()

        self.learning_rate = max(self.learning_rate, np.floor([X.shape[0]/12])[0])

        if X.shape[0]<10:
            raise ValueError("Not enough samples. Must be at least 10 samples.")

        if X.shape[0]-1 < self.n_neighbors:
            self.n_neighbors = X.shape[0]-1

        if self.descent_mode=="recursive":
            self.gradient_descent_recursive(self.max_iter, X)
        elif self.descent_mode=="iterative":
            self.gradient_descent_iterative(self.max_iter, X, self.__momentum)
        
        if classes is not None:
            self.element_classes = classes
        
        t1 = time.time()
        
        tdiff = t1-t0
        t_iter = tdiff/self.max_iter
        print("Embedding process finished")
        print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(tdiff)))
        print('Time/Iteration:', time.strftime("%H:%M:%S", time.gmtime(t_iter)))

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

        Note: Does not update the value of the embedding_history parameter.
        """

        distances_original = similarities.euclidean_distance_neighbors(data, n_neighbors=self.n_neighbors)
        affinities_original = similarities.joint_probabilities(distances_original, self.perplexity, self.perplexity_tolerance)
        
        
        return self.__gradient_descent_recursive(t,affinities_original)

    def __gradient_descent_recursive(self,t,affinities_original):
        if t<0:
            raise ValueError("t must be a positive Integer")

        if t == 0:
            new_embed = self.initial_embed(data=affinities_original, zeros=True)
        elif t == 1:
            if self.embed is not None:
                new_embed = self.embed
            else:
                new_embed = self.initial_embed(data=affinities_original)
        else:
            if self.embed==None:
                raise ValueError("Embed can only be None for t<2")
            
            if self.embedding_current_t==t-1:
                embed_t1 = self.embed
            else:
                embed_t1 = self.__gradient_descent_recursive(t-1,affinities_original)
            embed_t2 = self.__gradient_descent_recursive(t-2,affinities_original)
            
            distances_embed = similarities.euclidean_distance(embed_t1)
            affinities_current = similarities.joint_probabilities(distances_embed,self.perplexity,self.perplexity_tolerance)
            grad = gradient(affinities_original,self.early_exaggeration*affinities_current,self.embed)

            new_embed = embed_t1 + self.learning_rate * grad + self.__momentum(t)*(embed_t1-embed_t2)
            self.embed = new_embed
        
        self.embedding_current_t = t
        return new_embed

    def gradient_descent_iterative(self, t, data, momentum_term, return_evolution=False):
        """Performs the gradient descent in an iterative manner

        Parameters
        ----------
        t: int
            The amount of iterations to perform.

        data : ndarray of shape (n_samples, n_features)
            An array of the data where each row is a sample and each column is a feature. 

        return_evolution: boolean. default = False.
            Wether or not to return the history of all embeddings.
        
        Returns
        -------
        embed : ndarray of shape (n_samples, n_dimensions) or (t, n_samples, n_dimensions)
            Contains the resulting embedding after t iterations.
            If return_evolution is True, returns the entire history of embeddings.
        
        Note: Updates the value of the embedding_history parameter.
        """
        
        #con los vecinos indicados
        distances_original = similarities.euclidean_distance_neighbors(data,n_neighbors=self.n_neighbors)

        affinities_original = similarities.joint_probabilities(distances_original, self.perplexity, self.perplexity_tolerance)

        n_points = data.shape[0]
        Y = np.zeros(shape=(t, n_points, self.n_dimensions))
        Y[0] = self.initial_embed(data=data, zeros=True)

        
        Y[1] = self.embed if self.embed is not None else self.initial_embed(data=data)
        
        self.cost_history = np.zeros(shape=t)
        self.cost_history[0] = self.best_cost

        for i in range(1,t-1):
            
            # print("i: {}".format(i))

            distances_embed = similarities.euclidean_distance_neighbors(Y[i], n_neighbors=self.n_neighbors, return_squared=False)
            affinities_current = similarities.joint_probabilities(distances_embed,self.perplexity,self.perplexity_tolerance, distribution='t-student')

            #   con early exaggeration
            grad = gradient(affinities_original,self.early_exaggeration*affinities_current,Y[i],n_neighbors=self.n_neighbors,embed_distances=distances_embed)
            #   sin early exaggeration
            # grad = gradient(affinities_original,affinities_current,Y[i],n_neighbors=self.n_neighbors,embed_distances=distances_embed)
            
            cost = kl_divergence(affinities_original, affinities_current)
            self.cost_history[i] = cost
            
            Y[i+1] = Y[i] - self.learning_rate*grad + momentum_term(i+1)*(Y[i]-Y[i-1])
        


        distances_embed = similarities.euclidean_distance_neighbors(Y[t-1], n_neighbors=self.n_neighbors)
        affinities_current = similarities.joint_probabilities(distances_embed,self.perplexity,self.perplexity_tolerance, distribution='t-student')
        
        cost = kl_divergence(affinities_original, affinities_current)
        
        self.cost_history[t-1] = cost
        
        maxim = np.max(self.cost_history)
        self.cost_history[0] = maxim

        self.best_cost = np.min(self.cost_history)
        self.best_iter = np.where(self.cost_history==self.best_cost)[0][0]

        self.embedding_current_t = t-1
        self.embedding_history = Y
        self.embed = Y[t-1]
        
        if return_evolution:
            return Y
        else:
            return Y[t-1]

    def display_embed(self, t:int=None):
        """Displays the resulting embedding.

        Parameters
        ----------
        t: int, Optional.
            The embedding iteration to display.
            If none is given, displays the last iteration.
        """
        if t is not None:
            if self.descent_mode!="iterative" and t!=-1:
                raise ValueError("Previous embeddings available only when descent mode is Iterative")
            elif t<0:
                t = self.embedding_current_t
            elif t>self.embedding_current_t:
                raise ValueError("Cannot show embedding for t>{}".format(self.embedding_current_t))
            else:
                t = self.best_iter
            embed = self.embedding_history[t]
        else:
            if self.descent_mode=="iterative":
                t = self.best_iter
                embed = self.embedding_history[t]
            else:
                t = self.embedding_current_t
                embed = self.embed

        if self.element_classes is not None:
            labels = self.element_classes.astype(str)

        if self.n_dimensions>3:
            raise ValueError("Display of embedding not available for more than 3 dimensions. I am limited by the technology of my time")
        else:
            if self.n_dimensions<3:
                if self.n_dimensions==1:
                    x = embed
                    y = np.ones_like(x)
                else:
                    embed_T = embed.T
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
            else:
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                embed_T = embed.T
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
            plt.title("Result for embedding at t={}".format(t+1))
            plt.show()