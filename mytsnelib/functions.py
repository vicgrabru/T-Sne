import numpy as np
import matplotlib.pyplot as plt
from mytsnelib import similarities
import time



def print_efficiency(t0_ns, t1_ns, *, n_iters=None, n_ms_digits=None):
    t_diff = (t1_ns-t0_ns)*1e-9
    t_diff_clean = time.gmtime(t_diff)
    t_diff_exact_ms = t_diff - t_diff_clean.tm_sec

    if n_ms_digits is None or n_ms_digits<=0:
        n = -3
    else:
        n = n_ms_digits
    
    print("=================================================================")
    print("Embedding process finished")
    print("Execution time (min:sec): {}.{}".format(time.strftime("%M:%S", t_diff_clean), str(t_diff_exact_ms)[2:2+n]))
    if n_iters is not None:
        t_iter = t_diff/n_iters
        t_iter_clean = time.gmtime(t_iter)
        t_iter_exact_ms = t_iter - t_iter_clean.tm_sec
        print("Time/Iteration (s): {}.{}".format(time.strftime("%S", t_iter_clean), str(t_iter_exact_ms)[2:2+n]))
    print("=================================================================")


#===Gradiente====================================================================================================
def gradient(P, Q, y, y_dist, caso="safe") -> np.ndarray:
    match caso:
        case "safe":
            return __gradient_safe(P,Q,y,y_dist)
        case "forces":
            return __gradient_forces(P,Q,y,y_dist)
        case "forces_v2":
            return __gradient_forces_v2(P,Q,y,y_dist)
        case _:
            raise ValueError("Only accepted cases are safe, forces, and forces_v2")
#----------------------------------------------------------------------------------------------------------------
def __gradient_safe(P, Q, y, y_dist) -> np.ndarray:
    pq_diff = P - Q
    y_diff = np.expand_dims(y,1) - np.expand_dims(y,0)

    aux = 1. / (1. + y_dist)
    return 4. * np.sum(np.expand_dims(pq_diff, 2) * y_diff * np.expand_dims(aux,2), axis=1)
def __gradient_forces(P, Q, y, y_dist) -> np.ndarray:
    y_diff = np.expand_dims(y,1) - np.expand_dims(y,0)

    # paso 1: obtener Z
    dists = 1./(1.+y_dist)
    np.fill_diagonal(dists, 0.)
    z = dists.sum()

    # paso 2: calcular fuerzas atractivas y repulsivas
    # fuerzas atractivas
    pq = np.multiply(P,Q)*z
    np.fill_diagonal(pq, 0.)
    attractive = np.multiply(np.expand_dims(pq, 2), y_diff)

    # fuerzas repulsivas
    q2 = np.power(Q, 2)*z
    np.fill_diagonal(q2, 0.)
    repulsive = np.multiply(np.expand_dims(q2, 2), y_diff)
    #np.fill_diagonal(repulsive, 0)

    # paso 3: combinacion
    return 4*(np.sum(attractive, 1) - np.sum(repulsive, 1))
def __gradient_forces_v2(P, Q, y, y_dist) -> np.ndarray:
    """
    Optimizaciones sacadas de esta pagina
    Optimizacion 1 -> https://opentsne.readthedocs.io/en/latest/tsne_algorithm.html#attractive-forces
    Optimizacion 2 -> https://opentsne.readthedocs.io/en/latest/tsne_algorithm.html#repulsive-forces
    """
    y_diff = np.expand_dims(y,1) - np.expand_dims(y,0)

    # paso 1: obtener Z
    dists = 1/(1+y_dist)
    np.fill_diagonal(dists, 0.)
    z = dists.sum()

    # paso 2: calcular fuerzas atractivas y repulsivas
    
    # fuerzas atractivas
    # Optimizacion 1: considerar solo los 3*Perplexity vecinos mas cercanos
    
    pq = np.multiply(P,Q)*z
    np.fill_diagonal(pq, 0.)
    attractive = np.multiply(np.expand_dims(pq, 2), y_diff)

    # fuerzas repulsivas
    # Optimizacion 2: TODO
    q2 = np.power(Q, 2)*z
    np.fill_diagonal(q2, 0.)
    repulsive = np.multiply(np.expand_dims(q2, 2), y_diff)

    # paso 3: combinacion
    return 4*(np.sum(attractive, 1) - np.sum(repulsive, 1))

#===Funcion de coste: Kullback-Leibler divergence================================================================
def kl_divergence(high_dimension_p, low_dimension_q) -> float:
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
    aux = np.log(high_dimension_p/np.maximum(low_dimension_q, 1e-10))
    result = np.multiply(high_dimension_p, aux)

    return np.sum(result)

class TSne():
    """Class for performing the T-Sne embedding.
    Parameters
    ----------
    n_dimensions : int, default=2
        The number of dimensions in the embedding space.
    perplexity : float, default=30.0
        TODO: explicarlo
    perplexity_tolerance: float, default=1e-10
        TODO: explicarlo
    n_neighbors : int, default=10
        TODO: explicarlo
    metric : str, default='euclidean'
        The metric for the distance calculations
    init_method : str, default="random"
        TODO: explicarlo
    init_embed : ndarray of shape (n_samples, n_components), default=None
        TODO: explicarlo
    early_exaggeration : float, default=None
        TODO: explicarlo
    learning_rate : float, default=500.
        TODO: explicarlo
    max_iter : int, default=1000
        TODO: explicarlo
    momentum_params : array-like of shape (3,), default=[250.,0.5,0.8]
        TODO: explicarlo
    seed : int, default=None
        TODO: explicarlo
    verbose : int, default=0
        Verbosity level (all levels include all info from previous levels):
            0 displays no info
            1 displays total execution time and time / iteration
            2 displays
        TODO: explicarlo
    iters_check : int, default=50
        TODO: explicarlo

    Attributes
    ----------
    embed_history

    
    """


    def __init__(self, *, n_dimensions=2, perplexity=30., perplexity_tolerance=1e-10, n_neighbors = 10,
                 metric='euclidean', init_method="random", init_embed=None, early_exaggeration:float=None,
                 learning_rate=500., max_iter=1000, momentum_params=[250.,0.5,0.8], seed:int=None, verbose=0, iters_check=50):
        
        #===validacion de parametros=================================================================================
        self.__input_validation(n_dimensions, perplexity, perplexity_tolerance, n_neighbors, metric, init_method, init_embed,early_exaggeration, learning_rate, max_iter, momentum_params, seed, verbose, iters_check)

        #===inicializacion de la clase===============================================================================
        self.n_dimensions = n_dimensions
        self.perplexity = perplexity
        self.perplexity_tolerance = perplexity_tolerance
        self.metric = metric
        self.init_method = init_method
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.init_embed = init_embed
        self.momentum_params = momentum_params
        self.early_exaggeration = 1. if early_exaggeration is None else early_exaggeration
        self.n_neighbors = 3*perplexity + 1 if n_neighbors is None else n_neighbors
        self.verbose = verbose
        self.iters_check = iters_check
        self.random_state = np.random.RandomState(int(time.time())) if seed is None else np.random.RandomState(seed)

        #===parametros auxiliares====================================================================================
        self.element_classes = None
        self.embed_history = []
        self.q_history = []
        self.cost_history = []
        self.cost_history_embed = []
        self.best_iter_cost = None
        self.fitting_done = False
        self.prev_embed_dist = None
        self.embed_dist_history = []
        
        #===parametros para las metricas de rendimiento==============================================================
        self.t_diff_distancias_og = None
        self.t_diff_p = None
        self.t_diff_dist_embed = []
        self.t_diff_q = []
        self.t_diff_grad = []
    def __input_validation(self,n_dimensions,perplexity,perplexity_tolerance,n_neighbors,metric,init_method,init_embed,
                           early_exaggeration,learning_rate,max_iter,momentum_params, seed, verbose, iters_check):
        accepted_methods = ["random", "precomputed"]
        accepted_metrics = ["euclidean"]
        accepted_momentum_param_types = [np.float64,np.float32]
        invalid_numbers = np.array([np.nan, np.inf])

        if n_dimensions is not None: # n_dimensions: int
            if not isinstance(n_dimensions, int):
                raise ValueError("n_dimensions must be of int type")
            elif n_dimensions in invalid_numbers:
                raise ValueError("n_dimensions must be finite and not NaN")
            elif n_dimensions<1:
                raise ValueError("n_dimensions must be a positive number")
            elif n_dimensions>3:
                print("**Warning: If you use more than 3 dimensions, you will not be able to display the embedding**")
        if perplexity is not None: # perplexity: float
            if not isinstance(perplexity, float):
                raise ValueError("perplexity must be of float type")
            elif perplexity in invalid_numbers:
                raise ValueError("perplexity cannot be infinite or NaN")
            elif perplexity <= 0.:
                raise ValueError("perplexity must be a positive number")
        if perplexity_tolerance is not None: # perplexity_tolerance: float
            if not isinstance(perplexity_tolerance, float):
                raise ValueError("perplexity_tolerance must be of float type")
            elif perplexity_tolerance in invalid_numbers:
                raise ValueError("perplexity_tolerance must be finite and not NaN")
            elif perplexity_tolerance < 0:
                raise ValueError("perplexity_tolerance must be a positive number or 0")
        if n_neighbors is not None: # n_neighbors: int
            if not isinstance(n_neighbors, int):
                raise ValueError("n_neighbors must be of int type")
            elif n_neighbors in invalid_numbers:
                raise ValueError("n_neighbors must be finite and not NaN")
            elif n_neighbors <0:
                raise ValueError("n_neighbors must be at least 0")
        if metric is not None: # metric: str
            if not isinstance(metric, str):
                raise ValueError("metric must be of str type")
            elif metric not in accepted_metrics:
                raise ValueError("Only currently accepted metric is euclidean")
        if init_method is not None: # init_method: str
            if not isinstance(init_method, str):
                raise ValueError("init_method must be of str type")
            else: 
                if init_method not in accepted_methods:
                    raise ValueError("Only init_method values accepted are random and precomputed")
        if init_embed is not None: # init_embed: ndarray of shape (n_samples, n_features)
            if isinstance(init_embed, np.ndarray):
                if not isinstance(init_embed.ndtype, np.number):
                    raise ValueError("Data type of the initial embedding must be a number")
                elif len(np.intersect1d(invalid_numbers, np.reshape(init_embed, [1,-1])))>0:
                    raise ValueError("init_embed cant have NaN or an infinite number")
            else:
                raise ValueError("init_embed must be a ndarray")
        if early_exaggeration is not None: # early_exaggeration: float
            if not isinstance(early_exaggeration, float):
                raise ValueError("early_exaggeration must be of float type")
            elif early_exaggeration in invalid_numbers:
                raise ValueError("early_exaggeration must be finite and not NaN")
            elif early_exaggeration <=0.:
                raise ValueError("early_exaggeration must be a positive number")
        if learning_rate is not None: # learning_rate: float
            if not isinstance(learning_rate, float):
                raise ValueError("learning_rate must be of float type")
            elif learning_rate in invalid_numbers:
                raise ValueError("learning_rate must be finite and not NaN")
            elif learning_rate <=0.:
                raise ValueError("learning_rate must be a positive number")
        if max_iter is not None: # max_iter: int
            if not isinstance(max_iter, int):
                raise ValueError("max_iter must be of int type")
            elif max_iter in invalid_numbers:
                raise ValueError("max_iter must be finite and not NaN")
            elif max_iter <1:
                raise ValueError("max_iter must be a positive number")
        if momentum_params is not None: # momentum_params: ndarray of shape (3,)
            if not isinstance(momentum_params, np.ndarray):
                if np.array(momentum_params).shape!=(3,):
                    raise ValueError("momentum_params must be a ndarray of shape (3,)")
            elif momentum_params.shape!=(3,):
                raise ValueError("momentum_params must be a ndarray of shape (3,)")
            elif momentum_params.dtype not in accepted_momentum_param_types:
                raise ValueError("The elements of momentum_params must be float(at least float32)")
            elif len(np.intersect1d(momentum_params, invalid_numbers))>0:
                raise ValueError("init_embed cant have NaN or an infinite number")
            elif np.min(momentum_params)<=0.:
                raise ValueError("All elements must be positive numbers")
            elif not (momentum_params[0]).is_integer():
                raise ValueError("The time threshold must be a whole number")
        if seed is not None: # seed: int
            if not isinstance(seed, int):
                raise ValueError("seed must be an integer")
            elif seed<0:
                raise ValueError("seed must be a positive integer")
        if verbose is not None: # verbose: int
            if not isinstance(verbose, int):
                raise ValueError("verbose must be an integer")
            elif verbose<0:
                raise ValueError("verbose must be a positive number")
        if iters_check is not None: #iters_check: int
            if not isinstance(iters_check, int):
                raise ValueError("iters_check must be an integer")
            elif iters_check<1:
                raise ValueError("iters_check must be at least 1")
            elif iters_check>max_iter:
                raise ValueError("iters_check cannot be greater than max_iter")
    def __fit_input_validation(self, input, *, embed:np.ndarray=None):
        
        if isinstance(input, np.ndarray):
            result = input
        elif isinstance(input, (tuple, list)):
            result = np.array(input)
        else:
            raise ValueError("The given input is not array-like")
        
        if len(result)<10:
            raise ValueError("Not enough samples. Must be at least 10 samples.")
        if embed is not None:
            if len(input) != len(embed):
                raise ValueError("The input data must have the same number of samples as the given embedding")
        if len(result) <= self.n_neighbors:
            raise ValueError("The number of samples cannot be lower than the number of neighbors")
        return result
    def __rand_embed(self, *, n_samples:int, n_dimensions:int) -> np.ndarray:
        """Returns a random embedding following the given parameters.

        Parameters
        ----------
        n_samples: int.
            The number of samples to have in the embedding.
        n_dimensions: int.
            The number of dimentsions in the embedding.
        
        Returns
        ----------
        embed: ndarray of shape (n_samples, n_features).
            Random embedding that follows a standard distribution.

        """
        assert n_samples is not None
        assert n_dimensions is not None

        result = self.random_state.standard_normal(size=(n_samples, n_dimensions))
        return result

    def fit(self, input, classes:np.ndarray=None, compute_cost=True, measure_efficiency=False) -> np.ndarray:
        """Fit the given data and perform the embedding

        Parameters
        ----------
        X: array of shape (n_samples, n_features).
            The data to fit.
        classes: 1-D array of size (n_samples). Optional.
            Array with the class that each element in X belongs to.
        """

        X = self.__fit_input_validation(input, embed=self.init_embed)

        if self.verbose>0:
            t0_g = time.time_ns()
        
        self.learning_rate = max(self.learning_rate, np.floor(len(X)/12.))

        #====distancias_og=======================================================================================================================================
        if measure_efficiency:  # mediciones de tiempo
            t_0 = time.time_ns()
        dist_original = similarities.pairwise_euclidean_distance(input)
        if measure_efficiency:  # mediciones de tiempo
            self.t_diff_distancias_og = (time.time_ns()-t_0)*1e-9
        
        #====p===================================================================================================================================================
        if measure_efficiency:  # mediciones de tiempo
            t_0 = time.time_ns()
        p = similarities.joint_probabilities_gaussian(dist_original, self.perplexity, self.perplexity_tolerance)
        if measure_efficiency:  # mediciones de tiempo
            self.t_diff_p = (time.time_ns()-t_0)*1e-9
        

        if self.init_embed is None:
            self.init_embed = self.__rand_embed(n_samples=len(input), n_dimensions=self.n_dimensions)

        self.__gradient_descent(self.max_iter, p, compute_cost, measure_efficiency)

        if self.verbose>0:
            print_efficiency(t0_g, time.time_ns(), n_iters=self.max_iter, n_ms_digits=6)

        if classes is not None:
            self.element_classes = classes

        
        result = np.array(self.embed_history[-1])
        self.fitting_done = True

        if compute_cost:
            aux = np.argmin(self.cost_history)
            if aux==len(self.cost_history)-1:
                self.best_iter_cost = -1
            else:
                self.best_iter_cost = max(0, (aux-1)*self.iters_check)

        
        if measure_efficiency:  # mediciones de tiempo
            self.__print_time_analytics()

        return result
    def __gradient_descent(self, t, p, compute_cost=True, measure_efficiency=False):
        y = self.init_embed
        
        #====dist_embed==========================================================================================================================================
        if measure_efficiency:  # mediciones de tiempo
            t_0 = time.time_ns()
        dist_embed = similarities.pairwise_euclidean_distance(y)
        if measure_efficiency:  # mediciones de tiempo
            aux = (time.time_ns()-t_0)*1e-9
            self.t_diff_dist_embed.append(aux); self.t_diff_dist_embed.append(aux)
        
        #====q===================================================================================================================================================
        if measure_efficiency:  # mediciones de tiempo
            t_0 = time.time_ns()
        q = similarities.joint_probabilities_student(dist_embed)
        if measure_efficiency:  # mediciones de tiempo
            aux = (time.time_ns()-t_0)*1e-9
            self.t_diff_q.append(aux); self.t_diff_q.append(aux)
        
        self.prev_embed_dist = dist_embed
        self.embed_history.append(y); self.embed_history.append(y)
        self.embed_dist_history.append(dist_embed); self.embed_dist_history.append(dist_embed)
        self.q_history.append(q); self.q_history.append(q)

        if compute_cost:
            cost = kl_divergence(p, q)
            self.cost_history.append(cost); self.cost_history.append(cost)
            self.cost_history_embed.append(y); self.cost_history_embed.append(y)

        lr = self.learning_rate
        early = self.early_exaggeration
        momentum = self.momentum_params[1]

        for i in range(2,t):
            #====grad================================================================================================================================================
            if measure_efficiency:  # mediciones de tiempo
                t_0 = time.time_ns()
            
            grad = gradient(p, q*early, y, dist_embed, caso="safe")

            if measure_efficiency:  # mediciones de tiempo
                self.t_diff_grad.append((time.time_ns()-t_0)*1e-9)
            
            #====embed===============================================================================================================================================
            #y{i} = y{i-1} + learning_rate*gradiente + momentum(t) * (y{i-1} - y{i-2})
            y = self.embed_history[-1] - lr*grad + momentum*(self.embed_history[-1]-self.embed_history[-2])

            #====momentum change=====================================================================================================================================
            if i==self.momentum_params[0]:
                early = 1
                momentum = self.momentum_params[2]

            #====dist_embed==========================================================================================================================================
            if measure_efficiency:  # mediciones de tiempo
                t_0 = time.time_ns()
            dist_embed = similarities.pairwise_euclidean_distance(y)
            if measure_efficiency:  # mediciones de tiempo
                self.t_diff_dist_embed.append((time.time_ns()-t_0)*1e-9)
            
            
            #====q===================================================================================================================================================
            if measure_efficiency:  # mediciones de tiempo
                t_0 = time.time_ns()
            q = similarities.joint_probabilities_student(dist_embed)
            if measure_efficiency:  # mediciones de tiempo
                self.t_diff_q.append((time.time_ns()-t_0)*1e-9)
            
            
            self.embed_history.append(y)
            self.prev_embed_dist = dist_embed
            self.embed_dist_history.append(dist_embed)
            self.q_history.append(q)

            if compute_cost:
                if i%self.iters_check==0 or i==t-1:
                    cost = kl_divergence(p, q)
                    self.cost_history.append(cost)
                    self.cost_history_embed.append(y)
                

    def display_embed(self, *, display_best_iter_cost=False, t:int=-1, title=None):
        """Displays the resulting embedding.

        Parameters
        ----------
        display_best_iter: bool, Optional.
            Whether or not to display the iteration with the lowest cost.
            If True, the "t" parameter is ignored
        t: int, Optional.
            The embedding iteration to display.
        """
        assert self.fitting_done
        if display_best_iter_cost:
            t = self.best_iter_cost
        elif t not in range(-1,self.max_iter):
            raise ValueError("Cannot show embedding for values of t that are not within the range [-1, {})=".format(self.max_iter))
        
        embed = np.array(self.embed_history[t])
        embed_T = embed.T

        if self.element_classes is not None:
            labels = self.element_classes.astype(str)
        match self.n_dimensions:
            case 3:
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
            case 2:
                x = embed_T[0]
                y = embed_T[1]

                if self.element_classes is not None:
                    for i in range(len(x)):
                        plt.plot(x[i],y[i],marker='o',linestyle='', markersize=5, label=labels[i])
                    handles, labels = plt.gca().get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    plt.legend(by_label.values(), by_label.keys(), draggable=True)
                else:
                    for i in range(len(x)):
                        plt.plot(x[i],y[i],marker='o',linestyle='', markersize=8)
                print("")
            case 1:
                x = embed
                y = np.ones_like(x)

                if self.element_classes is not None:
                    for i in range(len(x)):
                        plt.plot(x[i],y[i],marker='o',linestyle='', markersize=5, label=labels[i])
                    handles, labels = plt.gca().get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    plt.legend(by_label.values(), by_label.keys(), draggable=True)
                else:
                    for i in range(0,x.shape[0]):
                        plt.plot(x[i],y[i],marker='o',linestyle='', markersize=8)
            case _:
                raise ValueError("Display of embedding not available for more than 3 dimensions. I am limited by the technology of my time")
            
        if t==-1:
            t = len(self.embed_history)

        if title is None:
            if display_best_iter_cost:
                title = "Best embedding for kl divergence, achieved at t={} out of {} iterations".format(t, self.max_iter)
            else:
                title = "Embedding at t={} out of {} iterations".format(t, self.max_iter)
        plt.title(title)
        plt.show()
    
    #===Devolver embeddings especificos==============================================================================
    def get_final_embedding(self) -> np.ndarray:
        assert self.fitting_done
        return self.embed_history[-1]
    def get_best_embedding_cost(self) -> np.ndarray:
        assert self.fitting_done
        return self.embed_history[self.best_iter_cost]
    
    #=============Analisis de eficiencia de cada metodo===================================================================================================
    def __print_time_analytics(self):
        print("======================================================================")
    
        # t_diff_distancias_og = None
        self.__print_time_diff(self.t_diff_distancias_og, "pairwise_euclidean_distances(data)")
    
        # t_diff_p = None
        self.__print_time_diff(self.t_diff_p, "joint_probabilities_gaussian")
    
        # t_diff_dist_embed = []
        avg_t_diff_dist_embed = self.__compute_average_time(np.array(self.t_diff_dist_embed))
        self.__print_time_diff(avg_t_diff_dist_embed, "pairwise_euclidean_distances(embed)")
    
        # t_diff_q = []
        avg_t_diff_q = self.__compute_average_time(np.array(self.t_diff_q))
        self.__print_time_diff(avg_t_diff_q, "joint_probabilities_student")
    
        # t_diff_grad = []
        avg_t_diff_grad = self.__compute_average_time(np.array(self.t_diff_grad))
        self.__print_time_diff(avg_t_diff_grad, "gradient")
        print("======================================================================")
    def __compute_average_time(self, time_list):
        return np.sum(time_list)/len(time_list)
    def __print_time_diff(self, t_diff, method):
        print("{}: {}".format(method, str(t_diff)))
    